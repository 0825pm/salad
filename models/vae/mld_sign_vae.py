"""
Part-aware MLD-VAE for sign language (120D → 13 part tokens)

Architecture (방안 E):
  Encoder:
    [B, T, 120] → Part Split (13 groups) → Per-part Linear + Part Emb
    → [B, T, 13, D] → Temporal Attention Pool → [B, 13, D]
    → Part Self-Attention (SkipTransformerEncoder) → μ, σ → z [B, 13, D]

  Decoder (MLD-VAE style):
    z [13, B, D] (memory) + Temporal queries [T, B, D] + PE
    → SkipTransformerDecoder (cross-attend to 13 part tokens)
    → Linear(D, 120) → [B, T, 120]

Based on MotionGPT3 MLD-VAE with part-aware modifications.
"""

import copy
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# Part definitions for 120D SMPL-X (jaw/expression removed)
# =============================================================================

PART_SPLITS_120D = [
    ('root',          0,    3),   #  3D
    ('spine_chest',   3,   15),   # 12D
    ('neck_shoulder', 15,  30),   # 15D
    ('l_thumb',       30,  39),   #  9D
    ('l_index',       39,  48),   #  9D
    ('l_middle',      48,  57),   #  9D
    ('l_ring',        57,  66),   #  9D
    ('l_pinky',       66,  75),   #  9D
    ('r_thumb',       75,  84),   #  9D
    ('r_index',       84,  93),   #  9D
    ('r_middle',      93, 102),   #  9D
    ('r_ring',       102, 111),   #  9D
    ('r_pinky',      111, 120),   #  9D
]

NUM_PARTS = len(PART_SPLITS_120D)  # 13
HAND_PART_INDICES = list(range(3, 13))  # 10 finger parts


# =============================================================================
# Utility
# =============================================================================

def lengths_to_mask(lengths, device, max_len=None):
    if max_len is None:
        max_len = max(lengths)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len) < torch.tensor(lengths, device=device).unsqueeze(1)
    return mask  # [B, T], True = valid


# =============================================================================
# Positional Encoding (learned, from MLD-VAE)
# =============================================================================

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """x: [seq_len, batch, d_model]"""
        seq_len = x.shape[0]
        positions = torch.arange(seq_len, device=x.device)
        x = x + self.pe(positions).unsqueeze(1)
        return self.dropout(x)


# =============================================================================
# Temporal Attention Pooling
# =============================================================================

class TemporalAttentionPool(nn.Module):
    """Per-part temporal pooling via cross-attention with a learnable query."""
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        x:    [B*J, T, D]
        mask: [B*J, T] (True = valid)
        Returns: [B*J, D]
        """
        q = self.query.expand(x.size(0), -1, -1)  # [B*J, 1, D]
        key_padding_mask = ~mask if mask is not None else None
        out, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask)
        return self.norm(out.squeeze(1))  # [B*J, D]


# =============================================================================
# SkipTransformerEncoder (from MotionGPT3, simplified)
# =============================================================================

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class SkipTransformerEncoder(nn.Module):
    """U-Net style skip-connection encoder. num_layers must be odd."""
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.d_model = encoder_layer.d_model if hasattr(encoder_layer, 'd_model') else encoder_layer.self_attn.embed_dim
        self.num_layers = num_layers
        self.norm = norm
        assert num_layers % 2 == 1, f"num_layers must be odd, got {num_layers}"

        n_block = (num_layers - 1) // 2
        self.input_blocks = _get_clones(encoder_layer, n_block)
        self.middle_block = copy.deepcopy(encoder_layer)
        self.output_blocks = _get_clones(encoder_layer, n_block)
        self.linear_blocks = _get_clones(nn.Linear(2 * self.d_model, self.d_model), n_block)

    def forward(self, src, src_key_padding_mask=None):
        x = src
        xs = []
        for block in self.input_blocks:
            x = block(x, src_key_padding_mask=src_key_padding_mask)
            xs.append(x)
        x = self.middle_block(x, src_key_padding_mask=src_key_padding_mask)
        for block, linear in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = block(x, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x


# =============================================================================
# SkipTransformerDecoder (from MotionGPT3, simplified)
# =============================================================================

class SkipTransformerDecoder(nn.Module):
    """U-Net style skip-connection decoder with cross-attention. num_layers must be odd."""
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.d_model = decoder_layer.d_model if hasattr(decoder_layer, 'd_model') else decoder_layer.self_attn.embed_dim
        self.num_layers = num_layers
        self.norm = norm
        assert num_layers % 2 == 1, f"num_layers must be odd, got {num_layers}"

        n_block = (num_layers - 1) // 2
        self.input_blocks = _get_clones(decoder_layer, n_block)
        self.middle_block = copy.deepcopy(decoder_layer)
        self.output_blocks = _get_clones(decoder_layer, n_block)
        self.linear_blocks = _get_clones(nn.Linear(2 * self.d_model, self.d_model), n_block)

    def forward(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = tgt
        xs = []
        for block in self.input_blocks:
            x = block(x, memory,
                      tgt_key_padding_mask=tgt_key_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask)
            xs.append(x)
        x = self.middle_block(x, memory,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        for block, linear in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = block(x, memory,
                      tgt_key_padding_mask=tgt_key_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x


# =============================================================================
# Transformer layers (with d_model attribute for SkipTransformer)
# =============================================================================

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation='gelu'):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()

    def forward(self, src, src_key_padding_mask=None):
        # Pre-norm
        x = self.norm1(src)
        x, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(x)
        x = self.norm2(src)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        src = src + self.dropout2(x)
        return src


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation='gelu'):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()

    def forward(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Self-attention (pre-norm)
        x = self.norm1(tgt)
        x, _ = self.self_attn(x, x, x, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(x)
        # Cross-attention
        x = self.norm2(tgt)
        x, _ = self.cross_attn(x, memory, memory, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(x)
        # FFN
        x = self.norm3(tgt)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        tgt = tgt + self.dropout3(x)
        return tgt


# =============================================================================
# MldSignVae — Part-aware MLD-VAE
# =============================================================================

class MldSignVae(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.latent_dim = getattr(opt, 'mld_latent_dim', 256)
        self.num_parts = NUM_PARTS  # 13
        self.pose_dim = getattr(opt, 'pose_dim', 120)
        ff_size = getattr(opt, 'mld_ff_size', 1024)
        n_heads = getattr(opt, 'mld_n_heads', 4)
        enc_layers = getattr(opt, 'mld_enc_layers', 3)  # odd, for 13 tokens
        dec_layers = getattr(opt, 'mld_dec_layers', 9)   # odd, for temporal reconstruction
        dropout = getattr(opt, 'dropout', 0.1)
        self.predict_length = getattr(opt, 'predict_length', False)
        self._max_motion_length = getattr(opt, 'max_motion_length', 400)

        D = self.latent_dim

        # ── Per-part input projections ──
        self.part_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(end - start, D),
                nn.GELU(),
                nn.Linear(D, D),
            )
            for _, start, end in PART_SPLITS_120D
        ])

        # ── Part embedding (learnable, not positional) ──
        self.part_embedding = nn.Parameter(torch.randn(self.num_parts, D) * 0.02)

        # ── Temporal attention pooling ──
        self.temporal_pool = TemporalAttentionPool(D, n_heads, dropout)

        # ── Part self-attention encoder ──
        enc_layer = EncoderLayer(D, n_heads, ff_size, dropout)
        enc_norm = nn.LayerNorm(D)
        self.part_encoder = SkipTransformerEncoder(enc_layer, enc_layers, enc_norm)

        # ── Distribution ──
        self.dist_layer = nn.Linear(D, 2 * D)

        # ── Decoder ──
        self.query_pe = LearnedPositionalEncoding(D, max_len=1000, dropout=dropout)
        dec_layer = DecoderLayer(D, n_heads, ff_size, dropout)
        dec_norm = nn.LayerNorm(D)
        self.decoder = SkipTransformerDecoder(dec_layer, dec_layers, dec_norm)

        # ── Output projection ──
        self.output_layer = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, self.pose_dim),
        )

        # ── Length predictor (optional): z [B, 13, D] → scalar ──
        if self.predict_length:
            self.length_predictor = nn.Sequential(
                nn.Linear(D, D // 2),
                nn.GELU(),
                nn.Linear(D // 2, 1),
            )  # applied to mean-pooled z → [B, 1]

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    # ─────────────────────────────────────────────
    # Encoder
    # ─────────────────────────────────────────────

    def encode(self, features, lengths=None):
        """
        features: [B, T, 120]
        lengths:  list of int (valid frames per sample) or None (all T)
        Returns:  z [B, 13, D], loss_dict
        """
        B, T, _ = features.shape
        if lengths is None:
            lengths = [T] * B

        # ── Part split + projection ──
        parts = []
        for i, (name, start, end) in enumerate(PART_SPLITS_120D):
            p = features[:, :, start:end]          # [B, T, d_i]
            p = self.part_projections[i](p)        # [B, T, D]
            p = p + self.part_embedding[i]         # broadcast part embedding
            parts.append(p)

        x = torch.stack(parts, dim=2)             # [B, T, 13, D]

        # ── Temporal attention pooling (per part) ──
        J, D = self.num_parts, self.latent_dim
        x = x.permute(0, 2, 1, 3).reshape(B * J, T, D)   # [B*13, T, D]

        # Build mask for pooling
        mask = lengths_to_mask(lengths, features.device, max_len=T)  # [B, T]
        mask = mask.unsqueeze(1).expand(-1, J, -1).reshape(B * J, T)  # [B*13, T]

        x = self.temporal_pool(x, mask)            # [B*13, D]
        x = x.reshape(B, J, D)                    # [B, 13, D]

        # ── Part self-attention ──
        x = self.part_encoder(x)                   # [B, 13, D]

        # ── Distribution ──
        dist = self.dist_layer(x)                  # [B, 13, 2D]
        mu, logvar = dist.chunk(2, dim=-1)         # [B, 13, D] each

        # Reparameterize
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)       # [B, 13, D]

        loss_kl = 0.5 * torch.mean(mu.pow(2) + logvar.exp() - logvar - 1.0)

        return z, {"loss_kl": loss_kl}

    # ─────────────────────────────────────────────
    # Decoder
    # ─────────────────────────────────────────────

    def decode(self, z, lengths):
        """
        z:       [B, 13, D]
        lengths: list of int (output T per sample)
        Returns: [B, T, 120]
        """
        B = z.shape[0]
        T = max(lengths)

        # Temporal queries (zeros + learned PE)
        # Decoder is batch_first, so queries: [B, T, D]
        queries = torch.zeros(B, T, self.latent_dim, device=z.device)
        # PE expects [seq, batch, D] format, reshape temporarily
        queries_t = queries.permute(1, 0, 2)       # [T, B, D]
        queries_t = self.query_pe(queries_t)        # [T, B, D]
        queries = queries_t.permute(1, 0, 2)       # [B, T, D]

        # Mask for variable-length outputs
        mask = lengths_to_mask(lengths, z.device, max_len=T)  # [B, T]

        # Cross-attend: queries → z (memory)
        output = self.decoder(
            tgt=queries,                            # [B, T, D]
            memory=z,                               # [B, 13, D]
            tgt_key_padding_mask=~mask,             # [B, T]
        )                                           # [B, T, D]

        output = self.output_layer(output)          # [B, T, 120]
        return output

    # ─────────────────────────────────────────────
    # Length prediction
    # ─────────────────────────────────────────────

    def predict_len(self, z):
        """z: [B, 13, D] → predicted normalized length [B] in (0, 1)"""
        pooled = z.mean(dim=1)                     # [B, D]
        return self.length_predictor(pooled).squeeze(-1).sigmoid()  # [B], 0~1

    # ─────────────────────────────────────────────
    # Forward (compatible with VAETrainer interface)
    # ─────────────────────────────────────────────

    def forward(self, x, lengths=None):
        """
        x: [B, T, 120] (normalized)
        Returns: (recon [B, T, 120], loss_dict)
        """
        x = x.detach().float()
        B, T, _ = x.shape
        if lengths is None:
            lengths = [T] * B

        z, loss_dict = self.encode(x, lengths)
        recon = self.decode(z, lengths)

        if self.predict_length:
            pred_lengths = self.predict_len(z)      # [B], 0~1 (sigmoid)
            max_len = getattr(self, '_max_motion_length', max(lengths))
            gt_lengths = torch.tensor(lengths, device=x.device, dtype=torch.float32) / max_len  # 0~1
            loss_dict["loss_length"] = F.l1_loss(pred_lengths, gt_lengths)

        return recon, loss_dict