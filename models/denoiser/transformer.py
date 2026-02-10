import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from models.skeleton.conv import STConv

# def _get_core_adj_matrix():
#     out = torch.zeros(7, 7, dtype=torch.float32)
#     out[0, [1, 2, 3]] = 1
#     out[1, 0] = 1
#     out[2, 0] = 1
#     out[3, [0, 4, 5, 6]] = 1
#     out[4, 3] = 1
#     out[5, 3] = 1
#     out[6, 3] = 1
#     return out

def _get_core_edges():
    return [(0, 1), (0, 2), (0, 3), (3, 4), (3, 5), (3, 6)]

def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return x * (scale + 1) + shift


class DenseFiLM(nn.Module):
    def __init__(self, opt):
        super(DenseFiLM, self).__init__()
        self.linear = nn.Sequential(
            nn.SiLU(),
            nn.Linear(opt.latent_dim, opt.latent_dim * 2),
        )

    def forward(self, cond):
        """
        cond: [B, D]
        """
        cond = self.linear(cond)
        cond = cond[:, None, None, :] # unsqueeze for skeleto-temporal dimensions
        scale, shift = cond.chunk(2, dim=-1)
        return scale, shift


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, batch_first=True):
        super(MultiheadAttention, self).__init__()
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, average_attn_weights=False):
        """
        query: [B, T1, D]
        key: [B, T2, D]
        value: [B, T2, D]
        key_padding_mask: [B, T2]
        """
        B, T1, D = query.size()
        _, T2, _ = key.size()

        # linear transformation
        query = self.Wq(query).view(B, T1, self.n_heads, self.head_dim).transpose(1, 2)
        key = self.Wk(key).view(B, T2, self.n_heads, self.head_dim).transpose(1, 2)
        value = self.Wv(value).view(B, T2, self.n_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(key_padding_mask[:, None, None, :], -1e9)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)

        # concat heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T1, D)

        # linear transformation
        attn_output = self.Wo(attn_output)

        if need_weights:
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)
            return attn_output, attn_weights
        else:
            return attn_output, None
    
    def forward_with_fixed_attn_weights(self, attn_weights, value):
        """
        Assume that the attention weights are already computed.
        """
        B, H, _, T2 = attn_weights.size()
        D = value.size(-1)

        # linear transformation
        value = self.Wv(value).view(B, T2, self.n_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention
        attn_output = torch.matmul(attn_weights, value)

        # concat heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, -1, D)

        # linear transformation
        attn_output = self.Wo(attn_output)

        return attn_output, attn_weights


# ═══════════════════════════════════════════════════════════════
#  LIMM: Local Information Modeling Module (Light-T2M, AAAI 2025)
#  Replaces Temporal Attention with lightweight 1D convolutions
# ═══════════════════════════════════════════════════════════════

class LIMM(nn.Module):
    """
    Pointwise Conv → Depthwise Conv (local kernel) → Pointwise Conv + residual.
    Operates on the temporal axis: input [B, T, D] → output [B, T, D].
    """
    def __init__(self, dim, kernel_size=7, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Conv1d(dim, dim, 1)
        self.dw  = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.pw2 = nn.Conv1d(dim, dim, 1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """x: [B, T, D]"""
        res = x
        x = self.norm(x).transpose(1, 2)   # [B, D, T]
        x = self.act(self.pw1(x))
        x = self.act(self.dw(x))
        x = self.pw2(x).transpose(1, 2)    # [B, T, D]
        x = self.drop(x)
        return x + res


# ═══════════════════════════════════════════════════════════════
#  ATII: Adaptive Textual Information Injector (Light-T2M, AAAI 2025)
#  Replaces Cross Attention with channel-wise gating
# ═══════════════════════════════════════════════════════════════

class ATII(nn.Module):
    """
    text_emb → mean pool → MLP → sigmoid → channel gating on motion features.
    Input: motion [B, T*J, D], text [B, N, D]  →  Output: [B, T*J, D]
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm_motion = nn.LayerNorm(dim)
        self.norm_text = nn.LayerNorm(dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.gate_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.drop = nn.Dropout(dropout)

    def forward(self, motion, text, text_mask=None):
        """
        motion: [B, L, D]  (L = T*J)
        text:   [B, N, D]
        text_mask: [B, N] (True = padding → ignore)
        """
        text_normed = self.norm_text(text)

        # mean pool over valid text tokens
        if text_mask is not None:
            valid = (~text_mask).unsqueeze(-1).float()  # [B, N, 1]
            text_pooled = (text_normed * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            text_pooled = text_normed.mean(dim=1)  # [B, D]

        # channel gate
        gate = torch.sigmoid(self.gate_mlp(text_pooled))  # [B, D]
        gate = gate.unsqueeze(1)  # [B, 1, D]

        motion_normed = self.norm_motion(motion)
        out = motion_normed * gate * self.gate_scale
        out = self.drop(out)
        return out


# ═══════════════════════════════════════════════════════════════
#  STTransformerLayer: Original full-attention block
# ═══════════════════════════════════════════════════════════════

class STTransformerLayer(nn.Module):
    """
    Setting
        - Normalization first
    """
    def __init__(self, opt):
        super(STTransformerLayer, self).__init__()
        self.opt = opt
        
        # skeletal attention
        self.skel_attn = MultiheadAttention(opt.latent_dim, opt.n_heads, opt.dropout, batch_first=True)
        self.skel_norm = nn.LayerNorm(opt.latent_dim)
        self.skel_dropout = nn.Dropout(opt.dropout)

        # temporal attention
        self.temp_attn = MultiheadAttention(opt.latent_dim, opt.n_heads, opt.dropout, batch_first=True)
        self.temp_norm = nn.LayerNorm(opt.latent_dim)
        self.temp_dropout = nn.Dropout(opt.dropout)

        # cross attention
        self.cross_attn = MultiheadAttention(opt.latent_dim, opt.n_heads, opt.dropout, batch_first=True)
        self.cross_src_norm = nn.LayerNorm(opt.latent_dim)
        self.cross_tgt_norm = nn.LayerNorm(opt.latent_dim)
        self.cross_dropout = nn.Dropout(opt.dropout)

        # ffn
        self.ffn_linear1 = nn.Linear(opt.latent_dim, opt.ff_dim)
        self.ffn_linear2 = nn.Linear(opt.ff_dim, opt.latent_dim)
        self.ffn_norm = nn.LayerNorm(opt.latent_dim)
        self.ffn_dropout = nn.Dropout(opt.dropout)

        # activation
        self.act = F.relu if opt.activation == "relu" else F.gelu

        # FiLM
        self.skel_film = DenseFiLM(opt)
        self.temp_film = DenseFiLM(opt)
        self.cross_film = DenseFiLM(opt)
        self.ffn_film = DenseFiLM(opt)

    def _sa_block(self, x, fixed_attn=None):
        x = self.skel_norm(x)
        if fixed_attn is None:
            x, attn = self.skel_attn.forward(x, x, x, need_weights=True, average_attn_weights=False)
        else:
            x, attn = self.skel_attn.forward_with_fixed_attn_weights(fixed_attn, x)
        x = self.skel_dropout(x)
        return x, attn

    def _ta_block(self, x, mask=None, fixed_attn=None):
        x = self.temp_norm(x)
        if fixed_attn is None:
            x, attn = self.temp_attn.forward(x, x, x, key_padding_mask=mask, need_weights=True, average_attn_weights=False)
        else:
            x, attn = self.temp_attn.forward_with_fixed_attn_weights(fixed_attn, x)
        x = self.temp_dropout(x)
        return x, attn

    def _ca_block(self, x, mem, mask=None, fixed_attn=None):
        x = self.cross_src_norm(x)
        mem = self.cross_tgt_norm(mem)
        if fixed_attn is None:
            x, attn = self.cross_attn.forward(x, mem, mem, key_padding_mask=mask, need_weights=True, average_attn_weights=False)
        else:
            x, attn = self.cross_attn.forward_with_fixed_attn_weights(fixed_attn, mem)
        x = self.cross_dropout(x)
        return x, attn
    
    def _ff_block(self, x):
        x = self.ffn_norm(x)
        x = self.ffn_linear1(x)
        x = self.act(x)
        x = self.ffn_linear2(x)
        x = self.ffn_dropout(x)
        return x
    
    def forward(self, x, memory, cond, x_mask=None, memory_mask=None,
                skel_attn=None, temp_attn=None, cross_attn=None):

        B, T, J, D = x.size()

        # diffusion timestep embedding
        skel_cond = self.skel_film(cond)
        temp_cond = self.temp_film(cond)
        cross_cond = self.cross_film(cond)
        ffn_cond = self.ffn_film(cond)

        # temporal attention
        ta_out, ta_weight = self._ta_block(x.transpose(1, 2).reshape(B * J, T, D),
                                            mask=x_mask,
                                            fixed_attn=temp_attn)
        ta_out = ta_out.reshape(B, J, T, D).transpose(1, 2)
        ta_out = featurewise_affine(ta_out, temp_cond)
        x = x + ta_out

        # skeletal attention
        sa_out, sa_weight = self._sa_block(x.reshape(B * T, J, D),
                                            fixed_attn=skel_attn)
        sa_out = sa_out.reshape(B, T, J, D)
        sa_out = featurewise_affine(sa_out, skel_cond)
        x = x + sa_out
    
        # cross attention
        ca_out, ca_weight = self._ca_block(x.reshape(B, T * J, D),
                                        memory,
                                        mask=memory_mask,
                                        fixed_attn=cross_attn)
        ca_out = ca_out.reshape(B, T, J, D)
        ca_out = featurewise_affine(ca_out, cross_cond)
        x = x + ca_out

        # feed-forward
        ff_out = self._ff_block(x)
        ff_out = featurewise_affine(ff_out, ffn_cond)
        x = x + ff_out

        attn_weights = (sa_weight, ta_weight, ca_weight)

        return x, attn_weights


# ═══════════════════════════════════════════════════════════════
#  STTransformerLayerLight: LIMM temporal + ATII text injection
#  Same forward signature as STTransformerLayer for drop-in use
# ═══════════════════════════════════════════════════════════════

class STTransformerLayerLight(nn.Module):
    """
    Lightweight variant of STTransformerLayer:
      - Temporal Attention → LIMM (1D conv, local)     [if use_limm]
      - Cross Attention    → ATII (channel gating)     [if use_atii]
      - Skeletal Attention → kept (only 7 joints, already cheap)
      - FFN                → kept
    """
    def __init__(self, opt, use_limm=True, use_atii=True):
        super().__init__()
        self.opt = opt
        self.use_limm = use_limm
        self.use_atii = use_atii

        # ── Skeletal attention (always full — 7 joints is tiny) ──
        self.skel_attn = MultiheadAttention(opt.latent_dim, opt.n_heads, opt.dropout, batch_first=True)
        self.skel_norm = nn.LayerNorm(opt.latent_dim)
        self.skel_dropout = nn.Dropout(opt.dropout)

        # ── Temporal: LIMM or full attention ──
        if use_limm:
            self.limm = LIMM(opt.latent_dim, kernel_size=7, dropout=opt.dropout)
        else:
            self.temp_attn = MultiheadAttention(opt.latent_dim, opt.n_heads, opt.dropout, batch_first=True)
            self.temp_norm = nn.LayerNorm(opt.latent_dim)
            self.temp_dropout = nn.Dropout(opt.dropout)

        # ── Text injection: ATII or full cross attention ──
        if use_atii:
            self.atii = ATII(opt.latent_dim, dropout=opt.dropout)
        else:
            self.cross_attn = MultiheadAttention(opt.latent_dim, opt.n_heads, opt.dropout, batch_first=True)
            self.cross_src_norm = nn.LayerNorm(opt.latent_dim)
            self.cross_tgt_norm = nn.LayerNorm(opt.latent_dim)
            self.cross_dropout = nn.Dropout(opt.dropout)

        # ── FFN (always full) ──
        self.ffn_linear1 = nn.Linear(opt.latent_dim, opt.ff_dim)
        self.ffn_linear2 = nn.Linear(opt.ff_dim, opt.latent_dim)
        self.ffn_norm = nn.LayerNorm(opt.latent_dim)
        self.ffn_dropout = nn.Dropout(opt.dropout)
        self.act = F.relu if opt.activation == "relu" else F.gelu

        # ── FiLM (4 modulations, same structure) ──
        self.skel_film = DenseFiLM(opt)
        self.temp_film = DenseFiLM(opt)
        self.cross_film = DenseFiLM(opt)
        self.ffn_film = DenseFiLM(opt)

    def _sa_block(self, x, fixed_attn=None):
        x = self.skel_norm(x)
        if fixed_attn is None:
            x, attn = self.skel_attn.forward(x, x, x, need_weights=True, average_attn_weights=False)
        else:
            x, attn = self.skel_attn.forward_with_fixed_attn_weights(fixed_attn, x)
        x = self.skel_dropout(x)
        return x, attn

    def _ff_block(self, x):
        x = self.ffn_norm(x)
        x = self.ffn_linear1(x)
        x = self.act(x)
        x = self.ffn_linear2(x)
        x = self.ffn_dropout(x)
        return x

    def forward(self, x, memory, cond, x_mask=None, memory_mask=None,
                skel_attn=None, temp_attn=None, cross_attn=None):
        """Same signature as STTransformerLayer.forward for drop-in compatibility."""

        B, T, J, D = x.size()

        # FiLM conditions from diffusion timestep
        skel_cond = self.skel_film(cond)
        temp_cond = self.temp_film(cond)
        cross_cond = self.cross_film(cond)
        ffn_cond = self.ffn_film(cond)

        # ── Temporal ──
        if self.use_limm:
            # LIMM: per-joint 1D conv over time axis (LIMM has internal residual)
            x_temp = x.transpose(1, 2).reshape(B * J, T, D)  # [B*J, T, D]
            ta_out = self.limm(x_temp)                         # [B*J, T, D] with residual
            ta_out = ta_out.reshape(B, J, T, D).transpose(1, 2)  # [B, T, J, D]
            delta = ta_out - x
            delta = featurewise_affine(delta, temp_cond)
            x = x + delta
            ta_weight = None
        else:
            ta_out = self.temp_norm(x.transpose(1, 2).reshape(B * J, T, D))
            if temp_attn is None:
                ta_out, ta_weight = self.temp_attn.forward(ta_out, ta_out, ta_out, key_padding_mask=x_mask, need_weights=True, average_attn_weights=False)
            else:
                ta_out, ta_weight = self.temp_attn.forward_with_fixed_attn_weights(temp_attn, ta_out)
            ta_out = self.temp_dropout(ta_out)
            ta_out = ta_out.reshape(B, J, T, D).transpose(1, 2)
            ta_out = featurewise_affine(ta_out, temp_cond)
            x = x + ta_out

        # ── Skeletal attention (always full) ──
        sa_out, sa_weight = self._sa_block(x.reshape(B * T, J, D),
                                            fixed_attn=skel_attn)
        sa_out = sa_out.reshape(B, T, J, D)
        sa_out = featurewise_affine(sa_out, skel_cond)
        x = x + sa_out

        # ── Text injection ──
        if self.use_atii:
            atii_out = self.atii(x.reshape(B, T * J, D), memory, text_mask=memory_mask)
            atii_out = atii_out.reshape(B, T, J, D)
            atii_out = featurewise_affine(atii_out, cross_cond)
            x = x + atii_out
            ca_weight = None
        else:
            x_flat = self.cross_src_norm(x.reshape(B, T * J, D))
            mem = self.cross_tgt_norm(memory)
            if cross_attn is None:
                ca_out, ca_weight = self.cross_attn.forward(x_flat, mem, mem, key_padding_mask=memory_mask, need_weights=True, average_attn_weights=False)
            else:
                ca_out, ca_weight = self.cross_attn.forward_with_fixed_attn_weights(cross_attn, mem)
            ca_out = self.cross_dropout(ca_out)
            ca_out = ca_out.reshape(B, T, J, D)
            ca_out = featurewise_affine(ca_out, cross_cond)
            x = x + ca_out

        # ── FFN ──
        ff_out = self._ff_block(x)
        ff_out = featurewise_affine(ff_out, ffn_cond)
        x = x + ff_out

        attn_weights = (sa_weight, ta_weight, ca_weight)
        return x, attn_weights


# ═══════════════════════════════════════════════════════════════
#  SkipTransformer: U-Net style skip connections
# ═══════════════════════════════════════════════════════════════

class SkipTransformer(nn.Module):
    def __init__(self, opt):
        super(SkipTransformer, self).__init__()
        self.opt = opt
        if self.opt.n_layers % 2 != 1:
            raise ValueError(f"n_layers should be odd for SkipTransformer, but got {self.opt.n_layers}")
        
        use_limm = getattr(opt, 'use_limm', False)
        use_atii = getattr(opt, 'use_atii', False)

        # transformer encoder
        self.input_blocks = nn.ModuleList()
        self.middle_block = STTransformerLayer(opt)  # middle always full
        self.output_blocks = nn.ModuleList()
        self.skip_blocks = nn.ModuleList()

        for i in range((self.opt.n_layers - 1) // 2):
            # input blocks: use Light if LIMM/ATII enabled
            if use_limm or use_atii:
                self.input_blocks.append(STTransformerLayerLight(opt, use_limm=use_limm, use_atii=use_atii))
            else:
                self.input_blocks.append(STTransformerLayer(opt))

            # output blocks: always full attention
            self.output_blocks.append(STTransformerLayer(opt))
            self.skip_blocks.append(nn.Linear(opt.latent_dim * 2, opt.latent_dim))
        
        if use_limm or use_atii:
            n_light = (self.opt.n_layers - 1) // 2
            n_full = self.opt.n_layers - n_light
            print(f"  [SkipTransformer] {n_light} Light blocks (LIMM={use_limm}, ATII={use_atii}) + {n_full} Full blocks")

    def forward(self, x, timestep_emb, word_emb, sa_mask=None, ca_mask=None, need_attn=False,
                fixed_sa=None, fixed_ta=None, fixed_ca=None):
        """
        x: [B, T, J, D]
        timestep_emb: [B, D]
        word_emb: [B, N, D]
        sa_mask: [B, T]
        ca_mask: [B, N]

        fixed_sa: [bsz*nframes, nlayers, nheads, njoints, njoints]
        fixed_ta: [bsz*njoints, nlayers, nheads, nframes, nframes]
        fixed_ca: [bsz, nlayers, nheads, nframes*njoints, dclip]
        """
        # B, T, J, D = x.size()
        
        xs = []

        attn_weights = [[], [], []]
        layer_idx = 0
        for i, block in enumerate(self.input_blocks):
            sa = None if fixed_sa is None else fixed_sa[:, layer_idx]
            ta = None if fixed_ta is None else fixed_ta[:, layer_idx]
            ca = None if fixed_ca is None else fixed_ca[:, layer_idx]

            x, attns = block(x, word_emb, timestep_emb, x_mask=sa_mask, memory_mask=ca_mask,
                             skel_attn=sa, temp_attn=ta, cross_attn=ca)
            xs.append(x)
            for j in range(len(attn_weights)):
                attn_weights[j].append(attns[j])
            layer_idx += 1
        
        sa = None if fixed_sa is None else fixed_sa[:, layer_idx]
        ta = None if fixed_ta is None else fixed_ta[:, layer_idx]
        ca = None if fixed_ca is None else fixed_ca[:, layer_idx]
        x, attns = self.middle_block(x, word_emb, timestep_emb, x_mask=sa_mask, memory_mask=ca_mask,
                                     skel_attn=sa, temp_attn=ta, cross_attn=ca)
        
        for j in range(len(attn_weights)):
            attn_weights[j].append(attns[j])
        layer_idx += 1

        for (block, skip) in zip(self.output_blocks, self.skip_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = skip(x)

            sa = None if fixed_sa is None else fixed_sa[:, layer_idx]
            ta = None if fixed_ta is None else fixed_ta[:, layer_idx]
            ca = None if fixed_ca is None else fixed_ca[:, layer_idx]

            x, attns = block(x, word_emb, timestep_emb, x_mask=sa_mask, memory_mask=ca_mask,
                             skel_attn=sa, temp_attn=ta, cross_attn=ca)
            
            for j in range(len(attn_weights)):
                attn_weights[j].append(attns[j])
            layer_idx += 1

        if need_attn:
            for j in range(len(attn_weights)):
                # Filter out None entries from Light blocks (LIMM/ATII return None)
                valid = [w for w in attn_weights[j] if w is not None]
                attn_weights[j] = torch.stack(valid, dim=1) if valid else None
        else:
            for j in range(len(attn_weights)):
                attn_weights[j] = None

        return x, attn_weights