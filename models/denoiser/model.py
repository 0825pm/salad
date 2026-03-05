"""
This code was inspired by the denoiser implementation in the Motion Latent Diffusion
    - https://github.com/ChenFengYe/motion-latent-diffusion/blob/main/mld/models/architectures/mld_denoiser.py
"""

from typing import List
import torch
import torch.nn as nn

from models.denoiser.clip import build_text_encoder, get_text_encoder_dim
from models.denoiser.embedding import TimestepEmbedding, PositionalEmbedding
from models.denoiser.transformer import SkipTransformer

class InputProcess(nn.Module):
    def __init__(self, opt, in_features):
        super(InputProcess, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, opt.latent_dim),
            nn.ReLU(),
            nn.Linear(opt.latent_dim, opt.latent_dim),
        )

    def forward(self, x):
        return self.layers(x)

class OutputProcess(nn.Module):
    def __init__(self, opt, out_features):
        super(OutputProcess, self).__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(opt.latent_dim),
            nn.Linear(opt.latent_dim, opt.latent_dim),
            nn.ReLU(),
            nn.Linear(opt.latent_dim, out_features),
        )
    
    def forward(self, x):
        return self.layers(x)

class Denoiser(nn.Module):
    def __init__(self, opt, vae_dim):
        super(Denoiser, self).__init__()

        self.opt = opt
        self.latent_dim = opt.latent_dim

        ## ── PATCH: use factory for text encoder (CLIP or XLM-R) ──
        self.clip_dim = get_text_encoder_dim(opt)
        ## ── end PATCH ──

        # input & output process
        self.input_process = InputProcess(opt, vae_dim)
        self.output_process = OutputProcess(opt, vae_dim)
        
        # timestep embedding
        self.timestep_emb = TimestepEmbedding(self.latent_dim)

        ## ── PATCH: build text encoder via factory ──
        self.clip_model = build_text_encoder(opt)
        ## ── end PATCH ──
        self.word_emb = nn.Linear(self.clip_dim, self.latent_dim)
        
        # positional embedding
        self.pos_emb = PositionalEmbedding(self.latent_dim, opt.dropout)

        # transformer
        self.transformer = SkipTransformer(opt)

        # cache for text embedding
        self._cache_word_emb = None
        self._cache_ca_mask = None
        self._cache_tokens_pos = None

        # track fine-tune mode
        self._finetune_text_encoder = getattr(opt, 'finetune_text_encoder', 'none')
    
    @property
    def is_text_encoder_trainable(self):
        return self._finetune_text_encoder != 'none'

    def parameters_without_clip(self):
        """Denoiser-only parameters (excludes text encoder)."""
        return [param for name, param in self.named_parameters() if "clip_model" not in name]

    def text_encoder_trainable_parameters(self):
        """Trainable text encoder parameters (empty if frozen)."""
        if hasattr(self.clip_model, 'trainable_parameters'):
            return self.clip_model.trainable_parameters()
        return []

    def all_trainable_parameters(self):
        """All trainable parameters: denoiser + text encoder (if fine-tuning)."""
        return [p for p in self.parameters() if p.requires_grad]

    def state_dict_without_clip(self):
        """State dict excluding frozen text encoder weights (and cache)."""
        state_dict = self.state_dict()
        remove_weights = [e for e in state_dict.keys() if "clip_model." in e or "_cache_" in e]
        for e in remove_weights:
            del state_dict[e]
        return state_dict

    def text_encoder_state_dict(self):
        """State dict for trainable text encoder parameters only."""
        if hasattr(self.clip_model, 'trainable_state_dict'):
            return self.clip_model.trainable_state_dict()
        return {}

    def load_text_encoder_state_dict(self, state_dict):
        """Load trainable text encoder parameters from checkpoint."""
        if hasattr(self.clip_model, 'load_trainable_state_dict'):
            self.clip_model.load_trainable_state_dict(state_dict)
    
    def remove_clip_cache(self):
        self._cache_word_emb = None
        self._cache_ca_mask = None
        self._cache_tokens_pos = None

    def forward(self, x, timestep_emb, text=None, text_emb=None, len_mask=None, need_attn=False,
                fixed_sa=None, fixed_ta=None, fixed_ca=None, use_cached_clip=False):
        """
        sample: [B, T, J, D]
        timestep: [B,] or [1,]
        text: list of str (raw text) — used when text_emb is None
        text_emb: (word_emb[B,T,D], attn_mask[B,T], token_pos[B]) — precomputed, skip encoder
        """

        # input process
        x = self.input_process(x)
        B, T, J, D = x.size()

        # diffusion timestep embedding
        timestep_emb = self.timestep_emb(timestep_emb).expand(B, D)

        # text embedding — 3 sources: text_emb (precomputed) > clip cache > encode_text
        if text_emb is not None:
            word_emb, ca_mask, token_pos = text_emb
            word_emb = word_emb.to(x.device)
            ca_mask = ca_mask.to(x.device)
            token_pos = token_pos.to(x.device)
            word_emb = self.word_emb(word_emb)
        elif use_cached_clip and all([e is not None for e in [self._cache_word_emb, self._cache_ca_mask, self._cache_tokens_pos]]):
            word_emb = self._cache_word_emb
            ca_mask = self._cache_ca_mask
            token_pos = self._cache_tokens_pos
        else:
            word_emb, ca_mask, token_pos = self.clip_model.encode_text(text)
            word_emb = self.word_emb(word_emb)
            if use_cached_clip:
                self._cache_word_emb = word_emb
                self._cache_ca_mask = ca_mask
                self._cache_tokens_pos = token_pos
        
        # positional embedding
        x = x.reshape(B, T * J, D)
        x = self.pos_emb.forward(x)
        x = x.reshape(B, T, J, D)

        # attention masks
        if len_mask is not None:
            # [B, T] -> [B*J, T]
            len_mask = len_mask.repeat_interleave(J, dim=0)

        # transformer
        x, attn_weights = self.transformer.forward(x, timestep_emb, word_emb,
                                                   sa_mask=None if len_mask is None else ~len_mask,
                                                   ca_mask=~ca_mask,
                                                   need_attn=need_attn,
                                                   fixed_sa=fixed_sa,
                                                   fixed_ta=fixed_ta,
                                                   fixed_ca=fixed_ca)

        # output process
        x = self.output_process(x)

        return x, attn_weights