"""
models/sign/model.py — Sign Language VAE (sign10_vel 전용)

기존 SALAD의 models/vae/model.py 를 건드리지 않고,
sign10_vel 210D에 특화된 VAE를 별도 파일로 구성.

구조:
  [B,T,210] → Sign10VelEncoder → [B,T,10,D]
            → STConvEncoder (SALAD 재사용, sign10 adj) → pooled latent
            → MultiLinear → μ, logσ² → z
            → STConvDecoder (SALAD 재사용) → [B,T,10,D]
            → Sign10VelDecoder → [B,T,210]

STConvEncoder/Decoder는 sign10 adjacency를 자동으로 사용
(pool.py에서 dataset='sign', skeleton_mode='sign10_vel' 분기).
"""

import torch
import torch.nn as nn

from models.skeleton.linear import MultiLinear
from models.vae.encdec import STConvEncoder, STConvDecoder
from models.sign.encdec import Sign10VelEncoder, Sign10VelDecoder


class SignVAE(nn.Module):
    """VAE for sign10_vel (210D) representation.

    Unlike the original SALAD VAE which handles t2m/kit/sign via branching,
    this is a dedicated model for sign10_vel with dual-stream encoder.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        # ── Motion Encoder/Decoder (sign10_vel specific) ──
        self.motion_enc = Sign10VelEncoder(opt)
        self.motion_dec = Sign10VelDecoder(opt)

        # ── Spatio-Temporal Conv (SALAD 재사용) ──
        # STConvEncoder는 opt.dataset_name='sign', opt.skeleton_mode='sign10_vel'로
        # sign10 adjacency를 자동 사용
        self.conv_enc = STConvEncoder(opt)
        self.conv_dec = STConvDecoder(opt, self.conv_enc)

        # ── Distribution layer ──
        num_parts = 10  # sign10 always has 10 super-joints
        self.dist = MultiLinear(opt.latent_dim, opt.latent_dim * 2, num_parts)

    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def encode(self, x):
        """
        Args:
            x: [B, T, 210]
        Returns:
            z: [B, T', 10, D]  (T' = T / 2^n_layers due to temporal pooling)
            loss_dict: {'loss_kl': scalar}
        """
        x = self.motion_enc(x)     # [B, T, 10, D]
        x = self.conv_enc(x)       # [B, T', J', D]  (pooled)
        x = self.dist(x)           # [B, T', J', 2D]
        mu, logvar = x.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        loss_kl = 0.5 * torch.mean(torch.pow(mu, 2) + torch.exp(logvar) - logvar - 1.0)
        return z, {"loss_kl": loss_kl}

    def decode(self, z):
        """
        Args:
            z: [B, T', J', D]
        Returns:
            [B, T, 210]
        """
        x = self.conv_dec(z)       # [B, T, 10, D]
        x = self.motion_dec(x)     # [B, T, 210]
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Args:
            x: [B, T, 210]
        Returns:
            recon: [B, T, 210]
            loss_dict: {'loss_kl': scalar}
        """
        x = x.detach().float()
        z, loss_dict = self.encode(x)
        recon = self.decode(z)
        return recon, loss_dict
