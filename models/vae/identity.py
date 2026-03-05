"""
IdentityVAE — drop-in VAE replacement for direct diffusion (--no_vae).

기존 VAE:   [B, T, 133] → encode → [B, T//4, 7, 32]   (temporal 4x downsample, 7 parts)
IdentityVAE: [B, T, 135] → encode → [B, T,   45,  3]   (no downsample, 45 joints, xyz)

Skeleton Attention이 45 joints 간의 공간 관계를 학습.
Denoiser(opt, vae_dim=3) → InputProcess: Linear(3 → latent_dim)
"""

import torch
import torch.nn as nn


class IdentityVAE(nn.Module):
    def __init__(self, num_joints, coords_dim=3):
        """
        Args:
            num_joints: number of joints (e.g. 45)
            coords_dim: per-joint dimension (default 3 for xyz)
        """
        super().__init__()
        self.num_joints = num_joints
        self.coords_dim = coords_dim
        self.latent_dim = coords_dim   # Denoiser uses this as vae_dim

    def freeze(self):
        pass

    def encode(self, x):
        """x: [B, T, D] → z: [B, T, J, 3], loss_dict"""
        B, T, D = x.shape
        assert D == self.num_joints * self.coords_dim, \
            f"Expected D={self.num_joints * self.coords_dim}, got {D}"
        z = x.view(B, T, self.num_joints, self.coords_dim)
        return z, {"loss_kl": torch.tensor(0.0, device=x.device)}

    def decode(self, x):
        """x: [B, T, J, 3] → [B, T, D]"""
        B, T, J, C = x.shape
        return x.view(B, T, J * C)

    def forward(self, x):
        z, loss_dict = self.encode(x)
        return self.decode(z), loss_dict

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def parameters(self):
        return iter([])