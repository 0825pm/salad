"""
models/sign/encdec.py — Sign10_vel Encoder/Decoder for VAE

핵심 설계:
  - 각 super-joint는 rotation과 velocity를 별도 스트림으로 처리
  - Body joints (SJ0-3): rotation-only → 단일 MLP
  - Hand joints (SJ4-9): dual-stream + gated fusion
    h_rot = MLP_rot(rot)
    h_vel = MLP_vel(vel)
    gate  = σ(Linear([h_rot; h_vel]))
    h     = h_rot + gate ⊙ h_vel

기존 SALAD 코드와의 관계:
  - STConvEncoder / STConvDecoder 는 models/vae/encdec.py 에서 import하여 재사용
  - SignMotionEncoder / SignMotionDecoder (기존 7part/finger용)는 건드리지 않음
  - 이 파일의 클래스는 sign10_vel 전용
"""

import torch
import torch.nn as nn

from models.skeleton.conv import get_activation
from utils.sign10_config import (
    SIGN_SPLITS_SIGN10_VEL,
    SIGN10_ROT_DIMS,
    SIGN10_VEL_DIMS,
    SIGN10_VEL_MASK,
    HAND_INDICES_SIGN10,
)


# ═══════════════════════════════════════════════════════════════════════
# Building Blocks
# ═══════════════════════════════════════════════════════════════════════

class PartMLP(nn.Module):
    """2-layer MLP for a single super-joint stream."""
    def __init__(self, in_dim, out_dim, activation='gelu'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            get_activation(activation),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GatedFusion(nn.Module):
    """Gated fusion of rotation and velocity streams.

    gate = σ(W · [h_rot; h_vel] + b)
    output = h_rot + gate ⊙ h_vel
    """
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

    def forward(self, h_rot, h_vel):
        gate = self.gate(torch.cat([h_rot, h_vel], dim=-1))
        return h_rot + gate * h_vel


# ═══════════════════════════════════════════════════════════════════════
# Sign10Vel Encoder: 210D → [B, T, 10, D]
# ═══════════════════════════════════════════════════════════════════════

class Sign10VelEncoder(nn.Module):
    """Per-part dual-stream encoder for 210D sign10_vel representation.

    Body joints (no velocity):  rot → MLP → h ∈ R^D
    Hand joints (with velocity): rot,vel → dual MLP → gated fusion → h ∈ R^D

    Output: [B, T, 10, D] — same format as SignMotionEncoder for STConv compatibility.
    """

    def __init__(self, opt):
        super().__init__()
        self.latent_dim = opt.latent_dim
        self.num_parts = 10
        self.total_splits = list(SIGN_SPLITS_SIGN10_VEL)
        self.rot_dims = list(SIGN10_ROT_DIMS)
        self.vel_dims = list(SIGN10_VEL_DIMS)
        self.vel_mask = list(SIGN10_VEL_MASK)

        act = getattr(opt, 'activation', 'gelu')

        # Rotation stream (all 10 joints)
        self.rot_proj = nn.ModuleList()
        for i in range(self.num_parts):
            self.rot_proj.append(PartMLP(self.rot_dims[i], self.latent_dim, act))

        # Velocity stream + gate (hand joints only)
        self.vel_proj = nn.ModuleList()
        self.fusion = nn.ModuleList()
        for i in range(self.num_parts):
            if self.vel_mask[i]:
                self.vel_proj.append(PartMLP(self.vel_dims[i], self.latent_dim, act))
                self.fusion.append(GatedFusion(self.latent_dim))
            else:
                self.vel_proj.append(None)
                self.fusion.append(None)

    def forward(self, x):
        """
        Args:
            x: [B, T, 210]
        Returns:
            [B, T, 10, D]
        """
        parts = torch.split(x, self.total_splits, dim=-1)

        out = []
        for i in range(self.num_parts):
            if self.vel_mask[i]:
                # Split into rotation and velocity
                rot_dim = self.rot_dims[i]
                rot = parts[i][..., :rot_dim]
                vel = parts[i][..., rot_dim:]

                h_rot = self.rot_proj[i](rot)
                h_vel = self.vel_proj[i](vel)
                h = self.fusion[i](h_rot, h_vel)
            else:
                # Rotation only
                h = self.rot_proj[i](parts[i])

            out.append(h)

        return torch.stack(out, dim=2)  # [B, T, 10, D]


# ═══════════════════════════════════════════════════════════════════════
# Sign10Vel Decoder: [B, T, 10, D] → 210D
# ═══════════════════════════════════════════════════════════════════════

class Sign10VelDecoder(nn.Module):
    """Per-part decoder for 210D sign10_vel representation.

    Body joints: h → MLP → rot_pred
    Hand joints: h → MLP_rot → rot_pred
                 h → MLP_vel → vel_pred
                 output = [rot_pred; vel_pred]

    Input: [B, T, 10, D] — from STConvDecoder.
    Output: [B, T, 210]
    """

    def __init__(self, opt):
        super().__init__()
        self.latent_dim = opt.latent_dim
        self.num_parts = 10
        self.rot_dims = list(SIGN10_ROT_DIMS)
        self.vel_dims = list(SIGN10_VEL_DIMS)
        self.vel_mask = list(SIGN10_VEL_MASK)

        act = getattr(opt, 'activation', 'gelu')

        # Rotation decoder (all joints)
        self.rot_proj = nn.ModuleList()
        for i in range(self.num_parts):
            self.rot_proj.append(nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                get_activation(act),
                nn.Linear(self.latent_dim, self.rot_dims[i]),
            ))

        # Velocity decoder (hand joints only)
        self.vel_proj = nn.ModuleList()
        for i in range(self.num_parts):
            if self.vel_mask[i]:
                self.vel_proj.append(nn.Sequential(
                    nn.Linear(self.latent_dim, self.latent_dim),
                    get_activation(act),
                    nn.Linear(self.latent_dim, self.vel_dims[i]),
                ))
            else:
                self.vel_proj.append(None)

    def forward(self, x):
        """
        Args:
            x: [B, T, 10, D]
        Returns:
            [B, T, 210]
        """
        parts = []
        for i in range(self.num_parts):
            h = x[:, :, i]  # [B, T, D]
            rot_pred = self.rot_proj[i](h)

            if self.vel_mask[i]:
                vel_pred = self.vel_proj[i](h)
                parts.append(torch.cat([rot_pred, vel_pred], dim=-1))
            else:
                parts.append(rot_pred)

        return torch.cat(parts, dim=-1)  # [B, T, 210]
