"""
models/sign/trainer.py — Sign10_vel VAE Trainer

Part-aware loss 설계:
  1. 각 super-joint별 reconstruction loss 개별 계산
  2. Hand joints (SJ4-9) → hand_weight (default 2.0) 가중
  3. Rotation vs Velocity 분리 가중: vel_weight (default 0.5)
  4. Velocity consistency loss (optional): 예측된 rot의 frame diff ≈ 예측된 vel

기존 SALAD의 models/vae/trainer.py 와 독립적으로 동작.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
from collections import OrderedDict

from utils.sign10_config import (
    SIGN_SPLITS_SIGN10_VEL,
    SIGN10_ROT_DIMS,
    SIGN10_VEL_DIMS,
    SIGN10_VEL_MASK,
    HAND_INDICES_SIGN10,
    SIGN_PART_NAMES_SIGN10,
)


class SignPartLoss(nn.Module):
    """Part-aware reconstruction loss for 210D sign10_vel.

    Computes MSE per super-joint with configurable weights:
      - hand_weight: multiplier for hand joints (SJ4-9)
      - vel_weight: multiplier for velocity components within each joint
      - vel_consist_weight: consistency between predicted rotation diff and velocity
    """

    def __init__(self, hand_weight=2.0, vel_weight=0.5, vel_consist_weight=0.1):
        super().__init__()
        self.total_splits = list(SIGN_SPLITS_SIGN10_VEL)
        self.rot_dims = list(SIGN10_ROT_DIMS)
        self.vel_dims = list(SIGN10_VEL_DIMS)
        self.vel_mask = list(SIGN10_VEL_MASK)
        self.hand_indices = set(HAND_INDICES_SIGN10)
        self.part_names = list(SIGN_PART_NAMES_SIGN10)
        self.num_parts = 10

        self.hand_weight = hand_weight
        self.vel_weight = vel_weight
        self.vel_consist_weight = vel_consist_weight

    def forward(self, pred, target, return_parts=False):
        """
        Args:
            pred:   [B, T, 210]
            target: [B, T, 210]
            return_parts: if True, also return per-part loss dict

        Returns:
            total_loss: scalar
            loss_parts: dict (optional) — per-part losses for logging
        """
        pred_parts = torch.split(pred, self.total_splits, dim=-1)
        target_parts = torch.split(target, self.total_splits, dim=-1)

        loss_parts = OrderedDict()
        total_loss = 0.0
        total_rot_loss = 0.0
        total_vel_loss = 0.0
        vel_consist_loss = 0.0

        for i in range(self.num_parts):
            part_weight = self.hand_weight if i in self.hand_indices else 1.0

            if self.vel_mask[i]:
                rd = self.rot_dims[i]

                # Separate rotation and velocity loss
                rot_loss = F.mse_loss(pred_parts[i][..., :rd],
                                      target_parts[i][..., :rd])
                vel_loss = F.mse_loss(pred_parts[i][..., rd:],
                                      target_parts[i][..., rd:])

                part_loss = part_weight * (rot_loss + self.vel_weight * vel_loss)

                # Velocity consistency: predicted rot diff ≈ predicted vel
                if self.vel_consist_weight > 0:
                    pred_rot = pred_parts[i][..., :rd]       # [B, T, rd]
                    pred_vel = pred_parts[i][..., rd:]       # [B, T, rd]
                    rot_diff = torch.zeros_like(pred_rot)
                    rot_diff[:, 1:] = pred_rot[:, 1:] - pred_rot[:, :-1]
                    vc_loss = F.mse_loss(pred_vel, rot_diff)
                    vel_consist_loss += part_weight * vc_loss

                total_rot_loss += part_weight * rot_loss
                total_vel_loss += part_weight * vel_loss
                loss_parts[f'rot/{self.part_names[i]}'] = rot_loss.item()
                loss_parts[f'vel/{self.part_names[i]}'] = vel_loss.item()
            else:
                part_loss = part_weight * F.mse_loss(pred_parts[i], target_parts[i])
                total_rot_loss += part_loss
                loss_parts[f'rot/{self.part_names[i]}'] = part_loss.item() / part_weight

            total_loss += part_loss

        # Normalize by number of parts
        total_loss = total_loss / self.num_parts
        vel_consist_loss = vel_consist_loss / max(sum(self.vel_mask), 1)

        # Add velocity consistency
        if self.vel_consist_weight > 0:
            total_loss = total_loss + self.vel_consist_weight * vel_consist_loss

        # Summary metrics
        loss_parts['loss_rot'] = total_rot_loss.item() / self.num_parts
        loss_parts['loss_vel'] = total_vel_loss.item() / max(sum(self.vel_mask), 1)
        if self.vel_consist_weight > 0:
            loss_parts['loss_vel_consist'] = vel_consist_loss.item()

        if return_parts:
            return total_loss, loss_parts
        return total_loss


class SignVAETrainer:
    """Trainer for SignVAE (sign10_vel).

    기존 SALAD trainer와 독립적. 핵심 차이:
      - SignPartLoss 사용 (part-aware, rot/vel 분리)
      - HumanML3D evaluation 없음 (수어 데이터셋)
      - 체크포인트 구조는 SALAD 호환 유지 (vae key)
    """

    def __init__(self, opt, model):
        self.opt = opt
        self.model = model.to(opt.device)
        self.device = opt.device

        # Loss
        self.part_loss_fn = SignPartLoss(
            hand_weight=getattr(opt, 'hand_weight', 2.0),
            vel_weight=getattr(opt, 'vel_weight', 0.5),
            vel_consist_weight=getattr(opt, 'vel_consist_weight', 0.1),
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=getattr(opt, 'lr', 2e-4),
            weight_decay=getattr(opt, 'weight_decay', 0.0),
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=opt.max_epoch,
            eta_min=getattr(opt, 'lr_min', 1e-6),
        )

        # Loss weights
        self.lambda_kl = getattr(opt, 'lambda_kl', 1e-5)

        # Logging
        self.log_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'log')
        self.model_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'model')
        self.meta_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'meta')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)

        self.writer = SummaryWriter(self.log_dir)
        self.global_step = 0
        self.best_val_loss = float('inf')

    def _save_checkpoint(self, epoch, filename='latest.tar'):
        state = {
            'vae': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
        }
        torch.save(state, pjoin(self.model_dir, filename))

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location='cpu')
        self.model.load_state_dict(ckpt['vae'])
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt.get('epoch', 0) + 1
        self.global_step = ckpt.get('global_step', 0)
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f'Loaded checkpoint from {path}, epoch={start_epoch-1}')
        return start_epoch

    def _train_one_epoch(self, loader):
        self.model.train()
        epoch_losses = {}
        count = 0

        for batch in loader:
            if isinstance(batch, (list, tuple)):
                motion = batch[0]
            else:
                motion = batch

            motion = motion.to(self.device)  # [B, T, 210]

            # Forward
            recon, loss_dict = self.model(motion)

            # Part-aware reconstruction loss
            loss_recon, part_losses = self.part_loss_fn(recon, motion, return_parts=True)

            # KL loss
            loss_kl = loss_dict['loss_kl']

            # Total
            loss = loss_recon + self.lambda_kl * loss_kl

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Logging
            self.global_step += 1
            batch_losses = {
                'loss': loss.item(),
                'loss_recon': loss_recon.item(),
                'loss_kl': loss_kl.item(),
                **part_losses,
            }
            for k, v in batch_losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v
            count += 1

            # Tensorboard (every 100 steps)
            if self.global_step % 100 == 0:
                for k, v in batch_losses.items():
                    self.writer.add_scalar(f'train/{k}', v, self.global_step)

        return {k: v / count for k, v in epoch_losses.items()}

    @torch.no_grad()
    def _validate(self, loader):
        self.model.eval()
        epoch_losses = {}
        count = 0

        for batch in loader:
            if isinstance(batch, (list, tuple)):
                motion = batch[0]
            else:
                motion = batch

            motion = motion.to(self.device)

            recon, loss_dict = self.model(motion)
            loss_recon, part_losses = self.part_loss_fn(recon, motion, return_parts=True)
            loss_kl = loss_dict['loss_kl']
            loss = loss_recon + self.lambda_kl * loss_kl

            batch_losses = {
                'loss': loss.item(),
                'loss_recon': loss_recon.item(),
                'loss_kl': loss_kl.item(),
                **part_losses,
            }
            for k, v in batch_losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v
            count += 1

        return {k: v / count for k, v in epoch_losses.items()}

    def train(self, train_loader, val_loader):
        """Main training loop."""
        opt = self.opt
        start_epoch = 0

        # Resume
        latest_path = pjoin(self.model_dir, 'latest.tar')
        if os.path.exists(latest_path) and getattr(opt, 'resume', True):
            start_epoch = self._load_checkpoint(latest_path)

        print(f'\n{"="*60}')
        print(f'SignVAE Training — sign10_vel 210D')
        print(f'  latent_dim={opt.latent_dim}, epochs={opt.max_epoch}')
        print(f'  hand_weight={self.part_loss_fn.hand_weight}, '
              f'vel_weight={self.part_loss_fn.vel_weight}, '
              f'vel_consist_weight={self.part_loss_fn.vel_consist_weight}')
        print(f'  lambda_kl={self.lambda_kl}')
        print(f'{"="*60}\n')

        for epoch in range(start_epoch, opt.max_epoch):
            t0 = time.time()

            train_losses = self._train_one_epoch(train_loader)
            val_losses = self._validate(val_loader)
            self.scheduler.step()

            # Log
            dt = time.time() - t0
            lr = self.optimizer.param_groups[0]['lr']
            print(f'[Epoch {epoch:04d}] '
                  f'train_loss={train_losses["loss"]:.5f} '
                  f'val_loss={val_losses["loss"]:.5f} '
                  f'recon={val_losses["loss_recon"]:.5f} '
                  f'kl={val_losses["loss_kl"]:.5f} '
                  f'lr={lr:.2e} ({dt:.1f}s)')

            for k, v in val_losses.items():
                self.writer.add_scalar(f'val/{k}', v, epoch)
            self.writer.add_scalar('lr', lr, epoch)

            # Save checkpoints
            self._save_checkpoint(epoch, 'latest.tar')

            if epoch % getattr(opt, 'save_every', 10) == 0:
                self._save_checkpoint(epoch, f'net_epoch{epoch:04d}.tar')

            if val_losses['loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['loss']
                self._save_checkpoint(epoch, 'net_best.tar')
                print(f'  → Best val loss: {self.best_val_loss:.5f}')

        print(f'\nTraining complete. Best val loss: {self.best_val_loss:.5f}')
        self.writer.close()
