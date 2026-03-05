"""
Train VAE for sign language data (223D: 133D rotation + 90D hand velocity).

Usage:
    # Default: 223D with hand velocity + part-wise std clamping
    python train_sign_vae.py \
        --name sign_vae_223d_v1 \
        --dataset_name sign \
        --sign_dataset how2sign \
        --data_root ./dataset/How2Sign \
        --mean_path ./dataset/How2Sign/mean.pt \
        --std_path ./dataset/How2Sign/std.pt \
        --window_size 64 \
        --batch_size 256 \
        --max_epoch 50

    # 133D only (no hand velocity)
    python train_sign_vae.py \
        --name sign_vae_133d_v1 \
        --dataset_name sign \
        --no_hand_vel \
        ...

Data flow:
    PKL(179D) → 133D(on-the-fly) → normalize(mean_133, std_133_clamped)
              → append_hand_vel → 223D → VAE

    Encoder: [B,T,223] → _group_features() → 7 super-joints
             SJ3(lhand): rot_45D + vel_45D = 90D input
             SJ4(rhand): rot_45D + vel_45D = 90D input
           → per-joint Linear → [B,T,7,latent_dim]
           → STConv (graph attention) → pooling → latent

    Decoder: latent → unpool → STConv → per-joint Linear
           → _ungroup_features() → [B,T,223] = [rot_133 | vel_90]

    Mean/Std:  항상 133D로 저장 (velocity는 normalized data에서 계산되므로 별도 통계 불필요)
"""
import os
from os.path import join as pjoin

import torch
from torch.utils.data import DataLoader
import numpy as np

from options.vae_option import arg_parse
from models.vae.model import VAE
from models.vae.trainer import VAETrainer
from data.sign_dataset import SignMotionDataset
from data.load_sign_data import load_mean_std
from utils.fixseed import fixseed

os.environ["OMP_NUM_THREADS"] = "1"

if __name__ == "__main__":
    opt = arg_parse(True)
    fixseed(opt.seed)

    assert opt.dataset_name == "sign", "This script is for sign language data only."

    # ── mean / std (always 133D, with part-wise clamping) ──
    if opt.mean_path and opt.std_path:
        mean, std = load_mean_std(
            opt.mean_path, opt.std_path,
            partwise_clamp=True,       # prevent normalization explosion
        )
    else:
        raise ValueError("--mean_path and --std_path are required for sign dataset")

    # Save 133D clamped mean/std for later use (denoiser, visualization)
    np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
    np.save(pjoin(opt.meta_dir, 'std.npy'), std)

    print(f'\n=== Sign VAE Training Config ===')
    print(f'  Mean shape: {mean.shape}, Std shape: {std.shape}')
    print(f'  pose_dim: {opt.pose_dim} ({"223D = 133D rot + 90D hand vel" if opt.use_hand_vel else "133D rot only"})')
    print(f'  skeleton_mode: {opt.skeleton_mode}')
    print(f'  use_hand_vel: {opt.use_hand_vel}')
    print(f'  Part-wise std clamping: enabled')
    print(f'  Std range after clamping: [{std.min():.6f}, {std.max():.6f}]')
    print(f'================================\n')

    # ── model ──
    net = VAE(opt)
    num_params = sum(p.numel() for p in net.parameters())
    print(f'Total trainable parameters: {num_params/1_000_000:.2f}M')

    # Print encoder splits for verification
    if hasattr(net.motion_enc, 'splits'):
        print(f'  Encoder splits (per-node input dims): {net.motion_enc.splits}')
        print(f'  Sum: {sum(net.motion_enc.splits)}D')

    # ── dataset ──
    train_dataset = SignMotionDataset(opt, mean, std, split='train')
    val_dataset   = SignMotionDataset(opt, mean, std, split='val')

    train_loader = DataLoader(
        train_dataset, batch_size=opt.batch_size, drop_last=True,
        num_workers=opt.num_workers, shuffle=True, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=opt.batch_size, drop_last=True,
        num_workers=opt.num_workers, shuffle=True, pin_memory=True)

    print(f'Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples')

    # Quick shape check
    sample = train_dataset[0]
    print(f'Sample shape: {sample.shape} (expected [W={opt.window_size}, D={opt.pose_dim}])')
    assert sample.shape == (opt.window_size, opt.pose_dim), \
        f"Shape mismatch: got {sample.shape}, expected ({opt.window_size}, {opt.pose_dim})"

    # ── train (no HumanML3D eval for sign) ──
    trainer = VAETrainer(opt, net)
    trainer.train(train_loader, val_loader, eval_val_loader=None, eval_wrapper=None, plot_eval=None)
