"""
train_sign10_vae.py — Sign10_vel (210D) VAE 학습 스크립트

기존 train_sign_vae.py와 독립적.
models/sign/model.py의 SignVAE + models/sign/trainer.py의 SignVAETrainer 사용.

Usage:
    python train_sign10_vae.py \
        --name sign_vae_210d_v1 \
        --sign_dataset how2sign_csl_phoenix \
        --data_root ./dataset/How2Sign \
        --csl_root ./dataset/CSL-Daily \
        --phoenix_root ./dataset/Phoenix_2014T \
        --mean_path ./dataset/CSL-Daily/mean.pt \
        --std_path ./dataset/CSL-Daily/std.pt \
        --window_size 64 \
        --batch_size 256 \
        --max_epoch 50 \
        --latent_dim 32 \
        --hand_weight 2.0 \
        --vel_weight 0.5 \
        --vel_consist_weight 0.1
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from os.path import join as pjoin

from utils.fixseed import fixseed
from utils.sign10_config import load_mean_std_sign10
from data.sign_dataset import SignMotionDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train Sign10_vel VAE (210D)')

    # ── Experiment ──
    parser.add_argument('--name', type=str, required=True, help='experiment name')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')

    # ── Data ──
    parser.add_argument('--sign_dataset', type=str, default='how2sign',
                        choices=['how2sign', 'csl', 'how2sign_csl', 'how2sign_csl_phoenix'])
    parser.add_argument('--data_root', type=str, required=True, help='How2Sign root')
    parser.add_argument('--csl_root', type=str, default=None)
    parser.add_argument('--phoenix_root', type=str, default=None)
    parser.add_argument('--mean_path', type=str, required=True)
    parser.add_argument('--std_path', type=str, required=True)
    parser.add_argument('--joint_root', type=str, default=None, help='precomputed joints dir')

    # ── Data processing ──
    parser.add_argument('--window_size', type=int, default=64)
    parser.add_argument('--min_motion_length', type=int, default=40)
    parser.add_argument('--max_motion_length', type=int, default=400)
    parser.add_argument('--unit_length', type=int, default=4)

    # ── Model ──
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_extra_layers', type=int, default=1)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--norm', type=str, default='none')
    parser.add_argument('--dropout', type=float, default=0.1)

    # ── Loss weights ──
    parser.add_argument('--hand_weight', type=float, default=2.0,
                        help='loss weight multiplier for hand super-joints')
    parser.add_argument('--vel_weight', type=float, default=0.5,
                        help='loss weight multiplier for velocity components')
    parser.add_argument('--vel_consist_weight', type=float, default=0.1,
                        help='velocity consistency loss weight')
    parser.add_argument('--lambda_kl', type=float, default=1e-5)

    # ── Training ──
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lr_min', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--no_resume', dest='resume', action='store_false')

    opt = parser.parse_args()

    # ── Fixed settings for sign10_vel ──
    opt.dataset_name = 'sign'
    opt.skeleton_mode = 'sign10_vel'
    opt.pose_dim = 210
    opt.joints_num = 10
    opt.contact_joints = []

    opt.device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')

    # ── Meta dir ──
    opt.meta_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'meta')
    os.makedirs(opt.meta_dir, exist_ok=True)

    return opt


def main():
    opt = parse_args()
    fixseed(opt.seed)

    print('=' * 60)
    print(f'Sign10_vel VAE Training: {opt.name}')
    print(f'  skeleton_mode: {opt.skeleton_mode}')
    print(f'  pose_dim: {opt.pose_dim} (10 super-joints)')
    print(f'  latent_dim: {opt.latent_dim}')
    print(f'  device: {opt.device}')
    print('=' * 60)

    # ── Load mean/std (120D sign10 order) ──
    mean, std = load_mean_std_sign10(opt.mean_path, opt.std_path)
    print(f'Mean shape: {mean.shape}, Std shape: {std.shape}')
    assert mean.shape == (120,), f'Expected mean shape (120,), got {mean.shape}'

    # Save for eval/vis
    np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
    np.save(pjoin(opt.meta_dir, 'std.npy'), std)

    # ── Build datasets ──
    # sign10_vel 모드: SignMotionDataset에서 120D로 로딩 → normalize → 210D 변환
    train_dataset = SignMotionDataset(opt, mean, std, split='train')
    val_dataset = SignMotionDataset(opt, mean, std, split='val')

    train_loader = DataLoader(
        train_dataset, batch_size=opt.batch_size, drop_last=True,
        num_workers=opt.num_workers, shuffle=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=opt.batch_size, drop_last=True,
        num_workers=opt.num_workers, shuffle=False, pin_memory=True,
    )

    print(f'Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples')

    # ── Build model ──
    from models.sign.model import SignVAE
    model = SignVAE(opt)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {num_params/1_000_000:.2f}M')

    # ── Save opt ──
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'opt.txt')
    os.makedirs(os.path.dirname(opt_path), exist_ok=True)
    with open(opt_path, 'w') as f:
        for k, v in sorted(vars(opt).items()):
            f.write(f'{k}: {v}\n')

    # ── Train ──
    from models.sign.trainer import SignVAETrainer
    trainer = SignVAETrainer(opt, model)
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
