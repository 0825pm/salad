"""
Train VAE for sign language data.
Usage:
    python train_sign_vae.py \
        --name sign_vae_v1 \
        --dataset_name sign \
        --sign_dataset how2sign \
        --data_root ./dataset/How2Sign \
        --mean_path ./dataset/How2Sign/mean.pt \
        --std_path ./dataset/How2Sign/std.pt \
        --window_size 64 \
        --batch_size 256 \
        --max_epoch 50
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

    # ── mean / std ──
    if opt.mean_path and opt.std_path:
        mean, std = load_mean_std(opt.mean_path, opt.std_path)
    else:
        raise ValueError("--mean_path and --std_path are required for sign dataset")

    # save mean/std for later use
    np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
    np.save(pjoin(opt.meta_dir, 'std.npy'), std)
    print(f'Mean shape: {mean.shape}, Std shape: {std.shape}')

    # ── model ──
    net = VAE(opt)
    num_params = sum(p.numel() for p in net.parameters())
    print(f'Total trainable parameters: {num_params/1_000_000:.2f}M')

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

    # ── train (no HumanML3D eval for sign) ──
    trainer = VAETrainer(opt, net)
    trainer.train(train_loader, val_loader, eval_val_loader=None, eval_wrapper=None, plot_eval=None)
