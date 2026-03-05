"""
Train Denoiser for sign language data.

Supports both original 133D (7part/finger) and 210D sign10_vel modes.
For sign10_vel, loads SignVAE instead of standard VAE, and uses 120D mean/std.

Usage:
    # Original 133D (7part)
    python train_sign_denoiser.py \
        --name sign_denoiser_v1 \
        --vae_name sign_vae_v1 \
        --dataset_name sign \
        --sign_dataset how2sign_csl_phoenix \
        --data_root ./dataset/How2Sign \
        --mean_path ./dataset/CSL-Daily/mean.pt \
        --std_path ./dataset/CSL-Daily/std.pt \
        --batch_size 64 --max_epoch 2000 \
        --prediction_type v_prediction

    # 210D sign10_vel
    python train_sign_denoiser.py \
        --name sign10_denoiser_v1 \
        --vae_name sign_vae_210d_v1 \
        --dataset_name sign \
        --skeleton_mode sign10_vel \
        --sign_dataset how2sign_csl_phoenix \
        --data_root ./dataset/How2Sign \
        --csl_root ./dataset/CSL-Daily \
        --phoenix_root ./dataset/Phoenix_2014T \
        --mean_path ./dataset/CSL-Daily/mean.pt \
        --std_path ./dataset/CSL-Daily/std.pt \
        --batch_size 64 --max_epoch 2000 \
        --text_encoder xlm-roberta \
        --prediction_type v_prediction \
        --hand_weight 2.0 \
        --use_text_cache
"""
import os
import glob
from os.path import join as pjoin

import torch
from torch.utils.data import DataLoader
import numpy as np
from diffusers import DDIMScheduler

from options.denoiser_option import arg_parse
from models.denoiser.model import Denoiser
from models.denoiser.trainer import DenoiserTrainer
from data.sign_dataset import SignText2MotionDataset
from utils.get_opt import get_opt
from utils.fixseed import fixseed

os.environ["OMP_NUM_THREADS"] = "1"


def load_vae(opt):
    """Load pretrained VAE from checkpoint.

    Dispatches to SignVAE (210D) or standard VAE (133D) based on
    skeleton_mode saved in the VAE's opt.txt.
    """
    vae_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vae_name)
    vae_opt = get_opt(pjoin(vae_dir, 'opt.txt'), opt.device)

    # ── Dispatch VAE model class ──
    skeleton_mode = getattr(vae_opt, 'skeleton_mode', '7part')
    if skeleton_mode == 'sign10_vel':
        from models.sign.model import SignVAE
        model = SignVAE(vae_opt)
        print(f'  VAE type: SignVAE (sign10_vel, 210D)')
    else:
        from models.vae.model import VAE
        model = VAE(vae_opt)
        print(f'  VAE type: VAE ({skeleton_mode})')

    # ── Load checkpoint ──
    loaded = False
    for ckpt_name in ['net_best_fid.tar', 'latest.tar']:
        ckpt_path = pjoin(vae_dir, 'model', ckpt_name)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            # SignVAE saves under 'vae' key (same convention)
            state_key = 'vae' if 'vae' in ckpt else 'model'
            model.load_state_dict(ckpt[state_key])
            print(f'  Loaded VAE from {ckpt_path}')
            loaded = True
            break

    if not loaded:
        # Try glob for epoch checkpoints
        ckpts = sorted(glob.glob(pjoin(vae_dir, 'model', 'net_epoch*.tar')))
        if ckpts:
            ckpt = torch.load(ckpts[-1], map_location='cpu')
            state_key = 'vae' if 'vae' in ckpt else 'model'
            model.load_state_dict(ckpt[state_key])
            print(f'  Loaded VAE from {ckpts[-1]}')
        else:
            raise FileNotFoundError(f'No VAE checkpoint found in {vae_dir}/model/')

    model.freeze()
    return model, vae_opt


def load_mean_std_for_mode(opt):
    """Load mean/std based on skeleton_mode.

    sign10_vel → 120D (sign10 order) mean/std
    7part/finger → 133D mean/std
    """
    if opt.skeleton_mode == 'sign10_vel':
        from utils.sign10_config import load_mean_std_sign10
        mean, std = load_mean_std_sign10(opt.mean_path, opt.std_path)
        print(f'  Mean/std: 120D sign10 order (shape={mean.shape})')
    else:
        from data.load_sign_data import load_mean_std
        mean, std = load_mean_std(opt.mean_path, opt.std_path)
        print(f'  Mean/std: 133D (shape={mean.shape})')
    return mean, std


if __name__ == "__main__":
    opt = arg_parse(True)
    fixseed(opt.seed)

    assert opt.dataset_name == "sign", "This script is for sign language data only."

    print("=" * 60)
    print(f"Sign Denoiser Training: {opt.name}")
    print(f"  skeleton_mode: {opt.skeleton_mode}")
    print(f"  pose_dim:      {opt.pose_dim}")
    print(f"  joints_num:    {opt.joints_num}")
    print(f"  vae_name:      {opt.vae_name}")
    print(f"  text_encoder:  {opt.text_encoder}")
    print(f"  hand_weight:   {opt.hand_weight}")
    print("=" * 60)

    # ── VAE ──
    print("\n[1/4] Loading VAE...")
    vae, vae_opt = load_vae(opt)

    # ── Denoiser ──
    print("\n[2/4] Building Denoiser...")
    denoiser = Denoiser(opt, vae_opt.latent_dim)
    num_params = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
    print(f'  Total trainable parameters: {num_params/1_000_000:.2f}M')
    print(f'  VAE latent_dim: {vae_opt.latent_dim}')

    # ── Scheduler ──
    scheduler = DDIMScheduler(
        num_train_timesteps=opt.num_train_timesteps,
        beta_start=opt.beta_start,
        beta_end=opt.beta_end,
        beta_schedule=opt.beta_schedule,
        prediction_type=opt.prediction_type,
        clip_sample=False,
    )

    # ── mean / std ──
    print("\n[3/4] Loading mean/std...")
    if opt.mean_path and opt.std_path:
        mean, std = load_mean_std_for_mode(opt)
    else:
        # Fall back to saved mean/std from VAE training
        vae_meta = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vae_name, 'meta')
        mean = np.load(pjoin(vae_meta, 'mean.npy'))
        std  = np.load(pjoin(vae_meta, 'std.npy'))
        print(f'  Mean/std from VAE meta: shape={mean.shape}')

    # ── dataset ──
    print("\n[4/4] Building datasets...")
    train_dataset = SignText2MotionDataset(opt, mean, std, split='train')
    val_dataset   = SignText2MotionDataset(opt, mean, std, split='val')

    train_loader = DataLoader(
        train_dataset, batch_size=opt.batch_size, drop_last=True,
        num_workers=opt.num_workers, shuffle=True, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=opt.batch_size, drop_last=True,
        num_workers=opt.num_workers, shuffle=True, pin_memory=True)

    print(f'  Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples')

    # ── train (no HumanML3D eval for sign) ──
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    trainer = DenoiserTrainer(opt, denoiser, vae, scheduler)
    trainer.train(train_loader, val_loader, eval_val_loader=None, eval_wrapper=None, plot_eval=None)