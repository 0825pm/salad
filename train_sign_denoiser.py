"""
Train Denoiser for sign language data.
Usage:
    python train_sign_denoiser.py \
        --name sign_denoiser_v1 \
        --vae_name sign_vae_v1 \
        --dataset_name sign \
        --sign_dataset how2sign \
        --data_root ./dataset/How2Sign \
        --mean_path ./dataset/How2Sign/mean.pt \
        --std_path ./dataset/How2Sign/std.pt \
        --batch_size 64 \
        --max_epoch 2000 \
        --prediction_type v_prediction
"""
import os
from os.path import join as pjoin

import torch
from torch.utils.data import DataLoader
import numpy as np
from diffusers import DDIMScheduler

from options.denoiser_option import arg_parse
from models.vae.model import VAE
from models.denoiser.model import Denoiser
from models.denoiser.trainer import DenoiserTrainer
from data.sign_dataset import SignText2MotionDataset
from data.load_sign_data import load_mean_std
from utils.get_opt import get_opt
from utils.fixseed import fixseed

os.environ["OMP_NUM_THREADS"] = "1"


def load_vae(opt):
    """Load pretrained VAE from checkpoint."""
    vae_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vae_name)
    vae_opt = get_opt(pjoin(vae_dir, 'opt.txt'), opt.device)

    model = VAE(vae_opt)

    # Try multiple checkpoint names
    for ckpt_name in ['net_best_fid.tar', 'latest.tar']:
        ckpt_path = pjoin(vae_dir, 'model', ckpt_name)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(ckpt["vae"])
            print(f'Loaded VAE from {ckpt_path}')
            break
    else:
        # Try glob for epoch checkpoints
        import glob
        ckpts = sorted(glob.glob(pjoin(vae_dir, 'model', 'net_epoch*.tar')))
        if ckpts:
            ckpt = torch.load(ckpts[-1], map_location='cpu')
            model.load_state_dict(ckpt["vae"])
            print(f'Loaded VAE from {ckpts[-1]}')
        else:
            raise FileNotFoundError(f'No VAE checkpoint found in {vae_dir}/model/')

    model.freeze()
    return model, vae_opt


if __name__ == "__main__":
    opt = arg_parse(True)
    fixseed(opt.seed)

    assert opt.dataset_name == "sign", "This script is for sign language data only."

    # ── VAE ──
    vae, vae_opt = load_vae(opt)

    # ── Denoiser ──
    denoiser = Denoiser(opt, vae_opt.latent_dim)
    num_params = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
    print(f'Total trainable parameters of all models: {num_params/1_000_000:.2f}M')

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
    if opt.mean_path and opt.std_path:
        mean, std = load_mean_std(opt.mean_path, opt.std_path)
    else:
        # Fall back to saved mean/std from VAE training
        vae_meta = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vae_name, 'meta')
        mean = np.load(pjoin(vae_meta, 'mean.npy'))
        std  = np.load(pjoin(vae_meta, 'std.npy'))

    # ── dataset ──
    train_dataset = SignText2MotionDataset(opt, mean, std, split='train')
    val_dataset   = SignText2MotionDataset(opt, mean, std, split='val')

    train_loader = DataLoader(
        train_dataset, batch_size=opt.batch_size, drop_last=True,
        num_workers=opt.num_workers, shuffle=True, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=opt.batch_size, drop_last=True,
        num_workers=opt.num_workers, shuffle=True, pin_memory=True)

    print(f'Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples')

    # ── train (no HumanML3D eval for sign) ──
    trainer = DenoiserTrainer(opt, denoiser, vae, scheduler)
    trainer.train(train_loader, val_loader, eval_val_loader=None, eval_wrapper=None, plot_eval=None)
