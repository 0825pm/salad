"""
vis_sign_diffusion.py — GT vs Diffusion Generation 비교 (133D axis-angle)

SALAD sign Denoiser → 텍스트 조건부 생성 결과 시각화

Usage:
    cd ~/Projects/research/salad

    python vis_sign_diffusion.py \
        --denoiser_checkpoint checkpoints/sign/sign_denoiser_v1/model/latest.tar \
        --vae_name sign_vae_all \
        --data_root /home/user/Projects/research/SOKE/data/How2Sign \
        --csl_root /home/user/Projects/research/SOKE/data/CSL-Daily \
        --phoenix_root /home/user/Projects/research/SOKE/data/Phoenix_2014T \
        --mean_path /home/user/Projects/research/SOKE/data/CSL-Daily/mean.pt \
        --std_path /home/user/Projects/research/SOKE/data/CSL-Daily/std.pt \
        --smplx_path deps/smpl_models/ \
        --sign_dataset how2sign_csl_phoenix \
        --split val --num_samples 5

    # Custom text (no GT comparison, generation only)
    python vis_sign_diffusion.py \
        --denoiser_checkpoint ... --vae_name ... \
        --text "a person signs hello" \
        --gen_length 120 \
        --mean_path ... --std_path ... --smplx_path ...
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from argparse import Namespace
from diffusers import DDIMScheduler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.feats2joints import aa133_to_joints_np
from utils.get_opt import get_opt
from os.path import join as pjoin


# =============================================================================
# Constants — SMPL-X joint indices
# =============================================================================
SMPLX_UPPER_BODY = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
SMPLX_LHAND = list(range(25, 40))
SMPLX_RHAND = list(range(40, 55))
SMPLX_VALID = SMPLX_UPPER_BODY + SMPLX_LHAND + SMPLX_RHAND


# =============================================================================
# Skeleton Visualization (shared with vis_sign_recon.py)
# =============================================================================

def get_connections(num_joints):
    upper_body = [
        (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
        (9, 13), (13, 16), (16, 18), (18, 20),
        (9, 14), (14, 17), (17, 19), (19, 21),
    ]
    hand_connections = []
    if num_joints >= 55:
        for finger in range(5):
            base = 25 + finger * 3
            if base + 2 < num_joints:
                hand_connections.extend([(20, base), (base, base+1), (base+1, base+2)])
        for finger in range(5):
            base = 40 + finger * 3
            if base + 2 < num_joints:
                hand_connections.extend([(21, base), (base, base+1), (base+1, base+2)])
    return [(i, j) for i, j in upper_body + hand_connections if i < num_joints and j < num_joints]


def normalize_to_root(joints, root_idx=9):
    if len(joints.shape) == 3:
        root = joints[:, root_idx:root_idx+1, :]
    else:
        root = joints[root_idx:root_idx+1, :]
    return joints - root


def compute_metrics(gt_feats, gen_feats, gt_joints, gen_joints):
    metrics = {}
    T = min(gt_feats.shape[0], gen_feats.shape[0])
    metrics['feat_rmse'] = np.sqrt(np.mean((gt_feats[:T] - gen_feats[:T]) ** 2))
    metrics['feat_mae'] = np.mean(np.abs(gt_feats[:T] - gen_feats[:T]))
    if gt_joints is not None and gen_joints is not None:
        T = min(gt_joints.shape[0], gen_joints.shape[0])
        J = min(gt_joints.shape[1], gen_joints.shape[1])
        mpjpe = np.sqrt(np.sum((gt_joints[:T, :J] - gen_joints[:T, :J]) ** 2, axis=-1)).mean()
        metrics['mpjpe'] = mpjpe
    return metrics


def save_comparison_video(left_joints, right_joints, save_path, title='',
                          fps=25, viewport=1.0, metrics=None,
                          left_label='GT', right_label='Generated'):
    """Side-by-side skeleton video"""
    T = min(left_joints.shape[0], right_joints.shape[0])
    J = min(left_joints.shape[1], right_joints.shape[1])

    root_idx = 9 if J > 21 else 0
    left = normalize_to_root(left_joints[:T, :J].copy(), root_idx)
    right = normalize_to_root(right_joints[:T, :J].copy(), root_idx)

    if viewport > 0:
        x_lim = (-viewport, viewport)
        y_lim = (-viewport, viewport)
    else:
        valid_idx = [i for i in SMPLX_VALID if i < J]
        all_data = np.concatenate([left[:, valid_idx], right[:, valid_idx]], axis=0)
        all_x, all_y = all_data[:, :, 0].flatten(), all_data[:, :, 1].flatten()
        max_range = max(all_x.max() - all_x.min(), all_y.max() - all_y.min(), 0.1) * 1.2
        x_mid = (all_x.max() + all_x.min()) / 2
        y_mid = (all_y.max() + all_y.min()) / 2
        x_lim = (x_mid - max_range/2, x_mid + max_range/2)
        y_lim = (y_mid - max_range/2, y_mid + max_range/2)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))
    main_title = title
    if metrics:
        main_title += f"\nRMSE={metrics.get('feat_rmse', 0):.4f}  "
        main_title += f"MPJPE={metrics.get('mpjpe', 0):.4f}"
    fig.suptitle(main_title, fontsize=10)

    for ax, label in [(ax_l, left_label), (ax_r, right_label)]:
        ax.set_title(label, fontsize=12, fontweight='bold',
                     color='blue' if label == left_label else 'red')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')
        ax.axis('off')

    connections = get_connections(J)
    colors = {'body': 'blue', 'lhand': 'red', 'rhand': 'green'}
    ub_idx = [i for i in SMPLX_UPPER_BODY if i < J]

    elements = []
    for ax, data in [(ax_l, left), (ax_r, right)]:
        lines = []
        for (i, j) in connections:
            if i >= 40 or j >= 40:
                c, lw = colors['rhand'], 1.0
            elif i >= 25 or j >= 25:
                c, lw = colors['lhand'], 1.0
            else:
                c, lw = colors['body'], 1.5
            line, = ax.plot([], [], color=c, linewidth=lw, alpha=0.8)
            lines.append((line, i, j))
        bs = ax.scatter([], [], c=colors['body'], s=10, zorder=5)
        ls = ax.scatter([], [], c=colors['lhand'], s=5, zorder=5)
        rs = ax.scatter([], [], c=colors['rhand'], s=5, zorder=5)
        elements.append((lines, bs, ls, rs, data))

    frame_text = fig.text(0.5, 0.02, '', ha='center', fontsize=9, color='gray')
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])

    def update(frame):
        f = min(frame, T - 1)
        frame_text.set_text(f'Frame {f}/{T-1}')
        for (lines, bs, ls, rs, data) in elements:
            fd = data[f]
            x, y = fd[:, 0], fd[:, 1]
            for (line, i, j) in lines:
                line.set_data([x[i], x[j]], [y[i], y[j]])
            bs.set_offsets(np.c_[x[ub_idx], y[ub_idx]])
            if J > 25:
                ls.set_offsets(np.c_[x[25:40], y[25:40]])
            if J > 40:
                rs.set_offsets(np.c_[x[40:55], y[40:55]])
        return []

    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False)
    try:
        anim.save(save_path, writer=FFMpegWriter(fps=fps, bitrate=5000))
    except Exception:
        anim.save(save_path.replace('.mp4', '.gif'), writer='pillow', fps=min(fps, 10))
    plt.close(fig)


def save_single_video(joints, save_path, title='', fps=25, viewport=1.0):
    """Single skeleton video (for text-only generation without GT)"""
    T, J = joints.shape[:2]
    root_idx = 9 if J > 21 else 0
    data = normalize_to_root(joints.copy(), root_idx)

    if viewport > 0:
        x_lim = (-viewport, viewport)
        y_lim = (-viewport, viewport)
    else:
        valid_idx = [i for i in SMPLX_VALID if i < J]
        d = data[:, valid_idx]
        all_x, all_y = d[:, :, 0].flatten(), d[:, :, 1].flatten()
        margin = max(all_x.max() - all_x.min(), all_y.max() - all_y.min(), 0.1) * 1.2
        x_lim = ((all_x.max()+all_x.min())/2 - margin/2, (all_x.max()+all_x.min())/2 + margin/2)
        y_lim = ((all_y.max()+all_y.min())/2 - margin/2, (all_y.max()+all_y.min())/2 + margin/2)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.suptitle(title, fontsize=10)
    ax.set_xlim(x_lim); ax.set_ylim(y_lim)
    ax.set_aspect('equal'); ax.axis('off')

    connections = get_connections(J)
    colors = {'body': 'blue', 'lhand': 'red', 'rhand': 'green'}
    ub_idx = [i for i in SMPLX_UPPER_BODY if i < J]

    lines = []
    for (i, j) in connections:
        if i >= 40 or j >= 40: c, lw = colors['rhand'], 1.0
        elif i >= 25 or j >= 25: c, lw = colors['lhand'], 1.0
        else: c, lw = colors['body'], 1.5
        line, = ax.plot([], [], color=c, linewidth=lw, alpha=0.8)
        lines.append((line, i, j))
    bs = ax.scatter([], [], c=colors['body'], s=10, zorder=5)
    ls = ax.scatter([], [], c=colors['lhand'], s=5, zorder=5)
    rs = ax.scatter([], [], c=colors['rhand'], s=5, zorder=5)

    frame_text = fig.text(0.5, 0.02, '', ha='center', fontsize=9, color='gray')
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    def update(frame):
        f = min(frame, T - 1)
        frame_text.set_text(f'Frame {f}/{T-1}')
        fd = data[f]; x, y = fd[:, 0], fd[:, 1]
        for (line, i, j) in lines:
            line.set_data([x[i], x[j]], [y[i], y[j]])
        bs.set_offsets(np.c_[x[ub_idx], y[ub_idx]])
        if J > 25: ls.set_offsets(np.c_[x[25:40], y[25:40]])
        if J > 40: rs.set_offsets(np.c_[x[40:55], y[40:55]])
        return []

    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False)
    try:
        anim.save(save_path, writer=FFMpegWriter(fps=fps, bitrate=5000))
    except Exception:
        anim.save(save_path.replace('.mp4', '.gif'), writer='pillow', fps=min(fps, 10))
    plt.close(fig)


# =============================================================================
# Model builders
# =============================================================================

def build_vae(vae_dir, device):
    """Load pretrained VAE from checkpoint directory."""
    vae_opt = get_opt(pjoin(vae_dir, 'opt.txt'), device)

    from models.vae.model import VAE
    vae = VAE(vae_opt)

    for ckpt_name in ['net_best_fid.tar', 'latest.tar']:
        ckpt_path = pjoin(vae_dir, 'model', ckpt_name)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            vae.load_state_dict(ckpt["vae"])
            print(f"  VAE loaded from {ckpt_path}")
            break
    else:
        tars = sorted(glob.glob(pjoin(vae_dir, 'model', 'net_epoch*.tar')))
        if tars:
            ckpt = torch.load(tars[-1], map_location='cpu')
            vae.load_state_dict(ckpt["vae"])
            print(f"  VAE loaded from {tars[-1]}")
        else:
            raise FileNotFoundError(f'No VAE checkpoint in {vae_dir}/model/')

    vae.freeze()
    return vae, vae_opt


def build_denoiser(opt, vae_opt, ckpt_path, device):
    """Load pretrained Denoiser from checkpoint."""
    from models.denoiser.model import Denoiser
    denoiser = Denoiser(opt, vae_opt.latent_dim)

    ckpt = torch.load(ckpt_path, map_location='cpu')
    missing, unexpected = denoiser.load_state_dict(ckpt["denoiser"], strict=False)
    # Only clip_model keys should be missing (they're frozen & not saved)
    assert all(k.startswith("clip_model.") for k in missing), \
        f"Unexpected missing keys: {[k for k in missing if not k.startswith('clip_model.')]}"

    epoch = ckpt.get('epoch', '?')
    total_iter = ckpt.get('total_iter', '?')
    print(f"  Denoiser loaded from {ckpt_path} (epoch={epoch}, iter={total_iter})")

    denoiser.eval().to(device)
    return denoiser


def lengths_to_mask(lengths):
    max_frames = torch.max(lengths)
    mask = torch.arange(max_frames, device=lengths.device).expand(
        len(lengths), max_frames) < lengths.unsqueeze(1)
    return mask


@torch.no_grad()
def generate_motion(denoiser, vae, scheduler, text, motion_ref, m_len,
                    device, cond_scale=7.5, num_inference_steps=50):
    """
    Text-conditioned motion generation.

    Args:
        text:       list of str [B]
        motion_ref: [B, T, 133] — reference motion (for latent shape only)
        m_len:      [B] — target lengths
        cond_scale: classifier-free guidance scale
    Returns:
        pred_motion: [B, T, 133]
    """
    denoiser.eval()

    motion_ref = motion_ref.to(device, dtype=torch.float32)
    m_lens_latent = m_len.to(device, dtype=torch.long) // 4
    len_mask = lengths_to_mask(m_lens_latent)

    # Get latent shape from reference
    z, _ = vae.encode(motion_ref)
    latents = torch.randn_like(z)
    latents = latents * scheduler.init_noise_sigma

    len_mask = F.pad(len_mask, (0, latents.shape[1] - len_mask.shape[1]),
                     mode="constant", value=False)
    latents = latents * len_mask[..., None, None].float()

    # Classifier-free guidance: unconditional + conditional
    use_cfg = cond_scale > 1.0
    if use_cfg:
        input_text = [""] * len(text) + list(text)
    else:
        input_text = list(text)

    # Reverse diffusion
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps.to(device)

    for timestep in timesteps:
        if use_cfg:
            input_latents = torch.cat([latents] * 2, dim=0)
            input_len_mask = torch.cat([len_mask] * 2, dim=0)
        else:
            input_latents = latents
            input_len_mask = len_mask

        pred, _ = denoiser.forward(input_latents, timestep, input_text,
                                    len_mask=input_len_mask, use_cached_clip=True)

        if use_cfg:
            pred_uncond, pred_cond = torch.chunk(pred, 2, dim=0)
            pred = pred_uncond + cond_scale * (pred_cond - pred_uncond)

        latents = scheduler.step(pred, timestep, latents).prev_sample
        latents = latents * len_mask[..., None, None].float()

    # Decode
    pred_motion = vae.decode(latents)
    if isinstance(pred_motion, (tuple, list)):
        pred_motion = pred_motion[0]

    denoiser.remove_clip_cache()
    return pred_motion


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='SALAD Sign Diffusion: GT vs Generated Visualization')

    # Model paths
    parser.add_argument('--denoiser_checkpoint', required=True, help='.tar checkpoint path')
    parser.add_argument('--vae_name', required=True, help='VAE experiment name')
    parser.add_argument('--checkpoints_dir', default='./checkpoints')

    # Data paths
    parser.add_argument('--data_root', default=None, help='How2Sign data root')
    parser.add_argument('--csl_root', default=None)
    parser.add_argument('--phoenix_root', default=None)
    parser.add_argument('--mean_path', required=True)
    parser.add_argument('--std_path', required=True)
    parser.add_argument('--smplx_path', default='deps/smpl_models/smplx')

    # Custom text mode (no GT)
    parser.add_argument('--text', default=None, nargs='+',
                        help='Custom text prompt(s) for generation (no GT comparison)')
    parser.add_argument('--gen_length', type=int, default=120,
                        help='Generation length in frames (for --text mode)')

    # Dataset mode (GT comparison)
    parser.add_argument('--sign_dataset', default='how2sign',
                        choices=['how2sign', 'csl', 'how2sign_csl', 'how2sign_csl_phoenix'])
    parser.add_argument('--split', default='val')
    parser.add_argument('--num_samples', type=int, default=5)

    # Generation config
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--cond_scale', type=float, default=7.5)

    # Denoiser arch (must match training config)
    parser.add_argument('--text_encoder', default='xlm-roberta', choices=['clip', 'xlm-roberta'])
    parser.add_argument('--clip_version', default='ViT-B/32')
    parser.add_argument('--xlmr_version', default='xlm-roberta-base')
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=5)
    parser.add_argument('--ff_dim', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--use_limm', action='store_true',
                        help='Must match training config if model was trained with LIMM')
    parser.add_argument('--use_atii', action='store_true',
                        help='Must match training config if model was trained with ATII')

    # Diffusion config
    parser.add_argument('--num_train_timesteps', type=int, default=1000)
    parser.add_argument('--beta_start', type=float, default=0.00085)
    parser.add_argument('--beta_end', type=float, default=0.012)
    parser.add_argument('--beta_schedule', default='scaled_linear')
    parser.add_argument('--prediction_type', default='v_prediction')

    # Output
    parser.add_argument('--output', default='vis_diffusion_output')
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--viewport', type=float, default=0.5)
    parser.add_argument('--device', default='cuda:0')

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = os.path.join(args.output, f'gen_{timestamp}')
    os.makedirs(output_root, exist_ok=True)

    # ── Load mean/std ──
    from data.load_sign_data import load_mean_std
    mean, std = load_mean_std(args.mean_path, args.std_path)
    mean_t = torch.tensor(mean, dtype=torch.float32)
    std_t = torch.tensor(std, dtype=torch.float32)

    print("=" * 60)
    print("SALAD Sign Diffusion: Text → Motion Visualization")
    print(f"  denoiser:    {args.denoiser_checkpoint}")
    print(f"  vae:         {args.vae_name}")
    print(f"  text_encoder:{args.text_encoder}")
    print(f"  LIMM:        {args.use_limm}")
    print(f"  ATII:        {args.use_atii}")
    print(f"  cond_scale:  {args.cond_scale}")
    print(f"  steps:       {args.num_inference_steps}")
    print(f"  output:      {output_root}")
    print("=" * 60)

    # =========================================================================
    # 1. Load VAE
    # =========================================================================
    print("\n[1/3] Loading VAE...")
    vae_dir = pjoin(args.checkpoints_dir, 'sign', args.vae_name)
    vae, vae_opt = build_vae(vae_dir, device)
    vae.to(device)

    # =========================================================================
    # 2. Load Denoiser + Scheduler
    # =========================================================================
    print("\n[2/3] Loading Denoiser...")
    denoiser_opt = Namespace(
        dataset_name='sign',
        text_encoder=args.text_encoder,
        clip_version=args.clip_version,
        xlmr_version=args.xlmr_version,
        latent_dim=args.latent_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        kernel_size=3,
        norm='layer',
        activation='gelu',
        device=device,
        cond_scale=args.cond_scale,
        classifier_free_guidance=args.cond_scale > 1.0,
        num_inference_timesteps=args.num_inference_steps,
        use_limm=args.use_limm,
        use_atii=args.use_atii,
    )

    denoiser = build_denoiser(denoiser_opt, vae_opt, args.denoiser_checkpoint, device)

    scheduler = DDIMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        prediction_type=args.prediction_type,
        clip_sample=False,
    )

    num_params = sum(p.numel() for p in denoiser.parameters())
    print(f"  Denoiser params: {num_params/1e6:.2f}M")

    # =========================================================================
    # 3. Generate & Visualize
    # =========================================================================

    if args.text is not None:
        # ── Custom text mode (no GT) ──
        print(f"\n[3/3] Generating from custom text...")
        out_dir = os.path.join(output_root, 'custom')
        os.makedirs(out_dir, exist_ok=True)

        T_gen = (args.gen_length // 4) * 4
        assert T_gen >= 4, f"gen_length must be >= 4, got {args.gen_length}"

        for idx, text_prompt in enumerate(args.text):
            print(f"\n  [{idx+1}/{len(args.text)}] \"{text_prompt}\" (T={T_gen})")

            # Dummy reference motion for shape
            dummy_motion = torch.zeros(1, T_gen, 133).to(device)
            m_len = torch.tensor([T_gen], dtype=torch.long)

            gen_motion = generate_motion(
                denoiser, vae, scheduler,
                text=[text_prompt], motion_ref=dummy_motion, m_len=m_len,
                device=device, cond_scale=args.cond_scale,
                num_inference_steps=args.num_inference_steps,
            )
            gen_np = gen_motion[0].cpu().numpy()  # [T, 133]

            # FK
            try:
                gen_joints = aa133_to_joints_np(
                    gen_np, mean_t, std_t, args.smplx_path, device=str(device))
            except Exception as e:
                print(f"    FK failed: {e}")
                continue

            if np.isnan(gen_joints).any():
                print(f"    ⚠️ NaN in joints, skip.")
                continue

            safe_text = text_prompt[:50].replace(' ', '_').replace('/', '_')
            video_path = os.path.join(out_dir, f'{idx:03d}_{safe_text}.mp4')
            title = f'"{text_prompt}" (T={T_gen})'
            save_single_video(gen_joints, video_path, title, args.fps, args.viewport)
            print(f"    Saved: {video_path}")

    else:
        # ── Dataset mode (GT vs Generated) ──
        print(f"\n[3/3] Generating GT vs Generated videos...")
        from data.sign_dataset import _build_annotations, _load_one

        assert args.data_root is not None, "--data_root required for dataset mode"

        all_data = _build_annotations(
            split=args.split,
            dataset_name=args.sign_dataset,
            data_root=args.data_root,
            csl_root=args.csl_root,
            phoenix_root=args.phoenix_root,
        )
        print(f"  [{args.split}] {len(all_data)} samples")

        all_metrics = []
        total_count = 0

        src_indices = {}
        for i, item in enumerate(all_data):
            s = item.get('src', 'how2sign')
            src_indices.setdefault(s, []).append(i)

        DS_LABELS = {'how2sign': 'H2S', 'csl': 'CSL', 'phoenix': 'Phoenix'}

        for src_key, indices in src_indices.items():
            if not indices:
                continue

            ds_label = DS_LABELS.get(src_key, src_key)
            n = min(args.num_samples, len(indices))
            sel = [indices[int(i)] for i in np.linspace(0, len(indices)-1, n)]

            out_dir = os.path.join(output_root, ds_label)
            os.makedirs(out_dir, exist_ok=True)
            print(f"\n  [{ds_label}] {len(indices)} total, visualizing {n}")

            for idx_in_sel, dataset_idx in enumerate(sel):
                ann = all_data[dataset_idx]
                name = ann.get('name', f'sample_{dataset_idx}')
                text = ann.get('text', '')

                # Raw load
                poses, text_loaded, name_loaded = _load_one(ann, args.csl_root, args.phoenix_root)
                if poses is None:
                    print(f"    [{idx_in_sel+1}/{n}] {name} — load failed, skip.")
                    continue
                if name_loaded: name = name_loaded
                if text_loaded: text = text_loaded

                # Normalize + 4-multiple crop
                poses_norm = (poses - mean) / (std + 1e-10)
                motion_norm = torch.from_numpy(poses_norm).float()
                T_len = motion_norm.shape[0]
                T_crop = (T_len // 4) * 4
                if T_crop < 4:
                    print(f"    [{idx_in_sel+1}/{n}] {name} — too short ({T_len}), skip.")
                    continue
                motion_norm = motion_norm[:T_crop]
                T_len = T_crop

                # ── Generate ──
                motion_gpu = motion_norm.unsqueeze(0).to(device)
                m_len = torch.tensor([T_len], dtype=torch.long)

                gen_motion = generate_motion(
                    denoiser, vae, scheduler,
                    text=[text], motion_ref=motion_gpu, m_len=m_len,
                    device=device, cond_scale=args.cond_scale,
                    num_inference_steps=args.num_inference_steps,
                )
                gen_np = gen_motion[0].cpu().numpy()
                gt_np = motion_norm.numpy()

                # ── FK ──
                try:
                    gt_joints = aa133_to_joints_np(
                        gt_np, mean_t, std_t, args.smplx_path, device=str(device))
                    gen_joints = aa133_to_joints_np(
                        gen_np, mean_t, std_t, args.smplx_path, device=str(device))
                except Exception as e:
                    print(f"    [{idx_in_sel+1}/{n}] {name} — FK failed: {e}")
                    continue

                # ── Metrics ──
                gt_raw = gt_np * std + mean
                gen_raw = gen_np * std + mean
                metrics = compute_metrics(gt_raw, gen_raw, gt_joints, gen_joints)
                metrics['name'] = name
                metrics['src'] = src_key
                metrics['text'] = text
                metrics['T'] = T_len
                all_metrics.append(metrics)

                J = gt_joints.shape[1]
                text_short = text[:60] + '...' if len(text) > 60 else text
                print(f"    [{idx_in_sel+1}/{n}] {name} (T={T_len}, J={J}) "
                      f"RMSE={metrics['feat_rmse']:.4f} MPJPE={metrics.get('mpjpe', 0):.4f}")
                print(f"      text: \"{text_short}\"")

                if np.isnan(gt_joints).any() or np.isnan(gen_joints).any():
                    print(f"      ⚠️ NaN in joints, skip video.")
                    continue

                # ── Save video ──
                safe_name = str(name)[:40].replace('/', '_').replace('\\', '_')
                video_path = os.path.join(out_dir, f'{idx_in_sel:03d}_{safe_name}.mp4')
                title = f'{name} [{ds_label}] (T={T_len})\n"{text_short}"'
                save_comparison_video(gt_joints, gen_joints, video_path,
                                      title, args.fps, args.viewport, metrics,
                                      left_label='GT', right_label='Generated')
                total_count += 1

        # ── Summary ──
        print(f"\n{'=' * 60}")
        print(f"Done. {total_count} videos saved to {output_root}")

        if all_metrics:
            rmses = [m['feat_rmse'] for m in all_metrics]
            mpjpes = [m['mpjpe'] for m in all_metrics if 'mpjpe' in m]
            print(f"\n  Overall RMSE:  {np.mean(rmses):.6f} ± {np.std(rmses):.6f}")
            if mpjpes:
                print(f"  Overall MPJPE: {np.mean(mpjpes):.6f} ± {np.std(mpjpes):.6f}")

            for src_key in ['how2sign', 'csl', 'phoenix']:
                ms = [m for m in all_metrics if m['src'] == src_key]
                if ms:
                    r = np.mean([m['feat_rmse'] for m in ms])
                    j = np.mean([m['mpjpe'] for m in ms if 'mpjpe' in m])
                    print(f"  [{DS_LABELS[src_key]}] RMSE={r:.6f}  MPJPE={j:.6f}  (n={len(ms)})")

            csv_path = os.path.join(output_root, 'metrics.csv')
            with open(csv_path, 'w') as f:
                f.write('name,src,T,text,feat_rmse,feat_mae,mpjpe\n')
                for m in all_metrics:
                    t = m.get('text', '').replace(',', ';')
                    f.write(f"{m['name']},{m['src']},{m['T']},\"{t}\","
                            f"{m['feat_rmse']:.6f},{m['feat_mae']:.6f},"
                            f"{m.get('mpjpe', 0):.6f}\n")
            print(f"\n  Metrics saved to {csv_path}")

        print("=" * 60)


if __name__ == '__main__':
    main()
