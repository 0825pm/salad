#!/usr/bin/env python3
"""
vis_sign10_denoiser.py — Sign10_vel Denoiser (210D) GT vs Generated 시각화

모든 데이터셋(H2S, CSL, Phoenix) × 모든 split(train, val, test) 한번에 시각화.

Pipeline:
  GT:    133D raw → norm(133D) → FK → 45-joint skeleton
  Gen:   text → Denoiser(DDIM) → latent → SignVAE.decode → 210D
         → strip vel → denorm(120D) → raw120 → pad133D
         → norm(133D) → FK → 45-joint skeleton

Usage:
    cd ~/Projects/research/salad

    python vis_sign10_denoiser.py \
        --denoiser_name sign10_denoiser_v1 \
        --vae_name sign_vae_210d_v1 \
        --data_root /home/user/Projects/research/SOKE/data/How2Sign \
        --csl_root /home/user/Projects/research/SOKE/data/CSL-Daily \
        --phoenix_root /home/user/Projects/research/SOKE/data/Phoenix_2014T \
        --mean_path /home/user/Projects/research/SOKE/data/CSL-Daily/mean.pt \
        --std_path /home/user/Projects/research/SOKE/data/CSL-Daily/std.pt \
        --smplx_path deps/smpl_models/ \
        --num_samples 5

    # 특정 split만
    python vis_sign10_denoiser.py ... --splits val

    # Custom text (GT 없이 생성만)
    python vis_sign10_denoiser.py ... \
        --text "a person signs hello" "thank you" \
        --gen_length 120
"""
import os
import sys
import glob
import json
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

from utils.sign10_config import (
    sign10_vel_to_rotation, reorder_from_sign10,
    load_mean_std_sign10, pad_to_133, rotation_to_sign10_vel,
)
from utils.feats2joints import aa133_to_joints_np
from utils.get_opt import get_opt
from data.sign_dataset import _build_annotations, _load_one
from data.load_sign_data import load_mean_std as load_mean_std_133
from os.path import join as pjoin


# =============================================================================
# Skeleton config — 45-joint layout
# =============================================================================

SMPLX_SELECT_IDX = [22] + [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] \
                   + list(range(25, 40)) + list(range(40, 55))

ROOT_IDX = 4   # spine3 in local 45-joint order

BODY_IDX  = list(range(0, 15))
LHAND_IDX = list(range(15, 30))
RHAND_IDX = list(range(30, 45))

BODY_CONNECTIONS = [
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 8),
    (4, 6), (6, 9), (9, 11), (11, 13),
    (4, 7), (7, 10), (10, 12), (12, 14),
    (8, 0),
]

def _hand_connections(wrist_idx, hand_offset):
    conns = []
    for finger in range(5):
        base = hand_offset + finger * 3
        conns.append((wrist_idx, base))
        conns.append((base, base + 1))
        conns.append((base + 1, base + 2))
    return conns

LHAND_CONNECTIONS = _hand_connections(13, 15)
RHAND_CONNECTIONS = _hand_connections(14, 30)
ALL_CONNECTIONS = BODY_CONNECTIONS + LHAND_CONNECTIONS + RHAND_CONNECTIONS

DS_LABELS = {'how2sign': 'H2S', 'csl': 'CSL', 'phoenix': 'Phoenix'}


# =============================================================================
# Helpers
# =============================================================================

def normalize_to_root(joints, root_idx=ROOT_IDX):
    if len(joints.shape) == 3:
        return joints - joints[:, root_idx:root_idx+1, :]
    return joints - joints[root_idx:root_idx+1, :]


def compute_metrics(gt_joints, gen_joints):
    """MPJPE 계산 (body / hand / jaw)."""
    T = min(gt_joints.shape[0], gen_joints.shape[0])
    J = min(gt_joints.shape[1], gen_joints.shape[1])
    gt, gn = gt_joints[:T, :J], gen_joints[:T, :J]
    per_joint = np.sqrt(np.sum((gt - gn) ** 2, axis=-1))

    body_idx = [i for i in range(1, 15) if i < J]
    hand_idx = [i for i in range(15, 45) if i < J]
    jaw_idx  = [0] if J > 0 else []

    return {
        'all_mpjpe':  float(per_joint.mean()),
        'body_mpjpe': float(per_joint[:, body_idx].mean()) if body_idx else 0.0,
        'hand_mpjpe': float(per_joint[:, hand_idx].mean()) if hand_idx else 0.0,
        'jaw_mpjpe':  float(per_joint[:, jaw_idx].mean())  if jaw_idx  else 0.0,
    }


def lengths_to_mask(lengths):
    max_frames = torch.max(lengths)
    mask = torch.arange(max_frames, device=lengths.device).expand(
        len(lengths), max_frames) < lengths.unsqueeze(1)
    return mask


def gen_210d_to_joints(gen_210d_np, mean_sign10, std_sign10, mean_133, std_133,
                       smplx_path, device='cpu'):
    """210D generated → FK → joints [T, 45, 3].

    Pipeline: 210D → strip vel → 120D norm → denorm → raw120 → pad133
              → norm(133D) → FK → select 45 joints
    """
    sign10_norm = sign10_vel_to_rotation(gen_210d_np)            # [T, 120]
    sign10_denorm = sign10_norm * (std_sign10 + 1e-10) + mean_sign10
    raw120 = reorder_from_sign10(sign10_denorm)                   # [T, 120]
    aa133 = pad_to_133(raw120)                                    # [T, 133]
    aa133_norm = (aa133 - mean_133) / (std_133 + 1e-10)          # normalize for FK

    joints_full = aa133_to_joints_np(
        aa133_norm, mean_133, std_133, smplx_path, device=device)
    return joints_full[:, SMPLX_SELECT_IDX]                       # [T, 45, 3]


def raw_to_joints(poses, mean_133, std_133, smplx_path, device='cpu'):
    """120D or 133D raw → FK → joints [T, 45, 3].

    _load_one returns 120D for sign data (jaw/expr dropped).
    We pad to 133D before FK.
    """
    D = poses.shape[-1]
    if D == 120:
        poses_133 = pad_to_133(poses)  # append zero jaw(3) + expr(10)
    elif D == 133:
        poses_133 = poses
    else:
        raise ValueError(f"Expected 120D or 133D, got {D}")

    poses_norm = (poses_133 - mean_133) / (std_133 + 1e-10)
    joints_full = aa133_to_joints_np(
        poses_norm, mean_133, std_133, smplx_path, device=device)
    return joints_full[:, SMPLX_SELECT_IDX]


# =============================================================================
# Video saving
# =============================================================================

def save_comparison_video(gt_joints, gen_joints, save_path, title='',
                          fps=25, viewport=0.5, metrics=None):
    """GT(왼쪽) vs Generated(오른쪽) 2D skeleton video."""
    T = min(gt_joints.shape[0], gen_joints.shape[0])
    gt = normalize_to_root(gt_joints[:T].copy())
    gn = normalize_to_root(gen_joints[:T].copy())

    if viewport > 0:
        lim = (-viewport, viewport)
    else:
        all_data = np.concatenate([gt, gn], axis=0)
        r = max(all_data[:,:,0].ptp(), all_data[:,:,1].ptp(), 0.1) * 1.2
        cx = (all_data[:,:,0].max() + all_data[:,:,0].min()) / 2
        cy = (all_data[:,:,1].max() + all_data[:,:,1].min()) / 2
        lim = None  # per-axis below

    fig, (ax_gt, ax_gen) = plt.subplots(1, 2, figsize=(12, 6))

    main_title = title
    if metrics:
        main_title += f"\nbody={metrics['body_mpjpe']:.4f}  hand={metrics['hand_mpjpe']:.4f}  jaw={metrics['jaw_mpjpe']:.4f}"
    fig.suptitle(main_title, fontsize=10)

    labels = [('GT (133D original)', 'blue'), ('Generated (Denoiser→210D)', 'darkgreen')]
    for ax, (label, color) in zip([ax_gt, ax_gen], labels):
        ax.set_title(label, fontsize=11, fontweight='bold', color=color)
        if viewport > 0:
            ax.set_xlim(lim); ax.set_ylim(lim)
        else:
            ax.set_xlim(cx - r/2, cx + r/2); ax.set_ylim(cy - r/2, cy + r/2)
        ax.set_aspect('equal'); ax.axis('off')

    colors = {'body': '#2196F3', 'lhand': '#F44336', 'rhand': '#4CAF50'}
    elements = []
    for ax, data in [(ax_gt, gt), (ax_gen, gn)]:
        lines = []
        for (i, j) in ALL_CONNECTIONS:
            if i >= 30 or j >= 30:
                c, lw = colors['rhand'], 1.0
            elif i >= 15 or j >= 15:
                c, lw = colors['lhand'], 1.0
            else:
                c, lw = colors['body'], 1.5
            line, = ax.plot([], [], color=c, linewidth=lw, alpha=0.8)
            lines.append((line, i, j))
        bs = ax.scatter([], [], c=colors['body'],  s=12, zorder=5)
        ls = ax.scatter([], [], c=colors['lhand'], s=5,  zorder=5)
        rs = ax.scatter([], [], c=colors['rhand'], s=5,  zorder=5)
        elements.append((lines, bs, ls, rs, data))

    frame_text = fig.text(0.5, 0.02, '', ha='center', fontsize=9, color='gray')
    plt.tight_layout(rect=[0, 0.04, 1, 0.90])

    def update(frame):
        f = min(frame, T - 1)
        frame_text.set_text(f'Frame {f}/{T-1}')
        for (lines, bs, ls, rs, data) in elements:
            fd = data[f]
            x, y = fd[:, 0], fd[:, 1]
            for (line, i, j) in lines:
                line.set_data([x[i], x[j]], [y[i], y[j]])
            bs.set_offsets(np.c_[x[BODY_IDX], y[BODY_IDX]])
            ls.set_offsets(np.c_[x[LHAND_IDX], y[LHAND_IDX]])
            rs.set_offsets(np.c_[x[RHAND_IDX], y[RHAND_IDX]])
        return []

    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False)
    try:
        anim.save(save_path, writer=FFMpegWriter(fps=fps, bitrate=5000))
    except Exception:
        anim.save(save_path.replace('.mp4', '.gif'), writer='pillow', fps=min(fps, 10))
    plt.close(fig)


def save_single_video(joints, save_path, title='', fps=25, viewport=0.5):
    """Single skeleton video (generated only, no GT)."""
    T = joints.shape[0]
    data = normalize_to_root(joints.copy())

    if viewport > 0:
        lim = (-viewport, viewport)
    else:
        r = max(data[:,:,0].ptp(), data[:,:,1].ptp(), 0.1) * 1.2
        cx, cy = data[:,:,0].mean(), data[:,:,1].mean()
        lim = None

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.suptitle(title, fontsize=10)
    if viewport > 0:
        ax.set_xlim(lim); ax.set_ylim(lim)
    else:
        ax.set_xlim(cx - r/2, cx + r/2); ax.set_ylim(cy - r/2, cy + r/2)
    ax.set_aspect('equal'); ax.axis('off')

    colors = {'body': '#2196F3', 'lhand': '#F44336', 'rhand': '#4CAF50'}
    lines_objs = []
    for (i, j) in ALL_CONNECTIONS:
        if i >= 30 or j >= 30:
            c, lw = colors['rhand'], 1.0
        elif i >= 15 or j >= 15:
            c, lw = colors['lhand'], 1.0
        else:
            c, lw = colors['body'], 1.5
        line, = ax.plot([], [], color=c, linewidth=lw, alpha=0.8)
        lines_objs.append((line, i, j))
    bs = ax.scatter([], [], c=colors['body'],  s=12, zorder=5)
    ls = ax.scatter([], [], c=colors['lhand'], s=5,  zorder=5)
    rs = ax.scatter([], [], c=colors['rhand'], s=5,  zorder=5)
    frame_text = fig.text(0.5, 0.02, '', ha='center', fontsize=9, color='gray')
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])

    def update(frame):
        f = min(frame, T - 1)
        frame_text.set_text(f'Frame {f}/{T-1}')
        fd = data[f]
        x, y = fd[:, 0], fd[:, 1]
        for (line, i, j) in lines_objs:
            line.set_data([x[i], x[j]], [y[i], y[j]])
        bs.set_offsets(np.c_[x[BODY_IDX], y[BODY_IDX]])
        ls.set_offsets(np.c_[x[LHAND_IDX], y[LHAND_IDX]])
        rs.set_offsets(np.c_[x[RHAND_IDX], y[RHAND_IDX]])
        return []

    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False)
    try:
        anim.save(save_path, writer=FFMpegWriter(fps=fps, bitrate=5000))
    except Exception:
        anim.save(save_path.replace('.mp4', '.gif'), writer='pillow', fps=min(fps, 10))
    plt.close(fig)


# =============================================================================
# Model loaders
# =============================================================================

def load_vae(vae_dir, device):
    """Load SignVAE (210D) or standard VAE based on opt.txt."""
    vae_opt = get_opt(pjoin(vae_dir, 'opt.txt'), device)
    skeleton_mode = getattr(vae_opt, 'skeleton_mode', '7part')

    if skeleton_mode == 'sign10_vel':
        from models.sign.model import SignVAE
        model = SignVAE(vae_opt)
    else:
        from models.vae.model import VAE
        model = VAE(vae_opt)

    for ckpt_name in ['net_best_fid.tar', 'latest.tar']:
        ckpt_path = pjoin(vae_dir, 'model', ckpt_name)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            key = 'vae' if 'vae' in ckpt else 'model'
            model.load_state_dict(ckpt[key])
            print(f"  VAE loaded: {ckpt_path} (mode={skeleton_mode})")
            break
    else:
        ckpts = sorted(glob.glob(pjoin(vae_dir, 'model', 'net_epoch*.tar')))
        if ckpts:
            ckpt = torch.load(ckpts[-1], map_location='cpu')
            key = 'vae' if 'vae' in ckpt else 'model'
            model.load_state_dict(ckpt[key])
            print(f"  VAE loaded: {ckpts[-1]}")
        else:
            raise FileNotFoundError(f'No VAE checkpoint in {vae_dir}/model/')

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, vae_opt


def load_denoiser(denoiser_dir, vae_opt, device):
    """Load Denoiser from checkpoint."""
    denoiser_opt = get_opt(pjoin(denoiser_dir, 'opt.txt'), device)

    from models.denoiser.model import Denoiser
    denoiser = Denoiser(denoiser_opt, vae_opt.latent_dim)

    for ckpt_name in ['net_best_fid.tar', 'latest.tar']:
        ckpt_path = pjoin(denoiser_dir, 'model', ckpt_name)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            missing, unexpected = denoiser.load_state_dict(ckpt["denoiser"], strict=False)
            assert all(k.startswith("clip_model.") for k in missing), \
                f"Unexpected missing: {[k for k in missing if not k.startswith('clip_model.')]}"
            # ── Load LoRA / fine-tuned text encoder weights ──
            if "text_encoder" in ckpt:
                denoiser.load_text_encoder_state_dict(ckpt["text_encoder"])
                print(f"  Loaded text encoder weights (mode={ckpt.get('finetune_text_encoder', '?')})")
            epoch = ckpt.get('epoch', '?')
            it = ckpt.get('total_iter', '?')
            print(f"  Denoiser loaded: {ckpt_path} (epoch={epoch}, iter={it})")
            break
    else:
        ckpts = sorted(glob.glob(pjoin(denoiser_dir, 'model', 'net_epoch*.tar')))
        if ckpts:
            ckpt = torch.load(ckpts[-1], map_location='cpu')
            denoiser.load_state_dict(ckpt["denoiser"], strict=False)
            print(f"  Denoiser loaded: {ckpts[-1]}")
        else:
            raise FileNotFoundError(f'No denoiser checkpoint in {denoiser_dir}/model/')

    denoiser.eval().to(device)
    return denoiser, denoiser_opt


# =============================================================================
# Generation
# =============================================================================

@torch.no_grad()
def generate_motion(denoiser, vae, scheduler, text_list, motion_ref,
                    m_len, device, cond_scale=7.5, num_steps=50):
    """Text-conditioned motion generation → [B, T, 210]."""
    motion_ref = motion_ref.to(device, dtype=torch.float32)
    m_lens_latent = m_len.to(device, dtype=torch.long) // 4
    len_mask = lengths_to_mask(m_lens_latent)

    # Get latent shape from reference
    z, _ = vae.encode(motion_ref)
    latents = torch.randn_like(z) * scheduler.init_noise_sigma

    len_mask = F.pad(len_mask, (0, latents.shape[1] - len_mask.shape[1]),
                     mode="constant", value=False)
    latents = latents * len_mask[..., None, None].float()

    use_cfg = cond_scale > 1.0
    input_text = ([""] * len(text_list) + list(text_list)) if use_cfg else list(text_list)

    scheduler.set_timesteps(num_steps)
    for timestep in scheduler.timesteps.to(device):
        if use_cfg:
            inp_lat = torch.cat([latents] * 2, dim=0)
            inp_mask = torch.cat([len_mask] * 2, dim=0)
        else:
            inp_lat, inp_mask = latents, len_mask

        pred, _ = denoiser.forward(inp_lat, timestep, text=input_text,
                                   len_mask=inp_mask, use_cached_clip=True)
        if use_cfg:
            pred_u, pred_c = torch.chunk(pred, 2, dim=0)
            pred = pred_u + cond_scale * (pred_c - pred_u)

        latents = scheduler.step(pred, timestep, latents).prev_sample
        latents = latents * len_mask[..., None, None].float()

    pred_motion = vae.decode(latents)
    if isinstance(pred_motion, (tuple, list)):
        pred_motion = pred_motion[0]

    denoiser.remove_clip_cache()
    return pred_motion


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Sign10_vel Denoiser: GT vs Generated (all datasets × splits)')

    # Model
    parser.add_argument('--denoiser_name', required=True, help='Denoiser experiment name')
    parser.add_argument('--vae_name', required=True, help='VAE experiment name')
    parser.add_argument('--checkpoints_dir', default='./checkpoints')

    # Data
    parser.add_argument('--data_root', required=True, help='How2Sign root')
    parser.add_argument('--csl_root', default=None)
    parser.add_argument('--phoenix_root', default=None)
    parser.add_argument('--mean_path', required=True)
    parser.add_argument('--std_path', required=True)
    parser.add_argument('--smplx_path', default='deps/smpl_models/')
    parser.add_argument('--sign_dataset', default='how2sign_csl_phoenix',
                    choices=['how2sign', 'csl', 'phoenix', 'how2sign_csl', 'how2sign_csl_phoenix'])

    # Splits & samples
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        help='Splits to visualize (default: train val test)')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Samples per dataset per split')

    # Custom text mode
    parser.add_argument('--text', nargs='+', default=None,
                        help='Custom text prompt(s) — no GT, generation only')
    parser.add_argument('--gen_length', type=int, default=120)

    # Generation config
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--cond_scale', type=float, default=7.5)

    # Output
    parser.add_argument('--output', default='vis_denoiser_output')
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--viewport', type=float, default=0.5)
    parser.add_argument('--device', default='cuda:0')

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = os.path.join(args.output, f'{args.denoiser_name}_{timestamp}')
    os.makedirs(output_root, exist_ok=True)

    print("=" * 70)
    print("Sign10_vel Denoiser Visualization")
    print(f"  denoiser:  {args.denoiser_name}")
    print(f"  vae:       {args.vae_name}")
    print(f"  splits:    {args.splits}")
    print(f"  samples:   {args.num_samples} per dataset per split")
    print(f"  cond_scale:{args.cond_scale}")
    print(f"  steps:     {args.num_inference_steps}")
    print(f"  output:    {output_root}")
    print("=" * 70)

    # ── Load models ──
    print("\n[1/4] Loading VAE...")
    vae_dir = pjoin(args.checkpoints_dir, 'sign', args.vae_name)
    vae, vae_opt = load_vae(vae_dir, device)
    skeleton_mode = getattr(vae_opt, 'skeleton_mode', '7part')
    pose_dim = getattr(vae_opt, 'pose_dim', 133)
    print(f"  skeleton_mode={skeleton_mode}, pose_dim={pose_dim}")

    print("\n[2/4] Loading Denoiser...")
    denoiser_dir = pjoin(args.checkpoints_dir, 'sign', args.denoiser_name)
    denoiser, denoiser_opt = load_denoiser(denoiser_dir, vae_opt, device)

    scheduler = DDIMScheduler(
        num_train_timesteps=getattr(denoiser_opt, 'num_train_timesteps', 1000),
        beta_start=getattr(denoiser_opt, 'beta_start', 0.00085),
        beta_end=getattr(denoiser_opt, 'beta_end', 0.012),
        beta_schedule=getattr(denoiser_opt, 'beta_schedule', 'scaled_linear'),
        prediction_type=getattr(denoiser_opt, 'prediction_type', 'v_prediction'),
        clip_sample=False,
    )

    num_params = sum(p.numel() for p in denoiser.parameters())
    print(f"  Denoiser params: {num_params/1e6:.2f}M")

    # ── Load mean/std ──
    print("\n[3/4] Loading mean/std...")
    mean_sign10, std_sign10 = load_mean_std_sign10(args.mean_path, args.std_path)
    mean_raw, std_raw = load_mean_std_133(args.mean_path, args.std_path)

    # mean/std가 120D일 수 있음 → 133D로 패딩 (jaw=0/1, expr=0/1)
    if mean_raw.shape[-1] == 120:
        mean_133 = np.concatenate([mean_raw, np.zeros(13, dtype=mean_raw.dtype)])
        std_133  = np.concatenate([std_raw,  np.ones(13, dtype=std_raw.dtype)])
        print(f"  mean/std 120D → padded to 133D (jaw/expr: mean=0, std=1)")
    else:
        mean_133 = mean_raw
        std_133  = std_raw
    print(f"  120D sign10: mean={mean_sign10.shape}, 133D: mean={mean_133.shape}")

    # =========================================================================
    # Custom text mode
    # =========================================================================
    if args.text is not None:
        print(f"\n[4/4] Custom text generation ({len(args.text)} prompts)...")
        out_dir = os.path.join(output_root, 'custom_text')
        os.makedirs(out_dir, exist_ok=True)

        T_gen = (args.gen_length // 4) * 4
        assert T_gen >= 4

        for idx, text_prompt in enumerate(args.text):
            print(f"\n  [{idx+1}/{len(args.text)}] \"{text_prompt}\" (T={T_gen})")

            # Dummy reference (for latent shape only)
            dummy = torch.zeros(1, T_gen, pose_dim).to(device)
            m_len = torch.tensor([T_gen], dtype=torch.long)

            gen_210 = generate_motion(
                denoiser, vae, scheduler,
                [text_prompt], dummy, m_len, device,
                cond_scale=args.cond_scale, num_steps=args.num_inference_steps,
            )
            gen_np = gen_210[0].cpu().numpy()  # [T, 210]

            try:
                gen_joints = gen_210d_to_joints(
                    gen_np, mean_sign10, std_sign10, mean_133, std_133,
                    args.smplx_path, device=str(device))
            except Exception as e:
                print(f"    FK failed: {e}")
                continue

            if np.isnan(gen_joints).any():
                print(f"    NaN in joints, skip.")
                continue

            safe_text = text_prompt[:50].replace(' ', '_').replace('/', '_')
            video_path = os.path.join(out_dir, f'{idx:03d}_{safe_text}.mp4')
            save_single_video(gen_joints, video_path,
                              f'"{text_prompt}" (T={T_gen})', args.fps, args.viewport)
            print(f"    Saved: {video_path}")

        print(f"\nCustom text done → {out_dir}")
        return

    # =========================================================================
    # Dataset mode: all datasets × all splits
    # =========================================================================
    print(f"\n[4/4] Generating GT vs Generated across all datasets × splits...")

    all_metrics = []
    total_videos = 0

    for split in args.splits:
        print(f"\n{'='*60}")
        print(f"  Split: {split}")
        print(f"{'='*60}")

        # Build annotations for this split
        try:
            all_data = _build_annotations(
                split, args.sign_dataset, args.data_root,
                args.csl_root, args.phoenix_root)
        except Exception as e:
            print(f"  ⚠️ Failed to build annotations for {split}: {e}")
            continue

        if not all_data:
            print(f"  No data for split={split}, skip.")
            continue

        # Group by source
        src_indices = {}
        for i, item in enumerate(all_data):
            src_indices.setdefault(item['src'], []).append(i)

        for src_key, indices in src_indices.items():
            if not indices:
                continue

            ds_label = DS_LABELS.get(src_key, src_key)
            n = min(args.num_samples, len(indices))
            sel = [indices[int(i)] for i in np.linspace(0, len(indices)-1, n)]

            out_dir = os.path.join(output_root, split, ds_label)
            os.makedirs(out_dir, exist_ok=True)
            print(f"\n  [{split}/{ds_label}] {len(indices)} total, visualizing {n}")

            for idx_s, dataset_idx in enumerate(sel):
                ann = all_data[dataset_idx]
                name = ann.get('name', f'sample_{dataset_idx}')
                text = ann.get('text', '')
                text_short = text[:60] + '...' if len(text) > 60 else text

                # ── Load GT (133D) ──
                poses_133, text_loaded, name_loaded = _load_one(
                    ann, args.csl_root, args.phoenix_root, skeleton_mode='7part')
                if poses_133 is None:
                    print(f"    [{idx_s+1}/{n}] {name[:40]} — load failed, skip.")
                    continue
                if text_loaded: text = text_loaded
                if name_loaded: name = name_loaded

                T_len = poses_133.shape[0]
                T_crop = (T_len // 4) * 4
                if T_crop < 4:
                    print(f"    [{idx_s+1}/{n}] {name[:40]} — too short ({T_len}), skip.")
                    continue
                poses_133 = poses_133[:T_crop]
                T_len = T_crop

                # ── GT FK ──
                try:
                    gt_joints = raw_to_joints(
                        poses_133, mean_133, std_133, args.smplx_path, device=str(device))
                except Exception as e:
                    print(f"    [{idx_s+1}/{n}] {name[:40]} — FK(GT) failed: {e}")
                    continue

                # ── Generate: need 210D reference for latent shape ──
                # Load sign10, normalize, convert to 210D
                poses_sign10, _, _ = _load_one(
                    ann, args.csl_root, args.phoenix_root, skeleton_mode='sign10')
                if poses_sign10 is None:
                    print(f"    [{idx_s+1}/{n}] {name[:40]} — sign10 load failed, skip.")
                    continue
                poses_sign10 = poses_sign10[:T_crop]
                sign10_norm = (poses_sign10 - mean_sign10) / (std_sign10 + 1e-10)
                sign10_vel = rotation_to_sign10_vel(sign10_norm)  # [T, 210]

                motion_ref = torch.from_numpy(sign10_vel).float().unsqueeze(0).to(device)
                m_len = torch.tensor([T_len], dtype=torch.long)

                gen_210 = generate_motion(
                    denoiser, vae, scheduler,
                    [text], motion_ref, m_len, device,
                    cond_scale=args.cond_scale, num_steps=args.num_inference_steps,
                )
                gen_np = gen_210[0].cpu().numpy()  # [T, 210]

                # ── Generated FK ──
                try:
                    gen_joints = gen_210d_to_joints(
                        gen_np, mean_sign10, std_sign10, mean_133, std_133,
                        args.smplx_path, device=str(device))
                except Exception as e:
                    print(f"    [{idx_s+1}/{n}] {name[:40]} — FK(Gen) failed: {e}")
                    continue

                # ── Metrics ──
                metrics = compute_metrics(gt_joints, gen_joints)
                metrics['name'] = name
                metrics['src'] = src_key
                metrics['split'] = split
                metrics['text'] = text
                metrics['T'] = T_len
                all_metrics.append(metrics)

                print(f"    [{idx_s+1}/{n}] {name[:40]} (T={T_len}) "
                      f"body={metrics['body_mpjpe']:.4f} hand={metrics['hand_mpjpe']:.4f}")
                print(f"      text: \"{text_short}\"")

                if np.isnan(gt_joints).any() or np.isnan(gen_joints).any():
                    print(f"      ⚠️ NaN in joints, skip video.")
                    continue

                # ── Save video ──
                safe_name = str(name)[:40].replace('/', '_').replace('\\', '_')
                video_path = os.path.join(out_dir, f'{idx_s:03d}_{safe_name}.mp4')
                vid_title = f'{name[:30]} [{ds_label}/{split}] (T={T_len})'
                save_comparison_video(gt_joints, gen_joints, video_path,
                                      vid_title, args.fps, args.viewport, metrics)
                total_videos += 1

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"Done. {total_videos} videos generated → {output_root}")

    if all_metrics:
        # Per split × dataset summary
        for split in args.splits:
            split_m = [m for m in all_metrics if m['split'] == split]
            if not split_m:
                continue
            print(f"\n  [{split}] ({len(split_m)} samples)")
            for src_key in ['how2sign', 'csl', 'phoenix']:
                src_m = [m for m in split_m if m['src'] == src_key]
                if not src_m:
                    continue
                body = np.mean([m['body_mpjpe'] for m in src_m])
                hand = np.mean([m['hand_mpjpe'] for m in src_m])
                jaw  = np.mean([m['jaw_mpjpe'] for m in src_m])
                all_ = np.mean([m['all_mpjpe'] for m in src_m])
                ds_label = DS_LABELS.get(src_key, src_key)
                print(f"    {ds_label:8s}: all={all_:.4f}  body={body:.4f}  hand={hand:.4f}  jaw={jaw:.4f}")

        # Overall
        body_all = np.mean([m['body_mpjpe'] for m in all_metrics])
        hand_all = np.mean([m['hand_mpjpe'] for m in all_metrics])
        all_all  = np.mean([m['all_mpjpe'] for m in all_metrics])
        print(f"\n  Overall ({len(all_metrics)} samples): all={all_all:.4f}  body={body_all:.4f}  hand={hand_all:.4f}")

        # Save metrics JSON
        metrics_path = os.path.join(output_root, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2, default=str)
        print(f"  Metrics saved: {metrics_path}")

    print(f"{'='*70}")


if __name__ == '__main__':
    main()