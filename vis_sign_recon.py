"""
vis_sign_recon.py — GT vs VAE Reconstruction 비교 (133D axis-angle)

SALAD sign VAE → GT vs Recon skeleton 비교 영상

Usage:
    cd ~/Projects/research/salad

    # 최신 체크포인트
    python vis_sign_recon.py \
        --checkpoint checkpoints/sign/sign_vae_v2/model/latest.tar \
        --data_root ./dataset/How2Sign \
        --mean_path ./dataset/How2Sign/mean.pt \
        --std_path ./dataset/How2Sign/std.pt \
        --viewport 0.5 --num_samples 5

    # CPU only
    python vis_sign_recon.py --checkpoint ... --device cpu
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.feats2joints import aa133_to_joints_np


# =============================================================================
# Constants — SMPL-X joint indices (same as original vis_6d_recon.py)
# =============================================================================
SMPLX_UPPER_BODY = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
SMPLX_LHAND = list(range(25, 40))
SMPLX_RHAND = list(range(40, 55))
SMPLX_VALID = SMPLX_UPPER_BODY + SMPLX_LHAND + SMPLX_RHAND


# =============================================================================
# Skeleton Visualization (unchanged from vis_6d_recon.py)
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


def compute_metrics(gt_feats, recon_feats, gt_joints, recon_joints):
    metrics = {}
    metrics['feat_rmse'] = np.sqrt(np.mean((gt_feats - recon_feats) ** 2))
    metrics['feat_mae'] = np.mean(np.abs(gt_feats - recon_feats))
    if gt_joints is not None and recon_joints is not None:
        T = min(gt_joints.shape[0], recon_joints.shape[0])
        J = min(gt_joints.shape[1], recon_joints.shape[1])
        mpjpe = np.sqrt(np.sum((gt_joints[:T, :J] - recon_joints[:T, :J]) ** 2, axis=-1)).mean()
        metrics['mpjpe'] = mpjpe
    return metrics


def save_comparison_video(gt_joints, recon_joints, save_path, title='',
                          fps=25, viewport=1.0, metrics=None):
    """GT(왼쪽) vs Recon(오른쪽) side-by-side skeleton video"""
    T = min(gt_joints.shape[0], recon_joints.shape[0])
    J = min(gt_joints.shape[1], recon_joints.shape[1])

    root_idx = 9 if J > 21 else 0
    gt = normalize_to_root(gt_joints[:T, :J].copy(), root_idx)
    recon = normalize_to_root(recon_joints[:T, :J].copy(), root_idx)

    if viewport > 0:
        x_lim = (-viewport, viewport)
        y_lim = (-viewport, viewport)
    else:
        valid_idx = [i for i in SMPLX_VALID if i < J]
        all_data = np.concatenate([gt[:, valid_idx], recon[:, valid_idx]], axis=0)
        all_x, all_y = all_data[:, :, 0].flatten(), all_data[:, :, 1].flatten()
        max_range = max(all_x.max() - all_x.min(), all_y.max() - all_y.min(), 0.1) * 1.2
        x_mid = (all_x.max() + all_x.min()) / 2
        y_mid = (all_y.max() + all_y.min()) / 2
        x_lim = (x_mid - max_range/2, x_mid + max_range/2)
        y_lim = (y_mid - max_range/2, y_mid + max_range/2)

    fig, (ax_gt, ax_recon) = plt.subplots(1, 2, figsize=(12, 6))
    main_title = title
    if metrics:
        main_title += f"\nRMSE={metrics.get('feat_rmse', 0):.4f}  "
        main_title += f"MPJPE={metrics.get('mpjpe', 0):.4f}"
    fig.suptitle(main_title, fontsize=10)

    for ax, label in [(ax_gt, 'GT'), (ax_recon, 'Recon')]:
        ax.set_title(label, fontsize=12, fontweight='bold',
                     color='blue' if label == 'GT' else 'red')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')
        ax.axis('off')

    connections = get_connections(J)
    colors = {'body': 'blue', 'lhand': 'red', 'rhand': 'green'}
    ub_idx = [i for i in SMPLX_UPPER_BODY if i < J]

    elements = []
    for ax, data in [(ax_gt, gt), (ax_recon, recon)]:
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


# =============================================================================
# Checkpoint finder
# =============================================================================

def find_checkpoint(ckpt_arg, model_dir=None):
    """Find checkpoint: 'latest' or specific path."""
    if ckpt_arg == 'latest' and model_dir:
        path = os.path.join(model_dir, 'latest.tar')
        if os.path.exists(path):
            return path
        # Try glob for any .tar
        tars = sorted(glob.glob(os.path.join(model_dir, '*.tar')))
        if tars:
            return tars[-1]
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")
    elif os.path.isfile(ckpt_arg):
        return ckpt_arg
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_arg}")


# =============================================================================
# Build VAE from options (minimal opt reconstruction)
# =============================================================================

def build_vae_from_opt(opt_overrides=None):
    """Build VAE model with sign config."""
    from argparse import Namespace
    skeleton_mode = (opt_overrides or {}).get('skeleton_mode', '7part')
    from utils.sign_paramUtil import get_sign_config
    _, _, _, num_parts = get_sign_config(skeleton_mode)
    opt = Namespace(
        dataset_name='sign',
        skeleton_mode=skeleton_mode,
        joints_num=num_parts,
        pose_dim=133,
        latent_dim=32,
        kernel_size=3,
        n_layers=2,
        n_extra_layers=1,
        norm='none',
        activation='gelu',
        dropout=0.1,
    )
    if opt_overrides:
        for k, v in opt_overrides.items():
            setattr(opt, k, v)

    from models.vae.model import VAE
    return VAE(opt), opt


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='SALAD Sign VAE: GT vs Recon Visualization')
    parser.add_argument('--checkpoint', required=True, help='path to .tar checkpoint or "latest"')
    parser.add_argument('--model_dir', default=None, help='model dir (used with --checkpoint latest)')
    parser.add_argument('--data_root', required=True, help='How2Sign data root')
    parser.add_argument('--csl_root', default=None)
    parser.add_argument('--phoenix_root', default=None)
    parser.add_argument('--mean_path', required=True)
    parser.add_argument('--std_path', required=True)
    parser.add_argument('--smplx_path', default='deps/smpl_models/',
                        help='path to SMPL-X model directory')
    parser.add_argument('--output', default='vis_sign_output')
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--split', default='val')
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--viewport', type=float, default=0.5, help='0=auto')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--sign_dataset', default='how2sign',
                        choices=['how2sign', 'csl', 'how2sign_csl', 'how2sign_csl_phoenix'])
    # VAE arch (must match training config)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_extra_layers', type=int, default=1)
    parser.add_argument('--skeleton_mode', type=str, default='7part',
                        choices=['7part', 'finger'])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    ckpt_path = find_checkpoint(args.checkpoint, args.model_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = os.path.join(args.output, f'recon_{timestamp}')
    os.makedirs(output_root, exist_ok=True)

    # Load mean/std
    mean = torch.load(args.mean_path, map_location='cpu').float()
    std = torch.load(args.std_path, map_location='cpu').float()
    # Filter to 133D if needed (179D → 133D: drop lower body 36D + shape 10D)
    if mean.shape[0] == 179:
        keep = list(range(36, 126)) + list(range(126+10, 179))  # skip [0:36] lower, [126:136] shape
        mean = mean[keep]
        std = std[keep]
    assert mean.shape[0] == 133, f"Expected 133D mean, got {mean.shape[0]}"

    print("=" * 60)
    print("SALAD Sign VAE Reconstruction: GT vs Recon (133D)")
    print(f"  checkpoint:  {ckpt_path}")
    print(f"  smplx model: {args.smplx_path}")
    print(f"  output:      {output_root}")
    print("=" * 60)

    # =========================================================================
    # 1. Load Dataset (raw, no window crop)
    # =========================================================================
    print("\n[1/3] Loading dataset...")
    from data.sign_dataset import _build_annotations, _load_one

    # Build annotation list only (no __getitem__ needed)
    all_data = _build_annotations(
        split=args.split,
        dataset_name=args.sign_dataset,
        data_root=args.data_root,
        csl_root=args.csl_root,
        phoenix_root=args.phoenix_root,
    )
    print(f"  [{args.split}] {len(all_data)} samples")

    # =========================================================================
    # 2. Load VAE
    # =========================================================================
    print("\n[2/3] Loading VAE...")
    vae, opt = build_vae_from_opt({
        'skeleton_mode': args.skeleton_mode,
        'latent_dim': args.latent_dim,
        'n_layers': args.n_layers,
        'n_extra_layers': args.n_extra_layers,
    })

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('vae', ckpt.get('state_dict', ckpt))
    missing, unexpected = vae.load_state_dict(state, strict=False)
    vae.eval().to(device)

    epoch = ckpt.get('epoch', '?')
    total_iter = ckpt.get('total_iter', '?')
    total_params = sum(p.numel() for p in vae.parameters())
    print(f"  Epoch: {epoch}, Iter: {total_iter}, Params: {total_params/1e6:.2f}M")
    if missing:
        print(f"  Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    # =========================================================================
    # 3. GT vs Recon
    # =========================================================================
    print(f"\n[3/3] Generating GT vs Recon videos...")
    all_metrics = []
    total_count = 0

    # Group by source
    src_indices = {}
    for i, item in enumerate(all_data):
        s = item.get('src', 'how2sign')
        src_indices.setdefault(s, []).append(i)

    DS_LABELS = {'how2sign': 'H2S', 'csl': 'CSL', 'phoenix': 'Phoenix'}
    mean_np, std_np = mean.numpy(), std.numpy()

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

            # Raw load → normalize (full length, no crop)
            poses, text, name_loaded = _load_one(ann, args.csl_root, args.phoenix_root)
            if poses is None:
                print(f"    [{idx_in_sel+1}/{n}] {name} — load failed, skip.")
                continue
            if name_loaded:
                name = name_loaded

            # Normalize
            poses_norm = (poses - mean_np) / (std_np + 1e-10)
            motion_norm = torch.from_numpy(poses_norm).float()
            T_len = motion_norm.shape[0]

            # Ensure 4-multiple for VAE temporal pooling
            T_crop = (T_len // 4) * 4
            if T_crop < 4:
                print(f"    [{idx_in_sel+1}/{n}] {name} — too short ({T_len}), skip.")
                continue
            motion_norm = motion_norm[:T_crop]
            T_len = T_crop

            if motion_norm.abs().max() < 1e-8:
                print(f"    [{idx_in_sel+1}/{n}] {name} — zeros, skip.")
                continue

            # --- VAE forward ---
            with torch.no_grad():
                motion_gpu = motion_norm.unsqueeze(0).to(device)  # [1, T, 133]
                recon_gpu, _ = vae(motion_gpu)
                recon_norm = recon_gpu[0].cpu()   # [T, 133]

            gt_np = motion_norm.numpy()
            recon_np = recon_norm.numpy()

            # --- 133D → joints via SMPL-X FK ---
            try:
                gt_joints = aa133_to_joints_np(
                    gt_np, mean, std, args.smplx_path, device=str(device))
                recon_joints = aa133_to_joints_np(
                    recon_np, mean, std, args.smplx_path, device=str(device))
            except Exception as e:
                print(f"    [{idx_in_sel+1}/{n}] {name} — FK failed: {e}")
                continue

            # --- Denormalized features for RMSE ---
            gt_raw = gt_np * std_np + mean_np
            recon_raw = recon_np * std_np + mean_np

            # --- Metrics ---
            metrics = compute_metrics(gt_raw, recon_raw, gt_joints, recon_joints)
            metrics['name'] = name
            metrics['src'] = src_key
            metrics['T'] = T_len
            all_metrics.append(metrics)

            J = gt_joints.shape[1]
            print(f"    [{idx_in_sel+1}/{n}] {name} (T={T_len}, J={J}) "
                  f"RMSE={metrics['feat_rmse']:.4f} MPJPE={metrics.get('mpjpe', 0):.4f}")

            if np.isnan(gt_joints).any() or np.isnan(recon_joints).any():
                print(f"      ⚠️ NaN in joints, skip video.")
                continue

            # --- Save video ---
            safe_name = str(name)[:40].replace('/', '_').replace('\\', '_')
            video_path = os.path.join(out_dir, f'{idx_in_sel:03d}_{safe_name}.mp4')
            title = f'{name} [{ds_label}] (T={T_len})'
            save_comparison_video(gt_joints, recon_joints, video_path,
                                  title, args.fps, args.viewport, metrics)
            total_count += 1

    # =========================================================================
    # Summary
    # =========================================================================
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
            f.write('name,src,T,feat_rmse,feat_mae,mpjpe\n')
            for m in all_metrics:
                f.write(f"{m['name']},{m['src']},{m['T']},"
                        f"{m['feat_rmse']:.6f},{m['feat_mae']:.6f},"
                        f"{m.get('mpjpe', 0):.6f}\n")
        print(f"\n  Metrics saved to {csv_path}")

    print("=" * 60)


if __name__ == '__main__':
    main()