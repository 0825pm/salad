"""
vis_sign_recon.py — GT vs VAE Reconstruction 검증 + 시각화 (223D / 133D)

검증 항목:
  1. VAE forward shape: 223D 입력 → 223D 출력 확인
  2. Reconstruction 품질: per-part RMSE (body/hand/face/vel)
  3. FK 시각화: GT vs Recon skeleton side-by-side 영상
  4. Per-part MPJPE: body / hand / jaw 분리 집계
  5. 분포 비교: GT vs Recon normalized value 분포

Usage:
    cd ~/Projects/research/salad

    # 223D VAE (default)
    python vis_sign_recon.py \
        --checkpoint latest \
        --model_dir checkpoints/sign/sign_vae_223d_v1/model \
        --data_root /home/user/Projects/research/SOKE/data/How2Sign \
        --csl_root /home/user/Projects/research/SOKE/data/CSL-Daily \
        --phoenix_root /home/user/Projects/research/SOKE/data/Phoenix_2014T \
        --mean_path /home/user/Projects/research/SOKE/data/CSL-Daily/mean.pt \
        --std_path /home/user/Projects/research/SOKE/data/CSL-Daily/std.pt \
        --sign_dataset how2sign_csl_phoenix \
        --smplx_path deps/smpl_models \
        --num_samples 5 --visualize --plot

    # 133D VAE (no hand velocity)
    python vis_sign_recon.py --no_hand_vel ...

    # CPU only
    python vis_sign_recon.py --device cpu ...
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.load_sign_data import (
    load_mean_std, append_hand_velocity, strip_hand_velocity,
    PART_RANGES_133,
)
from data.sign_dataset import _build_annotations, _load_one


# =============================================================================
# Constants — SMPL-X joint indices
# =============================================================================
SMPLX_UPPER_BODY = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
SMPLX_LHAND = list(range(25, 40))
SMPLX_RHAND = list(range(40, 55))
SMPLX_VALID = SMPLX_UPPER_BODY + SMPLX_LHAND + SMPLX_RHAND

# Part ranges for reconstruction analysis (133D rotation only)
RECON_PART_RANGES = PART_RANGES_133

# 223D velocity parts
VEL_PART_RANGES = {
    'lhand_vel': (133, 178),
    'rhand_vel': (178, 223),
}

DS_LABELS = {'how2sign': 'H2S', 'csl': 'CSL', 'phoenix': 'Phoenix'}


# =============================================================================
# Skeleton Visualization — verify_120d_loader.py style
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
    return [(i, j) for i, j in upper_body + hand_connections
            if i < num_joints and j < num_joints]


def normalize_to_root(joints, root_idx=9):
    if len(joints.shape) == 3:
        return joints - joints[:, root_idx:root_idx+1, :]
    return joints - joints[root_idx:root_idx+1, :]


# =============================================================================
# Metrics — per-part decomposition
# =============================================================================

def compute_feat_metrics(gt_raw, recon_raw):
    """Per-part feature-space RMSE/MAE (133D rotation only)."""
    T = min(gt_raw.shape[0], recon_raw.shape[0])
    gt, rec = gt_raw[:T], recon_raw[:T]
    metrics = {}

    # Overall
    metrics['feat_rmse'] = np.sqrt(np.mean((gt - rec) ** 2))
    metrics['feat_mae']  = np.mean(np.abs(gt - rec))

    # Per-part
    D = min(gt.shape[-1], rec.shape[-1], 133)
    for name, (s, e) in RECON_PART_RANGES.items():
        if e <= D:
            part_rmse = np.sqrt(np.mean((gt[..., s:e] - rec[..., s:e]) ** 2))
            metrics[f'feat_{name}_rmse'] = part_rmse

    return metrics


def compute_joint_metrics(gt_joints, recon_joints):
    """Per-region MPJPE in joint space."""
    T = min(gt_joints.shape[0], recon_joints.shape[0])
    J = min(gt_joints.shape[1], recon_joints.shape[1])
    gt, rec = gt_joints[:T, :J], recon_joints[:T, :J]
    per_joint = np.sqrt(np.sum((gt - rec) ** 2, axis=-1))

    body_idx = [i for i in SMPLX_UPPER_BODY if i < J]
    hand_idx = [i for i in SMPLX_LHAND + SMPLX_RHAND if i < J]
    jaw_idx  = [22] if J > 22 else []

    return {
        'all_mpjpe':  per_joint.mean(),
        'body_mpjpe': per_joint[:, body_idx].mean() if body_idx else 0,
        'hand_mpjpe': per_joint[:, hand_idx].mean() if hand_idx else 0,
        'jaw_mpjpe':  per_joint[:, jaw_idx].mean()  if jaw_idx  else 0,
        'per_joint_mean': per_joint.mean(axis=0),
    }


# =============================================================================
# Side-by-side video — verify_120d_loader.py style
# =============================================================================

def save_comparison_video(gt_joints, recon_joints, save_path, title='',
                          fps=25, viewport=0.5, diff_stats=None):
    """GT(왼쪽) vs Recon(오른쪽) side-by-side 2D skeleton video"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

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
        all_x = all_data[:, :, 0].flatten()
        all_y = all_data[:, :, 1].flatten()
        max_range = max(all_x.max() - all_x.min(), all_y.max() - all_y.min(), 0.1) * 1.2
        x_mid = (all_x.max() + all_x.min()) / 2
        y_mid = (all_y.max() + all_y.min()) / 2
        x_lim = (x_mid - max_range/2, x_mid + max_range/2)
        y_lim = (y_mid - max_range/2, y_mid + max_range/2)

    fig, (ax_gt, ax_rec) = plt.subplots(1, 2, figsize=(12, 6))
    main_title = title
    if diff_stats:
        main_title += f"\nbody MPJPE={diff_stats['body_mpjpe']:.4f}  "
        main_title += f"hand MPJPE={diff_stats['hand_mpjpe']:.4f}  "
        main_title += f"jaw MPJPE={diff_stats['jaw_mpjpe']:.4f}"
    fig.suptitle(main_title, fontsize=10)

    labels = [('GT (ground truth)', '#2196F3'), ('Recon (VAE output)', '#E91E63')]
    for ax, (label, color) in zip([ax_gt, ax_rec], labels):
        ax.set_title(label, fontsize=12, fontweight='bold', color=color)
        ax.set_xlim(x_lim); ax.set_ylim(y_lim)
        ax.set_aspect('equal'); ax.axis('off')

    connections = get_connections(J)
    colors = {'body': '#2196F3', 'lhand': '#F44336', 'rhand': '#4CAF50'}
    ub_idx = [i for i in SMPLX_UPPER_BODY if i < J]

    elements = []
    for ax, data in [(ax_gt, gt), (ax_rec, recon)]:
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
        bs = ax.scatter([], [], c=colors['body'],  s=10, zorder=5)
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
    if ckpt_arg == 'latest' and model_dir:
        path = os.path.join(model_dir, 'latest.tar')
        if os.path.exists(path):
            return path
        tars = sorted(glob.glob(os.path.join(model_dir, '*.tar')))
        if tars:
            return tars[-1]
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")
    elif os.path.isfile(ckpt_arg):
        return ckpt_arg
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_arg}")


# =============================================================================
# Build VAE (223D-aware)
# =============================================================================

def build_vae_from_opt(opt_overrides=None):
    """Build VAE model with sign config (supports 223D)."""
    from argparse import Namespace
    from utils.sign_paramUtil import get_sign_config

    skeleton_mode = (opt_overrides or {}).get('skeleton_mode', '7part')
    use_hand_vel = (opt_overrides or {}).get('use_hand_vel', True)
    _, _, _, num_parts = get_sign_config(skeleton_mode, use_hand_vel)

    opt = Namespace(
        dataset_name='sign',
        skeleton_mode=skeleton_mode,
        use_hand_vel=use_hand_vel,
        joints_num=num_parts,
        pose_dim=223 if use_hand_vel else 133,
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
        opt.pose_dim = 223 if opt.use_hand_vel else 133

    from models.vae.model import VAE
    return VAE(opt), opt


# =============================================================================
# Test 1: VAE forward shape verification
# =============================================================================

def test_vae_shape(vae, use_hand_vel, device):
    """VAE 입출력 shape 확인."""
    print("\n" + "=" * 60)
    print("TEST 1: VAE Forward Shape Verification")
    print("=" * 60)

    D = 223 if use_hand_vel else 133
    B, T = 2, 64
    x = torch.randn(B, T, D).to(device)

    with torch.no_grad():
        out, loss_dict = vae(x)

    ok_shape = out.shape == x.shape
    ok_kl = 'loss_kl' in loss_dict and torch.isfinite(loss_dict['loss_kl'])

    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape} -> {'PASS' if ok_shape else 'FAIL'}")
    print(f"  KL:     {loss_dict.get('loss_kl', 'MISSING'):.6f} -> {'PASS' if ok_kl else 'FAIL'}")
    print(f"  Encoder splits: {vae.motion_enc.splits}")

    return ok_shape and ok_kl


# =============================================================================
# Test 2: Per-part reconstruction quality (feature space)
# =============================================================================

def test_reconstruction_quality(all_data, csl_root, phoenix_root,
                                 vae, mean_np, std_np, use_hand_vel,
                                 device, num_samples=10):
    """Part별 feature-space RMSE 분석."""
    print("\n" + "=" * 60)
    print("TEST 2: Per-part Reconstruction Quality (feature RMSE)")
    print("=" * 60)

    all_metrics = []
    indices = np.linspace(0, len(all_data) - 1, min(num_samples, len(all_data)), dtype=int)

    for i in indices:
        ann = all_data[int(i)]
        name = ann.get('name', f'idx_{i}')[:40]
        src  = ann.get('src', '?')

        poses, _, _ = _load_one(ann, csl_root, phoenix_root)
        if poses is None:
            print(f"  [{src}] {name} -- SKIP (load failed)")
            continue

        # Normalize + crop
        poses_norm = (poses - mean_np) / (std_np + 1e-10)
        T_crop = (poses_norm.shape[0] // 4) * 4
        if T_crop < 4:
            continue
        poses_norm = poses_norm[:T_crop]

        # Append velocity if 223D
        if use_hand_vel:
            vae_input = append_hand_velocity(poses_norm)
        else:
            vae_input = poses_norm

        # VAE forward
        with torch.no_grad():
            inp = torch.from_numpy(vae_input).float().unsqueeze(0).to(device)
            out, _ = vae(inp)
            recon_all = out[0].cpu().numpy()

        # Strip velocity for feature comparison (133D)
        recon_133 = strip_hand_velocity(recon_all) if use_hand_vel else recon_all

        # Denormalize
        gt_raw    = poses_norm[:T_crop] * std_np + mean_np
        recon_raw = recon_133 * std_np + mean_np

        metrics = compute_feat_metrics(gt_raw, recon_raw)
        metrics['name'] = name
        metrics['src']  = src
        metrics['T']    = T_crop
        all_metrics.append(metrics)

        print(f"  [{src}] {name} (T={T_crop}) -- "
              f"RMSE={metrics['feat_rmse']:.4f}  "
              f"hand={metrics.get('feat_lhand_rmse', 0):.4f}/{metrics.get('feat_rhand_rmse', 0):.4f}  "
              f"body={metrics.get('feat_upper_a_rmse', 0):.4f}")

    if all_metrics:
        print(f"\n  {'Part':<14} {'Mean RMSE':>10} {'Std':>10}")
        print("  " + "-" * 36)

        # Aggregate per-part
        for name in list(RECON_PART_RANGES.keys()):
            key = f'feat_{name}_rmse'
            vals = [m[key] for m in all_metrics if key in m]
            if vals:
                print(f"  {name:<14} {np.mean(vals):>10.6f} {np.std(vals):>10.6f}")

        overall = [m['feat_rmse'] for m in all_metrics]
        print(f"  {'overall':<14} {np.mean(overall):>10.6f} {np.std(overall):>10.6f}")

    return all_metrics


# =============================================================================
# Test 3: FK visualization — GT vs Recon side-by-side
# =============================================================================

def test_visualization(all_data, csl_root, phoenix_root,
                       vae, mean_np, std_np, use_hand_vel,
                       smplx_path, output_dir, device,
                       num_samples=5, fps=25, viewport=0.5):
    """GT vs Recon FK skeleton side-by-side 영상 생성."""
    print("\n" + "=" * 60)
    print("TEST 3: FK Visualization (GT vs Recon side-by-side)")
    print("=" * 60)

    try:
        from utils.feats2joints import aa133_to_joints_np
    except ImportError:
        print("  !! utils.feats2joints not found -- skipping")
        print("     (salad 프로젝트 루트에서 실행해야 합니다)")
        return [], False

    os.makedirs(output_dir, exist_ok=True)

    mean_t = torch.from_numpy(mean_np).float()
    std_t  = torch.from_numpy(std_np).float()

    # Group by source
    src_indices = {}
    for i, item in enumerate(all_data):
        src_indices.setdefault(item.get('src', 'how2sign'), []).append(i)

    all_stats = []
    count = 0

    for src_key, indices in src_indices.items():
        if not indices:
            continue
        ds_label = DS_LABELS.get(src_key, src_key)
        n = min(num_samples, len(indices))
        sel = [indices[int(i)] for i in np.linspace(0, len(indices)-1, n)]

        out_dir = os.path.join(output_dir, ds_label)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n  [{ds_label}] {len(indices)} total, visualizing {n}")

        for idx_s, dataset_idx in enumerate(sel):
            ann = all_data[dataset_idx]
            name = ann.get('name', f'sample_{dataset_idx}')

            # -- Load raw 133D --
            poses, text, name_loaded = _load_one(ann, csl_root, phoenix_root)
            if poses is None:
                print(f"    [{idx_s+1}/{n}] {name[:40]} -- load failed, skip.")
                continue
            if name_loaded:
                name = name_loaded

            # -- Normalize + crop --
            poses_norm = (poses - mean_np) / (std_np + 1e-10)
            T_raw = poses_norm.shape[0]
            T_crop = (T_raw // 4) * 4
            if T_crop < 4:
                print(f"    [{idx_s+1}/{n}] {name[:40]} -- too short ({T_raw}), skip.")
                continue
            poses_norm = poses_norm[:T_crop]

            # -- Append velocity if 223D --
            if use_hand_vel:
                vae_input = append_hand_velocity(poses_norm)
            else:
                vae_input = poses_norm

            if np.abs(vae_input).max() < 1e-8:
                print(f"    [{idx_s+1}/{n}] {name[:40]} -- zeros, skip.")
                continue

            # -- VAE forward --
            with torch.no_grad():
                inp = torch.from_numpy(vae_input).float().unsqueeze(0).to(device)
                out, _ = vae(inp)
                recon_all = out[0].cpu().numpy()

            # -- Strip velocity -> 133D for FK --
            gt_norm_133    = poses_norm
            recon_norm_133 = strip_hand_velocity(recon_all) if use_hand_vel else recon_all

            # -- FK: 133D -> SMPL-X joints --
            try:
                gt_joints = aa133_to_joints_np(
                    gt_norm_133, mean_t, std_t, smplx_path, device=str(device))
            except Exception as e:
                print(f"    [{idx_s+1}/{n}] {name[:40]} -- FK(GT) failed: {e}")
                continue

            try:
                recon_joints = aa133_to_joints_np(
                    recon_norm_133, mean_t, std_t, smplx_path, device=str(device))
            except Exception as e:
                print(f"    [{idx_s+1}/{n}] {name[:40]} -- FK(Recon) failed: {e}")
                continue

            # -- Feature-space metrics --
            gt_raw    = gt_norm_133 * std_np + mean_np
            recon_raw = recon_norm_133 * std_np + mean_np
            feat_metrics = compute_feat_metrics(gt_raw, recon_raw)

            # -- Joint-space metrics --
            joint_metrics = compute_joint_metrics(gt_joints, recon_joints)

            stats = {**feat_metrics, **joint_metrics}
            stats['name'] = name
            stats['src']  = src_key
            stats['T']    = T_crop
            all_stats.append(stats)

            J = gt_joints.shape[1]
            print(f"    [{idx_s+1}/{n}] {name[:40]} (T={T_crop}, J={J})")
            print(f"      body MPJPE={stats['body_mpjpe']:.6f}  "
                  f"hand MPJPE={stats['hand_mpjpe']:.6f}  "
                  f"jaw MPJPE={stats['jaw_mpjpe']:.6f}  "
                  f"all MPJPE={stats['all_mpjpe']:.6f}")
            print(f"      feat RMSE={stats['feat_rmse']:.6f}  "
                  f"lhand={stats.get('feat_lhand_rmse', 0):.6f}  "
                  f"rhand={stats.get('feat_rhand_rmse', 0):.6f}")

            if np.isnan(gt_joints).any() or np.isnan(recon_joints).any():
                print(f"      !! NaN in joints, skip video.")
                continue

            # -- Save video --
            safe_name = str(name)[:40].replace('/', '_').replace('\\', '_')
            video_path = os.path.join(out_dir, f'{idx_s:03d}_{safe_name}.mp4')
            title = f'{name[:40]} [{ds_label}] (T={T_crop})'
            save_comparison_video(gt_joints, recon_joints, video_path,
                                  title, fps, viewport, stats)
            count += 1

    # -- Summary --
    print(f"\n  Generated {count} videos in {output_dir}")

    if all_stats:
        body = np.mean([s['body_mpjpe'] for s in all_stats])
        hand = np.mean([s['hand_mpjpe'] for s in all_stats])
        jaw  = np.mean([s['jaw_mpjpe']  for s in all_stats])
        all_ = np.mean([s['all_mpjpe']  for s in all_stats])
        rmse = np.mean([s['feat_rmse']  for s in all_stats])

        print(f"\n  === Aggregate Results ===")
        print(f"  body MPJPE:  {body:.6f}")
        print(f"  hand MPJPE:  {hand:.6f}")
        print(f"  jaw  MPJPE:  {jaw:.6f}")
        print(f"  all  MPJPE:  {all_:.6f}")
        print(f"  feat RMSE:   {rmse:.6f}")

        for sk in ['how2sign', 'csl', 'phoenix']:
            ms = [s for s in all_stats if s['src'] == sk]
            if ms:
                b = np.mean([s['body_mpjpe'] for s in ms])
                h = np.mean([s['hand_mpjpe'] for s in ms])
                j = np.mean([s['jaw_mpjpe']  for s in ms])
                r = np.mean([s['feat_rmse']  for s in ms])
                print(f"  [{DS_LABELS[sk]}] body={b:.6f}  hand={h:.6f}  "
                      f"jaw={j:.6f}  RMSE={r:.6f}  (n={len(ms)})")

        # Save CSV
        csv_path = os.path.join(output_dir, 'recon_metrics.csv')
        with open(csv_path, 'w') as f:
            f.write('name,src,T,feat_rmse,feat_mae,'
                    'body_mpjpe,hand_mpjpe,jaw_mpjpe,all_mpjpe')
            for name in RECON_PART_RANGES.keys():
                f.write(f',feat_{name}_rmse')
            f.write('\n')
            for s in all_stats:
                f.write(f"{s['name']},{s['src']},{s['T']},"
                        f"{s['feat_rmse']:.6f},{s['feat_mae']:.6f},"
                        f"{s['body_mpjpe']:.6f},{s['hand_mpjpe']:.6f},"
                        f"{s['jaw_mpjpe']:.6f},{s['all_mpjpe']:.6f}")
                for name in RECON_PART_RANGES.keys():
                    f.write(f",{s.get(f'feat_{name}_rmse', 0):.6f}")
                f.write('\n')
        print(f"\n  Metrics CSV: {csv_path}")

    return all_stats, count > 0


# =============================================================================
# Test 4: Distribution comparison — GT vs Recon
# =============================================================================

def test_distribution_comparison(all_data, csl_root, phoenix_root,
                                  vae, mean_np, std_np, use_hand_vel,
                                  output_dir, device, num_samples=50):
    """GT vs Recon normalized value 분포 비교 (per-part)."""
    print("\n" + "=" * 60)
    print("TEST 4: Distribution Comparison (GT vs Recon, per-part)")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping")
        return

    os.makedirs(output_dir, exist_ok=True)

    gt_all, recon_all = [], []
    indices = np.linspace(0, len(all_data) - 1, min(num_samples, len(all_data)), dtype=int)

    for i in indices:
        ann = all_data[int(i)]
        poses, _, _ = _load_one(ann, csl_root, phoenix_root)
        if poses is None:
            continue

        poses_norm = (poses - mean_np) / (std_np + 1e-10)
        T_crop = (poses_norm.shape[0] // 4) * 4
        if T_crop < 4:
            continue
        poses_norm = poses_norm[:T_crop]

        if use_hand_vel:
            vae_input = append_hand_velocity(poses_norm)
        else:
            vae_input = poses_norm

        with torch.no_grad():
            inp = torch.from_numpy(vae_input).float().unsqueeze(0).to(device)
            out, _ = vae(inp)
            recon = out[0].cpu().numpy()

        recon_133 = strip_hand_velocity(recon) if use_hand_vel else recon
        gt_all.append(poses_norm[:T_crop])
        recon_all.append(recon_133)

    if not gt_all:
        print("  No data loaded!")
        return

    gt_cat    = np.concatenate(gt_all, axis=0)
    recon_cat = np.concatenate(recon_all, axis=0)
    print(f"  Total frames: {gt_cat.shape[0]} from {len(gt_all)} samples")

    # -- Per-part distribution plot --
    parts = RECON_PART_RANGES
    n_parts = len(parts)
    fig, axes = plt.subplots(2, n_parts, figsize=(4 * n_parts, 8))
    fig.suptitle('Normalized Value Distribution: GT (top) vs Recon (bottom)',
                 fontsize=14, y=1.01)

    for col, (name, (s, e)) in enumerate(parts.items()):
        for row, (data, label, color) in enumerate([
            (gt_cat, 'GT', '#2196F3'), (recon_cat, 'Recon', '#E91E63')
        ]):
            ax = axes[row, col]
            vals = data[:, s:e].flatten()
            ax.hist(vals, bins=100, density=True, alpha=0.7, color=color,
                    edgecolor='black', linewidth=0.3)
            ax.set_title(f'{label}: {name}\n[{s}:{e}] ({e-s}D)', fontsize=9)
            ax.set_xlim(-5, 5)
            mu = vals.mean()
            ax.axvline(mu, color='red', linestyle='--', linewidth=0.8,
                       label=f'u={mu:.2f}')
            ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(output_dir, 'gt_vs_recon_distributions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Distribution plot: {path}")

    # -- Per-dim RMSE bar plot --
    T = min(gt_cat.shape[0], recon_cat.shape[0])
    per_dim_rmse = np.sqrt(np.mean((gt_cat[:T] - recon_cat[:T]) ** 2, axis=0))

    fig2, ax2 = plt.subplots(figsize=(14, 4))
    ax2.bar(range(133), per_dim_rmse, color='#E91E63', edgecolor='none', alpha=0.8)
    for name, (s, e) in parts.items():
        ax2.axvspan(s, e, alpha=0.08,
                    color='#F44336' if name in ['lhand', 'rhand'] else '#2196F3')
        ax2.text((s + e) / 2, per_dim_rmse.max() * 0.92, name,
                 ha='center', fontsize=8, rotation=45)
    ax2.set_xlabel('Dimension index (133D)')
    ax2.set_ylabel('RMSE')
    ax2.set_title('Per-dimension Reconstruction RMSE (133D rotation)')
    plt.tight_layout()
    path2 = os.path.join(output_dir, 'per_dim_rmse.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Per-dim RMSE plot: {path2}")

    # -- Std comparison bar --
    fig3, ax3 = plt.subplots(figsize=(14, 3))
    ax3.bar(range(133), std_np, color='steelblue', edgecolor='none',
            alpha=0.6, label='Std (clamped)')
    ax3.bar(range(133), per_dim_rmse, color='#E91E63', edgecolor='none',
            alpha=0.6, label='Recon RMSE')
    ax3.axhline(0.01, color='red', linestyle='--', linewidth=1,
                label='danger threshold (0.01)')
    for name, (s, e) in parts.items():
        ax3.text((s + e) / 2, std_np.max() * 0.9, name, ha='center', fontsize=7)
    ax3.set_xlabel('Dimension index')
    ax3.set_ylabel('Value')
    ax3.set_title('Std vs Reconstruction RMSE per dimension')
    ax3.legend(fontsize=8)
    plt.tight_layout()
    path3 = os.path.join(output_dir, 'std_vs_rmse.png')
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Std vs RMSE plot: {path3}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SALAD Sign VAE: Reconstruction Verification + Visualization')

    parser.add_argument('--checkpoint', required=True,
                        help='path to .tar checkpoint or "latest"')
    parser.add_argument('--model_dir', default=None,
                        help='model dir (used with --checkpoint latest)')
    parser.add_argument('--data_root', required=True, help='How2Sign data root')
    parser.add_argument('--csl_root', default=None)
    parser.add_argument('--phoenix_root', default=None)
    parser.add_argument('--mean_path', required=True, help='path to mean.pt (179D)')
    parser.add_argument('--std_path', required=True, help='path to std.pt (179D)')
    parser.add_argument('--smplx_path', default='deps/smpl_models/',
                        help='path to SMPL-X model directory')

    parser.add_argument('--sign_dataset', default='how2sign',
                        choices=['how2sign', 'csl', 'how2sign_csl', 'how2sign_csl_phoenix'])
    parser.add_argument('--split', default='val')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--num_stat_samples', type=int, default=50,
                        help='samples for distribution analysis')

    parser.add_argument('--visualize', action='store_true',
                        help='Generate FK skeleton videos (requires smplx)')
    parser.add_argument('--plot', action='store_true', default=True,
                        help='Generate distribution plots')

    # VAE arch (must match training config)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_extra_layers', type=int, default=1)
    parser.add_argument('--skeleton_mode', type=str, default='7part',
                        choices=['7part', 'finger'])
    parser.add_argument('--use_hand_vel', action='store_true', default=True,
                        help='223D mode (default)')
    parser.add_argument('--no_hand_vel', dest='use_hand_vel', action='store_false',
                        help='133D mode (no hand velocity)')

    parser.add_argument('--output', default='vis_sign_output')
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--viewport', type=float, default=0.5, help='0=auto')
    parser.add_argument('--device', default='cuda:0')

    args = parser.parse_args()

    device_str = args.device if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    ckpt_path = find_checkpoint(args.checkpoint, args.model_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pose_tag = '223d' if args.use_hand_vel else '133d'
    output_dir = os.path.join(args.output, f'recon_{pose_tag}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # Setup
    # =========================================================================
    print("=" * 60)
    print("SALAD Sign VAE: Reconstruction Verification")
    print(f"  checkpoint:    {ckpt_path}")
    print(f"  pose_dim:      {pose_tag.upper()}")
    print(f"  use_hand_vel:  {args.use_hand_vel}")
    print(f"  dataset:       {args.sign_dataset}")
    print(f"  split:         {args.split}")
    print(f"  samples:       {args.num_samples} (vis) / {args.num_stat_samples} (stats)")
    print(f"  output:        {output_dir}")
    print("=" * 60)

    # -- Load mean/std (always 133D, with partwise clamping) --
    print("\n[0] Loading mean/std (133D, partwise clamped)...")
    mean_np, std_np = load_mean_std(args.mean_path, args.std_path, partwise_clamp=True)
    print(f"  mean: {mean_np.shape}, std: {std_np.shape}")
    print(f"  std range: [{std_np.min():.6f}, {std_np.max():.6f}]")

    # -- Load annotations --
    print("\n[0] Building annotations...")
    all_data = _build_annotations(
        args.split, args.sign_dataset, args.data_root,
        args.csl_root, args.phoenix_root)

    # -- Load VAE --
    print("\n[0] Loading VAE...")
    vae, opt = build_vae_from_opt({
        'skeleton_mode': args.skeleton_mode,
        'use_hand_vel': args.use_hand_vel,
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
    print(f"  Encoder splits: {vae.motion_enc.splits}")
    if missing:
        print(f"  Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    # =========================================================================
    # Run tests
    # =========================================================================
    results = {}

    results['T1_vae_shape'] = test_vae_shape(vae, args.use_hand_vel, device)

    recon_metrics = test_reconstruction_quality(
        all_data, args.csl_root, args.phoenix_root,
        vae, mean_np, std_np, args.use_hand_vel,
        device, args.num_samples)
    results['T2_recon_quality'] = len(recon_metrics) > 0

    if args.visualize:
        vis_stats, vis_ok = test_visualization(
            all_data, args.csl_root, args.phoenix_root,
            vae, mean_np, std_np, args.use_hand_vel,
            args.smplx_path, os.path.join(output_dir, 'videos'), device,
            args.num_samples, args.fps, args.viewport)
        results['T3_visualization'] = vis_ok

    if args.plot:
        test_distribution_comparison(
            all_data, args.csl_root, args.phoenix_root,
            vae, mean_np, std_np, args.use_hand_vel,
            os.path.join(output_dir, 'plots'), device,
            args.num_stat_samples)
        results['T4_distribution'] = True

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        if isinstance(result, bool):
            status = 'PASS' if result else 'FAIL'
            print(f"  {name}: {status}")
        else:
            print(f"  {name}: {result}")

    print(f"\n  Output saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()