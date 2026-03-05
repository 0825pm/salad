"""
vis_sign_120d.py — 133D vs 120D (jaw/expr 제거) SMPL-X FK 비교 시각화

133D SMPL-X axis-angle에서 jaw(3D)+expression(10D)을 제거한 120D가
body/hand 모션을 보존하는지 검증.

  133D: [root 3 | upper_a 12 | upper_b 15 | lhand 45 | rhand 45 | jaw 3 | expr 10]
  120D: [root 3 | upper_a 12 | upper_b 15 | lhand 45 | rhand 45]  ← [:120]

Usage:
    cd ~/Projects/research/salad

    python vis_sign_120d.py \
        --data_root /home/user/Projects/research/SOKE/data/How2Sign \
        --csl_root  /home/user/Projects/research/SOKE/data/CSL-Daily \
        --phoenix_root /home/user/Projects/research/SOKE/data/Phoenix_2014T \
        --mean_path /home/user/Projects/research/SOKE/data/CSL-Daily/mean.pt \
        --std_path  /home/user/Projects/research/SOKE/data/CSL-Daily/std.pt \
        --smplx_path deps/smpl_models/ \
        --sign_dataset how2sign_csl_phoenix \
        --split val --num_samples 3
"""

import os
import sys
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
# SMPL-X skeleton (FK output) — same as vis_sign_recon.py
# =============================================================================
SMPLX_UPPER_BODY = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
SMPLX_LHAND = list(range(25, 40))
SMPLX_RHAND = list(range(40, 55))
SMPLX_VALID = SMPLX_UPPER_BODY + SMPLX_LHAND + SMPLX_RHAND


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
        return joints - joints[:, root_idx:root_idx+1, :]
    return joints - joints[root_idx:root_idx+1, :]


# =============================================================================
# 120D slicing utilities
# =============================================================================

def slice_to_120d(poses_133):
    """133D → 120D: jaw(3)+expr(10) 제거"""
    return poses_133[..., :120]


def pad_to_133d(poses_120):
    """120D → 133D: jaw=0, expr=0 패딩"""
    shape = list(poses_120.shape)
    shape[-1] = 13  # jaw(3) + expr(10)
    zeros = np.zeros(shape, dtype=poses_120.dtype)
    return np.concatenate([poses_120, zeros], axis=-1)


# =============================================================================
# Video rendering
# =============================================================================

def save_comparison_video(gt_joints, sliced_joints, save_path, title='',
                          fps=25, viewport=0.5, diff_stats=None):
    """133D(왼쪽) vs 120D-padded(오른쪽) side-by-side skeleton video"""
    T = min(gt_joints.shape[0], sliced_joints.shape[0])
    J = min(gt_joints.shape[1], sliced_joints.shape[1])

    root_idx = 9 if J > 21 else 0
    gt = normalize_to_root(gt_joints[:T, :J].copy(), root_idx)
    sl = normalize_to_root(sliced_joints[:T, :J].copy(), root_idx)

    if viewport > 0:
        x_lim = (-viewport, viewport)
        y_lim = (-viewport, viewport)
    else:
        valid_idx = [i for i in SMPLX_VALID if i < J]
        all_data = np.concatenate([gt[:, valid_idx], sl[:, valid_idx]], axis=0)
        all_x, all_y = all_data[:, :, 0].flatten(), all_data[:, :, 1].flatten()
        max_range = max(all_x.max() - all_x.min(), all_y.max() - all_y.min(), 0.1) * 1.2
        x_mid, y_mid = (all_x.max() + all_x.min()) / 2, (all_y.max() + all_y.min()) / 2
        x_lim = (x_mid - max_range/2, x_mid + max_range/2)
        y_lim = (y_mid - max_range/2, y_mid + max_range/2)

    fig, (ax_gt, ax_sl) = plt.subplots(1, 2, figsize=(12, 6))
    main_title = title
    if diff_stats:
        main_title += f"\nbody MPJPE={diff_stats['body_mpjpe']:.4f}  "
        main_title += f"hand MPJPE={diff_stats['hand_mpjpe']:.4f}  "
        main_title += f"jaw MPJPE={diff_stats['jaw_mpjpe']:.4f}"
    fig.suptitle(main_title, fontsize=10)

    labels = [('133D (original)', 'blue'), ('120D (no jaw/expr)', 'darkgreen')]
    for ax, (label, color) in zip([ax_gt, ax_sl], labels):
        ax.set_title(label, fontsize=12, fontweight='bold', color=color)
        ax.set_xlim(x_lim); ax.set_ylim(y_lim)
        ax.set_aspect('equal'); ax.axis('off')

    connections = get_connections(J)
    colors = {'body': '#2196F3', 'lhand': '#F44336', 'rhand': '#4CAF50'}
    ub_idx = [i for i in SMPLX_UPPER_BODY if i < J]

    elements = []
    for ax, data in [(ax_gt, gt), (ax_sl, sl)]:
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
# Diff analysis
# =============================================================================

def compute_diff_stats(gt_joints, sl_joints):
    """133D FK vs 120D-padded FK joint-level MPJPE 분석"""
    T = min(gt_joints.shape[0], sl_joints.shape[0])
    J = min(gt_joints.shape[1], sl_joints.shape[1])
    gt, sl = gt_joints[:T, :J], sl_joints[:T, :J]

    per_joint = np.sqrt(np.sum((gt - sl) ** 2, axis=-1))  # [T, J]

    body_idx = [i for i in SMPLX_UPPER_BODY if i < J]
    hand_idx = [i for i in SMPLX_LHAND + SMPLX_RHAND if i < J]
    # jaw = joint 22 in SMPL-X (if exists)
    jaw_idx = [22] if J > 22 else []

    stats = {
        'all_mpjpe': per_joint.mean(),
        'body_mpjpe': per_joint[:, body_idx].mean() if body_idx else 0,
        'hand_mpjpe': per_joint[:, hand_idx].mean() if hand_idx else 0,
        'jaw_mpjpe': per_joint[:, jaw_idx].mean() if jaw_idx else 0,
        'per_joint_mean': per_joint.mean(axis=0),  # [J]
    }
    return stats


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='120D slicing verification: 133D vs 120D (jaw/expr removed)')

    parser.add_argument('--data_root', required=True, help='How2Sign data root')
    parser.add_argument('--csl_root', default=None)
    parser.add_argument('--phoenix_root', default=None)
    parser.add_argument('--mean_path', required=True)
    parser.add_argument('--std_path', required=True)
    parser.add_argument('--smplx_path', default='deps/smpl_models/',
                        help='path to SMPL-X model directory')
    parser.add_argument('--output', default='vis_120d_output')
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--split', default='val')
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--viewport', type=float, default=0.5, help='0=auto')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--sign_dataset', default='how2sign',
                        choices=['how2sign', 'csl', 'how2sign_csl', 'how2sign_csl_phoenix'])
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = os.path.join(args.output, f'slice120d_{timestamp}')
    os.makedirs(output_root, exist_ok=True)

    # ── Load mean/std (133D) ──
    mean = torch.load(args.mean_path, map_location='cpu').float()
    std = torch.load(args.std_path, map_location='cpu').float()
    if mean.shape[0] == 179:
        keep = list(range(36, 126)) + list(range(136, 179))
        mean, std = mean[keep], std[keep]
    assert mean.shape[0] == 133, f"Expected 133D mean, got {mean.shape[0]}"

    mean_np = mean.numpy()
    std_np = std.numpy()

    # 120D versions
    mean_120 = mean_np[:120]
    std_120 = std_np[:120]

    print("=" * 60)
    print("120D Slicing Verification: 133D vs 120D (jaw/expr removed)")
    print(f"  133D layout: root(3) + upper_a(12) + upper_b(15) + lhand(45) + rhand(45) + jaw(3) + expr(10)")
    print(f"  120D layout: root(3) + upper_a(12) + upper_b(15) + lhand(45) + rhand(45)")
    print(f"  smplx: {args.smplx_path}")
    print(f"  output: {output_root}")
    print("=" * 60)

    # ── Feature-level statistics ──
    print(f"\n[Stats] jaw mean range:  [{mean_np[120:123].min():.4f}, {mean_np[120:123].max():.4f}]")
    print(f"[Stats] jaw std range:   [{std_np[120:123].min():.4f}, {std_np[120:123].max():.4f}]")
    print(f"[Stats] expr mean range: [{mean_np[123:133].min():.4f}, {mean_np[123:133].max():.4f}]")
    print(f"[Stats] expr std range:  [{std_np[123:133].min():.4f}, {std_np[123:133].max():.4f}]")

    # ── Load dataset ──
    from data.sign_dataset import _build_annotations, _load_one

    all_data = _build_annotations(
        split=args.split, dataset_name=args.sign_dataset,
        data_root=args.data_root, csl_root=args.csl_root,
        phoenix_root=args.phoenix_root)
    print(f"\n[{args.split}] {len(all_data)} samples loaded")

    # ── Group by source ──
    src_indices = {}
    for i, item in enumerate(all_data):
        src_indices.setdefault(item.get('src', 'how2sign'), []).append(i)

    DS_LABELS = {'how2sign': 'H2S', 'csl': 'CSL', 'phoenix': 'Phoenix'}
    all_stats = []
    total_count = 0

    for src_key, indices in src_indices.items():
        if not indices:
            continue
        ds_label = DS_LABELS.get(src_key, src_key)
        n = min(args.num_samples, len(indices))
        sel = [indices[int(i)] for i in np.linspace(0, len(indices)-1, n)]

        out_dir = os.path.join(output_root, ds_label)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n  [{ds_label}] {len(indices)} total, visualizing {n}")

        for idx_s, dataset_idx in enumerate(sel):
            ann = all_data[dataset_idx]
            name = ann.get('name', f'sample_{dataset_idx}')

            poses, text, name_loaded = _load_one(ann, args.csl_root, args.phoenix_root)
            if poses is None:
                print(f"    [{idx_s+1}/{n}] {name} — load failed, skip.")
                continue
            if name_loaded:
                name = name_loaded

            T_raw = poses.shape[0]
            T_crop = (T_raw // 4) * 4
            if T_crop < 4:
                print(f"    [{idx_s+1}/{n}] {name} — too short ({T_raw}), skip.")
                continue
            poses = poses[:T_crop]  # [T, 133]

            # ── Feature-level diff ──
            poses_120 = slice_to_120d(poses)         # [T, 120]
            poses_padded = pad_to_133d(poses_120)     # [T, 133] with jaw=0, expr=0

            feat_diff_jaw = np.abs(poses[:, 120:123]).mean()
            feat_diff_expr = np.abs(poses[:, 123:133]).mean()

            # ── Normalize for FK (aa133_to_joints_np expects normalized input) ──
            poses_norm = (poses - mean_np) / (std_np + 1e-10)
            padded_norm = (poses_padded - mean_np) / (std_np + 1e-10)

            # ── FK: 133D original ──
            try:
                gt_joints = aa133_to_joints_np(
                    poses_norm, mean_np, std_np, args.smplx_path, device=device)
            except Exception as e:
                print(f"    [{idx_s+1}/{n}] {name} — FK(133D) failed: {e}")
                continue

            # ── FK: 120D padded ──
            try:
                sl_joints = aa133_to_joints_np(
                    padded_norm, mean_np, std_np, args.smplx_path, device=device)
            except Exception as e:
                print(f"    [{idx_s+1}/{n}] {name} — FK(120D) failed: {e}")
                continue

            # ── Diff stats ──
            stats = compute_diff_stats(gt_joints, sl_joints)
            stats['name'] = name
            stats['src'] = src_key
            stats['T'] = T_crop
            stats['feat_jaw_mae'] = feat_diff_jaw
            stats['feat_expr_mae'] = feat_diff_expr
            all_stats.append(stats)

            J = gt_joints.shape[1]
            print(f"    [{idx_s+1}/{n}] {name} (T={T_crop}, J={J})")
            print(f"      body MPJPE={stats['body_mpjpe']:.6f}  "
                  f"hand MPJPE={stats['hand_mpjpe']:.6f}  "
                  f"jaw MPJPE={stats['jaw_mpjpe']:.6f}  "
                  f"all MPJPE={stats['all_mpjpe']:.6f}")
            print(f"      jaw feat MAE={feat_diff_jaw:.6f}  "
                  f"expr feat MAE={feat_diff_expr:.6f}")

            if np.isnan(gt_joints).any() or np.isnan(sl_joints).any():
                print(f"      NaN in joints, skip video.")
                continue

            # ── Save video ──
            safe_name = str(name)[:40].replace('/', '_').replace('\\', '_')
            video_path = os.path.join(out_dir, f'{idx_s:03d}_{safe_name}.mp4')
            title = f'{name} [{ds_label}] (T={T_crop})'
            save_comparison_video(gt_joints, sl_joints, video_path,
                                  title, args.fps, args.viewport, stats)
            total_count += 1

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"Done. {total_count} videos saved to {output_root}")

    if all_stats:
        print(f"\n  === Aggregate Results ===")
        body = np.mean([s['body_mpjpe'] for s in all_stats])
        hand = np.mean([s['hand_mpjpe'] for s in all_stats])
        jaw  = np.mean([s['jaw_mpjpe'] for s in all_stats])
        all_ = np.mean([s['all_mpjpe'] for s in all_stats])
        print(f"  body MPJPE:  {body:.6f}  (should be ~0)")
        print(f"  hand MPJPE:  {hand:.6f}  (should be ~0)")
        print(f"  jaw  MPJPE:  {jaw:.6f}   (expected diff — jaw zeroed)")
        print(f"  all  MPJPE:  {all_:.6f}")
        print()
        print(f"  Expected: body/hand MPJPE ≈ 0, jaw MPJPE > 0")
        print(f"  → 120D slicing preserves body+hand motion perfectly")
        print(f"  → Only jaw/expression affected (set to neutral)")

        for src_key in ['how2sign', 'csl', 'phoenix']:
            ms = [s for s in all_stats if s['src'] == src_key]
            if ms:
                b = np.mean([s['body_mpjpe'] for s in ms])
                h = np.mean([s['hand_mpjpe'] for s in ms])
                j = np.mean([s['jaw_mpjpe'] for s in ms])
                print(f"  [{DS_LABELS[src_key]}] body={b:.6f}  hand={h:.6f}  jaw={j:.6f}  (n={len(ms)})")

        csv_path = os.path.join(output_root, 'diff_stats.csv')
        with open(csv_path, 'w') as f:
            f.write('name,src,T,body_mpjpe,hand_mpjpe,jaw_mpjpe,all_mpjpe,feat_jaw_mae,feat_expr_mae\n')
            for s in all_stats:
                f.write(f"{s['name']},{s['src']},{s['T']},"
                        f"{s['body_mpjpe']:.6f},{s['hand_mpjpe']:.6f},"
                        f"{s['jaw_mpjpe']:.6f},{s['all_mpjpe']:.6f},"
                        f"{s['feat_jaw_mae']:.6f},{s['feat_expr_mae']:.6f}\n")
        print(f"\n  Stats saved to {csv_path}")

    print("=" * 60)


if __name__ == '__main__':
    main()
