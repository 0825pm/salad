"""
vis_sign_joints.py — 135D joint coordinates 시각화 (7-part grouped)

data_joint의 .npy 파일을 직접 읽어 skeleton animation 생성.
SMPL-X FK 불필요 — 이미 3D 좌표이므로 바로 시각화.

Usage:
    cd ~/Projects/research/salad

    # 단일 .npy 파일
    python vis_sign_joints.py \
        --npy_path /path/to/data_joint/How2Sign/train/poses/sample.npy \
        --output vis_joints_output

    # 데이터셋 모드 (여러 샘플)
    python vis_sign_joints.py \
        --joint_root /home/user/Projects/research/SOKE/data_joint \
        --data_root  /home/user/Projects/research/SOKE/data/How2Sign \
        --csl_root   /home/user/Projects/research/SOKE/data/CSL-Daily \
        --phoenix_root /home/user/Projects/research/SOKE/data/Phoenix_2014T \
        --sign_dataset how2sign_csl_phoenix \
        --split val --num_samples 5

    # mean/std 정규화 확인 (normalize → denormalize → 시각화)
    python vis_sign_joints.py \
        --joint_root ... --data_root ... \
        --verify_norm --num_samples 3
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# 45-joint skeleton definition (local indices, 7-part grouped)
# =============================================================================
#
# Layout (after reorder in convert_smplx_to_joints.py):
#   0:  pelvis       1:  spine1       2:  spine2       3:  spine3       (torso)
#   4:  L_collar     5:  L_shoulder   6:  L_elbow      7:  L_wrist     (L_arm)
#   8:  R_collar     9:  R_shoulder   10: R_elbow      11: R_wrist     (R_arm)
#   12-26: left hand (index,middle,pinky,ring,thumb × 3j)              (lhand)
#   27-41: right hand                                                   (rhand)
#   42: neck         43: head                                           (head_neck)
#   44: jaw                                                             (jaw)

ROOT_IDX = 3  # spine3 — 상체 중심

BODY_CONNECTIONS = [
    # spine
    (0, 1), (1, 2), (2, 3),
    # spine3 → neck → head → jaw
    (3, 42), (42, 43), (43, 44),
    # left arm
    (3, 4), (4, 5), (5, 6), (6, 7),
    # right arm
    (3, 8), (8, 9), (9, 10), (10, 11),
]

# SMPL-X hand: 5 fingers × 3 joints each
# Finger order within each hand: index, middle, pinky, ring, thumb
def _hand_connections(wrist_idx, hand_offset):
    """wrist → 5 finger chains"""
    conns = []
    for finger in range(5):
        base = hand_offset + finger * 3
        conns.append((wrist_idx, base))
        conns.append((base, base + 1))
        conns.append((base + 1, base + 2))
    return conns

LHAND_CONNECTIONS = _hand_connections(7, 12)   # L_wrist → local 12-26
RHAND_CONNECTIONS = _hand_connections(11, 27)  # R_wrist → local 27-41

ALL_CONNECTIONS = BODY_CONNECTIONS + LHAND_CONNECTIONS + RHAND_CONNECTIONS

BODY_IDX  = list(range(0, 12)) + [42, 43, 44]
LHAND_IDX = list(range(12, 27))
RHAND_IDX = list(range(27, 42))


# =============================================================================
# Visualization
# =============================================================================

def normalize_to_root(joints, root_idx=ROOT_IDX):
    if len(joints.shape) == 3:
        return joints - joints[:, root_idx:root_idx+1, :]
    return joints - joints[root_idx:root_idx+1, :]


def save_skeleton_video(joints, save_path, title='', fps=25, viewport=0.5):
    """Single skeleton video. joints: [T, 45, 3]"""
    T, J, _ = joints.shape
    data = normalize_to_root(joints.copy())

    if viewport > 0:
        x_lim = (-viewport, viewport)
        y_lim = (-viewport, viewport)
    else:
        all_x = data[:, :, 0].flatten()
        all_y = data[:, :, 1].flatten()
        margin = max(all_x.max() - all_x.min(), all_y.max() - all_y.min(), 0.1) * 1.2
        cx, cy = (all_x.max()+all_x.min())/2, (all_y.max()+all_y.min())/2
        x_lim = (cx - margin/2, cx + margin/2)
        y_lim = (cy - margin/2, cy + margin/2)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.suptitle(title, fontsize=10)
    ax.set_xlim(x_lim); ax.set_ylim(y_lim)
    ax.invert_yaxis()
    ax.set_aspect('equal'); ax.axis('off')

    colors = {'body': '#2196F3', 'lhand': '#F44336', 'rhand': '#4CAF50'}

    lines = []
    for (i, j) in ALL_CONNECTIONS:
        if i in RHAND_IDX or j in RHAND_IDX:
            c, lw = colors['rhand'], 1.0
        elif i in LHAND_IDX or j in LHAND_IDX:
            c, lw = colors['lhand'], 1.0
        else:
            c, lw = colors['body'], 1.5
        line, = ax.plot([], [], color=c, linewidth=lw, alpha=0.8)
        lines.append((line, i, j))

    body_sc = ax.scatter([], [], c=colors['body'], s=12, zorder=5)
    lh_sc   = ax.scatter([], [], c=colors['lhand'], s=5, zorder=5)
    rh_sc   = ax.scatter([], [], c=colors['rhand'], s=5, zorder=5)
    frame_text = fig.text(0.5, 0.02, '', ha='center', fontsize=9, color='gray')
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    def update(frame):
        f = min(frame, T - 1)
        fd = data[f]
        x, y = fd[:, 0], fd[:, 1]
        frame_text.set_text(f'Frame {f}/{T-1}')
        for (line, i, j) in lines:
            line.set_data([x[i], x[j]], [y[i], y[j]])
        body_sc.set_offsets(np.c_[x[BODY_IDX], y[BODY_IDX]])
        lh_sc.set_offsets(np.c_[x[LHAND_IDX], y[LHAND_IDX]])
        rh_sc.set_offsets(np.c_[x[RHAND_IDX], y[RHAND_IDX]])
        return []

    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False)
    try:
        anim.save(save_path, writer=FFMpegWriter(fps=fps, bitrate=5000))
    except Exception:
        anim.save(save_path.replace('.mp4', '.gif'), writer='pillow', fps=min(fps, 10))
    plt.close(fig)


def save_comparison_video(left_joints, right_joints, save_path, title='',
                          fps=25, viewport=0.5,
                          left_label='GT', right_label='Recon'):
    """Side-by-side skeleton video. joints: [T, 45, 3] each."""
    T = min(left_joints.shape[0], right_joints.shape[0])
    left  = normalize_to_root(left_joints[:T].copy())
    right = normalize_to_root(right_joints[:T].copy())

    if viewport > 0:
        x_lim = (-viewport, viewport)
        y_lim = (-viewport, viewport)
    else:
        all_pts = np.concatenate([left, right], axis=0)
        all_x, all_y = all_pts[:,:,0].flatten(), all_pts[:,:,1].flatten()
        margin = max(all_x.max()-all_x.min(), all_y.max()-all_y.min(), 0.1) * 1.2
        cx, cy = (all_x.max()+all_x.min())/2, (all_y.max()+all_y.min())/2
        x_lim = (cx-margin/2, cx+margin/2)
        y_lim = (cy-margin/2, cy+margin/2)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title, fontsize=10)

    for ax, label in [(ax_l, left_label), (ax_r, right_label)]:
        ax.set_title(label, fontsize=12, fontweight='bold',
                     color='blue' if label == left_label else 'red')
        ax.set_xlim(x_lim); ax.set_ylim(y_lim)
        ax.invert_yaxis()
        ax.set_aspect('equal'); ax.axis('off')

    colors = {'body': '#2196F3', 'lhand': '#F44336', 'rhand': '#4CAF50'}

    elements = []
    for ax, d in [(ax_l, left), (ax_r, right)]:
        ax_lines = []
        for (i, j) in ALL_CONNECTIONS:
            if i in RHAND_IDX or j in RHAND_IDX:
                c, lw = colors['rhand'], 1.0
            elif i in LHAND_IDX or j in LHAND_IDX:
                c, lw = colors['lhand'], 1.0
            else:
                c, lw = colors['body'], 1.5
            line, = ax.plot([], [], color=c, linewidth=lw, alpha=0.8)
            ax_lines.append((line, i, j))
        bs = ax.scatter([], [], c=colors['body'], s=12, zorder=5)
        ls = ax.scatter([], [], c=colors['lhand'], s=5, zorder=5)
        rs = ax.scatter([], [], c=colors['rhand'], s=5, zorder=5)
        elements.append((ax_lines, bs, ls, rs, d))

    frame_text = fig.text(0.5, 0.02, '', ha='center', fontsize=9, color='gray')
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])

    def update(frame):
        f = min(frame, T - 1)
        frame_text.set_text(f'Frame {f}/{T-1}')
        for (ax_lines, bs, ls, rs, d) in elements:
            fd = d[f]
            x, y = fd[:, 0], fd[:, 1]
            for (line, i, j) in ax_lines:
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
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Visualize 135D joint coordinates')

    # Mode 1: single .npy file
    parser.add_argument('--npy_path', default=None, help='single .npy file path')

    # Mode 2: dataset mode
    parser.add_argument('--joint_root', default=None,
                        help='data_joint root (e.g. .../SOKE/data_joint)')
    parser.add_argument('--data_root', default=None, help='How2Sign original root (for annotations)')
    parser.add_argument('--csl_root', default=None, help='CSL-Daily original root')
    parser.add_argument('--phoenix_root', default=None, help='Phoenix original root')
    parser.add_argument('--sign_dataset', default='how2sign',
                        choices=['how2sign', 'csl', 'phoenix', 'how2sign_csl_phoenix'])
    parser.add_argument('--split', default='val')
    parser.add_argument('--num_samples', type=int, default=5)

    # verify normalization round-trip
    parser.add_argument('--verify_norm', action='store_true',
                        help='normalize→denormalize and compare')

    # output
    parser.add_argument('--output', default='vis_joints_output')
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--viewport', type=float, default=0.0,
                        help='0=auto viewport')

    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = os.path.join(args.output, f'joints_{timestamp}')
    os.makedirs(output_root, exist_ok=True)

    print("=" * 60)
    print("Sign Language Joint Visualization (135D → 45 joints, 7-part grouped)")
    print(f"  output: {output_root}")
    print("=" * 60)

    # =====================================================================
    # Mode 1: Single .npy
    # =====================================================================
    if args.npy_path:
        print(f"\nLoading: {args.npy_path}")
        flat = np.load(args.npy_path)  # [T, 135]
        joints = flat.reshape(-1, 45, 3)
        print(f"  Shape: {flat.shape} → {joints.shape}")

        basename = os.path.splitext(os.path.basename(args.npy_path))[0]
        save_path = os.path.join(output_root, f'{basename}.mp4')
        save_skeleton_video(joints, save_path,
                            f'{basename} (T={joints.shape[0]})',
                            args.fps, args.viewport)
        print(f"  Saved: {save_path}")

        if args.verify_norm and args.joint_root:
            from data.load_sign_data import load_mean_std_joints
            mean, std = load_mean_std_joints(args.joint_root)
            normed = (flat - mean) / (std + 1e-10)
            denormed = normed * (std + 1e-10) + mean
            err = np.abs(flat - denormed).max()
            print(f"  Norm round-trip max error: {err:.2e}")

        print("\nDone!")
        return

    # =====================================================================
    # Mode 2: Dataset
    # =====================================================================
    assert args.joint_root and args.data_root, \
        "--joint_root and --data_root required for dataset mode"

    from data.sign_dataset import _build_annotations, _load_one_joint

    if args.verify_norm:
        from data.load_sign_data import load_mean_std_joints
        mean, std = load_mean_std_joints(args.joint_root)
        print(f"\nMean/std loaded from {args.joint_root}")
        print(f"  mean range: [{mean.min():.4f}, {mean.max():.4f}]")
        print(f"  std  range: [{std.min():.6f}, {std.max():.4f}]")

    all_data = _build_annotations(
        args.split, args.sign_dataset, args.data_root,
        args.csl_root, args.phoenix_root)
    print(f"\n[{args.split}] {len(all_data)} annotations loaded")

    # Group by source
    src_indices = {}
    for i, item in enumerate(all_data):
        src_indices.setdefault(item['src'], []).append(i)

    DS_LABELS = {'how2sign': 'H2S', 'csl': 'CSL', 'phoenix': 'Phoenix'}
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

        for sel_idx, dataset_idx in enumerate(sel):
            ann = all_data[dataset_idx]
            name = ann.get('name', f'sample_{dataset_idx}')
            text = ann.get('text', '')

            poses, _, name_loaded = _load_one_joint(ann, args.joint_root, args.split)
            if poses is None:
                print(f"    [{sel_idx+1}/{n}] {name} — load failed, skip.")
                continue
            if name_loaded:
                name = name_loaded

            joints = poses.reshape(-1, 45, 3)  # [T, 45, 3]
            T = joints.shape[0]

            print(f"    [{sel_idx+1}/{n}] {name} (T={T}) text=\"{text[:50]}\"")

            if args.verify_norm:
                # normalize → denormalize → compare
                normed = (poses - mean) / (std + 1e-10)
                denormed = normed * (std + 1e-10) + mean
                joints_dn = denormed.reshape(-1, 45, 3)
                err = np.abs(poses - denormed).max()
                print(f"      norm round-trip error: {err:.2e}")

                safe_name = str(name)[:40].replace('/', '_')
                video_path = os.path.join(out_dir, f'{sel_idx:03d}_{safe_name}_compare.mp4')
                title = f'{name} [{ds_label}] (T={T}) | norm err={err:.2e}'
                save_comparison_video(joints, joints_dn, video_path,
                                      title, args.fps, args.viewport,
                                      'Original', 'Norm→Denorm')
            else:
                safe_name = str(name)[:40].replace('/', '_')
                video_path = os.path.join(out_dir, f'{sel_idx:03d}_{safe_name}.mp4')
                title = f'{name} [{ds_label}] (T={T})\n"{text[:60]}"'
                save_skeleton_video(joints, video_path, title,
                                    args.fps, args.viewport)

            total_count += 1

    print(f"\n{'='*60}")
    print(f"Done. {total_count} videos saved to {output_root}")
    print("=" * 60)


if __name__ == '__main__':
    main()