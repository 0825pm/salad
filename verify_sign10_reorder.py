#!/usr/bin/env python3
"""
verify_sign10_reorder.py — sign10 + sign10_vel (210D) 파이프라인 검증 + 시각화

검증 항목:
  T1. Reorder 검증: raw120 <-> sign10 bijection, roundtrip
  T2. Velocity 검증: 120D <-> 210D roundtrip, velocity 값 정합성
  T3. Part별 통계: sign10 per-SJ raw/normalized 분포 + velocity 분포
  T4. 정규화 분석: sign10 120D mean/std 값 분포, 폭발 위험 차원 식별
  T5. 정규화 round-trip: sign10 norm → 210D(vel) → strip → denorm 오차
  T6. FK 시각화: GT(133D) vs 210D-roundtrip side-by-side
        GT:        179D → 133D → norm(133D) → aa133_to_joints_np
        Roundtrip: 179D → sign10 → norm(120D) → 210D(vel) → strip
                   → denorm(120D) → raw120 → pad133D → norm(133D)
                   → aa133_to_joints_np

Usage:
    cd ~/Projects/research/salad

    # 수학 검증만 (데이터 불필요)
    python verify_sign10_reorder.py --numeric_only

    # 전체 검증 + FK 시각화
    python verify_sign10_reorder.py \
        --data_root ./dataset/How2Sign \
        --mean_path ./dataset/How2Sign/mean.pt \
        --std_path  ./dataset/How2Sign/std.pt \
        --smplx_path deps/smpl_models/ \
        --sign_dataset how2sign --split val --num_samples 3 \
        --visualize
"""
import os
import sys
import argparse
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.sign_paramUtil import (
    REORDER_RAW120_TO_SIGN10, REORDER_SIGN10_TO_RAW120,
    reorder_to_sign10, reorder_from_sign10,
    SIGN_SPLITS_SIGN10, SIGN_SPLITS_SIGN10_VEL,
    SIGN_PART_NAMES_SIGN10,
    sign_adj_list_sign10, PART_LOSS_WEIGHTS_SIGN10,
    SIGN10_VEL_ROT_VEL_DIMS, SIGN10_VEL_INDICES,
    rotation_to_sign10_vel, sign10_vel_to_rotation,
    get_sign_config, get_pose_dim,
)


# =============================================================================
# Skeleton visualization config (vis_sign_joints.py 45-joint 방식)
# =============================================================================
#
# aa133_to_joints_np → SMPLX FK → [T, 127+, 3]
# SELECT_IDX로 45 joints 추출 → LOCAL index connections 사용
#
# 45-joint layout (local index):
#   0:     jaw         (SMPLX #22)
#   1:     pelvis      (SMPLX #0)
#   2:     spine1      (SMPLX #3)
#   3:     spine2      (SMPLX #6)
#   4:     spine3      (SMPLX #9)   ← ROOT
#   5:     neck        (SMPLX #12)
#   6:     L_collar    (SMPLX #13)
#   7:     R_collar    (SMPLX #14)
#   8:     head        (SMPLX #15)
#   9:     L_shoulder  (SMPLX #16)
#   10:    R_shoulder  (SMPLX #17)
#   11:    L_elbow     (SMPLX #18)
#   12:    R_elbow     (SMPLX #19)
#   13:    L_wrist     (SMPLX #20)
#   14:    R_wrist     (SMPLX #21)
#   15-29: left hand   (SMPLX #25-39)
#   30-44: right hand  (SMPLX #40-54)

SMPLX_SELECT_IDX = [22] + [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] \
                   + list(range(25, 40)) + list(range(40, 55))  # 45 joints

ROOT_IDX = 4   # spine3 in local 45-joint order

BODY_IDX  = list(range(0, 15))   # jaw + upper body (local)
LHAND_IDX = list(range(15, 30))  # left hand (local)
RHAND_IDX = list(range(30, 45))  # right hand (local)

BODY_CONNECTIONS = [
    # spine
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 8),
    # left arm
    (4, 6), (6, 9), (9, 11), (11, 13),
    # right arm
    (4, 7), (7, 10), (10, 12), (12, 14),
    # face
    (8, 0),
]

def _hand_connections(wrist_idx, hand_offset):
    """wrist → 5 finger chains (3 joints each)"""
    conns = []
    for finger in range(5):
        base = hand_offset + finger * 3
        conns.append((wrist_idx, base))
        conns.append((base, base + 1))
        conns.append((base + 1, base + 2))
    return conns

LHAND_CONNECTIONS = _hand_connections(13, 15)  # L_wrist → local 15-29
RHAND_CONNECTIONS = _hand_connections(14, 30)  # R_wrist → local 30-44
ALL_CONNECTIONS = BODY_CONNECTIONS + LHAND_CONNECTIONS + RHAND_CONNECTIONS


def normalize_to_root(joints, root_idx=ROOT_IDX):
    if len(joints.shape) == 3:
        return joints - joints[:, root_idx:root_idx+1, :]
    return joints - joints[root_idx:root_idx+1, :]


def pad_120_to_133(poses_120):
    """120D (raw order) → 133D by zero-padding jaw(3D) + expr(10D)."""
    shape = list(poses_120.shape)
    shape[-1] = 13
    zeros = np.zeros(shape, dtype=poses_120.dtype)
    return np.concatenate([poses_120, zeros], axis=-1)


def compute_diff_stats(gt_joints, sl_joints):
    """Compute MPJPE for body/hand/jaw parts using 45-joint local indices."""
    T = min(gt_joints.shape[0], sl_joints.shape[0])
    J = min(gt_joints.shape[1], sl_joints.shape[1])
    gt, sl = gt_joints[:T, :J], sl_joints[:T, :J]
    per_joint = np.sqrt(np.sum((gt - sl) ** 2, axis=-1))

    # 45-joint local indices
    body_idx = [i for i in range(1, 15) if i < J]   # pelvis~wrist (skip jaw=0)
    hand_idx = [i for i in range(15, 45) if i < J]   # lhand + rhand
    jaw_idx  = [0] if J > 0 else []                   # jaw = local 0

    return {
        'all_mpjpe':  per_joint.mean(),
        'body_mpjpe': per_joint[:, body_idx].mean() if body_idx else 0,
        'hand_mpjpe': per_joint[:, hand_idx].mean() if hand_idx else 0,
        'jaw_mpjpe':  per_joint[:, jaw_idx].mean()  if jaw_idx  else 0,
        'per_joint_mean': per_joint.mean(axis=0),
    }


def save_comparison_video(gt_joints, sl_joints, save_path, title='',
                          fps=25, viewport=0.5, diff_stats=None):
    """GT(왼쪽) vs Roundtrip(오른쪽) side-by-side 2D skeleton video.
    joints: [T, 45, 3] (45-joint local order after SELECT_IDX).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    T = min(gt_joints.shape[0], sl_joints.shape[0])
    gt = normalize_to_root(gt_joints[:T].copy())
    sl = normalize_to_root(sl_joints[:T].copy())

    if viewport > 0:
        x_lim = (-viewport, viewport)
        y_lim = (-viewport, viewport)
    else:
        all_data = np.concatenate([gt, sl], axis=0)
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

    labels = [('133D GT (original)', 'blue'),
              ('210D roundtrip (sign10_vel)', 'darkgreen')]
    for ax, (label, color) in zip([ax_gt, ax_sl], labels):
        ax.set_title(label, fontsize=12, fontweight='bold', color=color)
        ax.set_xlim(x_lim); ax.set_ylim(y_lim)
        # ax.invert_yaxis()
        ax.set_aspect('equal'); ax.axis('off')

    colors = {'body': '#2196F3', 'lhand': '#F44336', 'rhand': '#4CAF50'}

    elements = []
    for ax, data in [(ax_gt, gt), (ax_sl, sl)]:
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


# =============================================================================
# T1: Reorder verification (120D)
# =============================================================================

def test_reorder():
    print("\n" + "=" * 60)
    print("T1: Reorder Verification (raw120 <-> sign10)")
    print("=" * 60)

    x = np.arange(120, dtype=np.float64)
    sign10 = reorder_to_sign10(x)
    back = reorder_from_sign10(sign10)
    assert np.array_equal(x, back), "FAIL: roundtrip"
    print("  [PASS] roundtrip identity")

    assert len(set(REORDER_RAW120_TO_SIGN10)) == 120
    print("  [PASS] bijection (120 unique)")

    assert np.array_equal(np.argsort(REORDER_RAW120_TO_SIGN10), REORDER_SIGN10_TO_RAW120)
    print("  [PASS] inverse = argsort(forward)")

    assert sum(SIGN_SPLITS_SIGN10) == 120
    print(f"  [PASS] splits sum=120: {SIGN_SPLITS_SIGN10}")

    checks = [
        ("SJ0 neck",    slice(0,3),    [0,1,2]),
        ("SJ1 head",    slice(3,6),    [9,10,11]),
        ("SJ4 l_thumb", slice(30,39),  list(range(66,75))),
        ("SJ5 l_index", slice(39,48),  list(range(30,39))),
        ("SJ7 r_thumb", slice(75,84),  list(range(111,120))),
        ("SJ8 r_index", slice(84,93),  list(range(75,84))),
    ]
    for name, slc, expected in checks:
        assert list(sign10[slc].astype(int)) == expected, f"FAIL: {name}"
        print(f"  [PASS] {name}")

    for i, nb in enumerate(sign_adj_list_sign10):
        for j in nb:
            assert i in sign_adj_list_sign10[j]
    print("  [PASS] adjacency symmetric")

    batch = np.random.randn(8, 64, 120)
    assert np.allclose(batch, reorder_from_sign10(reorder_to_sign10(batch)))
    print("  [PASS] batch roundtrip [8,64,120]")

    return True


# =============================================================================
# T2: Velocity verification (120D <-> 210D)
# =============================================================================

def test_velocity():
    print("\n" + "=" * 60)
    print("T2: Velocity Verification (120D <-> 210D)")
    print("=" * 60)

    assert sum(SIGN_SPLITS_SIGN10_VEL) == 210
    print(f"  [PASS] splits sum=210: {SIGN_SPLITS_SIGN10_VEL}")

    for i, ((rd, vd), sp) in enumerate(zip(SIGN10_VEL_ROT_VEL_DIMS, SIGN_SPLITS_SIGN10_VEL)):
        assert rd + vd == sp
    print("  [PASS] rot+vel dims match splits")

    fake = np.random.randn(32, 120)
    assert np.allclose(fake, sign10_vel_to_rotation(rotation_to_sign10_vel(fake)))
    print("  [PASS] [T,120] -> [T,210] -> [T,120]")

    fake3 = np.random.randn(4, 64, 120)
    assert np.allclose(fake3, sign10_vel_to_rotation(rotation_to_sign10_vel(fake3)))
    print("  [PASS] [B,T,120] -> [B,T,210] -> [B,T,120]")

    rot_seq = np.random.randn(10, 120)
    v210 = rotation_to_sign10_vel(rot_seq)

    expected = np.zeros((10, 9))
    expected[1:] = rot_seq[1:, 30:39] - rot_seq[:-1, 30:39]
    assert np.allclose(expected, v210[:, 39:48])
    print("  [PASS] velocity values (l_thumb)")

    expected_r = np.zeros((10, 9))
    expected_r[1:] = rot_seq[1:, 84:93] - rot_seq[:-1, 84:93]
    assert np.allclose(expected_r, v210[:, 147:156])
    print("  [PASS] velocity values (r_index)")

    for name, (s, e) in SIGN10_VEL_INDICES.items():
        assert np.allclose(v210[0, s:e], 0)
    print("  [PASS] vel[t=0] = 0 (all hand SJs)")

    assert np.allclose(v210[:, :30], rot_seq[:, :30])
    print("  [PASS] body rotation unchanged in 210D")

    try:
        import torch
        t = torch.randn(4, 32, 120)
        assert torch.allclose(t, sign10_vel_to_rotation(rotation_to_sign10_vel(t)), atol=1e-6)
        print("  [PASS] PyTorch roundtrip")
    except ImportError:
        print("  [SKIP] PyTorch not available")

    return True


# =============================================================================
# T3: Part-wise statistics (sign10 layout)
# =============================================================================

def test_part_statistics(all_data, csl_root, phoenix_root, num_samples=50):
    """sign10 per-SJ raw 값 분포 분석."""
    print("\n" + "=" * 60)
    print("T3: Part-wise Raw Statistics (sign10 120D, before normalization)")
    print("=" * 60)

    from data.sign_dataset import _load_one

    all_poses = []
    indices = np.linspace(0, len(all_data) - 1, min(num_samples, len(all_data)), dtype=int)
    for i in indices:
        poses, _, _ = _load_one(all_data[int(i)], csl_root, phoenix_root,
                                skeleton_mode="sign10")
        if poses is not None:
            all_poses.append(poses)

    if not all_poses:
        print("  No data loaded!")
        return False

    concat = np.concatenate(all_poses, axis=0)
    print(f"\n  Total frames: {concat.shape[0]} from {len(all_poses)} samples")

    splits = SIGN_SPLITS_SIGN10
    names = SIGN_PART_NAMES_SIGN10

    print(f"\n  {'Part':<12} {'Dim':>5} {'Mean':>10} {'Std':>10} "
          f"{'Min':>10} {'Max':>10} {'Near0 Std':>10}")
    print("  " + "-" * 68)

    offset = 0
    for sj, (dim, name) in enumerate(zip(splits, names)):
        part = concat[:, offset:offset+dim]
        sd_per_dim = np.std(part, axis=0)
        n_near_zero = np.sum(sd_per_dim < 0.01)
        print(f"  {name:<12} {dim:>5} {part.mean():>10.4f} {sd_per_dim.mean():>10.4f} "
              f"{part.min():>10.4f} {part.max():>10.4f} {n_near_zero:>6}/{dim}")
        offset += dim

    return True


# =============================================================================
# T4: Normalization analysis
# =============================================================================

def test_normalization_analysis(mean, std):
    """sign10 120D mean/std — 정규화 폭발 위험 차원 식별."""
    print("\n" + "=" * 60)
    print("T4: Normalization Analysis (sign10 120D mean/std)")
    print("=" * 60)

    print(f"\n  Mean shape: {mean.shape}, Std shape: {std.shape}")
    print(f"  Overall mean range: [{mean.min():.6f}, {mean.max():.6f}]")
    print(f"  Overall std  range: [{std.min():.6f}, {std.max():.6f}]")

    splits = SIGN_SPLITS_SIGN10
    names = SIGN_PART_NAMES_SIGN10

    print(f"\n  {'Part':<12} {'Dims':>5} {'Mean(abs)':>10} {'Std(mean)':>10} "
          f"{'Std(min)':>10} {'Std<0.01':>10} {'Std<0.001':>10}")
    print("  " + "-" * 78)

    danger_dims = []
    offset = 0
    for sj, (dim, name) in enumerate(zip(splits, names)):
        part_mean = mean[offset:offset+dim]
        part_std  = std[offset:offset+dim]
        n_small_01  = np.sum(part_std < 0.01)
        n_small_001 = np.sum(part_std < 0.001)
        print(f"  {name:<12} {dim:>5} {np.abs(part_mean).mean():>10.6f} "
              f"{part_std.mean():>10.6f} {part_std.min():>10.6f} "
              f"{n_small_01:>10} {n_small_001:>10}")
        for d in range(dim):
            if part_std[d] < 0.01:
                danger_dims.append((offset + d, name, part_std[d]))
        offset += dim

    if danger_dims:
        print(f"\n  ⚠️  정규화 폭발 위험 차원 ({len(danger_dims)}개):")
        for d, part, s_val in danger_dims[:20]:
            print(f"    dim={d:>3} ({part}) — std={s_val:.6f} "
                  f"→ normalized scale ≈ ±{1.0/(s_val+1e-10):.0f}")
    else:
        print("\n  ✅ 정규화 폭발 위험 차원 없음")

    return len(danger_dims)


# =============================================================================
# T5: Full pipeline round-trip
# =============================================================================

def test_pipeline_roundtrip(all_data, csl_root, phoenix_root, mean, std, num_samples=5):
    """sign10: norm → 210D(vel) → strip → denorm → 원본과 비교."""
    print("\n" + "=" * 60)
    print("T5: Full Pipeline Round-trip")
    print("  sign10 120D → norm → 210D(vel) → strip → denorm → sign10 120D")
    print("=" * 60)

    from data.sign_dataset import _load_one

    max_errors = []
    indices = np.linspace(0, len(all_data) - 1, min(num_samples, len(all_data)), dtype=int)

    for i in indices:
        ann = all_data[int(i)]
        name = ann.get('name', f'idx_{i}')[:40]

        poses, _, _ = _load_one(ann, csl_root, phoenix_root, skeleton_mode="sign10")
        if poses is None:
            print(f"  {name} — SKIP")
            continue

        normed = (poses - mean) / (std + 1e-10)
        vel_210 = rotation_to_sign10_vel(normed)

        normed_back = sign10_vel_to_rotation(vel_210)
        denormed = normed_back * (std + 1e-10) + mean

        norm_err = np.abs(normed - normed_back).max()
        rot_err = np.abs(poses - denormed).max()
        max_errors.append(rot_err)
        print(f"  {name} — norm_err={norm_err:.2e}  rot_err={rot_err:.2e}  "
              f"[{'PASS' if rot_err < 1e-6 else 'FAIL'}]")

    if max_errors:
        overall = max(max_errors)
        ok = overall < 1e-6
        print(f"\n  Overall max error: {overall:.2e} → {'PASS' if ok else 'FAIL'}")
        return ok
    return True


# =============================================================================
# T6: FK Visualization — GT(133D) vs 210D-roundtrip side-by-side
#   verify_120d_loader.py 의 test_visualization 방식 그대로
#   aa133_to_joints_np 사용, 133D 정규화 경유
# =============================================================================

def test_visualization(all_data, csl_root, phoenix_root,
                       smplx_path, output_dir, mean_path, std_path,
                       mean_sign10, std_sign10,
                       num_samples=3, fps=25, viewport=0.5, device='cpu'):
    """
    GT(133D) vs 210D-roundtrip FK side-by-side.

    Path A (GT):
      179D → 133D raw → norm(133D mean/std) → aa133_to_joints_np

    Path B (Roundtrip):
      179D → sign10 120D raw → norm(sign10 120D mean/std) → 210D(vel)
      → strip vel → denorm(sign10 120D) → raw120 → pad_133D(jaw=0, expr=0)
      → norm(133D mean/std) → aa133_to_joints_np

    Expected: body/hand MPJPE ≈ 0, jaw MPJPE > 0
    """
    print("\n" + "=" * 60)
    print("T6: FK Visualization (133D GT vs 210D Roundtrip)")
    print("  GT:        179D → 133D → norm(133D) → FK")
    print("  Roundtrip: 179D → sign10 → norm(120D) → 210D(vel)")
    print("             → strip → denorm(120D) → pad133D → norm(133D) → FK")
    print("=" * 60)

    try:
        import torch
        from utils.feats2joints import aa133_to_joints_np
        from data.sign_dataset import _load_one
    except ImportError as e:
        print(f"  ⚠️  missing dependency: {e} — skipping")
        print("     (salad 프로젝트 루트에서 실행해야 합니다)")
        return False

    os.makedirs(output_dir, exist_ok=True)

    # ── 133D mean/std 로드 (vis_sign_120d.py 방식) ──
    mean_full = torch.load(mean_path, map_location='cpu').float()
    std_full  = torch.load(std_path,  map_location='cpu').float()
    if mean_full.shape[0] == 179:
        keep = list(range(36, 159)) + list(range(169, 179))
        mean_full, std_full = mean_full[keep], std_full[keep]
    assert mean_full.shape[0] == 133, f"Expected 133D mean, got {mean_full.shape[0]}"
    mean_133 = mean_full.numpy()
    std_133  = std_full.numpy()

    print(f"\n  133D mean/std loaded (for FK)")
    print(f"  sign10 120D mean/std loaded (for pipeline)")

    # ── Group by source ──
    src_indices = {}
    for i, item in enumerate(all_data):
        src_indices.setdefault(item.get('src', 'how2sign'), []).append(i)

    DS_LABELS = {'how2sign': 'H2S', 'csl': 'CSL', 'phoenix': 'Phoenix'}
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

            # ── Load 133D (for GT) ──
            poses_133, text, name_loaded = _load_one(
                ann, csl_root, phoenix_root, skeleton_mode="7part")
            if poses_133 is None:
                print(f"    [{idx_s+1}/{n}] {name[:40]} — load failed, skip.")
                continue
            if name_loaded:
                name = name_loaded

            T_raw = poses_133.shape[0]
            T_crop = (T_raw // 4) * 4
            if T_crop < 4:
                print(f"    [{idx_s+1}/{n}] {name[:40]} — too short ({T_raw}), skip.")
                continue
            poses_133 = poses_133[:T_crop]  # [T, 133]

            # ── Load sign10 120D (for roundtrip) ──
            poses_sign10, _, _ = _load_one(
                ann, csl_root, phoenix_root, skeleton_mode="sign10")
            if poses_sign10 is None:
                print(f"    [{idx_s+1}/{n}] {name[:40]} — sign10 load failed, skip.")
                continue
            poses_sign10 = poses_sign10[:T_crop]  # [T, 120]

            # ============================================================
            # Path A: GT 133D → norm(133D) → FK
            # ============================================================
            poses_133_norm = (poses_133 - mean_133) / (std_133 + 1e-10)

            try:
                gt_joints_full = aa133_to_joints_np(
                    poses_133_norm, mean_133, std_133, smplx_path, device=device)
                gt_joints = gt_joints_full[:, SMPLX_SELECT_IDX]  # [T, 45, 3]
            except Exception as e:
                print(f"    [{idx_s+1}/{n}] {name[:40]} — FK(GT) failed: {e}")
                continue

            # ============================================================
            # Path B: sign10 → norm(120D) → 210D(vel) → strip → denorm
            #          → raw120 → pad133D → norm(133D) → FK
            # ============================================================
            sign10_norm = (poses_sign10 - mean_sign10) / (std_sign10 + 1e-10)
            sign10_vel_210 = rotation_to_sign10_vel(sign10_norm)  # [T, 210]

            # ── 모델 출력 시뮬레이션 ──

            sign10_norm_back = sign10_vel_to_rotation(sign10_vel_210)  # [T, 120]
            sign10_denorm = sign10_norm_back * (std_sign10 + 1e-10) + mean_sign10
            raw120_back = reorder_from_sign10(sign10_denorm)       # [T, 120]
            padded_133 = pad_120_to_133(raw120_back)               # [T, 133]
            padded_133_norm = (padded_133 - mean_133) / (std_133 + 1e-10)

            try:
                rt_joints_full = aa133_to_joints_np(
                    padded_133_norm, mean_133, std_133, smplx_path, device=device)
                rt_joints = rt_joints_full[:, SMPLX_SELECT_IDX]  # [T, 45, 3]
            except Exception as e:
                print(f"    [{idx_s+1}/{n}] {name[:40]} — FK(RT) failed: {e}")
                continue

            # ── Feature-level diff ──
            feat_diff_jaw  = np.abs(poses_133[:, 120:123]).mean()
            feat_diff_expr = np.abs(poses_133[:, 123:133]).mean()

            # ── Diff stats ──
            stats = compute_diff_stats(gt_joints, rt_joints)
            stats['name'] = name
            stats['src'] = src_key
            stats['T'] = T_crop
            stats['feat_jaw_mae'] = feat_diff_jaw
            stats['feat_expr_mae'] = feat_diff_expr
            all_stats.append(stats)

            J = gt_joints.shape[1]  # should be 45
            print(f"    [{idx_s+1}/{n}] {name[:40]} (T={T_crop}, J={J})")
            print(f"      body MPJPE={stats['body_mpjpe']:.6f}  "
                  f"hand MPJPE={stats['hand_mpjpe']:.6f}  "
                  f"jaw MPJPE={stats['jaw_mpjpe']:.6f}  "
                  f"all MPJPE={stats['all_mpjpe']:.6f}")
            print(f"      jaw feat MAE={feat_diff_jaw:.6f}  "
                  f"expr feat MAE={feat_diff_expr:.6f}")

            if np.isnan(gt_joints).any() or np.isnan(rt_joints).any():
                print(f"      ⚠️ NaN in joints, skip video.")
                continue

            # ── Save video ──
            safe_name = str(name)[:40].replace('/', '_').replace('\\', '_')
            video_path = os.path.join(out_dir, f'{idx_s:03d}_{safe_name}.mp4')
            title = f'{name[:40]} [{ds_label}] (T={T_crop})'
            save_comparison_video(gt_joints, rt_joints, video_path,
                                  title, fps, viewport, stats)
            count += 1

    # ── Summary ──
    print(f"\n  Generated {count} videos in {output_dir}")

    if all_stats:
        body = np.mean([s['body_mpjpe'] for s in all_stats])
        hand = np.mean([s['hand_mpjpe'] for s in all_stats])
        jaw  = np.mean([s['jaw_mpjpe']  for s in all_stats])
        all_ = np.mean([s['all_mpjpe']  for s in all_stats])

        print(f"\n  === Aggregate Results ===")
        print(f"  body MPJPE:  {body:.6f}  (should be ~0)")
        print(f"  hand MPJPE:  {hand:.6f}  (should be ~0)")
        print(f"  jaw  MPJPE:  {jaw:.6f}   (expected diff — jaw zeroed)")
        print(f"  all  MPJPE:  {all_:.6f}")
        print(f"\n  Expected: body/hand MPJPE ≈ 0, jaw MPJPE > 0")
        print(f"  → 210D roundtrip preserves body+hand motion perfectly")

        for sk in ['how2sign', 'csl', 'phoenix']:
            ms = [s for s in all_stats if s['src'] == sk]
            if ms:
                b = np.mean([s['body_mpjpe'] for s in ms])
                h = np.mean([s['hand_mpjpe'] for s in ms])
                j = np.mean([s['jaw_mpjpe']  for s in ms])
                print(f"  [{DS_LABELS[sk]}] body={b:.6f}  hand={h:.6f}  jaw={j:.6f}  (n={len(ms)})")

        csv_path = os.path.join(output_dir, 'roundtrip_stats.csv')
        with open(csv_path, 'w') as f:
            f.write('name,src,T,body_mpjpe,hand_mpjpe,jaw_mpjpe,all_mpjpe,feat_jaw_mae,feat_expr_mae\n')
            for s in all_stats:
                f.write(f"{s['name']},{s['src']},{s['T']},"
                        f"{s['body_mpjpe']:.6f},{s['hand_mpjpe']:.6f},"
                        f"{s['jaw_mpjpe']:.6f},{s['all_mpjpe']:.6f},"
                        f"{s['feat_jaw_mae']:.6f},{s['feat_expr_mae']:.6f}\n")
        print(f"  Stats CSV: {csv_path}")

    return count > 0


# =============================================================================
# Static plots: sign10 rotation + velocity distributions
# =============================================================================

def plot_distributions(all_data, csl_root, phoenix_root, mean, std,
                       output_dir, num_samples=100):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping distribution plots")
        return

    from data.sign_dataset import _load_one

    os.makedirs(output_dir, exist_ok=True)

    all_poses = []
    indices = np.linspace(0, len(all_data) - 1, min(num_samples, len(all_data)), dtype=int)
    for i in indices:
        poses, _, _ = _load_one(all_data[int(i)], csl_root, phoenix_root,
                                skeleton_mode="sign10")
        if poses is not None:
            all_poses.append(poses)

    if not all_poses:
        return

    # ── Rotation distribution (normalized, 120D) ──
    all_normed = [((p - mean) / (std + 1e-10)) for p in all_poses]
    concat_norm = np.concatenate(all_normed, axis=0)

    splits = SIGN_SPLITS_SIGN10
    names = SIGN_PART_NAMES_SIGN10

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Sign10 Normalized Rotation Distribution (120D)', fontsize=14)
    offset = 0
    for sj, (dim, name) in enumerate(zip(splits, names)):
        ax = axes[sj // 5][sj % 5]
        vals = concat_norm[:, offset:offset+dim].flatten()
        ax.hist(vals, bins=100, density=True, alpha=0.7, edgecolor='black', linewidth=0.3)
        ax.set_title(f'SJ{sj} {name} ({dim}D)', fontsize=9)
        ax.axvline(0, color='red', linestyle='--', linewidth=0.5)
        ax.set_xlim(-8, 8)
        offset += dim
    plt.tight_layout()
    path = os.path.join(output_dir, 'sign10_rot_distributions.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Distribution plot: {path}")

    # ── Velocity distribution (from normalized rotation) ──
    all_vel = []
    for pn in all_normed:
        v210 = rotation_to_sign10_vel(pn)
        all_vel.append(v210)
    vel_concat = np.concatenate(all_vel, axis=0)

    hand_sjs = [(4,"l_thumb"),(5,"l_index"),(6,"l_mrp"),
                (7,"r_thumb"),(8,"r_index"),(9,"r_mrp")]

    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
    fig2.suptitle('Sign10 Hand Velocity Distribution', fontsize=14)
    for pi, (sj_idx, name) in enumerate(hand_sjs):
        ax = axes2[pi // 3][pi % 3]
        s, e = SIGN10_VEL_INDICES[name + "_vel"]
        v = vel_concat[:, s:e].flatten()
        ax.hist(v, bins=100, density=True, alpha=0.7, edgecolor='black', linewidth=0.3)
        ax.set_title(f'{name} vel ({e-s}D) std={v.std():.3f}', fontsize=9)
        ax.axvline(0, color='red', linestyle='--', linewidth=0.5)
        ax.set_xlim(-5, 5)
    plt.tight_layout()
    path2 = os.path.join(output_dir, 'sign10_vel_distributions.png')
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"  Velocity distribution plot: {path2}")

    # ── Std per-dim bar chart (sign10 120D) ──
    fig3, ax3 = plt.subplots(figsize=(14, 3))
    ax3.bar(range(120), std, color='steelblue', edgecolor='none')
    offset = 0
    for name, dim in zip(names, splits):
        ax3.axvspan(offset, offset+dim, alpha=0.08,
                     color='red' if dim <= 3 else 'green')
        ax3.text(offset + dim/2, std.max() * 0.9, name, ha='center', fontsize=7, rotation=45)
        offset += dim
    ax3.axhline(0.01, color='red', linestyle='--', linewidth=1, label='danger (0.01)')
    ax3.set_xlabel('Dimension index (sign10 order)')
    ax3.set_ylabel('Std value')
    ax3.set_title('Per-dimension Std (sign10 120D)')
    ax3.legend()
    plt.tight_layout()
    path3 = os.path.join(output_dir, 'std_per_dim_sign10.png')
    plt.savefig(path3, dpi=150)
    plt.close()
    print(f"  Std per-dim plot: {path3}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='sign10 + sign10_vel (210D) Pipeline Verification + Visualization')

    parser.add_argument('--numeric_only', action='store_true',
                        help='Run T1+T2 only (no data needed)')
    parser.add_argument('--data_root', default=None, help='How2Sign data root')
    parser.add_argument('--csl_root', default=None)
    parser.add_argument('--phoenix_root', default=None)
    parser.add_argument('--mean_path', default=None, help='path to mean.pt (179D)')
    parser.add_argument('--std_path', default=None, help='path to std.pt (179D)')

    parser.add_argument('--sign_dataset', default='how2sign',
                        choices=['how2sign', 'csl', 'phoenix',
                                 'how2sign_csl', 'how2sign_csl_phoenix'])
    parser.add_argument('--split', default='val')
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--num_stat_samples', type=int, default=100)

    parser.add_argument('--visualize', action='store_true',
                        help='Generate FK skeleton videos (requires smplx + feats2joints)')
    parser.add_argument('--smplx_path', default='deps/smpl_models/')
    parser.add_argument('--plot', action='store_true', default=True)

    parser.add_argument('--output', default='verify_sign10_output')
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--viewport', type=float, default=0.5, help='0=auto')
    parser.add_argument('--device', default='cuda:0')

    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("sign10 + sign10_vel (210D) Pipeline Verification")
    print(f"  dataset:  {args.sign_dataset}")
    print(f"  split:    {args.split}")
    print(f"  samples:  {args.num_samples} (tests) / {args.num_stat_samples} (stats)")
    print(f"  output:   {output_dir}")
    print("=" * 60)

    results = {}

    # ── T1 + T2: Always run ──
    results['T1_reorder'] = test_reorder()
    results['T2_velocity'] = test_velocity()

    if args.numeric_only:
        print("\nDone (numeric only).")
        _print_summary(results)
        return

    if not (args.mean_path and args.data_root):
        print("\nERROR: --mean_path and --data_root required for full verification.")
        _print_summary(results)
        return

    # ── Lazy imports (only needed with data) ──
    import torch
    from data.load_sign_data import load_mean_std
    from data.sign_dataset import _build_annotations, _load_one

    device = args.device if torch.cuda.is_available() else 'cpu'

    # ── Load annotations ──
    print("\n[0] Building annotations...")
    all_data = _build_annotations(
        args.split, args.sign_dataset, args.data_root,
        args.csl_root, args.phoenix_root)

    # ── Load sign10 120D mean/std ──
    print("\n[0] Loading mean/std (sign10 120D)...")
    mean_sign10, std_sign10 = load_mean_std(
        args.mean_path, args.std_path, skeleton_mode="sign10")
    print(f"  mean: {mean_sign10.shape}, std: {std_sign10.shape}")

    # ── Run tests ──
    results['T3_part_stats'] = test_part_statistics(
        all_data, args.csl_root, args.phoenix_root, args.num_stat_samples)

    n_danger = test_normalization_analysis(mean_sign10, std_sign10)
    results['T4_norm_danger_dims'] = n_danger

    results['T5_roundtrip'] = test_pipeline_roundtrip(
        all_data, args.csl_root, args.phoenix_root,
        mean_sign10, std_sign10, args.num_samples)

    if args.plot:
        plot_distributions(
            all_data, args.csl_root, args.phoenix_root,
            mean_sign10, std_sign10, output_dir, args.num_stat_samples)

    if args.visualize:
        results['T6_visualization'] = test_visualization(
            all_data, args.csl_root, args.phoenix_root,
            args.smplx_path, os.path.join(output_dir, 'videos'),
            args.mean_path, args.std_path,
            mean_sign10, std_sign10,
            args.num_samples, args.fps, args.viewport, device)

    _print_summary(results)
    print(f"\n  Output saved to: {output_dir}")


def _print_summary(results):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        if isinstance(result, bool):
            print(f"  {name}: {'✅ PASS' if result else '❌ FAIL'}")
        elif isinstance(result, int):
            print(f"  {name}: {result} {'⚠️ danger dims' if result > 0 else '✅ clean'}")
        else:
            print(f"  {name}: {result}")


if __name__ == '__main__':
    main()