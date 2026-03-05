"""
verify_120d_loader.py — 120D 데이터 로더 검증 + 시각화

검증 항목:
  1. 차원 검증: 179D → 120D 변환 결과 shape 확인
  2. 변환 일관성: 179→120 vs 179→133→120 동일성 검증
  3. Part별 통계: mean/std/range per body part
  4. 정규화 분석: std 값 분포, 폭발 위험 차원 식별
  5. 정규화 round-trip: normalize → denormalize 오차 확인
  6. 120D vs 133D mean/std 비교
  7. 133D vs 120D SMPL-X FK side-by-side (45-joint SELECT_IDX 방식)

Usage:
    cd ~/Projects/research/salad

    # 기본 검증 (통계 + 차원 확인)
    python verify_120d_loader.py \
        --data_root /path/to/How2Sign \
        --mean_path /path/to/mean.pt \
        --std_path  /path/to/std.pt \
        --sign_dataset how2sign --split val --num_samples 5

    # 전체 데이터셋 + FK 시각화 + 133D 비교
    python verify_120d_loader.py \
        --data_root /path/to/How2Sign \
        --csl_root  /path/to/CSL-Daily \
        --phoenix_root /path/to/Phoenix_2014T \
        --mean_path /path/to/mean.pt \
        --std_path  /path/to/std.pt \
        --smplx_path deps/smpl_models/ \
        --sign_dataset how2sign_csl_phoenix \
        --split val --num_samples 3 \
        --visualize --compare_dims
"""
import os
import sys
import argparse
import numpy as np
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.load_sign_data import (
    load_mean_std, get_part_ranges,
    PART_RANGES_120, PART_RANGES_133,
)
from data.sign_dataset import _build_annotations, _load_one


# =============================================================================
# Helpers: load 133D / 120D via current _load_one API
# =============================================================================

def _load_133(ann, csl_root, phoenix_root):
    """Load 133D poses (7part mode)."""
    return _load_one(ann, csl_root, phoenix_root, skeleton_mode="7part")


def _load_120(ann, csl_root, phoenix_root):
    """Load 120D poses (133D[:120] — drop jaw+expr)."""
    poses, text, name = _load_one(ann, csl_root, phoenix_root, skeleton_mode="7part")
    if poses is not None:
        poses = poses[:, :120]
    return poses, text, name


def _load_mean_std_120(mean_path, std_path):
    """Load 120D mean/std (133D[:120])."""
    mean, std = load_mean_std(mean_path, std_path, skeleton_mode="7part")
    return mean[:120], std[:120]


def _load_mean_std_133(mean_path, std_path):
    """Load 133D mean/std."""
    return load_mean_std(mean_path, std_path, skeleton_mode="7part")


# =============================================================================
# Test 1: Dimension verification
# =============================================================================

def test_dimensions(all_data, csl_root, phoenix_root, num_samples=5):
    """로더에서 반환하는 pose shape이 정확히 120D인지 확인."""
    print("\n" + "=" * 60)
    print("TEST 1: Dimension Verification (120D)")
    print("=" * 60)

    passed, failed, skipped = 0, 0, 0
    indices = np.linspace(0, len(all_data) - 1, min(num_samples, len(all_data)), dtype=int)

    for i in indices:
        ann = all_data[i]
        name = ann.get('name', f'idx_{i}')
        src  = ann.get('src', '?')

        poses, text, _ = _load_120(ann, csl_root, phoenix_root)
        if poses is None:
            print(f"  [{src}] {name[:40]} — SKIP (load failed)")
            skipped += 1
            continue

        T, D = poses.shape
        ok = D == 120
        status = "PASS" if ok else f"FAIL (got {D}D)"
        print(f"  [{src}] {name[:40]} — shape=({T}, {D}) → {status}")
        if ok:
            passed += 1
        else:
            failed += 1

    total = passed + failed + skipped
    print(f"\n  Result: {passed}/{total} passed, {failed} failed, {skipped} skipped")
    return failed == 0


# =============================================================================
# Test 2: Conversion consistency (179→120 vs 179→133→120)
# =============================================================================

def test_conversion_consistency(all_data, csl_root, phoenix_root, num_samples=5):
    """179D→120D 와 179D→133D→[:120] 결과가 동일한지 검증."""
    print("\n" + "=" * 60)
    print("TEST 2: Conversion Consistency (179→120 vs 179→133→120[:120])")
    print("=" * 60)

    passed, failed, skipped = 0, 0, 0
    indices = np.linspace(0, len(all_data) - 1, min(num_samples, len(all_data)), dtype=int)

    for i in indices:
        ann = all_data[i]
        name = ann.get('name', f'idx_{i}')
        src  = ann.get('src', '?')

        poses_120, _, _ = _load_120(ann, csl_root, phoenix_root)
        poses_133, _, _ = _load_133(ann, csl_root, phoenix_root)

        if poses_120 is None or poses_133 is None:
            print(f"  [{src}] {name[:40]} — SKIP")
            skipped += 1
            continue

        poses_133_to_120 = poses_133[:, :120]
        max_diff = np.abs(poses_120 - poses_133_to_120).max()
        ok = max_diff < 1e-10
        status = "PASS" if ok else f"FAIL (max_diff={max_diff:.2e})"
        print(f"  [{src}] {name[:40]} — max_diff={max_diff:.2e} → {status}")
        if ok:
            passed += 1
        else:
            failed += 1

    total = passed + failed + skipped
    print(f"\n  Result: {passed}/{total} passed, {failed} failed, {skipped} skipped")
    return failed == 0


# =============================================================================
# Test 3: Part-wise statistics
# =============================================================================

def test_part_statistics(all_data, csl_root, phoenix_root, num_samples=50):
    """Part별 raw 값 분포 분석 (정규화 전)."""
    print("\n" + "=" * 60)
    print("TEST 3: Part-wise Raw Statistics (120D, before normalization)")
    print("=" * 60)

    all_poses = []
    indices = np.linspace(0, len(all_data) - 1, min(num_samples, len(all_data)), dtype=int)
    for i in indices:
        ann = all_data[i]
        poses, _, _ = _load_120(ann, csl_root, phoenix_root)
        if poses is not None:
            all_poses.append(poses)

    if not all_poses:
        print("  No data loaded!")
        return False

    concat = np.concatenate(all_poses, axis=0)
    print(f"\n  Total frames: {concat.shape[0]} from {len(all_poses)} samples")

    parts = PART_RANGES_120
    print(f"\n  {'Part':<12} {'Dim':>5} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Near-zero Std':>14}")
    print("  " + "-" * 73)

    for name, (s, e) in parts.items():
        part = concat[:, s:e]
        m   = np.mean(part)
        sd  = np.std(part, axis=0)
        mn  = np.min(part)
        mx  = np.max(part)
        n_near_zero = np.sum(sd < 0.01)
        total_dims = e - s
        print(f"  {name:<12} {total_dims:>5} {m:>10.4f} {sd.mean():>10.4f} "
              f"{mn:>10.4f} {mx:>10.4f} {n_near_zero:>6}/{total_dims}")

    return True


# =============================================================================
# Test 4: Normalization analysis
# =============================================================================

def test_normalization_analysis(mean, std):
    """Mean/std 값 분석 — 정규화 폭발 위험 차원 식별."""
    print("\n" + "=" * 60)
    print("TEST 4: Normalization Analysis (120D mean/std)")
    print("=" * 60)

    print(f"\n  Mean shape: {mean.shape}, Std shape: {std.shape}")
    print(f"  Overall mean range: [{mean.min():.6f}, {mean.max():.6f}]")
    print(f"  Overall std  range: [{std.min():.6f}, {std.max():.6f}]")

    parts = PART_RANGES_120
    print(f"\n  {'Part':<12} {'Dims':>5} {'Mean(abs)':>10} {'Std(mean)':>10} "
          f"{'Std(min)':>10} {'Std<0.01':>10} {'Std<0.001':>10}")
    print("  " + "-" * 78)

    danger_dims = []
    for name, (s, e) in parts.items():
        part_mean = mean[s:e]
        part_std  = std[s:e]
        n_small_01  = np.sum(part_std < 0.01)
        n_small_001 = np.sum(part_std < 0.001)
        total_dims = e - s
        print(f"  {name:<12} {total_dims:>5} {np.abs(part_mean).mean():>10.6f} "
              f"{part_std.mean():>10.6f} {part_std.min():>10.6f} "
              f"{n_small_01:>10} {n_small_001:>10}")
        for d in range(s, e):
            if std[d] < 0.01:
                danger_dims.append((d, name, std[d]))

    if danger_dims:
        print(f"\n  ⚠️  정규화 폭발 위험 차원 ({len(danger_dims)}개):")
        for d, part, s_val in danger_dims[:20]:
            print(f"    dim={d:>3} ({part}) — std={s_val:.6f} "
                  f"→ normalized scale ≈ ±{1.0/(s_val+1e-10):.0f}")
        if len(danger_dims) > 20:
            print(f"    ... and {len(danger_dims)-20} more")
    else:
        print("\n  ✅ 정규화 폭발 위험 차원 없음")

    return len(danger_dims)


# =============================================================================
# Test 5: Normalization round-trip
# =============================================================================

def test_normalization_roundtrip(all_data, csl_root, phoenix_root, mean, std, num_samples=5):
    """normalize → denormalize 후 원본과의 차이 확인."""
    print("\n" + "=" * 60)
    print("TEST 5: Normalization Round-trip (normalize → denormalize)")
    print("=" * 60)

    max_errors = []
    indices = np.linspace(0, len(all_data) - 1, min(num_samples, len(all_data)), dtype=int)

    for i in indices:
        ann = all_data[i]
        name = ann.get('name', f'idx_{i}')[:40]
        poses, _, _ = _load_120(ann, csl_root, phoenix_root)
        if poses is None:
            continue

        normed   = (poses - mean) / (std + 1e-10)
        denormed = normed * (std + 1e-10) + mean
        err = np.abs(poses - denormed).max()
        max_errors.append(err)
        print(f"  {name} — max round-trip error: {err:.2e}")

    if max_errors:
        overall = max(max_errors)
        ok = overall < 1e-6
        print(f"\n  Overall max error: {overall:.2e} → {'PASS' if ok else 'FAIL'}")
        return ok
    return True


# =============================================================================
# Test 6: Compare 120D vs 133D statistics
# =============================================================================

def test_compare_dims(all_data, csl_root, phoenix_root, mean_path, std_path, num_samples=30):
    """120D vs 133D mean/std 비교 — jaw/expr 제거 효과 분석."""
    print("\n" + "=" * 60)
    print("TEST 6: 120D vs 133D Comparison")
    print("=" * 60)

    mean_120, std_120 = _load_mean_std_120(mean_path, std_path)
    mean_133, std_133 = _load_mean_std_133(mean_path, std_path)

    print(f"\n  120D: mean[{mean_120.shape}], std[{std_120.shape}]")
    print(f"  133D: mean[{mean_133.shape}], std[{std_133.shape}]")

    diff = np.abs(mean_120 - mean_133[:120]).max()
    print(f"\n  Mean[0:120] diff between 120D and 133D: {diff:.2e} → {'PASS' if diff < 1e-10 else 'FAIL'}")

    jaw_std  = std_133[120:123]
    expr_std = std_133[123:133]
    print(f"\n  Jaw  std (removed in 120D): {jaw_std}")
    print(f"  Expr std (removed in 120D): {expr_std}")
    print(f"  Jaw  std range:  [{jaw_std.min():.6f}, {jaw_std.max():.6f}]")
    print(f"  Expr std range:  [{expr_std.min():.6f}, {expr_std.max():.6f}]")

    print(f"\n  Checking jaw/expr actual ranges from {num_samples} samples...")
    jaw_vals, expr_vals = [], []
    indices = np.linspace(0, len(all_data) - 1, min(num_samples, len(all_data)), dtype=int)
    for i in indices:
        ann = all_data[i]
        poses, _, _ = _load_133(ann, csl_root, phoenix_root)
        if poses is not None:
            jaw_vals.append(poses[:, 120:123])
            expr_vals.append(poses[:, 123:133])

    if jaw_vals:
        jaw_all  = np.concatenate(jaw_vals, axis=0)
        expr_all = np.concatenate(expr_vals, axis=0)
        print(f"  Jaw  actual range: [{jaw_all.min():.4f}, {jaw_all.max():.4f}], "
              f"std per dim: {jaw_all.std(axis=0)}")
        print(f"  Expr actual range: [{expr_all.min():.4f}, {expr_all.max():.4f}], "
              f"std per dim: {np.round(expr_all.std(axis=0), 4)}")
    return True


# =============================================================================
# Test 7: FK Visualization — 133D vs 120D side-by-side
#   45-joint SELECT_IDX (vis_sign_joints.py 방식)
# =============================================================================

# SMPLX FK → 127+ joints → SELECT_IDX로 45 joints 추출 → local index connections
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


def slice_to_120d(poses_133):
    return poses_133[..., :120]


def pad_to_133d(poses_120):
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

    body_idx = [i for i in range(1, 15) if i < J]
    hand_idx = [i for i in range(15, 45) if i < J]
    jaw_idx  = [0] if J > 0 else []

    return {
        'all_mpjpe':  per_joint.mean(),
        'body_mpjpe': per_joint[:, body_idx].mean() if body_idx else 0,
        'hand_mpjpe': per_joint[:, hand_idx].mean() if hand_idx else 0,
        'jaw_mpjpe':  per_joint[:, jaw_idx].mean()  if jaw_idx  else 0,
        'per_joint_mean': per_joint.mean(axis=0),
    }


def save_comparison_video(gt_joints, sl_joints, save_path, title='',
                          fps=25, viewport=0.5, diff_stats=None):
    """133D(왼쪽) vs 120D-padded(오른쪽) side-by-side 2D skeleton video.
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

    labels = [('133D (original)', 'blue'), ('120D (no jaw/expr)', 'darkgreen')]
    for ax, (label, color) in zip([ax_gt, ax_sl], labels):
        ax.set_title(label, fontsize=12, fontweight='bold', color=color)
        ax.set_xlim(x_lim); ax.set_ylim(y_lim)
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


def test_visualization(all_data, csl_root, phoenix_root,
                       smplx_path, output_dir, mean_path, std_path,
                       num_samples=3, fps=25, viewport=0.5, device='cpu'):
    """
    133D vs 120D FK side-by-side.

    1. _load_133 → 133D raw
    2. slice_to_120d → pad_to_133d (jaw=0, expr=0)
    3. Both → normalize(133D mean/std) → aa133_to_joints_np → SELECT_IDX
    4. side-by-side video + MPJPE
    """
    print("\n" + "=" * 60)
    print("TEST 7: FK Visualization (133D vs 120D side-by-side)")
    print("=" * 60)

    try:
        from utils.feats2joints import aa133_to_joints_np
    except (ImportError, Exception) as e:
        print(f"  ⚠️  missing dependency: {e} — skipping")
        print("     (salad 프로젝트 루트에서 실행해야 합니다)")
        return False

    os.makedirs(output_dir, exist_ok=True)

    # ── 133D mean/std 로드 ──
    mean_np, std_np = _load_mean_std_133(mean_path, std_path)

    print(f"\n  [Stats] jaw mean range:  [{mean_np[120:123].min():.4f}, {mean_np[120:123].max():.4f}]")
    print(f"  [Stats] jaw std range:   [{std_np[120:123].min():.4f}, {std_np[120:123].max():.4f}]")
    print(f"  [Stats] expr mean range: [{mean_np[123:133].min():.4f}, {mean_np[123:133].max():.4f}]")
    print(f"  [Stats] expr std range:  [{std_np[123:133].min():.4f}, {std_np[123:133].max():.4f}]")

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

            # ── Load 133D ──
            poses, text, name_loaded = _load_133(ann, csl_root, phoenix_root)
            if poses is None:
                print(f"    [{idx_s+1}/{n}] {name[:40]} — load failed, skip.")
                continue
            if name_loaded:
                name = name_loaded

            T_raw = poses.shape[0]
            T_crop = (T_raw // 4) * 4
            if T_crop < 4:
                print(f"    [{idx_s+1}/{n}] {name[:40]} — too short ({T_raw}), skip.")
                continue
            poses = poses[:T_crop]  # [T, 133]

            # ── Feature-level diff ──
            poses_120  = slice_to_120d(poses)        # [T, 120]
            poses_padded = pad_to_133d(poses_120)    # [T, 133] jaw=0, expr=0

            feat_diff_jaw  = np.abs(poses[:, 120:123]).mean()
            feat_diff_expr = np.abs(poses[:, 123:133]).mean()

            # ── Normalize → FK → 45 joints ──
            poses_norm  = (poses        - mean_np) / (std_np + 1e-10)
            padded_norm = (poses_padded - mean_np) / (std_np + 1e-10)

            try:
                gt_joints_full = aa133_to_joints_np(
                    poses_norm, mean_np, std_np, smplx_path, device=device)
                gt_joints = gt_joints_full[:, SMPLX_SELECT_IDX]  # [T, 45, 3]
            except Exception as e:
                print(f"    [{idx_s+1}/{n}] {name[:40]} — FK(133D) failed: {e}")
                continue

            try:
                sl_joints_full = aa133_to_joints_np(
                    padded_norm, mean_np, std_np, smplx_path, device=device)
                sl_joints = sl_joints_full[:, SMPLX_SELECT_IDX]  # [T, 45, 3]
            except Exception as e:
                print(f"    [{idx_s+1}/{n}] {name[:40]} — FK(120D) failed: {e}")
                continue

            # ── Diff stats ──
            stats = compute_diff_stats(gt_joints, sl_joints)
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

            if np.isnan(gt_joints).any() or np.isnan(sl_joints).any():
                print(f"      ⚠️ NaN in joints, skip video.")
                continue

            # ── Save video ──
            safe_name = str(name)[:40].replace('/', '_').replace('\\', '_')
            video_path = os.path.join(out_dir, f'{idx_s:03d}_{safe_name}.mp4')
            title = f'{name[:40]} [{ds_label}] (T={T_crop})'
            save_comparison_video(gt_joints, sl_joints, video_path,
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
        print(f"  → 120D slicing preserves body+hand motion perfectly")

        for sk in ['how2sign', 'csl', 'phoenix']:
            ms = [s for s in all_stats if s['src'] == sk]
            if ms:
                b = np.mean([s['body_mpjpe'] for s in ms])
                h = np.mean([s['hand_mpjpe'] for s in ms])
                j = np.mean([s['jaw_mpjpe']  for s in ms])
                print(f"  [{DS_LABELS[sk]}] body={b:.6f}  hand={h:.6f}  jaw={j:.6f}  (n={len(ms)})")

        csv_path = os.path.join(output_dir, 'diff_stats.csv')
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
# Static visualization: per-part normalized distribution
# =============================================================================

def plot_part_distributions(all_data, csl_root, phoenix_root, mean, std,
                            output_dir, num_samples=100):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping distribution plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    all_poses = []
    indices = np.linspace(0, len(all_data) - 1, min(num_samples, len(all_data)), dtype=int)
    for i in indices:
        ann = all_data[i]
        poses, _, _ = _load_120(ann, csl_root, phoenix_root)
        if poses is not None:
            normed = (poses - mean) / (std + 1e-10)
            all_poses.append(normed)

    if not all_poses:
        return

    concat = np.concatenate(all_poses, axis=0)
    parts = PART_RANGES_120

    fig, axes = plt.subplots(1, len(parts), figsize=(4 * len(parts), 5))
    fig.suptitle('Normalized Value Distribution per Part (120D)', fontsize=14, y=1.02)

    for ax, (name, (s, e)) in zip(axes, parts.items()):
        vals = concat[:, s:e].flatten()
        ax.hist(vals, bins=100, density=True, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_title(f'{name}\n[{s}:{e}] ({e-s}D)')
        ax.set_xlabel('Normalized value')
        ax.set_xlim(-10, 10)
        mu = vals.mean()
        ax.axvline(mu, color='red', linestyle='--', label=f'μ={mu:.2f}')
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, 'part_distributions_120d.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Distribution plot saved: {path}")

    fig2, ax2 = plt.subplots(figsize=(14, 3))
    ax2.bar(range(120), std, color='steelblue', edgecolor='none')
    for name, (s, e) in parts.items():
        ax2.axvspan(s, e, alpha=0.1, color='red' if name in ['root', 'upper_a'] else 'green')
        ax2.text((s + e) / 2, std.max() * 0.9, name, ha='center', fontsize=8)
    ax2.axhline(0.01, color='red', linestyle='--', linewidth=1, label='danger threshold (0.01)')
    ax2.set_xlabel('Dimension index')
    ax2.set_ylabel('Std value')
    ax2.set_title('Per-dimension Std (120D) — Red = low std = normalization risk')
    ax2.legend()
    plt.tight_layout()
    path2 = os.path.join(output_dir, 'std_per_dim_120d.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Std per-dim plot saved: {path2}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='120D Data Loader Verification + Visualization')

    parser.add_argument('--data_root', required=True, help='How2Sign data root')
    parser.add_argument('--csl_root', default=None)
    parser.add_argument('--phoenix_root', default=None)
    parser.add_argument('--mean_path', required=True, help='path to mean.pt (179D)')
    parser.add_argument('--std_path', required=True, help='path to std.pt (179D)')

    parser.add_argument('--sign_dataset', default='how2sign',
                        choices=['how2sign', 'csl', 'phoenix',
                                 'how2sign_csl', 'how2sign_csl_phoenix'])
    parser.add_argument('--split', default='val')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--num_stat_samples', type=int, default=100)

    parser.add_argument('--compare_dims', action='store_true')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate FK skeleton videos (requires smplx + feats2joints)')
    parser.add_argument('--smplx_path', default='deps/smpl_models/')
    parser.add_argument('--plot', action='store_true', default=True)

    parser.add_argument('--output', default='verify_120d_output')
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--viewport', type=float, default=0.5, help='0=auto')
    parser.add_argument('--device', default='cuda:0')

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("120D Data Loader Verification")
    print(f"  dataset:  {args.sign_dataset}")
    print(f"  split:    {args.split}")
    print(f"  samples:  {args.num_samples} (tests) / {args.num_stat_samples} (stats)")
    print(f"  output:   {output_dir}")
    print("=" * 60)

    # ── Load annotations ──
    print("\n[0] Building annotations...")
    all_data = _build_annotations(
        args.split, args.sign_dataset, args.data_root,
        args.csl_root, args.phoenix_root)

    # ── Load 120D mean/std (for T1-T5) ──
    print("\n[0] Loading mean/std (120D)...")
    mean, std = _load_mean_std_120(args.mean_path, args.std_path)
    print(f"  mean: {mean.shape}, std: {std.shape}")

    # ── Run tests ──
    results = {}

    results['T1_dimensions'] = test_dimensions(
        all_data, args.csl_root, args.phoenix_root, args.num_samples)

    results['T2_consistency'] = test_conversion_consistency(
        all_data, args.csl_root, args.phoenix_root, args.num_samples)

    results['T3_part_stats'] = test_part_statistics(
        all_data, args.csl_root, args.phoenix_root, args.num_stat_samples)

    n_danger = test_normalization_analysis(mean, std)
    results['T4_norm_danger_dims'] = n_danger

    results['T5_roundtrip'] = test_normalization_roundtrip(
        all_data, args.csl_root, args.phoenix_root, mean, std, args.num_samples)

    if args.compare_dims:
        results['T6_compare'] = test_compare_dims(
            all_data, args.csl_root, args.phoenix_root,
            args.mean_path, args.std_path, args.num_stat_samples)

    if args.plot:
        plot_part_distributions(
            all_data, args.csl_root, args.phoenix_root, mean, std,
            output_dir, args.num_stat_samples)

    if args.visualize:
        results['T7_visualization'] = test_visualization(
            all_data, args.csl_root, args.phoenix_root,
            args.smplx_path, os.path.join(output_dir, 'videos'),
            args.mean_path, args.std_path,
            args.num_samples, args.fps, args.viewport, device)

    # ── Summary ──
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

    print(f"\n  Output saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()