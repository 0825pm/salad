"""
sign10_config.py — Sign10 / Sign10_vel super-joint 정의 및 변환 함수

=== Sign10 Super-Joint 정의 (10 nodes, 120D) ===

Raw 120D layout (SMPL-X upper body, jaw/expr 제거):
  [0:3]     root_pose        3D
  [3:30]    body_pose         27D  (9 joints × 3D)
  [30:75]   left_hand_pose    45D  (15 joints × 3D)
  [75:120]  right_hand_pose   45D  (15 joints × 3D)

Sign10 reorder → 120D:
  SJ0: root           [0:3]      3D   root_pose
  SJ1: spine           [3:15]    12D   body joints 1-4
  SJ2: shoulders       [15:24]    9D   body joints 5-7
  SJ3: arms            [24:30]    6D   body joints 8-9
  SJ4: l_thumb         [30:39]    9D   lhand thumb (joints 12,13,14)
  SJ5: l_index         [39:48]    9D   lhand index (joints 0,1,2)
  SJ6: l_mrp           [48:75]   27D   lhand middle+ring+pinky
  SJ7: r_thumb         [75:84]    9D   rhand thumb (joints 12,13,14)
  SJ8: r_index         [84:93]    9D   rhand index (joints 0,1,2)
  SJ9: r_mrp           [93:120]  27D   rhand middle+ring+pinky

=== Sign10_vel (10 nodes, 210D) ===

Body joints (SJ0-3): rotation only → 30D
Hand joints (SJ4-9): rotation + velocity → 180D  (90D rot + 90D vel)
Total: 30 + 180 = 210D
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Sign10 Constants
# ═══════════════════════════════════════════════════════════════════════════

SIGN_SPLITS_SIGN10 = [3, 12, 9, 6, 9, 9, 27, 9, 9, 27]  # J=10, sum=120

SIGN_SPLITS_SIGN10_VEL = [3, 12, 9, 6, 18, 18, 54, 18, 18, 54]  # J=10, sum=210

SIGN10_ROT_DIMS = [3, 12, 9, 6, 9, 9, 27, 9, 9, 27]
SIGN10_VEL_DIMS = [0, 0, 0, 0, 9, 9, 27, 9, 9, 27]
SIGN10_VEL_MASK = [False, False, False, False, True, True, True, True, True, True]

sign_adj_list_sign10 = [
    [1],                          # 0: root
    [0, 2],                       # 1: spine
    [1, 3],                       # 2: shoulders
    [2, 4, 5, 6, 7, 8, 9],       # 3: arms → all hand nodes
    [3],                          # 4: l_thumb
    [3],                          # 5: l_index
    [3],                          # 6: l_mrp
    [3],                          # 7: r_thumb
    [3],                          # 8: r_index
    [3],                          # 9: r_mrp
]

HAND_INDICES_SIGN10 = [4, 5, 6, 7, 8, 9]

SIGN_PART_NAMES_SIGN10 = [
    'root', 'spine', 'shoulders', 'arms',
    'l_thumb', 'l_index', 'l_mrp',
    'r_thumb', 'r_index', 'r_mrp',
]


# ═══════════════════════════════════════════════════════════════════════════
# Reorder Index Mapping: raw 120D ↔ sign10 120D
# ═══════════════════════════════════════════════════════════════════════════

def _build_sign10_reorder_indices():
    """Build index mapping: raw 120D → sign10 120D."""
    idx = []
    idx += list(range(0, 3))       # SJ0: root
    idx += list(range(3, 15))      # SJ1: spine (body joints 1-4)
    idx += list(range(15, 24))     # SJ2: shoulders (body joints 5-7)
    idx += list(range(24, 30))     # SJ3: arms (body joints 8-9)

    lh = 30  # left hand offset
    idx += list(range(lh + 36, lh + 45))  # SJ4: l_thumb (joints 12-14)
    idx += list(range(lh + 0, lh + 9))    # SJ5: l_index (joints 0-2)
    idx += list(range(lh + 9, lh + 18))   # SJ6: l_mrp — middle
    idx += list(range(lh + 27, lh + 36))  #              ring
    idx += list(range(lh + 18, lh + 27))  #              pinky

    rh = 75  # right hand offset
    idx += list(range(rh + 36, rh + 45))  # SJ7: r_thumb
    idx += list(range(rh + 0, rh + 9))    # SJ8: r_index
    idx += list(range(rh + 9, rh + 18))   # SJ9: r_mrp — middle
    idx += list(range(rh + 27, rh + 36))  #              ring
    idx += list(range(rh + 18, rh + 27))  #              pinky

    assert len(idx) == 120 and len(set(idx)) == 120
    return np.array(idx, dtype=np.int64)


_SIGN10_FWD_IDX = _build_sign10_reorder_indices()
_SIGN10_INV_IDX = np.argsort(_SIGN10_FWD_IDX)

# ── Backward-compat alias (load_sign_data.py에서 import) ──
REORDER_RAW120_TO_SIGN10 = _SIGN10_FWD_IDX
REORDER_SIGN10_TO_RAW120 = _SIGN10_INV_IDX


# ═══════════════════════════════════════════════════════════════════════════
# Reorder Functions
# ═══════════════════════════════════════════════════════════════════════════

def reorder_to_sign10(poses_raw120):
    """Raw 120D → Sign10 order 120D."""
    return poses_raw120[..., _SIGN10_FWD_IDX]


def reorder_from_sign10(poses_sign10):
    """Sign10 order 120D → Raw 120D."""
    return poses_sign10[..., _SIGN10_INV_IDX]


# ═══════════════════════════════════════════════════════════════════════════
# Velocity Conversion: 120D ↔ 210D
# ═══════════════════════════════════════════════════════════════════════════

def _build_rot_extract_indices():
    """210D에서 rotation 부분만 추출하는 인덱스."""
    indices = []
    offset = 0
    for rot_dim, vel_dim in zip(SIGN10_ROT_DIMS, SIGN10_VEL_DIMS):
        indices += list(range(offset, offset + rot_dim))
        offset += rot_dim + vel_dim
    assert offset == 210 and len(indices) == 120
    return np.array(indices, dtype=np.int64)


_ROT_EXTRACT_IDX = _build_rot_extract_indices()


def rotation_to_sign10_vel(poses_sign10_120):
    """Sign10 120D (normalized) → Sign10_vel 210D.

    Velocity = frame diff for hand joints only. First frame velocity = 0.
    """
    is_3d = poses_sign10_120.ndim == 3
    if not is_3d:
        poses_sign10_120 = poses_sign10_120[np.newaxis]

    B, T, D = poses_sign10_120.shape
    assert D == 120, f"Expected 120D input, got {D}D"

    rot_parts = np.split(poses_sign10_120, np.cumsum(SIGN10_ROT_DIMS[:-1]), axis=-1)

    output_parts = []
    for i in range(10):
        rot = rot_parts[i]
        if SIGN10_VEL_MASK[i]:
            vel = np.zeros_like(rot)
            vel[:, 1:] = rot[:, 1:] - rot[:, :-1]
            output_parts.append(np.concatenate([rot, vel], axis=-1))
        else:
            output_parts.append(rot)

    result = np.concatenate(output_parts, axis=-1)
    assert result.shape[-1] == 210

    if not is_3d:
        result = result[0]
    return result


def sign10_vel_to_rotation(poses_210):
    """Sign10_vel 210D → Sign10 120D (extract rotation, discard velocity)."""
    return poses_210[..., _ROT_EXTRACT_IDX]


# ═══════════════════════════════════════════════════════════════════════════
# Full Pipeline Helpers
# ═══════════════════════════════════════════════════════════════════════════

def raw133_to_sign10(poses_133):
    """133D raw axis-angle → 120D sign10 order (drop jaw+expr, reorder)."""
    return reorder_to_sign10(poses_133[..., :120])


def sign10_to_raw120(poses_sign10):
    """120D sign10 → 120D raw order."""
    return reorder_from_sign10(poses_sign10)


def pad_to_133(poses_raw120, jaw_val=0.0, expr_val=0.0):
    """120D raw → 133D (append zero jaw + expr)."""
    shape = list(poses_raw120.shape)
    shape[-1] = 3
    jaw = np.full(shape, jaw_val, dtype=poses_raw120.dtype)
    shape[-1] = 10
    expr = np.full(shape, expr_val, dtype=poses_raw120.dtype)
    return np.concatenate([poses_raw120, jaw, expr], axis=-1)


# ═══════════════════════════════════════════════════════════════════════════
# Dimension Conversion Helpers (backward-compat for load_sign_data.py)
# ═══════════════════════════════════════════════════════════════════════════

def _179_to_120_raw(poses_179):
    """SOKE 179D → 120D raw order.

    SOKE 179D layout:
      [0:36]    lower body (root_trans 3D + leg joints 33D) — DROP
      [36:39]   root_pose       3D
      [39:66]   body_pose       27D  (9 upper body joints)
      [66:111]  lhand_pose      45D
      [111:156] rhand_pose      45D
      [156:159] jaw_pose        3D   — DROP
      [159:169] shape           10D  — DROP
      [169:179] expression      10D  — DROP

    Output: 120D raw = root(3) + body(27) + lhand(45) + rhand(45)
    """
    return poses_179[..., 36:156]


def _179_to_120_sign10(poses_179):
    """SOKE 179D → 120D sign10 order."""
    raw120 = _179_to_120_raw(poses_179)
    return reorder_to_sign10(raw120)


def _133_to_120_sign10(poses_133):
    """133D (SALAD format) → 120D sign10 order.

    133D = root(3) + body(27) + lhand(45) + rhand(45) + jaw(3) + expr(10)
    Drop jaw+expr (last 13D), then reorder.
    """
    return reorder_to_sign10(poses_133[..., :120])


# ═══════════════════════════════════════════════════════════════════════════
# Load mean/std with sign10 support
# ═══════════════════════════════════════════════════════════════════════════

def load_mean_std_sign10(mean_path, std_path):
    """Load SOKE-format mean/std (.pt) → 120D sign10 order."""
    import torch

    mean_raw = torch.load(mean_path, map_location='cpu')
    std_raw = torch.load(std_path, map_location='cpu')

    if isinstance(mean_raw, torch.Tensor):
        mean_raw = mean_raw.numpy()
        std_raw = std_raw.numpy()

    D = mean_raw.shape[0]

    if D == 179:
        mean_120 = _179_to_120_sign10(mean_raw)
        std_120 = _179_to_120_sign10(std_raw)
    elif D == 133:
        mean_120 = _133_to_120_sign10(mean_raw)
        std_120 = _133_to_120_sign10(std_raw)
    elif D == 120:
        mean_120, std_120 = mean_raw, std_raw
    else:
        raise ValueError(f"Unexpected mean shape: {mean_raw.shape}")

    return mean_120, std_120