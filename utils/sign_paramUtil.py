"""
sign_paramUtil.py — Sign language skeleton parameters for SALAD.

기존 7part / finger 모드 + 새로운 sign10 / sign10_vel 모드.

[7part]      133D → 7 super-joints (original)
[finger]     133D → 15 tokens     (original)
[sign10]     120D → 10 super-joints (NEW: jaw/expr 제거, 손가락 그룹핑)
[sign10_vel] 210D → 10 super-joints (NEW: rotation 120D + hand velocity 90D)
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# 7part mode (default) — 기존 코드 유지
# ═══════════════════════════════════════════════════════════════════════════

SIGN_SPLITS_7PART = [3, 12, 15, 45, 45, 3, 10]   # J=7, sum=133

sign_adj_list_7part = [
    [1, 5],         # 0: root
    [0, 2],         # 1: upper_body_a
    [1, 3, 4],      # 2: upper_body_b
    [2],            # 3: left_hand
    [2],            # 4: right_hand
    [0, 6],         # 5: jaw
    [5],            # 6: expression
]

SIGN_PART_NAMES_7PART = [
    'root', 'upper_body_a', 'upper_body_b',
    'left_hand', 'right_hand', 'jaw', 'expression',
]

HAND_INDICES_7PART = [3, 4]


# ═══════════════════════════════════════════════════════════════════════════
# finger mode — 기존 코드 유지
# ═══════════════════════════════════════════════════════════════════════════

SIGN_SPLITS_FINGER = [
    3, 12, 15,
    9, 9, 9, 9, 9,
    9, 9, 9, 9, 9,
    3, 10,
]   # J=15, sum=133

sign_adj_list_finger = [
    [1, 13],
    [0, 2],
    [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    [2], [2], [2], [2], [2],
    [2], [2], [2], [2], [2],
    [0, 14],
    [13],
]

SIGN_PART_NAMES_FINGER = [
    'root', 'upper_body_a', 'upper_body_b',
    'l_thumb', 'l_index', 'l_middle', 'l_ring', 'l_pinky',
    'r_thumb', 'r_index', 'r_middle', 'r_ring', 'r_pinky',
    'jaw', 'expression',
]

HAND_INDICES_FINGER = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


# ═══════════════════════════════════════════════════════════════════════════
# sign10 / sign10_vel mode — NEW
# ═══════════════════════════════════════════════════════════════════════════

from utils.sign10_config import (
    SIGN_SPLITS_SIGN10,
    SIGN_SPLITS_SIGN10_VEL,
    sign_adj_list_sign10,
    HAND_INDICES_SIGN10,
    SIGN_PART_NAMES_SIGN10,
    # 변환 함수들도 re-export
    reorder_to_sign10,
    reorder_from_sign10,
    rotation_to_sign10_vel,
    sign10_vel_to_rotation,
    raw133_to_sign10,
    sign10_to_raw120,
    pad_to_133,
    load_mean_std_sign10,
    # ── backward-compat aliases (load_sign_data.py에서 import) ──
    REORDER_RAW120_TO_SIGN10,
    REORDER_SIGN10_TO_RAW120,
    _179_to_120_raw,
    _179_to_120_sign10,
    _133_to_120_sign10,
)


# ═══════════════════════════════════════════════════════════════════════════
# FK helper: Sign10 120D → SMPL-X input dict
# ═══════════════════════════════════════════════════════════════════════════

def sign10_to_smplx_input(flat_sign10):
    """Sign10 120D denormalized [BT, 120] → SMPL-X input dict for FK.

    aa133_to_joints와 동일 컨벤션:
      - 133D에서 36D lower body를 prepend(zeros)한 뒤 169D로 분해
      - global_orient = zeros(3)  (root orient는 body_pose 안에 포함)
      - body_pose = zeros(33, lower) + root(3) + upper_body(27) = 63D
      - left_hand_pose  = 45D
      - right_hand_pose = 45D
    """
    import torch

    # sign10 order → raw 120D order
    raw120 = flat_sign10[..., REORDER_SIGN10_TO_RAW120]  # [BT, 120]

    BT = raw120.shape[0]
    dev, dt = raw120.device, raw120.dtype

    return {
        'global_orient':   torch.zeros(BT, 3, device=dev, dtype=dt),
        'body_pose':       torch.cat([torch.zeros(BT, 33, device=dev, dtype=dt),
                                      raw120[:, :30]], dim=-1),   # 63D
        'left_hand_pose':  raw120[:, 30:75],                      # 45D
        'right_hand_pose': raw120[:, 75:120],                     # 45D
    }


# ═══════════════════════════════════════════════════════════════════════════
# Dispatch helpers
# ═══════════════════════════════════════════════════════════════════════════

# backward compat aliases (기존 코드에서 import 하는 경우)
sign_adj_list = sign_adj_list_7part
SIGN_SPLITS = SIGN_SPLITS_7PART


def get_sign_config(skeleton_mode='7part'):
    """Return (splits, adj_list, hand_indices, num_joints) for given mode."""
    if skeleton_mode == '7part':
        return SIGN_SPLITS_7PART, sign_adj_list_7part, HAND_INDICES_7PART, 7
    elif skeleton_mode == 'finger':
        return SIGN_SPLITS_FINGER, sign_adj_list_finger, HAND_INDICES_FINGER, 15
    elif skeleton_mode == 'sign10':
        return SIGN_SPLITS_SIGN10, sign_adj_list_sign10, HAND_INDICES_SIGN10, 10
    elif skeleton_mode == 'sign10_vel':
        return SIGN_SPLITS_SIGN10_VEL, sign_adj_list_sign10, HAND_INDICES_SIGN10, 10
    else:
        raise ValueError(f"Unknown skeleton_mode: {skeleton_mode}")


def get_pose_dim(skeleton_mode='7part'):
    """Return total pose dimension for given mode."""
    splits, _, _, _ = get_sign_config(skeleton_mode)
    return sum(splits)
    # 7part → 133, finger → 133, sign10 → 120, sign10_vel → 210


def get_part_names(skeleton_mode='7part'):
    """Return human-readable part names."""
    names = {
        '7part': SIGN_PART_NAMES_7PART,
        'finger': SIGN_PART_NAMES_FINGER,
        'sign10': SIGN_PART_NAMES_SIGN10,
        'sign10_vel': SIGN_PART_NAMES_SIGN10,
    }
    return names[skeleton_mode]