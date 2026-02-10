"""
Sign language skeleton parameters for SALAD.

133D SMPL-X upper body, two skeleton modes:

[7part] 133D → 7 super-joints (손 전체 = 1토큰)
  SJ0: root_pose        [0:3]     3D
  SJ1: upper_body_a     [3:15]   12D  (body joints 1-4)
  SJ2: upper_body_b     [15:30]  15D  (body joints 5-9)
  SJ3: left_hand        [30:75]  45D  (15 hand joints)
  SJ4: right_hand       [75:120] 45D  (15 hand joints)
  SJ5: jaw              [120:123]  3D
  SJ6: expression       [123:133] 10D

[finger] 133D → 15 tokens (손가락별 = 수형 구분 가능)
  0: root(3)  1: upper_a(12)  2: upper_b(15)
  3-7: l_thumb~l_pinky (9D each = 3joints×3D)
  8-12: r_thumb~r_pinky (9D each)
  13: jaw(3)  14: expression(10)
"""

# ── 7part mode (default) ──
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

HAND_INDICES_7PART = [3, 4]   # lhand, rhand


# ── finger mode (권장: 수어 수형 구분) ──
SIGN_SPLITS_FINGER = [
    3, 12, 15,                    # body: root, upper_a, upper_b
    9, 9, 9, 9, 9,               # lhand: thumb, index, middle, ring, pinky
    9, 9, 9, 9, 9,               # rhand: thumb, index, middle, ring, pinky
    3, 10,                        # face: jaw, expression
]   # J=15, sum=133

sign_adj_list_finger = [
    [1, 13],                      # 0: root
    [0, 2],                       # 1: upper_body_a
    [1, 3, 4, 5, 6, 7,           # 2: upper_body_b → all fingers
        8, 9, 10, 11, 12],
    [2],                          # 3: l_thumb
    [2],                          # 4: l_index
    [2],                          # 5: l_middle
    [2],                          # 6: l_ring
    [2],                          # 7: l_pinky
    [2],                          # 8: r_thumb
    [2],                          # 9: r_index
    [2],                          # 10: r_middle
    [2],                          # 11: r_ring
    [2],                          # 12: r_pinky
    [0, 14],                      # 13: jaw
    [13],                         # 14: expression
]

SIGN_PART_NAMES_FINGER = [
    'root', 'upper_body_a', 'upper_body_b',
    'l_thumb', 'l_index', 'l_middle', 'l_ring', 'l_pinky',
    'r_thumb', 'r_index', 'r_middle', 'r_ring', 'r_pinky',
    'jaw', 'expression',
]

HAND_INDICES_FINGER = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # 10 finger tokens


# ── dispatch helpers ──
def get_sign_config(skeleton_mode='7part'):
    """Return (splits, adj_list, hand_indices, num_joints) for given mode."""
    if skeleton_mode == '7part':
        return SIGN_SPLITS_7PART, sign_adj_list_7part, HAND_INDICES_7PART, 7
    elif skeleton_mode == 'finger':
        return SIGN_SPLITS_FINGER, sign_adj_list_finger, HAND_INDICES_FINGER, 15
    else:
        raise ValueError(f"Unknown skeleton_mode: {skeleton_mode}")


# ── backward compat aliases ──
SIGN_SPLITS = SIGN_SPLITS_7PART
sign_adj_list = sign_adj_list_7part
SIGN_NUM_JOINTS = 7
SIGN_PART_NAMES = SIGN_PART_NAMES_7PART