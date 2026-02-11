"""
load_sign_data.py — Sign language data loading functions.

기존 133D axis-angle + 135D joint coordinates 모두 지원.
joint_root가 주어지면 data_joint/{dataset}/에서 .npy 로드 (FK 불필요).

Ported from SOKE (mGPT/data/humanml/load_data.py).
"""
import pickle
import numpy as np
import os
import math
import torch

SMPLX_KEYS = [
    'smplx_root_pose',    # (3,)
    'smplx_body_pose',    # (63,)
    'smplx_lhand_pose',   # (45,)
    'smplx_rhand_pose',   # (45,)
    'smplx_jaw_pose',     # (3,)
    'smplx_shape',        # (10,)
    'smplx_expr',         # (10,)
]

BAD_H2S_IDS = {
    '0DU7wWLK-QU_0-8-rgb_front', '0ICZi26jdaQ_28-5-rgb_front',
    '0vNfEYst_tQ_11-8-rgb_front', '13X0vEMNm7M_8-5-rgb_front',
    '14weIYQswlE_23-8-rgb_front', '1B56XMJ-j1Q_13-8-rgb_front',
    '1P0oKY4FNyI_0-8-rgb_front', '1dpRaxOTfZs_0-8-rgb_front',
    '1ei1kVTw23A_29-8-rgb_front', '1spCnuBmWYk_0-8-rgb_front',
    '2-vXO7MMLJc_0-5-rgb_front', '21PbS6wnHtY_0-5-rgb_front',
    '3tyfxL2wO-M_0-8-rgb_front', 'BpYDl3AO4B8_0-1-rgb_front',
    'CH7AviIr0-0_14-8-rgb_front', 'CJ8RyW9pzKU_6-8-rgb_front',
    'D0T7ho08Q3o_25-2-rgb_front', 'Db5SUQvNsHc_18-1-rgb_front',
    'Eh697LCFjTw_0-3-rgb_front', 'F-p1IdedNbg_23-8-rgb_front',
    'aUBQCNegrYc_13-1-rgb_front', 'cvn7htBA8Xc_9-8-rgb_front',
    'czBrBQgZIuc_19-5-rgb_front', 'dbSAB8F8GYc_11-9-rgb_front',
    'doMosV-zfCI_7-2-rgb_front', 'dvBdWGLzayI_10-8-rgb_front',
    'eBrlZcccILg_26-3-rgb_front', '39FN42e41r0_17-1-rgb_front',
    'a4Nxq0QV_WA_9-3-rgb_front', 'fzrJBu2qsM8_11-8-rgb_front',
    'g3Cc_1-V31U_12-3-rgb_front',
}


def get_encoder_cache_name(text_encoder, clip_version='ViT-B/32', xlmr_version='xlm-roberta-base'):
    if text_encoder == 'xlm-roberta':
        return xlmr_version.replace('xlm-roberta-', 'xlmr-')
    return 'clip-' + clip_version.replace('/', '').replace('-', '').lower()


def _subsample(input_list, count):
    ss = float(len(input_list)) / count
    return [input_list[int(math.floor(i * ss))] for i in range(count)]


# ═══════════════════════════════════════════════════════════════════════
# 133D axis-angle loading (기존 — 변경 없음)
# ═══════════════════════════════════════════════════════════════════════

def _179_to_133(poses_179):
    poses = poses_179[:, (3 + 3 * 11):]
    poses = np.concatenate([poses[:, :-20], poses[:, -10:]], axis=1)
    return poses


def load_mean_std(mean_path, std_path):
    """Load SOKE-format mean/std (.pt) and filter 179D -> 133D."""
    mean = torch.load(mean_path, map_location='cpu')
    std  = torch.load(std_path,  map_location='cpu')
    mean = mean[(3 + 3 * 11):]
    std  = std[(3 + 3 * 11):]
    mean = torch.cat([mean[:-20], mean[-10:]], dim=0)
    std  = torch.cat([std[:-20],  std[-10:]],  dim=0)
    return mean.numpy(), std.numpy()


def load_h2s_sample(ann, data_dir):
    name = ann['name']
    fps  = ann['fps']
    base_dir = os.path.join(data_dir, name)
    if not os.path.isdir(base_dir):
        return None, None, None

    n_frames = len(os.listdir(base_dir))
    frame_list = [os.path.join(base_dir, f'{name}_{fid}_3D.pkl') for fid in range(n_frames)]
    if fps > 24:
        frame_list = _subsample(frame_list, count=int(24 * len(frame_list) / fps))
    if len(frame_list) < 4:
        return None, None, None

    clip_poses = np.zeros([len(frame_list), 179])
    for fid, fpath in enumerate(frame_list):
        with open(fpath, 'rb') as f:
            poses = pickle.load(f)
        clip_poses[fid] = np.concatenate([poses[k] for k in SMPLX_KEYS], 0)

    return _179_to_133(clip_poses), ann['text'], name


def load_csl_sample(ann, csl_root):
    name, text = ann['name'], ann['text']
    poses_dir = os.path.join(csl_root, 'poses', name)
    if not os.path.isdir(poses_dir):
        return None, None, None
    frame_list = sorted(os.listdir(poses_dir))
    if len(frame_list) < 4:
        return None, None, None

    clip_poses = np.zeros([len(frame_list), 179])
    for fid, fname in enumerate(frame_list):
        with open(os.path.join(poses_dir, fname), 'rb') as f:
            poses = pickle.load(f)
        clip_poses[fid] = np.concatenate([poses[k] for k in SMPLX_KEYS], 0)

    return _179_to_133(clip_poses), text, name


def load_phoenix_sample(ann, phoenix_root):
    name, text = ann['name'], ann['text']
    poses_dir = os.path.join(phoenix_root, name)
    if not os.path.isdir(poses_dir):
        return None, None, None
    frame_list = sorted(os.listdir(poses_dir))
    if len(frame_list) < 4:
        return None, None, None

    clip_poses = np.zeros([len(frame_list), 179])
    for fid, fname in enumerate(frame_list):
        with open(os.path.join(poses_dir, fname), 'rb') as f:
            poses = pickle.load(f)
        clip_poses[fid] = np.concatenate([poses[k] for k in SMPLX_KEYS], 0)

    return _179_to_133(clip_poses), text, name


# ═══════════════════════════════════════════════════════════════════════
# 135D joint coordinates loading (신규)
# ═══════════════════════════════════════════════════════════════════════
#
# data_joint 구조 (원본 미러링):
#   {joint_root}/How2Sign/{split}/poses/{name}.npy   [T, 135]
#   {joint_root}/CSL-Daily/poses/{name}.npy           [T, 135]
#   {joint_root}/Phoenix_2014T/{split}/{name}.npy     [T, 135]
#   {joint_root}/mean.npy, std.npy                    [135]
#
# 45-joint layout (local index):
#   0:     jaw
#   1-14:  upper body (pelvis, spine1, spine2, spine3, neck, L/R_collar,
#          head, L/R_shoulder, L/R_elbow, L/R_wrist)
#   15-29: left hand (15 joints)
#   30-44: right hand (15 joints)

def load_mean_std_joints(joint_root):
    """Load joint mean/std from data_joint root."""
    mean = np.load(os.path.join(joint_root, 'mean.npy'))
    std  = np.load(os.path.join(joint_root, 'std.npy'))
    return mean, std


def load_h2s_joint(ann, joint_dir):
    """
    joint_dir = {joint_root}/How2Sign/{split}/poses
    """
    name = ann['name']
    path = os.path.join(joint_dir, f'{name}.npy')
    if not os.path.exists(path):
        return None, None, None
    joints = np.load(path)  # [T, 135]
    if joints.shape[0] < 4:
        return None, None, None
    return joints, ann['text'], name


def load_csl_joint(ann, csl_joint_root):
    """
    csl_joint_root = {joint_root}/CSL-Daily
    Loads: {csl_joint_root}/poses/{name}.npy
    """
    name = ann['name']
    path = os.path.join(csl_joint_root, 'poses', f'{name}.npy')
    if not os.path.exists(path):
        return None, None, None
    joints = np.load(path)
    if joints.shape[0] < 4:
        return None, None, None
    return joints, ann['text'], name


def load_phoenix_joint(ann, phoenix_joint_root):
    """
    phoenix_joint_root = {joint_root}/Phoenix_2014T
    name already contains split prefix: e.g. 'train/11August_...'
    Loads: {phoenix_joint_root}/{name}.npy
    """
    name = ann['name']
    path = os.path.join(phoenix_joint_root, f'{name}.npy')
    if not os.path.exists(path):
        return None, None, None
    joints = np.load(path)
    if joints.shape[0] < 4:
        return None, None, None
    return joints, ann['text'], name