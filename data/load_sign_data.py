"""
Sign language data loading functions.
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
    """Text encoder → cache dir name.  e.g. xlm-roberta-base → 'xlmr-base'"""
    if text_encoder == 'xlm-roberta':
        return xlmr_version.replace('xlm-roberta-', 'xlmr-')
    return 'clip-' + clip_version.replace('/', '').replace('-', '').lower()


def _subsample(input_list, count):
    ss = float(len(input_list)) / count
    return [input_list[int(math.floor(i * ss))] for i in range(count)]


def _179_to_133(poses_179):
    """179D SMPL-X -> 133D: drop lower body (36D) + shape (10D), keep expr."""
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