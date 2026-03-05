"""
Sign language datasets for SALAD.
- SignMotionDataset:       VAE training  — __getitem__ returns motion [W, D]
- SignText2MotionDataset:  Denoiser training — __getitem__ returns (text, motion, m_length)

joint_root 옵션:
  None  → 기존 133D axis-angle (프레임별 pkl, SMPL-X FK 필요)
  경로  → 135D joint coordinates (사전 변환된 .npy, FK 불필요)

skeleton_mode 옵션:
  '7part'      → 133D axis-angle, 7 super-joints
  'finger'     → 133D axis-angle, 15 super-joints
  'sign10'     → 120D axis-angle (sign10 order), 10 super-joints
  'sign10_vel' → 210D (120D rotation + 90D velocity), 10 super-joints

Data loading from SOKE (H2S.py / dataset_m_vq_sign.py / dataset_t2m.py).
Return format matches SALAD (data/t2m_dataset.py).
"""
import os
import random
import pickle
import gzip
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from tqdm import tqdm
from copy import deepcopy

from data.load_sign_data import (
    load_h2s_sample, load_csl_sample, load_phoenix_sample,
    load_h2s_joint, load_csl_joint, load_phoenix_joint,
    BAD_H2S_IDS, get_encoder_cache_name,
)


# ─── Shared annotation builder ───────────────────────────────
# annotation은 항상 원본 data root에서 빌드 (CSV, gzip pickle)

def _build_annotations(split, dataset_name, data_root, csl_root=None, phoenix_root=None):
    """Build list of annotation dicts (mirrors SOKE H2S.py __init__)."""
    all_data = []

    # ── How2Sign ──
    if 'how2sign' in dataset_name:
        poses_dir = os.path.join(data_root, split, 'poses')
        csv_path  = os.path.join(data_root, split, 're_aligned',
                                 f'how2sign_realigned_{split}_preprocessed_fps.csv')
        csv = pd.read_csv(csv_path)
        csv['DURATION'] = csv['END_REALIGNED'] - csv['START_REALIGNED']
        csv = csv[csv['DURATION'] < 30].reset_index(drop=True)

        print(f'[{split}] loading how2sign... {len(csv)}')
        for _, row in tqdm(csv.iterrows(), total=len(csv), desc='how2sign'):
            name = row['SENTENCE_NAME']
            if name in BAD_H2S_IDS:
                continue
            all_data.append({
                'name': name, 'fps': row['fps'], 'text': row['SENTENCE'],
                'src': 'how2sign', '_poses_dir': poses_dir,
            })
        print(f'  how2sign: {len(all_data)}')

    # ── CSL-Daily ──
    csl_start = len(all_data)
    if 'csl' in dataset_name and csl_root:
        ann_file = 'csl_clean.train' if split == 'train' else f'csl_clean.{split}'
        with gzip.open(os.path.join(csl_root, ann_file), 'rb') as f:
            ann_list = pickle.load(f)
        print(f'[{split}] loading csl... {len(ann_list)}')
        for ann in tqdm(ann_list, desc='csl'):
            a = deepcopy(ann); a['src'] = 'csl'; all_data.append(a)
        print(f'  csl: {len(all_data) - csl_start}')

    # ── Phoenix-2014T ──
    phoenix_start = len(all_data)
    if 'phoenix' in dataset_name and phoenix_root:
        ann_file = 'phoenix14t.dev' if split == 'val' else f'phoenix14t.{split}'
        with gzip.open(os.path.join(phoenix_root, ann_file), 'rb') as f:
            ann_list = pickle.load(f)
        print(f'[{split}] loading phoenix... {len(ann_list)}')
        for ann in tqdm(ann_list, desc='phoenix'):
            a = deepcopy(ann); a['src'] = 'phoenix'; all_data.append(a)
        print(f'  phoenix: {len(all_data) - phoenix_start}')

    print(f'[{split}] total: {len(all_data)}')
    return all_data


# ─── Loading dispatchers ─────────────────────────────────────

def _load_one(ann, csl_root, phoenix_root, skeleton_mode='7part'):
    """axis-angle 로딩. skeleton_mode에 따라 차원 결정.

    Returns:
        poses: [T, D] where D depends on skeleton_mode:
            '7part'/'finger' → 133D
            'sign10'/'sign10_vel' → 120D (sign10 order, velocity는 나중에)
        text: str
        name: str
    """
    src = ann['src']

    # 항상 133D로 먼저 로딩
    if src == 'how2sign':
        poses, text, name = load_h2s_sample(ann, ann['_poses_dir'])
    elif src == 'csl':
        poses, text, name = load_csl_sample(ann, csl_root)
    elif src == 'phoenix':
        poses, text, name = load_phoenix_sample(ann, phoenix_root)
    else:
        return None, None, None

    if poses is None:
        return None, None, None

    # skeleton_mode에 따라 변환
    if skeleton_mode in ('sign10', 'sign10_vel'):
        from utils.sign10_config import reorder_to_sign10
        poses = poses[:, :120]             # 133D → 120D (drop jaw+expr)
        poses = reorder_to_sign10(poses)   # raw120 → sign10 order
    # 'finger' → 133D 그대로
    # '7part'  → 133D 그대로

    return poses, text, name


def _load_one_joint(ann, joint_root, split):
    """135D joint coordinates 로딩 (data_joint 구조)"""
    src = ann['src']
    if src == 'how2sign':
        joint_dir = os.path.join(joint_root, 'How2Sign', split, 'poses')
        return load_h2s_joint(ann, joint_dir)
    elif src == 'csl':
        csl_joint = os.path.join(joint_root, 'CSL-Daily')
        return load_csl_joint(ann, csl_joint)
    elif src == 'phoenix':
        phoenix_joint = os.path.join(joint_root, 'Phoenix_2014T')
        return load_phoenix_joint(ann, phoenix_joint)
    return None, None, None


# ─── Length adjustment ────────────────────────────────────────

def _adjust_length(poses, min_len, max_len, unit_length):
    """SOKE-style: upsample short / downsample long / center-crop middle."""
    assert min_len % unit_length == 0 and max_len % unit_length == 0
    T = poses.shape[0]
    if T < min_len:
        idx = np.linspace(0, T - 1, num=min_len, dtype=int)
        return poses[idx]
    elif T > max_len:
        idx = np.linspace(0, T - 1, num=max_len, dtype=int)
        return poses[idx]
    else:
        new_len = (T // unit_length) * unit_length
        start = (T - new_len) // 2
        return poses[start:start + new_len]


# ─── VAE Dataset ──────────────────────────────────────────────

class SignMotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split='train'):
        self.opt  = opt
        self.mean = mean
        self.std  = std
        self.split = split
        self.window_size    = opt.window_size
        self.unit_length    = getattr(opt, 'unit_length', 4)
        self.min_motion_len = getattr(opt, 'min_motion_length', 40)
        self.max_motion_len = getattr(opt, 'max_motion_length', 400)

        self.csl_root     = getattr(opt, 'csl_root', None)
        self.phoenix_root = getattr(opt, 'phoenix_root', None)
        self.joint_root   = getattr(opt, 'joint_root', None)

        # ── skeleton mode ──
        self.skeleton_mode = getattr(opt, 'skeleton_mode', '7part')
        self.use_sign10_vel = (self.skeleton_mode == 'sign10_vel')

        # annotation은 항상 원본 data_root 기준
        self.all_data = _build_annotations(
            split, opt.sign_dataset, opt.data_root,
            self.csl_root, self.phoenix_root)

    def inv_transform(self, data):
        """Inverse transform: denormalize, and for sign10_vel: 210D → 120D."""
        if self.use_sign10_vel:
            from utils.sign10_config import sign10_vel_to_rotation
            # 210D → 120D normalized sign10
            if isinstance(data, torch.Tensor):
                data_np = data.cpu().numpy()
                rot = sign10_vel_to_rotation(data_np)
                rot = torch.from_numpy(rot).to(data.device).float()
            else:
                rot = sign10_vel_to_rotation(data)
            return rot * self.std + self.mean  # denormalize 120D
        else:
            return data * self.std + self.mean

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        ann = self.all_data[idx]

        # joint_root 설정 시 135D joint, 아니면 axis-angle
        if self.joint_root:
            poses, text, name = _load_one_joint(ann, self.joint_root, self.split)
        else:
            # sign10_vel → 120D sign10 order로 로딩 (velocity 변환은 정규화 후)
            load_mode = 'sign10' if self.use_sign10_vel else self.skeleton_mode
            poses, text, name = _load_one(
                ann, self.csl_root, self.phoenix_root,
                skeleton_mode=load_mode,
            )

        if poses is None:
            return self.__getitem__(random.randint(0, len(self) - 1))

        # ── normalize (120D sign10 mean/std for sign10_vel, 133D for 7part) ──
        poses = (poses - self.mean) / (self.std + 1e-10)

        # ── sign10_vel: 120D → 210D (must be AFTER normalization) ──
        if self.use_sign10_vel:
            from utils.sign10_config import rotation_to_sign10_vel
            poses = rotation_to_sign10_vel(poses)  # [T, 120] → [T, 210]

        # ── length adjustment ──
        poses = _adjust_length(poses, self.min_motion_len, self.max_motion_len,
                               self.unit_length)

        T = poses.shape[0]
        if T >= self.window_size:
            start = random.randint(0, T - self.window_size)
            poses = poses[start:start + self.window_size]
        else:
            idx = np.linspace(0, T - 1, num=self.window_size, dtype=int)
            poses = poses[idx]

        return torch.from_numpy(poses).float()   # [W, D]  (D=210 for sign10_vel, 133 otherwise)


# ─── Denoiser Dataset ────────────────────────────────────────

def _resolve_text_cache_path(ann, split, data_root, csl_root, phoenix_root, encoder_name):
    src, name = ann['src'], ann['name']
    if src == 'how2sign':
        return os.path.join(data_root, split, 'text_emb', encoder_name, f'{name}.pt')
    elif src == 'csl':
        return os.path.join(csl_root, 'text_emb', encoder_name, f'{name}.pt')
    elif src == 'phoenix':
        return os.path.join(phoenix_root, 'text_emb', encoder_name, f'{name}.pt')
    return None


class SignText2MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split='train'):
        self.opt  = opt
        self.mean = mean
        self.std  = std
        self.split = split
        self.unit_length    = getattr(opt, 'unit_length', 4)
        self.min_motion_len = getattr(opt, 'min_motion_length', 40)
        self.max_motion_len = opt.max_motion_length
        self.pose_dim       = opt.pose_dim  # 133, 120, or 210

        self.data_root    = getattr(opt, 'data_root', None)
        self.csl_root     = getattr(opt, 'csl_root', None)
        self.phoenix_root = getattr(opt, 'phoenix_root', None)
        self.joint_root   = getattr(opt, 'joint_root', None)

        # ── skeleton mode ──
        self.skeleton_mode = getattr(opt, 'skeleton_mode', '7part')
        self.use_sign10_vel = (self.skeleton_mode == 'sign10_vel')

        self.all_data = _build_annotations(
            split, opt.sign_dataset, self.data_root,
            self.csl_root, self.phoenix_root)

        # ── text embedding cache ──
        self.use_text_cache = False
        self.cfg_drop_prob  = getattr(opt, 'cfg_drop_prob', 0.0)
        self.empty_text_emb = None
        self.encoder_cache_name = None

        text_encoder = getattr(opt, 'text_encoder', None)
        if text_encoder:
            self.encoder_cache_name = get_encoder_cache_name(
                text_encoder,
                getattr(opt, 'clip_version', 'ViT-B/32'),
                getattr(opt, 'xlmr_version', 'xlm-roberta-base'))

            sample_ann = self.all_data[0] if self.all_data else None
            if sample_ann:
                sample_path = _resolve_text_cache_path(
                    sample_ann, split, self.data_root,
                    self.csl_root, self.phoenix_root, self.encoder_cache_name)
                if sample_path and os.path.exists(sample_path):
                    self.use_text_cache = True
                    print(f"  Text cache ON: {self.encoder_cache_name}")

                    # empty embedding for CFG dropout
                    empty_dir = os.path.dirname(sample_path)
                    empty_path = os.path.join(empty_dir, '__empty__.pt')
                    if os.path.exists(empty_path):
                        self.empty_text_emb = torch.load(empty_path, map_location='cpu')
                        self.empty_text_emb = (
                            self.empty_text_emb['word_emb'],
                            self.empty_text_emb['attn_mask'],
                            self.empty_text_emb['token_pos'])
                else:
                    print(f"  Text cache not found, falling back to text strings.")

    def inv_transform(self, data):
        """Inverse transform: denormalize, and for sign10_vel: 210D → 120D."""
        if self.use_sign10_vel:
            from utils.sign10_config import sign10_vel_to_rotation
            if isinstance(data, torch.Tensor):
                data_np = data.cpu().numpy()
                rot = sign10_vel_to_rotation(data_np)
                rot = torch.from_numpy(rot).to(data.device).float()
            else:
                rot = sign10_vel_to_rotation(data)
            return rot * self.std + self.mean
        else:
            return data * self.std + self.mean

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        ann = self.all_data[idx]

        if self.joint_root:
            poses, text, name = _load_one_joint(ann, self.joint_root, self.split)
        else:
            # sign10_vel → 120D sign10 order로 로딩 (velocity 변환은 정규화 후)
            load_mode = 'sign10' if self.use_sign10_vel else self.skeleton_mode
            poses, text, name = _load_one(
                ann, self.csl_root, self.phoenix_root,
                skeleton_mode=load_mode,
            )

        if poses is None:
            return self.__getitem__(random.randint(0, len(self) - 1))

        # ── normalize (120D sign10 mean/std for sign10_vel, 133D for 7part) ──
        poses = (poses - self.mean) / (self.std + 1e-10)

        # ── sign10_vel: 120D → 210D (must be AFTER normalization) ──
        if self.use_sign10_vel:
            from utils.sign10_config import rotation_to_sign10_vel
            poses = rotation_to_sign10_vel(poses)  # [T, 120] → [T, 210]

        # ── length adjustment ──
        poses = _adjust_length(poses, self.min_motion_len, self.max_motion_len,
                               self.unit_length)
        m_length = poses.shape[0]

        # pad to max for batching
        if m_length < self.max_motion_len:
            pad = np.zeros((self.max_motion_len - m_length, poses.shape[1]))
            poses = np.concatenate([poses, pad], axis=0)

        # ── text output ──
        if self.use_text_cache:
            if self.cfg_drop_prob > 0 and random.random() < self.cfg_drop_prob:
                if self.empty_text_emb is not None:
                    text_out = self.empty_text_emb
                else:
                    text_out = (torch.zeros(1), torch.zeros(1), torch.tensor(0))
            else:
                cache_path = _resolve_text_cache_path(
                    ann, self.split, self.data_root, self.csl_root,
                    self.phoenix_root, self.encoder_cache_name)
                cache = torch.load(cache_path, map_location='cpu')
                text_out = (cache['word_emb'], cache['attn_mask'], cache['token_pos'])
        else:
            text_out = text

        return text_out, torch.from_numpy(poses).float(), m_length