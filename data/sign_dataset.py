"""
Sign language datasets for SALAD.
- SignMotionDataset:       VAE training  — __getitem__ returns motion [W, 133]
- SignText2MotionDataset:  Denoiser training — __getitem__ returns (text, motion, m_length)

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
    BAD_H2S_IDS, get_encoder_cache_name,
)


# ─── Shared annotation builder ───────────────────────────────

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


def _load_one(ann, csl_root, phoenix_root):
    src = ann['src']
    if src == 'how2sign':
        return load_h2s_sample(ann, ann['_poses_dir'])
    elif src == 'csl':
        return load_csl_sample(ann, csl_root)
    elif src == 'phoenix':
        return load_phoenix_sample(ann, phoenix_root)
    return None, None, None


def _adjust_length(poses, min_len, max_len, unit_length):
    """SOKE-style: upsample short / downsample long / center-crop middle.
    Always returns length that is a multiple of unit_length."""
    assert min_len % unit_length == 0 and max_len % unit_length == 0, \
        f"min_len({min_len}) and max_len({max_len}) must be multiples of unit_length({unit_length})"
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
# SALAD train_vae.py:  batch = motion tensor  →  trainer.train_forward(batch)

class SignMotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split='train'):
        self.opt  = opt
        self.mean = mean   # (133,) numpy
        self.std  = std
        self.window_size    = opt.window_size
        self.unit_length    = getattr(opt, 'unit_length', 4)
        self.min_motion_len = getattr(opt, 'min_motion_length', 40)
        self.max_motion_len = getattr(opt, 'max_motion_length', 400)

        self.csl_root     = getattr(opt, 'csl_root', None)
        self.phoenix_root = getattr(opt, 'phoenix_root', None)

        self.all_data = _build_annotations(
            split, opt.sign_dataset, opt.data_root,
            self.csl_root, self.phoenix_root)

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        ann = self.all_data[idx]
        poses, text, name = _load_one(ann, self.csl_root, self.phoenix_root)
        if poses is None:
            return self.__getitem__(random.randint(0, len(self) - 1))

        poses = (poses - self.mean) / (self.std + 1e-10)
        poses = _adjust_length(poses, self.min_motion_len, self.max_motion_len, self.unit_length)

        T = poses.shape[0]
        if T >= self.window_size:
            start = random.randint(0, T - self.window_size)
            poses = poses[start:start + self.window_size]
        else:
            idx = np.linspace(0, T - 1, num=self.window_size, dtype=int)
            poses = poses[idx]

        return torch.from_numpy(poses).float()   # [W, 133]


# ─── Denoiser Dataset ────────────────────────────────────────
# SALAD train_denoiser.py:  text, motion, m_lens = batch

def _resolve_text_cache_path(ann, split, data_root, csl_root, phoenix_root, encoder_name):
    """annotation → text embedding cache .pt 경로"""
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

        self.data_root    = getattr(opt, 'data_root', None)
        self.csl_root     = getattr(opt, 'csl_root', None)
        self.phoenix_root = getattr(opt, 'phoenix_root', None)

        self.all_data = _build_annotations(
            split, opt.sign_dataset, self.data_root,
            self.csl_root, self.phoenix_root)

        # ── text embedding cache ──
        self.use_text_cache = getattr(opt, 'use_text_cache', False)
        self.encoder_cache_name = None
        self.empty_text_emb = None
        self.cfg_drop_prob = getattr(opt, 'cond_drop_prob', 0.0) if split == 'train' else 0.0

        if self.use_text_cache:
            self.encoder_cache_name = get_encoder_cache_name(
                getattr(opt, 'text_encoder', 'clip'),
                getattr(opt, 'clip_version', 'ViT-B/32'),
                getattr(opt, 'xlmr_version', 'xlm-roberta-base'),
            )
            # Load empty embedding for CFG dropout
            for root in [self.data_root, self.csl_root, self.phoenix_root]:
                if root is None:
                    continue
                empty_path = os.path.join(root, 'text_emb', self.encoder_cache_name, '__empty__.pt')
                if os.path.exists(empty_path):
                    e = torch.load(empty_path, map_location='cpu')
                    self.empty_text_emb = (e['word_emb'], e['attn_mask'], e['token_pos'])
                    print(f"  [TextCache] loaded __empty__.pt from {root}")
                    break
            if self.empty_text_emb is None:
                print("  [TextCache] WARNING: __empty__.pt not found, CFG dropout will use zeros")

            # Verify cache exists for first sample
            sample_ann = self.all_data[0]
            sample_path = _resolve_text_cache_path(
                sample_ann, split, self.data_root, self.csl_root, self.phoenix_root, self.encoder_cache_name)
            if sample_path and os.path.exists(sample_path):
                print(f"  [TextCache] ON — encoder={self.encoder_cache_name}, split={split}")
            else:
                print(f"  [TextCache] WARNING: cache not found at {sample_path}")
                print(f"  [TextCache] Run cache_text_embeddings.py first! Falling back to text strings.")
                self.use_text_cache = False

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        ann = self.all_data[idx]
        poses, text, name = _load_one(ann, self.csl_root, self.phoenix_root)
        if poses is None:
            return self.__getitem__(random.randint(0, len(self) - 1))

        poses = (poses - self.mean) / (self.std + 1e-10)
        poses = _adjust_length(poses, self.min_motion_len, self.max_motion_len, self.unit_length)
        m_length = poses.shape[0]

        # pad to max for batching
        if m_length < self.max_motion_len:
            pad = np.zeros((self.max_motion_len - m_length, 133))
            poses = np.concatenate([poses, pad], axis=0)

        # ── text output: cached tensor tuple or raw string ──
        if self.use_text_cache:
            # CFG dropout
            if self.cfg_drop_prob > 0 and random.random() < self.cfg_drop_prob:
                if self.empty_text_emb is not None:
                    text_out = self.empty_text_emb
                else:
                    text_out = (torch.zeros(1), torch.zeros(1), torch.tensor(0))
            else:
                cache_path = _resolve_text_cache_path(
                    ann, self.split, self.data_root, self.csl_root, self.phoenix_root,
                    self.encoder_cache_name)
                cache = torch.load(cache_path, map_location='cpu')
                text_out = (cache['word_emb'], cache['attn_mask'], cache['token_pos'])
        else:
            text_out = text

        return text_out, torch.from_numpy(poses).float(), m_length