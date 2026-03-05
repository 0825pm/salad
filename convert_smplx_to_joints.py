"""
convert_smplx_to_joints.py — SMPL-X 179D PKL → 3D joint coordinates (.npy)

원본 구조 미러링:
  data/How2Sign/{split}/poses/{name}/       →  data_joint/How2Sign/{split}/poses/{name}.npy
  data/CSL-Daily/poses/{name}/              →  data_joint/CSL-Daily/poses/{name}.npy
  data/Phoenix_2014T/{split}/{name}/        →  data_joint/Phoenix_2014T/{split}/{name}.npy

Joint: 45 joints × 3 = 135D (7-part grouped)
  [0:12]   torso     — pelvis, spine1, spine2, spine3         (4j × 3D)
  [12:24]  L_arm     — L_collar, L_shoulder, L_elbow, L_wrist (4j × 3D)
  [24:36]  R_arm     — R_collar, R_shoulder, R_elbow, R_wrist (4j × 3D)
  [36:81]  lhand     — 15 joints (index, middle, pinky, ring, thumb × 3)
  [81:126] rhand     — 15 joints
  [126:132] head_neck — neck, head                            (2j × 3D)
  [132:135] jaw       — jaw                                   (1j × 3D)

Usage:
    python convert_smplx_to_joints.py \
        --smplx_path deps/smpl_models/ \
        --data_root /home/user/Projects/research/SOKE/data/How2Sign \
        --csl_root  /home/user/Projects/research/SOKE/data/CSL-Daily \
        --phoenix_root /home/user/Projects/research/SOKE/data/Phoenix_2014T \
        --output_root /home/user/Projects/research/SOKE/data_joint \
        --sign_dataset how2sign_csl_phoenix \
        --split all
"""

import os
import argparse
import pickle
import gzip
import math
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy

import smplx

# ─── Constants ─────────────────────────────────────────────────────────
SMPLX_KEYS = [
    'smplx_root_pose', 'smplx_body_pose', 'smplx_lhand_pose',
    'smplx_rhand_pose', 'smplx_jaw_pose', 'smplx_shape', 'smplx_expr',
]

# Grouped by 7-part for VAE: torso | L_arm | R_arm | lhand | rhand | head_neck | jaw
TORSO_IDX     = [0, 3, 6, 9]       # pelvis, spine1, spine2, spine3 (4j)
L_ARM_IDX     = [13, 16, 18, 20]   # L_collar, L_shoulder, L_elbow, L_wrist (4j)
R_ARM_IDX     = [14, 17, 19, 21]   # R_collar, R_shoulder, R_elbow, R_wrist (4j)
LHAND_IDX     = list(range(25, 40)) # 15 left hand joints (index,middle,pinky,ring,thumb)
RHAND_IDX     = list(range(40, 55)) # 15 right hand joints
HEAD_NECK_IDX = [12, 15]           # neck, head (2j)
JAW_IDX       = [22]               # jaw (1j)
SELECT_IDX    = TORSO_IDX + L_ARM_IDX + R_ARM_IDX + LHAND_IDX + RHAND_IDX + HEAD_NECK_IDX + JAW_IDX  # 45

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

# Post-processing: body vs hand joint indices (after reorder)
#   body: torso(0-3) + L_arm(4-7) + R_arm(8-11) + head_neck(42-43) + jaw(44) = 15 joints
#   hand: lhand(12-26) + rhand(27-41) = 30 joints
_BODY_JOINT_IDX = list(range(0, 12)) + list(range(42, 45))  # 15 joints
_HAND_JOINT_IDX = list(range(12, 42))                        # 30 joints


# ─── Helpers ───────────────────────────────────────────────────────────

def _subsample(lst, count):
    ss = float(len(lst)) / count
    return [lst[int(math.floor(i * ss))] for i in range(count)]


def _load_frames_to_179(frame_paths):
    clip = np.zeros([len(frame_paths), 179])
    for i, fp in enumerate(frame_paths):
        with open(fp, 'rb') as f:
            p = pickle.load(f)
        clip[i] = np.concatenate([p[k] for k in SMPLX_KEYS], 0)
    return clip


def poses179_to_joints(poses_179, body_model, device, chunk_size=512):
    """179D → SMPL-X FK → selected 45 joints [T, 45, 3]"""
    T = poses_179.shape[0]
    pt = torch.from_numpy(poses_179).float().to(device)
    zero_eye = torch.zeros(T, 3, device=device)

    all_j = []
    for s in range(0, T, chunk_size):
        e = min(s + chunk_size, T)
        with torch.no_grad():
            out = body_model(
                global_orient=pt[s:e, 0:3],
                body_pose=pt[s:e, 3:66],
                left_hand_pose=pt[s:e, 66:111],
                right_hand_pose=pt[s:e, 111:156],
                jaw_pose=pt[s:e, 156:159],
                expression=pt[s:e, 169:179],
                betas=pt[s:e, 159:169],
                leye_pose=zero_eye[s:e],
                reye_pose=zero_eye[s:e],
            )
        all_j.append(out.joints[:, SELECT_IDX, :])
    return torch.cat(all_j, 0).cpu().numpy()  # [T, 45, 3]


# ─── Post-processing: spike removal + smoothing ───────────────────────

def _remove_spikes(joints, vel_thresh=3.0):
    """
    Joint별 프레임간 velocity 기반 스파이크 탐지 → linear interpolation.
    joints: [T, J, 3]
    vel_thresh: median velocity의 몇 배를 outlier로 볼지
    """
    T, J, C = joints.shape
    cleaned = joints.copy()

    for j in range(J):
        for c in range(C):
            signal = cleaned[:, j, c]
            vel = np.abs(np.diff(signal))
            if vel.max() < 1e-8:
                continue
            median_vel = np.median(vel)
            if median_vel < 1e-8:
                median_vel = np.mean(vel) + 1e-8

            # 양방향 spike: frame i에서 갑자기 튀었다가 i+1에서 돌아오면 i가 spike
            spike_mask = np.zeros(T, dtype=bool)
            for i in range(1, T - 1):
                if vel[i-1] > vel_thresh * median_vel and vel[i] > vel_thresh * median_vel:
                    spike_mask[i] = True
            # 첫/끝 프레임
            if T > 2 and vel[0] > vel_thresh * median_vel:
                spike_mask[0] = True
            if T > 2 and vel[-1] > vel_thresh * median_vel:
                spike_mask[-1] = True

            # spike 프레임을 양쪽 정상 프레임으로 linear interpolation
            good_idx = np.where(~spike_mask)[0]
            if len(good_idx) < 2 or len(good_idx) == T:
                continue
            bad_idx = np.where(spike_mask)[0]
            signal[bad_idx] = np.interp(bad_idx, good_idx, signal[good_idx])

    return cleaned


def _smooth_savgol(joints, window_length=7, polyorder=3):
    """
    Savitzky-Golay filter. 형태 보존하면서 jitter 제거.
    joints: [T, J, 3]
    """
    from scipy.signal import savgol_filter

    T = joints.shape[0]
    # window_length는 홀수여야 하고 T보다 작아야 함
    wl = min(window_length, T)
    if wl % 2 == 0:
        wl -= 1
    if wl < polyorder + 2:
        return joints  # 너무 짧으면 smoothing 안 함

    smoothed = joints.copy()
    # [T, J*3] flatten해서 한번에 처리
    flat = smoothed.reshape(T, -1)
    for d in range(flat.shape[1]):
        flat[:, d] = savgol_filter(flat[:, d], wl, polyorder)
    return flat.reshape(joints.shape)


def postprocess_joints(joints, smooth=True, vel_thresh=3.0,
                       savgol_window=7, savgol_poly=3):
    """스파이크 제거 → Savitzky-Golay smoothing. joints: [T, 45, 3]
    body(torso+arms+head_neck+jaw): spike 제거 + savgol smoothing
    hand(lhand+rhand): spike만 관대하게 제거, savgol 미적용 (빠른 손동작 보존)

    7-part grouped layout:
      body = [0:12] torso+arms + [42:45] head_neck+jaw  (15 joints)
      hand = [12:42] lhand+rhand                        (30 joints)
    """
    body = joints[:, _BODY_JOINT_IDX, :].copy()   # [T, 15, 3]
    hand = joints[:, _HAND_JOINT_IDX, :].copy()   # [T, 30, 3]

    body = _remove_spikes(body, vel_thresh=vel_thresh)
    hand = _remove_spikes(hand, vel_thresh=vel_thresh * 3)  # 손가락은 훨씬 관대하게

    if smooth:
        body = _smooth_savgol(body, savgol_window, savgol_poly)
        # hand는 savgol 미적용 — 수어 손가락 동작은 고주파 신호가 의미 있음

    # Reassemble in original order
    result = joints.copy()
    result[:, _BODY_JOINT_IDX, :] = body
    result[:, _HAND_JOINT_IDX, :] = hand
    return result


def _convert_and_save(frame_paths, out_path, body_model, device, chunk_size,
                      smooth=True):
    """Load frames → FK → (optional) smooth → flatten → save .npy."""
    poses_179 = _load_frames_to_179(frame_paths)
    joints = poses179_to_joints(poses_179, body_model, device, chunk_size)
    if smooth:
        joints = postprocess_joints(joints)
    flat = joints.reshape(joints.shape[0], -1).astype(np.float32)  # [T, 135]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, flat)
    return flat


# ─── Per-dataset converters ───────────────────────────────────────────

def convert_how2sign(data_root, output_root, split, body_model, device, chunk, smooth=True):
    """
    Read:  {data_root}/{split}/poses/{name}/{name}_{fid}_3D.pkl
    Write: {output_root}/How2Sign/{split}/poses/{name}.npy
    """
    import pandas as pd

    poses_base = os.path.join(data_root, split, 'poses')
    out_base   = os.path.join(output_root, 'How2Sign', split, 'poses')
    os.makedirs(out_base, exist_ok=True)

    csv_path = os.path.join(data_root, split, 're_aligned',
                            f'how2sign_realigned_{split}_preprocessed_fps.csv')
    csv = pd.read_csv(csv_path)
    csv['DURATION'] = csv['END_REALIGNED'] - csv['START_REALIGNED']
    csv = csv[csv['DURATION'] < 30].reset_index(drop=True)

    train_data = []
    ok, fail = 0, 0

    for _, row in tqdm(csv.iterrows(), total=len(csv), desc=f'h2s-{split}'):
        name, fps = row['SENTENCE_NAME'], row['fps']
        if name in BAD_H2S_IDS:
            continue

        poses_dir = os.path.join(poses_base, name)
        if not os.path.isdir(poses_dir):
            fail += 1; continue

        n_frames = len(os.listdir(poses_dir))
        frames = [os.path.join(poses_dir, f'{name}_{fid}_3D.pkl')
                  for fid in range(n_frames)]
        if fps > 24:
            frames = _subsample(frames, int(24 * len(frames) / fps))
        if len(frames) < 4:
            fail += 1; continue

        try:
            flat = _convert_and_save(
                frames, os.path.join(out_base, f'{name}.npy'),
                body_model, device, chunk, smooth)
            if split == 'train':
                train_data.append(flat)
            ok += 1
        except Exception as e:
            print(f'  FAIL {name}: {e}'); fail += 1

    print(f'  h2s {split}: {ok} ok, {fail} fail')
    return train_data


def convert_csl(csl_root, output_root, split, body_model, device, chunk, smooth=True):
    """
    Read:  {csl_root}/poses/{name}/*.pkl
    Write: {output_root}/CSL-Daily/poses/{name}.npy
    """
    out_base = os.path.join(output_root, 'CSL-Daily', 'poses')
    os.makedirs(out_base, exist_ok=True)

    ann_file = 'csl_clean.train' if split == 'train' else f'csl_clean.{split}'
    with gzip.open(os.path.join(csl_root, ann_file), 'rb') as f:
        ann_list = pickle.load(f)

    train_data = []
    ok, fail = 0, 0

    for ann in tqdm(ann_list, desc=f'csl-{split}'):
        name = ann['name']
        poses_dir = os.path.join(csl_root, 'poses', name)
        if not os.path.isdir(poses_dir):
            fail += 1; continue

        frames = sorted([os.path.join(poses_dir, f) for f in os.listdir(poses_dir)])
        if len(frames) < 4:
            fail += 1; continue

        try:
            flat = _convert_and_save(
                frames, os.path.join(out_base, f'{name}.npy'),
                body_model, device, chunk, smooth)
            if split == 'train':
                train_data.append(flat)
            ok += 1
        except Exception as e:
            print(f'  FAIL {name}: {e}'); fail += 1

    print(f'  csl {split}: {ok} ok, {fail} fail')
    return train_data


def convert_phoenix(phoenix_root, output_root, split, body_model, device, chunk, smooth=True):
    """
    Read:  {phoenix_root}/{split}/{name}/*.pkl
    Write: {output_root}/Phoenix_2014T/{split}/{name}.npy
    """
    out_base = os.path.join(output_root, 'Phoenix_2014T')
    os.makedirs(out_base, exist_ok=True)

    # annotation: val→dev
    ann_name = 'phoenix14t.dev' if split == 'dev' else f'phoenix14t.{split}'
    with gzip.open(os.path.join(phoenix_root, ann_name), 'rb') as f:
        ann_list = pickle.load(f)

    train_data = []
    ok, fail = 0, 0

    for ann in tqdm(ann_list, desc=f'phoenix-{split}'):
        name = ann['name']  # e.g. 'train/11August_2010_...'
        poses_dir = os.path.join(phoenix_root, name)
        if not os.path.isdir(poses_dir):
            fail += 1; continue

        frames = sorted([os.path.join(poses_dir, f) for f in os.listdir(poses_dir)])
        if len(frames) < 4:
            fail += 1; continue

        try:
            out_path = os.path.join(out_base, f'{name}.npy')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            flat = _convert_and_save(
                frames, out_path,
                body_model, device, chunk, smooth)
            if split == 'train':
                train_data.append(flat)
            ok += 1
        except Exception as e:
            print(f'  FAIL {name}: {e}'); fail += 1

    print(f'  phoenix {split}: {ok} ok, {fail} fail')
    return train_data


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smplx_path', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True, help='How2Sign root')
    parser.add_argument('--csl_root', type=str, default=None)
    parser.add_argument('--phoenix_root', type=str, default=None)
    parser.add_argument('--output_root', type=str, required=True,
                        help='e.g. /home/user/Projects/research/SOKE/data_joint')
    parser.add_argument('--sign_dataset', type=str, default='how2sign',
                        choices=['how2sign', 'csl', 'phoenix', 'how2sign_csl_phoenix'])
    parser.add_argument('--split', type=str, default='all')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--chunk_size', type=int, default=512)
    parser.add_argument('--smooth', action='store_true', default=True,
                        help='spike removal + Savitzky-Golay (default: on)')
    parser.add_argument('--no-smooth', dest='smooth', action='store_false')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_root, exist_ok=True)

    print(f'Loading SMPL-X from {args.smplx_path} ...')
    print(f'Smoothing: {"ON (spike removal + Savitzky-Golay)" if args.smooth else "OFF"}')
    body_model = smplx.create(
        model_path=args.smplx_path, model_type='smplx',
        gender='neutral', use_pca=False, use_face_contour=False,
        batch_size=1,
    ).to(device).eval()

    train_all = []

    # ── How2Sign: splits = train/val/test ──
    if 'how2sign' in args.sign_dataset:
        h2s_splits = ['train', 'val', 'test'] if args.split == 'all' else [args.split]
        for sp in h2s_splits:
            print(f'\n=== How2Sign {sp} ===')
            t = convert_how2sign(args.data_root, args.output_root, sp,
                                 body_model, device, args.chunk_size, args.smooth)
            train_all.extend(t)

    # ── CSL-Daily: annotation splits = train/val/test, poses는 flat ──
    if 'csl' in args.sign_dataset and args.csl_root:
        csl_splits = ['train', 'val', 'test'] if args.split == 'all' else [args.split]
        for sp in csl_splits:
            print(f'\n=== CSL-Daily {sp} ===')
            t = convert_csl(args.csl_root, args.output_root, sp,
                            body_model, device, args.chunk_size, args.smooth)
            train_all.extend(t)

    # ── Phoenix: splits = train/dev/test ──
    if 'phoenix' in args.sign_dataset and args.phoenix_root:
        ph_splits = ['train', 'dev', 'test'] if args.split == 'all' else [args.split]
        for sp in ph_splits:
            print(f'\n=== Phoenix {sp} ===')
            t = convert_phoenix(args.phoenix_root, args.output_root, sp,
                                body_model, device, args.chunk_size, args.smooth)
            train_all.extend(t)

    # ── mean / std (train만) ──
    if train_all:
        print(f'\nComputing mean/std from {len(train_all)} train samples...')
        cat = np.concatenate(train_all, axis=0)
        mean = cat.mean(axis=0).astype(np.float32)
        std  = cat.std(axis=0).astype(np.float32)
        std[std < 1e-10] = 1e-10

        np.save(os.path.join(args.output_root, 'mean.npy'), mean)
        np.save(os.path.join(args.output_root, 'std.npy'), std)
        print(f'  mean {mean.shape} [{mean.min():.4f}, {mean.max():.4f}]')
        print(f'  std  {std.shape}  [{std.min():.6f}, {std.max():.4f}]')

    print(f'\nDone! → {args.output_root}/')


if __name__ == '__main__':
    main()