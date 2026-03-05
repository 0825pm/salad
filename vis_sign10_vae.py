#!/usr/bin/env python3
"""
vis_sign10_vae.py — Sign10 VAE (210D) GT vs Reconstruction 시각화

SignVAE 체크포인트 로딩 → GT vs Recon skeleton 비교 영상 + MPJPE 메트릭

Pipeline:
  GT:    133D → norm(133D) → FK → 45-joint skeleton
  Recon: sign10 120D → norm(120D) → 210D(vel)
         → SignVAE → 210D recon
         → strip vel → denorm(120D) → raw120 → pad133D
         → norm(133D) → FK → 45-joint skeleton

Usage:
    cd ~/Projects/research/salad

    python vis_sign10_vae.py \
        --checkpoint checkpoints/sign/sign_vae_210d_v1/model/latest.tar \
        --data_root ./dataset/How2Sign \
        --mean_path ./dataset/How2Sign/mean.pt \
        --std_path ./dataset/How2Sign/std.pt \
        --smplx_path deps/smpl_models/ \
        --num_samples 5 --viewport 0.5

    # CPU only
    python vis_sign10_vae.py --checkpoint ... --device cpu
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.sign_paramUtil import (
    reorder_from_sign10,
    rotation_to_sign10_vel, sign10_vel_to_rotation,
    get_sign_config,
)
from utils.sign10_config import (
    load_mean_std_sign10, pad_to_133,
)


# =============================================================================
# Skeleton config — 45-joint layout (verify_sign10_reorder.py 동일)
# =============================================================================

SMPLX_SELECT_IDX = [22] + [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] \
                   + list(range(25, 40)) + list(range(40, 55))  # 45 joints

ROOT_IDX = 4   # spine3 in local 45-joint order

BODY_IDX  = list(range(0, 15))
LHAND_IDX = list(range(15, 30))
RHAND_IDX = list(range(30, 45))

BODY_CONNECTIONS = [
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 8),
    (4, 6), (6, 9), (9, 11), (11, 13),
    (4, 7), (7, 10), (10, 12), (12, 14),
    (8, 0),
]

def _hand_connections(wrist_idx, hand_offset):
    conns = []
    for finger in range(5):
        base = hand_offset + finger * 3
        conns.append((wrist_idx, base))
        conns.append((base, base + 1))
        conns.append((base + 1, base + 2))
    return conns

LHAND_CONNECTIONS = _hand_connections(13, 15)
RHAND_CONNECTIONS = _hand_connections(14, 30)
ALL_CONNECTIONS = BODY_CONNECTIONS + LHAND_CONNECTIONS + RHAND_CONNECTIONS


def normalize_to_root(joints, root_idx=ROOT_IDX):
    if len(joints.shape) == 3:
        return joints - joints[:, root_idx:root_idx+1, :]
    return joints - joints[root_idx:root_idx+1, :]


# =============================================================================
# MPJPE 계산
# =============================================================================

def compute_diff_stats(gt_joints, recon_joints):
    """MPJPE: body/hand/jaw 분리 계산 (45-joint local index)."""
    T = min(gt_joints.shape[0], recon_joints.shape[0])
    J = min(gt_joints.shape[1], recon_joints.shape[1])
    gt, rc = gt_joints[:T, :J], recon_joints[:T, :J]
    per_joint = np.sqrt(np.sum((gt - rc) ** 2, axis=-1))

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


# =============================================================================
# Side-by-side video 생성
# =============================================================================

def save_comparison_video(gt_joints, recon_joints, save_path, title='',
                          fps=25, viewport=0.5, diff_stats=None):
    """GT(왼쪽) vs Recon(오른쪽) 2D skeleton video. joints: [T, 45, 3]."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    T = min(gt_joints.shape[0], recon_joints.shape[0])
    gt = normalize_to_root(gt_joints[:T].copy())
    rc = normalize_to_root(recon_joints[:T].copy())

    if viewport > 0:
        x_lim = (-viewport, viewport)
        y_lim = (-viewport, viewport)
    else:
        all_data = np.concatenate([gt, rc], axis=0)
        all_x = all_data[:, :, 0].flatten()
        all_y = all_data[:, :, 1].flatten()
        max_range = max(all_x.max() - all_x.min(),
                        all_y.max() - all_y.min(), 0.1) * 1.2
        x_mid = (all_x.max() + all_x.min()) / 2
        y_mid = (all_y.max() + all_y.min()) / 2
        x_lim = (x_mid - max_range/2, x_mid + max_range/2)
        y_lim = (y_mid - max_range/2, y_mid + max_range/2)

    fig, (ax_gt, ax_rc) = plt.subplots(1, 2, figsize=(12, 6))
    main_title = title
    if diff_stats:
        main_title += f"\nbody MPJPE={diff_stats['body_mpjpe']:.4f}  "
        main_title += f"hand MPJPE={diff_stats['hand_mpjpe']:.4f}  "
        main_title += f"jaw MPJPE={diff_stats['jaw_mpjpe']:.4f}"
    fig.suptitle(main_title, fontsize=10)

    labels = [('GT (133D original)', 'blue'),
              ('VAE Recon (210D → SignVAE)', 'darkgreen')]
    for ax, (label, color) in zip([ax_gt, ax_rc], labels):
        ax.set_title(label, fontsize=12, fontweight='bold', color=color)
        ax.set_xlim(x_lim); ax.set_ylim(y_lim)
        ax.set_aspect('equal'); ax.axis('off')

    colors = {'body': '#2196F3', 'lhand': '#F44336', 'rhand': '#4CAF50'}

    elements = []
    for ax, data in [(ax_gt, gt), (ax_rc, rc)]:
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
# Checkpoint / Model 로딩
# =============================================================================

def find_checkpoint(ckpt_arg, model_dir=None):
    """Find checkpoint: 'latest', specific path, or best_fid."""
    if ckpt_arg == 'latest' and model_dir:
        path = os.path.join(model_dir, 'latest.tar')
        if os.path.exists(path):
            return path
        tars = sorted(glob.glob(os.path.join(model_dir, '*.tar')))
        if tars:
            return tars[-1]
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")
    elif os.path.isfile(ckpt_arg):
        return ckpt_arg
    elif os.path.isdir(ckpt_arg):
        # checkpoint dir 직접 전달된 경우
        for name in ['net_best_fid.tar', 'latest.tar']:
            p = os.path.join(ckpt_arg, name)
            if os.path.exists(p):
                return p
        tars = sorted(glob.glob(os.path.join(ckpt_arg, '*.tar')))
        if tars:
            return tars[-1]
        raise FileNotFoundError(f"No checkpoint in dir: {ckpt_arg}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_arg}")


def build_sign_vae(opt_overrides=None):
    """SignVAE 모델 빌드 (sign10_vel 전용).

    기존 vis_sign_recon.py의 build_vae_from_opt와 유사하지만
    models/sign/model.py의 SignVAE를 사용.
    """
    from argparse import Namespace

    skeleton_mode = (opt_overrides or {}).get('skeleton_mode', 'sign10_vel')
    _, _, _, num_parts = get_sign_config(skeleton_mode)
    splits, _, _, _ = get_sign_config(skeleton_mode)
    pose_dim = sum(splits)

    opt = Namespace(
        dataset_name='sign',
        skeleton_mode=skeleton_mode,
        joints_num=num_parts,
        pose_dim=pose_dim,
        latent_dim=32,
        kernel_size=3,
        n_layers=2,
        n_extra_layers=1,
        norm='none',
        activation='gelu',
        dropout=0.1,
    )
    if opt_overrides:
        for k, v in opt_overrides.items():
            setattr(opt, k, v)

    from models.sign.model import SignVAE
    return SignVAE(opt), opt


def load_133_mean_std(mean_path, std_path):
    """179D/133D mean/std.pt → 133D numpy arrays."""
    mean_full = torch.load(mean_path, map_location='cpu').float()
    std_full  = torch.load(std_path,  map_location='cpu').float()
    if mean_full.shape[0] == 179:
        keep = list(range(36, 159)) + list(range(169, 179))
        mean_full, std_full = mean_full[keep], std_full[keep]
    assert mean_full.shape[0] == 133, f"Expected 133D mean, got {mean_full.shape[0]}"
    return mean_full.numpy(), std_full.numpy()


# =============================================================================
# Main Visualization
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Sign10 VAE (210D): GT vs Reconstruction Visualization')

    # ── 필수 ──
    parser.add_argument('--checkpoint', required=True,
                        help='path to .tar checkpoint, dir, or "latest"')
    parser.add_argument('--model_dir', default=None,
                        help='model dir (used with --checkpoint latest)')
    parser.add_argument('--data_root', required=True, help='How2Sign data root')
    parser.add_argument('--mean_path', required=True, help='path to mean.pt (179D)')
    parser.add_argument('--std_path',  required=True, help='path to std.pt (179D)')
    parser.add_argument('--smplx_path', default='deps/smpl_models/')

    # ── 데이터셋 ──
    parser.add_argument('--csl_root', default=None)
    parser.add_argument('--phoenix_root', default=None)
    parser.add_argument('--sign_dataset', default='how2sign',
                        choices=['how2sign', 'csl', 'phoenix',
                                 'how2sign_csl', 'how2sign_csl_phoenix'])
    parser.add_argument('--split', default='val')
    parser.add_argument('--num_samples', type=int, default=5)

    # ── VAE 아키텍처 (학습 설정과 일치해야 함) ──
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_extra_layers', type=int, default=1)

    # ── 출력 ──
    parser.add_argument('--output', default='vis_sign10_output')
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--viewport', type=float, default=0.5, help='0=auto')
    parser.add_argument('--device', default='cuda:0')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    ckpt_path = find_checkpoint(args.checkpoint, args.model_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = os.path.join(args.output, f'recon_{timestamp}')
    os.makedirs(output_root, exist_ok=True)

    print("=" * 60)
    print("Sign10 VAE (210D): GT vs Reconstruction")
    print(f"  checkpoint:  {ckpt_path}")
    print(f"  smplx model: {args.smplx_path}")
    print(f"  output:      {output_root}")
    print(f"  device:      {device}")
    print("=" * 60)

    # =====================================================================
    # 1. Mean / Std 로딩
    # =====================================================================
    print("\n[1/4] Loading mean/std...")

    # sign10 120D mean/std (VAE 입출력용)
    mean_sign10, std_sign10 = load_mean_std_sign10(args.mean_path, args.std_path)
    print(f"  sign10 120D: mean={mean_sign10.shape}, std={std_sign10.shape}")

    # 133D mean/std (FK용)
    mean_133, std_133 = load_133_mean_std(args.mean_path, args.std_path)
    print(f"  133D:        mean={mean_133.shape}, std={std_133.shape}")

    # =====================================================================
    # 2. Dataset 로딩
    # =====================================================================
    print("\n[2/4] Loading dataset...")
    from data.sign_dataset import _build_annotations, _load_one

    all_data = _build_annotations(
        split=args.split,
        dataset_name=args.sign_dataset,
        data_root=args.data_root,
        csl_root=args.csl_root,
        phoenix_root=args.phoenix_root,
    )
    print(f"  [{args.split}] {len(all_data)} samples")

    # =====================================================================
    # 3. SignVAE 로딩
    # =====================================================================
    print("\n[3/4] Loading SignVAE...")
    vae, opt = build_sign_vae({
        'skeleton_mode': 'sign10_vel',
        'latent_dim': args.latent_dim,
        'n_layers': args.n_layers,
        'n_extra_layers': args.n_extra_layers,
    })

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('vae', ckpt.get('state_dict', ckpt))
    missing, unexpected = vae.load_state_dict(state, strict=False)
    vae.eval().to(device)

    epoch = ckpt.get('epoch', '?')
    total_iter = ckpt.get('total_iter', '?')
    total_params = sum(p.numel() for p in vae.parameters())
    print(f"  Epoch: {epoch}, Iter: {total_iter}, Params: {total_params/1e6:.2f}M")
    if missing:
        print(f"  Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    # =====================================================================
    # 4. GT vs Recon 시각화
    # =====================================================================
    print(f"\n[4/4] Generating GT vs Recon videos...")
    from utils.feats2joints import aa133_to_joints_np

    # Group by source
    src_indices = {}
    for i, item in enumerate(all_data):
        src_indices.setdefault(item.get('src', 'how2sign'), []).append(i)

    DS_LABELS = {'how2sign': 'H2S', 'csl': 'CSL', 'phoenix': 'Phoenix'}
    all_stats = []
    total_count = 0

    for src_key, indices in src_indices.items():
        if not indices:
            continue

        ds_label = DS_LABELS.get(src_key, src_key)
        n = min(args.num_samples, len(indices))
        sel = [indices[int(i)] for i in np.linspace(0, len(indices)-1, n)]

        out_dir = os.path.join(output_root, ds_label)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n  [{ds_label}] {len(indices)} total, visualizing {n}")

        for idx_s, dataset_idx in enumerate(sel):
            ann = all_data[dataset_idx]
            name = ann.get('name', f'sample_{dataset_idx}')

            # ── Load 133D (GT용) ──
            poses_133, text, name_loaded = _load_one(
                ann, args.csl_root, args.phoenix_root, skeleton_mode="7part")
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

            # ── Load sign10 120D (VAE 입력용) ──
            poses_sign10, _, _ = _load_one(
                ann, args.csl_root, args.phoenix_root, skeleton_mode="sign10")
            if poses_sign10 is None:
                print(f"    [{idx_s+1}/{n}] {name[:40]} — sign10 load failed, skip.")
                continue
            poses_sign10 = poses_sign10[:T_crop]  # [T, 120]

            # ============================================================
            # Path A: GT → 133D → norm(133D) → FK
            # ============================================================
            poses_133_norm = (poses_133 - mean_133) / (std_133 + 1e-10)

            try:
                gt_joints_full = aa133_to_joints_np(
                    poses_133_norm, mean_133, std_133, args.smplx_path, device=str(device))
                gt_joints = gt_joints_full[:, SMPLX_SELECT_IDX]  # [T, 45, 3]
            except Exception as e:
                print(f"    [{idx_s+1}/{n}] {name[:40]} — FK(GT) failed: {e}")
                continue

            # ============================================================
            # Path B: sign10 → norm(120D) → 210D(vel) → SignVAE
            #          → 210D recon → strip vel → denorm → raw120
            #          → pad133D → norm(133D) → FK
            # ============================================================

            # sign10 → norm → 210D
            sign10_norm = (poses_sign10 - mean_sign10) / (std_sign10 + 1e-10)
            sign10_vel_210 = rotation_to_sign10_vel(sign10_norm)  # [T, 210]

            # VAE forward pass
            with torch.no_grad():
                motion_input = torch.from_numpy(sign10_vel_210).float().unsqueeze(0)  # [1, T, 210]
                motion_input = motion_input.to(device)
                recon_210, loss_dict = vae(motion_input)
                recon_210 = recon_210[0].cpu().numpy()  # [T, 210]

            # 210D → strip vel → 120D norm
            recon_sign10_norm = sign10_vel_to_rotation(recon_210)  # [T, 120]

            # denorm → raw120 → pad133
            recon_sign10_denorm = recon_sign10_norm * (std_sign10 + 1e-10) + mean_sign10
            recon_raw120 = reorder_from_sign10(recon_sign10_denorm)  # [T, 120]
            recon_133 = pad_to_133(recon_raw120)                     # [T, 133]

            # norm(133D) → FK
            recon_133_norm = (recon_133 - mean_133) / (std_133 + 1e-10)

            try:
                rc_joints_full = aa133_to_joints_np(
                    recon_133_norm, mean_133, std_133, args.smplx_path, device=str(device))
                rc_joints = rc_joints_full[:, SMPLX_SELECT_IDX]  # [T, 45, 3]
            except Exception as e:
                print(f"    [{idx_s+1}/{n}] {name[:40]} — FK(Recon) failed: {e}")
                continue

            # ── Feature-level diff (jaw/expr은 210D에서 제거되므로 차이 예상) ──
            feat_diff_jaw  = np.abs(poses_133[:, 120:123]).mean()
            feat_diff_expr = np.abs(poses_133[:, 123:133]).mean()

            # ── MPJPE ──
            stats = compute_diff_stats(gt_joints, rc_joints)
            stats['name'] = name
            stats['src'] = src_key
            stats['T'] = T_crop
            stats['kl'] = loss_dict.get('loss_kl', torch.tensor(0)).item()
            stats['feat_jaw_mae'] = feat_diff_jaw
            stats['feat_expr_mae'] = feat_diff_expr
            all_stats.append(stats)

            J = gt_joints.shape[1]
            print(f"    [{idx_s+1}/{n}] {name[:40]} (T={T_crop}, J={J})")
            print(f"      body MPJPE={stats['body_mpjpe']:.6f}  "
                  f"hand MPJPE={stats['hand_mpjpe']:.6f}  "
                  f"jaw MPJPE={stats['jaw_mpjpe']:.6f}  "
                  f"all MPJPE={stats['all_mpjpe']:.6f}")
            print(f"      KL={stats['kl']:.4f}  "
                  f"jaw feat MAE={feat_diff_jaw:.6f}  "
                  f"expr feat MAE={feat_diff_expr:.6f}")

            if np.isnan(gt_joints).any() or np.isnan(rc_joints).any():
                print(f"      ⚠️ NaN in joints, skip video.")
                continue

            # ── Video 저장 ──
            safe_name = str(name)[:40].replace('/', '_').replace('\\', '_')
            video_path = os.path.join(out_dir, f'{idx_s:03d}_{safe_name}.mp4')
            title = f'{name[:40]} [{ds_label}] (T={T_crop})'
            save_comparison_video(gt_joints, rc_joints, video_path,
                                  title, args.fps, args.viewport, stats)
            total_count += 1

    # =====================================================================
    # Summary
    # =====================================================================
    print(f"\n  Generated {total_count} videos in {output_root}")

    if all_stats:
        body = np.mean([s['body_mpjpe'] for s in all_stats])
        hand = np.mean([s['hand_mpjpe'] for s in all_stats])
        jaw  = np.mean([s['jaw_mpjpe']  for s in all_stats])
        all_ = np.mean([s['all_mpjpe']  for s in all_stats])
        kl   = np.mean([s['kl']         for s in all_stats])

        print(f"\n  === Aggregate Results (Epoch {epoch}) ===")
        print(f"  body MPJPE:  {body:.6f}")
        print(f"  hand MPJPE:  {hand:.6f}")
        print(f"  jaw  MPJPE:  {jaw:.6f}  (expected — jaw zeroed in 210D)")
        print(f"  all  MPJPE:  {all_:.6f}")
        print(f"  mean KL:     {kl:.4f}")

        # per-source
        for sk in ['how2sign', 'csl', 'phoenix']:
            ms = [s for s in all_stats if s['src'] == sk]
            if ms:
                b = np.mean([s['body_mpjpe'] for s in ms])
                h = np.mean([s['hand_mpjpe'] for s in ms])
                j = np.mean([s['jaw_mpjpe']  for s in ms])
                print(f"  [{DS_LABELS[sk]}] body={b:.6f}  hand={h:.6f}  "
                      f"jaw={j:.6f}  (n={len(ms)})")

        # CSV 저장
        csv_path = os.path.join(output_root, 'recon_stats.csv')
        with open(csv_path, 'w') as f:
            f.write('name,src,T,body_mpjpe,hand_mpjpe,jaw_mpjpe,'
                    'all_mpjpe,kl,feat_jaw_mae,feat_expr_mae\n')
            for s in all_stats:
                f.write(f"{s['name']},{s['src']},{s['T']},"
                        f"{s['body_mpjpe']:.6f},{s['hand_mpjpe']:.6f},"
                        f"{s['jaw_mpjpe']:.6f},{s['all_mpjpe']:.6f},"
                        f"{s['kl']:.6f},"
                        f"{s['feat_jaw_mae']:.6f},{s['feat_expr_mae']:.6f}\n")
        print(f"  Stats CSV: {csv_path}")

    print(f"\n  Output: {output_root}")


if __name__ == '__main__':
    main()
