#!/usr/bin/env python3
"""
verify_223d_pipeline.py — 223D VAE 파이프라인 차원 검증 (실 데이터 불필요)

테스트 항목:
  1. sign_paramUtil: 7part/finger splits for 133D vs 223D
  2. load_sign_data: partwise clamping, append_hand_velocity, strip_hand_velocity
  3. SignMotionEncoder: _group_features → [B,T,J,D]
  4. SignMotionDecoder: _ungroup_features → [B,T,223]
  5. Full VAE forward: [B,T,223] → latent → [B,T,223]
  6. Trainer _sign_loss: part-wise weighted loss
  7. Denormalization: 223D → 133D → FK-ready

Usage:
    cd ~/Projects/research/salad
    python verify_223d_pipeline.py
"""
import sys
import os
import numpy as np
import torch
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  ✓ {name}")
        passed += 1
    else:
        print(f"  ✗ {name}  — {detail}")
        failed += 1


def test_1_sign_param():
    """Test sign_paramUtil splits for 133D and 223D."""
    print("\n[1/7] sign_paramUtil splits")
    from utils.sign_paramUtil import get_sign_config

    # 7part, no vel
    splits, adj, hand_idx, nj = get_sign_config('7part', use_hand_vel=False)
    check("7part 133D splits sum", sum(splits) == 133, f"got {sum(splits)}")
    check("7part 133D num_joints", nj == 7, f"got {nj}")
    check("7part hand_indices", hand_idx == [3, 4], f"got {hand_idx}")

    # 7part, with vel
    splits_v, adj_v, hand_idx_v, nj_v = get_sign_config('7part', use_hand_vel=True)
    check("7part 223D splits sum", sum(splits_v) == 223, f"got {sum(splits_v)}")
    check("7part 223D hand dims", splits_v[3] == 90 and splits_v[4] == 90,
          f"lhand={splits_v[3]}, rhand={splits_v[4]}")
    check("7part adj_list unchanged", adj == adj_v)
    check("7part num_joints unchanged", nj == nj_v)

    # finger, with vel
    splits_f, _, _, nj_f = get_sign_config('finger', use_hand_vel=True)
    check("finger 223D splits sum", sum(splits_f) == 223, f"got {sum(splits_f)}")
    check("finger 223D num_joints", nj_f == 15)


def test_2_clamping():
    """Test partwise std clamping."""
    print("\n[2/7] Partwise std clamping")
    from data.load_sign_data import apply_partwise_clamping, PART_RANGES_133

    # Simulate extreme std (body barely moves)
    std_raw = np.random.uniform(0.0001, 0.001, 133).astype(np.float32)
    std_raw[30:120] = np.random.uniform(0.1, 1.0, 90)  # hands move a lot

    std_clamped = apply_partwise_clamping(std_raw)
    check("Clamped shape", std_clamped.shape == (133,))
    check("Root clamped ≥ 0.1", std_clamped[0:3].min() >= 0.1,
          f"min={std_clamped[0:3].min():.6f}")
    check("Upper_a clamped ≥ 0.1", std_clamped[3:15].min() >= 0.1)
    check("Hands preserved (low threshold)", 
          np.allclose(std_clamped[30:120], std_raw[30:120]),
          "hand std should not change (already > 0.01)")
    check("Jaw clamped ≥ 0.05", std_clamped[120:123].min() >= 0.05)


def test_3_hand_velocity():
    """Test append/strip hand velocity."""
    print("\n[3/7] Hand velocity append/strip")
    from data.load_sign_data import append_hand_velocity, strip_hand_velocity

    T = 64
    poses_133 = np.random.randn(T, 133).astype(np.float32)
    
    poses_223 = append_hand_velocity(poses_133)
    check("223D shape", poses_223.shape == (T, 223), f"got {poses_223.shape}")
    check("Rotation preserved", np.allclose(poses_223[:, :133], poses_133))
    
    # Velocity = diff of normalized rotation
    lhand_vel = poses_223[:, 133:178]
    check("First frame vel = 0", np.allclose(lhand_vel[0], 0.0))
    expected_vel = poses_133[1:, 30:75] - poses_133[:-1, 30:75]
    check("LHand vel correct", np.allclose(lhand_vel[1:], expected_vel, atol=1e-6))
    
    rhand_vel = poses_223[:, 178:223]
    expected_rvel = poses_133[1:, 75:120] - poses_133[:-1, 75:120]
    check("RHand vel correct", np.allclose(rhand_vel[1:], expected_rvel, atol=1e-6))

    stripped = strip_hand_velocity(poses_223)
    check("Strip → 133D", stripped.shape == (T, 133))
    check("Strip preserves rot", np.allclose(stripped, poses_133))


def test_4_encoder_grouping():
    """Test SignMotionEncoder _group_features."""
    print("\n[4/7] SignMotionEncoder grouping (223D → per-node)")
    from models.vae.encdec import SignMotionEncoder

    opt = Namespace(
        dataset_name='sign', skeleton_mode='7part', use_hand_vel=True,
        latent_dim=32, activation='gelu',
    )
    enc = SignMotionEncoder(opt)
    
    check("Encoder splits", enc.splits == [3, 12, 15, 90, 90, 3, 10],
          f"got {enc.splits}")
    
    B, T = 2, 16
    x = torch.randn(B, T, 223)
    
    # Test grouping
    parts = enc._group_features(x)
    check("7 parts", len(parts) == 7)
    check("Root dim", parts[0].shape == (B, T, 3))
    check("LHand dim", parts[3].shape == (B, T, 90), f"got {parts[3].shape}")
    check("RHand dim", parts[4].shape == (B, T, 90))
    
    # Verify lhand grouping: rot[30:75] + vel[133:178]
    expected_lhand = torch.cat([x[..., 30:75], x[..., 133:178]], dim=-1)
    check("LHand = rot+vel grouped", torch.allclose(parts[3], expected_lhand))
    
    # Full forward
    out = enc(x)
    check("Encoder output shape", out.shape == (B, T, 7, 32),
          f"got {out.shape}")


def test_5_decoder_ungrouping():
    """Test SignMotionDecoder _ungroup_features."""
    print("\n[5/7] SignMotionDecoder ungrouping (per-node → 223D)")
    from models.vae.encdec import SignMotionDecoder

    opt = Namespace(
        dataset_name='sign', skeleton_mode='7part', use_hand_vel=True,
        latent_dim=32, activation='gelu',
    )
    dec = SignMotionDecoder(opt)
    
    B, T = 2, 16
    x = torch.randn(B, T, 7, 32)
    out = dec(x)
    check("Decoder output shape", out.shape == (B, T, 223), f"got {out.shape}")
    
    # Verify: first 133D = rotation, last 90D = velocity
    rot_part = out[..., :133]
    vel_part = out[..., 133:223]
    check("Rot part 133D", rot_part.shape[-1] == 133)
    check("Vel part 90D", vel_part.shape[-1] == 90)


def test_6_vae_forward():
    """Test full VAE: [B,T,223] → latent → [B,T,223]."""
    print("\n[6/7] Full VAE forward pass")
    from models.vae.model import VAE

    opt = Namespace(
        dataset_name='sign', skeleton_mode='7part', use_hand_vel=True,
        latent_dim=32, kernel_size=3, n_layers=2, n_extra_layers=1,
        norm='none', activation='gelu', dropout=0.1,
        pose_dim=223, joints_num=7, contact_joints=[],
    )
    
    vae = VAE(opt)
    num_params = sum(p.numel() for p in vae.parameters())
    print(f"    VAE params: {num_params/1e6:.2f}M")
    
    B, T = 4, 64
    x = torch.randn(B, T, 223)
    
    # Forward
    out, loss_dict = vae(x)
    check("Output shape matches input", out.shape == x.shape,
          f"got {out.shape}, expected {x.shape}")
    check("loss_kl present", "loss_kl" in loss_dict)
    check("loss_kl finite", torch.isfinite(loss_dict["loss_kl"]))
    
    # Encode → decode consistency
    z, _ = vae.encode(x)
    check("Latent is 4D [B,T',J',D]", z.dim() == 4, f"got dim={z.dim()}")
    recon = vae.decode(z)
    check("Decode shape", recon.shape[-1] == 223, f"got {recon.shape}")
    
    # Also test 133D mode
    opt_133 = Namespace(**vars(opt))
    opt_133.use_hand_vel = False
    opt_133.pose_dim = 133
    vae_133 = VAE(opt_133)
    x_133 = torch.randn(B, T, 133)
    out_133, _ = vae_133(x_133)
    check("133D mode works too", out_133.shape == (B, T, 133))


def test_7_trainer_loss():
    """Test trainer _sign_loss with part-wise weighting."""
    print("\n[7/7] Trainer part-wise loss")
    from models.vae.trainer import VAETrainer, PART_LOSS_CONFIG_223

    opt = Namespace(
        dataset_name='sign', device='cpu', recon_loss='l1_smooth',
        is_train=True, log_dir='/tmp/test_vae_log',
        lambda_recon=1.0, lambda_vel=0.5, lambda_kl=0.02,
        kl_anneal_iters=0, pose_dim=223,
    )
    os.makedirs(opt.log_dir, exist_ok=True)

    # Dummy VAE
    from models.vae.model import VAE
    vae_opt = Namespace(
        dataset_name='sign', skeleton_mode='7part', use_hand_vel=True,
        latent_dim=32, kernel_size=3, n_layers=2, n_extra_layers=1,
        norm='none', activation='gelu', dropout=0.1,
        pose_dim=223, joints_num=7, contact_joints=[],
    )
    vae = VAE(vae_opt)
    trainer = VAETrainer(opt, vae)

    # Check part config
    check("Part config has vel parts", len(trainer._part_config) == 9,
          f"got {len(trainer._part_config)} (expected 7 rot + 2 vel)")
    
    # Names
    part_names = [p[0] for p in trainer._part_config]
    check("lhand_vel in parts", 'lhand_vel' in part_names)
    check("rhand_vel in parts", 'rhand_vel' in part_names)
    
    # Run loss
    B, T = 4, 64
    motion = torch.randn(B, T, 223)
    pred_motion = motion + torch.randn_like(motion) * 0.1
    loss_dict = {"loss_kl": torch.tensor(0.5)}
    
    loss, ld = trainer._sign_loss(motion, pred_motion, loss_dict)
    check("Loss is scalar", loss.dim() == 0)
    check("Loss is finite", torch.isfinite(loss))
    check("loss_recon in dict", "loss_recon" in ld)
    check("loss_vel in dict", "loss_vel" in ld)
    check("loss_lhand in dict", "loss_lhand" in ld)
    check("loss_rhand in dict", "loss_rhand" in ld)
    check("loss_lhand_vel in dict", "loss_lhand_vel" in ld)

    # Cleanup
    import shutil
    shutil.rmtree(opt.log_dir, ignore_errors=True)


if __name__ == '__main__':
    print("=" * 60)
    print("  223D VAE Pipeline Verification")
    print("=" * 60)

    test_1_sign_param()
    test_2_clamping()
    test_3_hand_velocity()
    test_4_encoder_grouping()
    test_5_decoder_ungrouping()
    test_6_vae_forward()
    test_7_trainer_loss()

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
    else:
        print("\n  All checks passed! VAE is ready for training.")
        print("\n  Training command:")
        print("    python train_sign_vae.py \\")
        print("      --name sign_vae_223d_v1 \\")
        print("      --dataset_name sign \\")
        print("      --sign_dataset how2sign \\")
        print("      --data_root ./dataset/How2Sign \\")
        print("      --mean_path ./dataset/How2Sign/mean.pt \\")
        print("      --std_path ./dataset/How2Sign/std.pt \\")
        print("      --window_size 64 --batch_size 256 --max_epoch 50")
