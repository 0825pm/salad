"""
feats2joints.py — 133D axis-angle → SMPL-X FK → 3D joints

133D layout (upper body only, lower body removed):
  [0:3]     root (pelvis) orientation
  [3:15]    upper body joints (spine1~chest, 4 joints × 3)
  [15:30]   neck~shoulders (5 joints × 3)
  [30:75]   left hand (15 joints × 3)
  [75:120]  right hand (15 joints × 3)
  [120:123] jaw (1 joint × 3)
  [123:133] expression (10D)

To feed SMPL-X we prepend 36D zeros for lower body → 169D total.
"""

import torch
import smplx
import os

# Default shape params from SOKE (average body shape)
DEFAULT_BETAS = torch.tensor([
    -0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
     0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842
])

_body_model_cache = {}


def _get_body_model(model_path, device='cpu'):
    """Load and cache SMPL-X neutral model."""
    key = (model_path, str(device))
    if key not in _body_model_cache:
        _body_model_cache[key] = smplx.create(
            model_path=model_path,
            model_type='smplx',
            gender='neutral',
            use_pca=False,
            use_face_contour=False,
            batch_size=1,
        ).to(device).eval()
    return _body_model_cache[key]


def aa133_to_joints(features, mean, std, smplx_model_path,
                    device='cpu', betas=None):
    """
    133D normalized features → SMPL-X FK → joints [B, T, J, 3]

    Args:
        features: [B, T, 133] normalized tensor
        mean: [133] tensor
        std: [133] tensor
        smplx_model_path: path to smplx model directory
            e.g., 'deps/smpl_models/smplx'
        device: 'cpu' or 'cuda:X'
        betas: [10] body shape params (optional, uses default if None)

    Returns:
        joints: [B, T, J, 3] — J depends on SMPL-X (typically 127+)
    """
    # Denormalize
    features = features * std.to(features) + mean.to(features)

    B, T, D = features.shape
    assert D == 133, f"Expected 133D features, got {D}"

    # Prepend 36D zeros for lower body → 169D
    zero_lower = torch.zeros(B, T, 36, device=features.device, dtype=features.dtype)
    full = torch.cat([zero_lower, features], dim=-1)  # [B, T, 169]
    full = full.view(B * T, -1)

    # Split into SMPL-X params (same mapping as SOKE H2S.py)
    root_pose   = full[:, 0:3]
    body_pose   = full[:, 3:66]     # 21 joints × 3 (includes lower body zeros)
    lhand_pose  = full[:, 66:111]   # 15 joints × 3
    rhand_pose  = full[:, 111:156]  # 15 joints × 3
    jaw_pose    = full[:, 156:159]
    expr        = full[:, 159:169]

    # Shape params
    if betas is None:
        betas = DEFAULT_BETAS
    betas = betas.to(features.device).unsqueeze(0).expand(B * T, -1)

    # Zero eye poses
    zero_eye = torch.zeros(B * T, 3, device=features.device, dtype=features.dtype)

    # SMPL-X forward
    model = _get_body_model(smplx_model_path, device=features.device)

    # Process in chunks to avoid OOM
    chunk_size = 512
    all_joints = []
    for start in range(0, B * T, chunk_size):
        end = min(start + chunk_size, B * T)
        with torch.no_grad():
            output = model(
                global_orient=root_pose[start:end],
                body_pose=body_pose[start:end],
                left_hand_pose=lhand_pose[start:end],
                right_hand_pose=rhand_pose[start:end],
                jaw_pose=jaw_pose[start:end],
                expression=expr[start:end],
                betas=betas[start:end],
                leye_pose=zero_eye[start:end],
                reye_pose=zero_eye[start:end],
            )
        all_joints.append(output.joints)

    joints = torch.cat(all_joints, dim=0)  # [B*T, J, 3]
    J = joints.shape[1]
    joints = joints.view(B, T, J, 3)
    return joints


def aa133_to_joints_np(motion_133, mean, std, smplx_model_path, device='cpu'):
    """
    Numpy convenience wrapper.

    Args:
        motion_133: [T, 133] numpy array (normalized)
        mean, std: [133] numpy or tensor
        smplx_model_path: path to smplx model dir

    Returns:
        joints: [T, J, 3] numpy array
    """
    import numpy as np
    motion_t = torch.from_numpy(motion_133).float().unsqueeze(0)  # [1, T, 133]
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, dtype=torch.float32)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, dtype=torch.float32)

    motion_t = motion_t.to(device)
    joints = aa133_to_joints(motion_t, mean, std, smplx_model_path, device=device)
    return joints[0].cpu().numpy()  # [T, J, 3]
