"""feats2joints.py - Sign10 210D/120D -> SMPL-X FK -> 3D joints"""

import torch
import numpy as np
from utils.sign_paramUtil import sign10_to_smplx_input, sign10_vel_to_rotation

DEFAULT_BETAS = torch.tensor([
    -0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
     0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842
])
_body_model_cache = {}


def _get_body_model(model_path, device="cpu"):
    key = (model_path, str(device))
    if key not in _body_model_cache:
        import smplx
        _body_model_cache[key] = smplx.create(
            model_path=model_path, model_type="smplx", gender="neutral",
            use_pca=False, use_face_contour=False, batch_size=1,
        ).to(device).eval()
    return _body_model_cache[key]


def _run_fk(smplx_in, smplx_model_path, device, betas, BT):
    """Common FK logic."""
    if betas is None:
        betas = DEFAULT_BETAS
    betas_exp = betas.to(device).unsqueeze(0).expand(BT, -1)
    zero3 = torch.zeros(BT, 3, device=device)
    zero10 = torch.zeros(BT, 10, device=device)
    model = _get_body_model(smplx_model_path, device)
    with torch.no_grad():
        output = model(
            global_orient=smplx_in["global_orient"].to(device),
            body_pose=smplx_in["body_pose"].to(device),
            left_hand_pose=smplx_in["left_hand_pose"].to(device),
            right_hand_pose=smplx_in["right_hand_pose"].to(device),
            jaw_pose=zero3, leye_pose=zero3, reye_pose=zero3,
            expression=zero10, betas=betas_exp,
        )
    return output.joints.cpu()


def sign10_vel_to_joints(features, mean, std, smplx_model_path,
                         device="cpu", betas=None):
    """210D sign10_vel normalized -> denorm rotation -> FK -> joints [B,T,J,3]"""
    mean_t = torch.from_numpy(mean).float() if isinstance(mean, np.ndarray) else mean
    std_t = torch.from_numpy(std).float() if isinstance(std, np.ndarray) else std

    # Strip velocity -> 120D rotation, then denormalize
    rot_120 = sign10_vel_to_rotation(features)  # [B, T, 120]
    rot_denorm = rot_120 * std_t.to(features.device) + mean_t.to(features.device)

    B, T, D = rot_denorm.shape
    assert D == 120
    flat = rot_denorm.view(B * T, 120)
    smplx_in = sign10_to_smplx_input(flat)
    joints = _run_fk(smplx_in, smplx_model_path, device, betas, B * T)
    return joints.view(B, T, -1, 3)


def sign10_to_joints(features, mean, std, smplx_model_path,
                     device="cpu", betas=None):
    """120D sign10 normalized -> FK -> joints [B,T,J,3]"""
    mean_t = torch.from_numpy(mean).float() if isinstance(mean, np.ndarray) else mean
    std_t = torch.from_numpy(std).float() if isinstance(std, np.ndarray) else std
    features = features * std_t.to(features.device) + mean_t.to(features.device)

    B, T, D = features.shape
    assert D == 120
    flat = features.view(B * T, 120)
    smplx_in = sign10_to_smplx_input(flat)
    joints = _run_fk(smplx_in, smplx_model_path, device, betas, B * T)
    return joints.view(B, T, -1, 3)

def aa133_to_joints(features, mean, std, smplx_model_path,
                    device='cpu', betas=None):
    """133D normalized features → SMPL-X FK → joints [B, T, J, 3]"""
    mean_t = torch.from_numpy(mean).float() if isinstance(mean, np.ndarray) else mean
    std_t = torch.from_numpy(std).float() if isinstance(std, np.ndarray) else std
    features = features * std_t.to(features.device) + mean_t.to(features.device)

    B, T, D = features.shape
    assert D == 133, f"Expected 133D, got {D}"

    # Prepend 36D zeros for lower body → 169D
    zero_lower = torch.zeros(B, T, 36, device=features.device, dtype=features.dtype)
    full = torch.cat([zero_lower, features], dim=-1)  # [B, T, 169]
    flat = full.view(B * T, -1)

    # Split into SMPL-X params
    root_pose   = flat[:, 0:3]
    body_pose   = flat[:, 3:66]      # 21 joints × 3
    lhand_pose  = flat[:, 66:111]    # 15 joints × 3
    rhand_pose  = flat[:, 111:156]   # 15 joints × 3
    jaw_pose    = flat[:, 156:159]
    expr        = flat[:, 159:169]

    if betas is None:
        betas = DEFAULT_BETAS
    betas_exp = betas.to(device).unsqueeze(0).expand(B * T, -1)
    zero3 = torch.zeros(B * T, 3, device=device)

    model = _get_body_model(smplx_model_path, device)
    chunk_size = 512
    all_joints = []
    for s in range(0, B * T, chunk_size):
        e = min(s + chunk_size, B * T)
        with torch.no_grad():
            output = model(
                global_orient=root_pose[s:e].to(device),
                body_pose=body_pose[s:e].to(device),
                left_hand_pose=lhand_pose[s:e].to(device),
                right_hand_pose=rhand_pose[s:e].to(device),
                jaw_pose=jaw_pose[s:e].to(device),
                expression=expr[s:e].to(device),
                betas=betas_exp[s:e],
                leye_pose=zero3[s:e],
                reye_pose=zero3[s:e],
            )
        all_joints.append(output.joints.cpu())

    joints = torch.cat(all_joints, dim=0)
    return joints.view(B, T, -1, 3)

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
