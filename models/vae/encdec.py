import torch
import torch.nn as nn

from models.skeleton.conv import ResSTConv, get_activation
from models.skeleton.pool import STPool, STUnpool
from utils.paramUtil import kit_adj_list, t2m_adj_list
from utils.sign_paramUtil import sign_adj_list, SIGN_SPLITS, get_sign_config
from utils.skeleton import adj_list_to_edges


# ══════════════════════════════════════════════════════════════
# Original SALAD: 263D HumanML3D
# ══════════════════════════════════════════════════════════════

class MotionEncoder(nn.Module):
    def __init__(self, opt):
        super(MotionEncoder, self).__init__()

        self.pose_dim = opt.pose_dim
        self.joints_num = (self.pose_dim + 1) // 12
        self.latent_dim = opt.latent_dim
        self.contact_joints = opt.contact_joints

        self.layers = nn.ModuleList()
        for i in range(self.joints_num):
            if i == 0:
                input_dim = 7
            elif i in self.contact_joints:
                input_dim = 13
            else:
                input_dim = 12
            self.layers.append(nn.Sequential(
                nn.Linear(input_dim, self.latent_dim),
                get_activation(opt.activation),
                nn.Linear(self.latent_dim, self.latent_dim),
            ))

    def forward(self, x):
        B, T, D = x.size()
        root, ric, rot, vel, contact = torch.split(
            x, [4, 3 * (self.joints_num - 1), 6 * (self.joints_num - 1),
                3 * self.joints_num, 4], dim=-1)
        ric = ric.reshape(B, T, self.joints_num - 1, 3)
        rot = rot.reshape(B, T, self.joints_num - 1, 6)
        vel = vel.reshape(B, T, self.joints_num, 3)

        joints = [torch.cat([root, vel[:, :, 0]], dim=-1)]
        for i in range(1, self.joints_num):
            joints.append(torch.cat([ric[:, :, i - 1], rot[:, :, i - 1], vel[:, :, i]], dim=-1))
        for cidx, jidx in enumerate(self.contact_joints):
            joints[jidx] = torch.cat([joints[jidx], contact[:, :, cidx, None]], dim=-1)

        out = []
        for i in range(self.joints_num):
            out.append(self.layers[i](joints[i]))
        out = torch.stack(out, dim=2)
        return out


class MotionDecoder(nn.Module):
    def __init__(self, opt):
        super(MotionDecoder, self).__init__()

        self.pose_dim = opt.pose_dim
        self.joints_num = (self.pose_dim + 1) // 12
        self.latent_dim = opt.latent_dim
        self.contact_joints = opt.contact_joints

        self.layers = nn.ModuleList()
        for i in range(self.joints_num):
            if i == 0:
                output_dim = 7
            elif i in self.contact_joints:
                output_dim = 13
            else:
                output_dim = 12
            self.layers.append(nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                get_activation(opt.activation),
                nn.Linear(self.latent_dim, output_dim),
            ))

    def forward(self, x):
        B, T, J, D = x.size()

        out = []
        for i in range(self.joints_num):
            out.append(self.layers[i](x[:, :, i]))

        root = out[0]
        ric_list, rot_list, vel_list = [], [], []
        for i in range(1, self.joints_num):
            ric = out[i][:, :, :3]
            rot = out[i][:, :, 3:9]
            vel = out[i][:, :, 9:12]
            ric_list.append(ric)
            rot_list.append(rot)
            vel_list.append(vel)

        contact = [out[i][:, :, -1] for i in self.contact_joints]

        ric = torch.stack(ric_list, dim=2).reshape(B, T, (J - 1) * 3)
        rot = torch.stack(rot_list, dim=2).reshape(B, T, (J - 1) * 6)
        vel = torch.stack(vel_list, dim=2).reshape(B, T, (J - 1) * 3)
        contact = torch.stack(contact, dim=2).reshape(B, T, len(self.contact_joints))

        motion = torch.cat([
            root[..., :4],
            ric,
            rot,
            torch.cat([root[..., 4:], vel], dim=-1),
            contact,
        ], dim=-1)

        return motion


# ══════════════════════════════════════════════════════════════
# Sign Language: 133D or 223D (with hand velocity)
# ══════════════════════════════════════════════════════════════

# 133D rot-only splits (7part):  [3, 12, 15, 45, 45, 3, 10]
# 223D rot+vel splits (7part):   [3, 12, 15, 90, 90, 3, 10]
#   → hand node gets 90D = 45D rot + 45D vel, richer for graph attention
#
# Data layout (223D): [rot_133D | lhand_vel_45D | rhand_vel_45D]
# Encoder groups: lhand_rot(45) + lhand_vel(45) → one 90D node input
# Decoder ungroups back to: [rot_133D | vel_90D]

# ── Finger rotation index ranges (within 133D) ──
# Left hand: [30:75] = 5 fingers × 9D = 45D
# Right hand: [75:120] = 5 fingers × 9D = 45D
_FINGER_ROT_RANGES_L = [(30 + i*9, 30 + (i+1)*9) for i in range(5)]
_FINGER_ROT_RANGES_R = [(75 + i*9, 75 + (i+1)*9) for i in range(5)]


class SignMotionEncoder(nn.Module):
    """Per-part linear projection: [B, T, D_in] → [B, T, J, latent_dim]

    D_in = 133 (rot only) or 223 (rot + hand velocity).
    When use_hand_vel=True, hand rot+vel are grouped per node before projection.
    """
    def __init__(self, opt):
        super().__init__()
        self.latent_dim = opt.latent_dim
        self.skeleton_mode = getattr(opt, 'skeleton_mode', '7part')
        self.use_hand_vel = getattr(opt, 'use_hand_vel', True)

        # splits reflect per-node input dims (90D for hand with vel)
        self.splits, _, _, _ = get_sign_config(
            self.skeleton_mode, self.use_hand_vel)

        self.layers = nn.ModuleList()
        for dim in self.splits:
            self.layers.append(nn.Sequential(
                nn.Linear(dim, self.latent_dim),
                get_activation(opt.activation),
                nn.Linear(self.latent_dim, self.latent_dim),
            ))

    def _group_features(self, x):
        """Regroup [rot_133, vel_90] → list of per-node feature tensors.

        Groups hand rotation + velocity into single tensors so each hand node
        gets a richer feature vector for graph attention.
        """
        rot = x[..., :133]
        lhand_vel = x[..., 133:178]   # 45D
        rhand_vel = x[..., 178:223]   # 45D

        if self.skeleton_mode == '7part':
            return [
                rot[..., 0:3],                                       # root:    3D
                rot[..., 3:15],                                      # upper_a: 12D
                rot[..., 15:30],                                     # upper_b: 15D
                torch.cat([rot[..., 30:75],  lhand_vel], dim=-1),    # lhand:   90D
                torch.cat([rot[..., 75:120], rhand_vel], dim=-1),    # rhand:   90D
                rot[..., 120:123],                                   # jaw:     3D
                rot[..., 123:133],                                   # expr:    10D
            ]
        else:  # finger mode
            parts = [rot[..., 0:3], rot[..., 3:15], rot[..., 15:30]]
            # Left hand: 5 fingers, each 9D rot + 9D vel = 18D
            for i in range(5):
                s, e = _FINGER_ROT_RANGES_L[i]
                parts.append(torch.cat([
                    rot[..., s:e],
                    lhand_vel[..., i*9:(i+1)*9],
                ], dim=-1))
            # Right hand: 5 fingers, each 9D rot + 9D vel = 18D
            for i in range(5):
                s, e = _FINGER_ROT_RANGES_R[i]
                parts.append(torch.cat([
                    rot[..., s:e],
                    rhand_vel[..., i*9:(i+1)*9],
                ], dim=-1))
            parts.extend([rot[..., 120:123], rot[..., 123:133]])
            return parts

    def forward(self, x):
        """x: [B, T, 133 or 223] → [B, T, J, latent_dim]"""
        if self.use_hand_vel:
            parts = self._group_features(x)
        else:
            parts = torch.split(x, self.splits, dim=-1)

        out = [self.layers[i](parts[i]) for i in range(len(self.splits))]
        return torch.stack(out, dim=2)


class SignMotionDecoder(nn.Module):
    """Per-part projection back: [B, T, J, latent_dim] → [B, T, D_out]

    D_out = 133 (rot only) or 223 (rot + hand velocity).
    When use_hand_vel=True, decoder outputs [rot_133, lhand_vel_45, rhand_vel_45].
    """
    def __init__(self, opt):
        super().__init__()
        self.latent_dim = opt.latent_dim
        self.skeleton_mode = getattr(opt, 'skeleton_mode', '7part')
        self.use_hand_vel = getattr(opt, 'use_hand_vel', True)

        self.splits, _, _, _ = get_sign_config(
            self.skeleton_mode, self.use_hand_vel)

        self.layers = nn.ModuleList()
        for dim in self.splits:
            self.layers.append(nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                get_activation(opt.activation),
                nn.Linear(self.latent_dim, dim),
            ))

    def _ungroup_features(self, raw_parts):
        """Separate rot and vel from hand node outputs → [rot_133, vel_90].

        Inverse of encoder's _group_features().
        """
        if self.skeleton_mode == '7part':
            lhand_full = raw_parts[3]    # 90D = 45 rot + 45 vel
            rhand_full = raw_parts[4]    # 90D = 45 rot + 45 vel

            rot_133 = torch.cat([
                raw_parts[0],             # root:    3D
                raw_parts[1],             # upper_a: 12D
                raw_parts[2],             # upper_b: 15D
                lhand_full[..., :45],     # lhand rot: 45D
                rhand_full[..., :45],     # rhand rot: 45D
                raw_parts[5],             # jaw:     3D
                raw_parts[6],             # expr:    10D
            ], dim=-1)
            vel_90 = torch.cat([
                lhand_full[..., 45:],     # lhand vel: 45D
                rhand_full[..., 45:],     # rhand vel: 45D
            ], dim=-1)

        else:  # finger mode
            rot_parts = [raw_parts[0], raw_parts[1], raw_parts[2]]
            vel_parts_l, vel_parts_r = [], []
            for i in range(5):
                finger = raw_parts[3 + i]   # 18D = 9 rot + 9 vel
                rot_parts.append(finger[..., :9])
                vel_parts_l.append(finger[..., 9:])
            for i in range(5):
                finger = raw_parts[8 + i]   # 18D = 9 rot + 9 vel
                rot_parts.append(finger[..., :9])
                vel_parts_r.append(finger[..., 9:])
            rot_parts.extend([raw_parts[13], raw_parts[14]])
            rot_133 = torch.cat(rot_parts, dim=-1)
            vel_90 = torch.cat(vel_parts_l + vel_parts_r, dim=-1)

        return torch.cat([rot_133, vel_90], dim=-1)   # [B, T, 223]

    def forward(self, x):
        """x: [B, T, J, latent_dim] → [B, T, 133 or 223]"""
        raw_parts = [self.layers[i](x[:, :, i]) for i in range(len(self.splits))]

        if self.use_hand_vel:
            return self._ungroup_features(raw_parts)
        else:
            return torch.cat(raw_parts, dim=-1)


# ══════════════════════════════════════════════════════════════
# STConv Encoder / Decoder  (shared, dataset-aware)
# ══════════════════════════════════════════════════════════════

class STConvEncoder(nn.Module):
    def __init__(self, opt):
        super(STConvEncoder, self).__init__()

        adj_lists = {
            "t2m": t2m_adj_list,
            "kit": kit_adj_list,
        }
        if opt.dataset_name == "sign":
            skeleton_mode = getattr(opt, 'skeleton_mode', '7part')
            _, self.adj_list, _, _ = get_sign_config(skeleton_mode)
        else:
            self.adj_list = adj_lists[opt.dataset_name]

        self.edge_list = [adj_list_to_edges(self.adj_list)]
        self.mapping_list = []

        self.layers = nn.ModuleList()
        for i in range(opt.n_layers):
            layers = []
            for _ in range(opt.n_extra_layers):
                layers.append(ResSTConv(
                    self.edge_list[-1],
                    opt.latent_dim,
                    opt.kernel_size,
                    activation=opt.activation,
                    norm=opt.norm,
                    dropout=opt.dropout
                ))
            layers.append(ResSTConv(
                self.edge_list[-1],
                opt.latent_dim,
                opt.kernel_size,
                activation=opt.activation,
                norm=opt.norm,
                dropout=opt.dropout
            ))

            pool = STPool(opt.dataset_name, i,
                          skeleton_mode=getattr(opt, 'skeleton_mode', '7part'))
            layers.append(pool)
            self.layers.append(nn.Sequential(*layers))

            self.edge_list.append(pool.new_edges)
            self.mapping_list.append(pool.skeleton_mapping)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class STConvDecoder(nn.Module):
    def __init__(self, opt, encoder: STConvEncoder):
        super(STConvDecoder, self).__init__()

        self.layers = nn.ModuleList()
        mapping_list = encoder.mapping_list.copy()
        edge_list = encoder.edge_list.copy()

        for i in range(opt.n_layers):
            layers = []
            layers.append(STUnpool(skeleton_mapping=mapping_list.pop()))
            edges = edge_list.pop()
            for _ in range(opt.n_extra_layers):
                layers.append(ResSTConv(
                    edge_list[-1],
                    opt.latent_dim,
                    opt.kernel_size,
                    activation=opt.activation,
                    norm=opt.norm,
                    dropout=opt.dropout
                ))
            layers.append(ResSTConv(
                edge_list[-1],
                opt.latent_dim,
                opt.kernel_size,
                activation=opt.activation,
                norm=opt.norm,
                dropout=opt.dropout
            ))
            self.layers.append(nn.Sequential(*layers))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
