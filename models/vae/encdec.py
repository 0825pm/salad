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
        root, ric, rot, vel, contact = torch.split(x, [4, 3 * (self.joints_num - 1), 6 * (self.joints_num - 1), 3 * self.joints_num, 4], dim=-1)
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
# NEW: Sign Language 133D  (7 super-joints)
# ══════════════════════════════════════════════════════════════

class SignMotionEncoder(nn.Module):
    """Per-part linear projection: [B, T, 133] → [B, T, J, D]"""
    def __init__(self, opt):
        super().__init__()
        self.latent_dim = opt.latent_dim
        skeleton_mode = getattr(opt, 'skeleton_mode', '7part')
        self.splits, _, _, _ = get_sign_config(skeleton_mode)

        self.layers = nn.ModuleList()
        for dim in self.splits:
            self.layers.append(nn.Sequential(
                nn.Linear(dim, self.latent_dim),
                get_activation(opt.activation),
                nn.Linear(self.latent_dim, self.latent_dim),
            ))

    def forward(self, x):
        """x: [B, T, 133]  →  [B, T, J, D]"""
        parts = torch.split(x, self.splits, dim=-1)
        out = [self.layers[i](parts[i]) for i in range(len(self.splits))]
        return torch.stack(out, dim=2)


class SignMotionDecoder(nn.Module):
    """Per-part projection back: [B, T, J, D] → [B, T, 133]"""
    def __init__(self, opt):
        super().__init__()
        self.latent_dim = opt.latent_dim
        skeleton_mode = getattr(opt, 'skeleton_mode', '7part')
        self.splits, _, _, _ = get_sign_config(skeleton_mode)

        self.layers = nn.ModuleList()
        for dim in self.splits:
            self.layers.append(nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                get_activation(opt.activation),
                nn.Linear(self.latent_dim, dim),
            ))

    def forward(self, x):
        """x: [B, T, J, D]  →  [B, T, 133]"""
        parts = [self.layers[i](x[:, :, i]) for i in range(len(self.splits))]
        return torch.cat(parts, dim=-1)


# ══════════════════════════════════════════════════════════════
# STConv Encoder / Decoder  (shared, dataset-aware)
# ══════════════════════════════════════════════════════════════

class STConvEncoder(nn.Module):
    def __init__(self, opt):
        super(STConvEncoder, self).__init__()

        ## ── PATCH: add sign adj_list (skeleton_mode aware) ──
        adj_lists = {
            "t2m": t2m_adj_list,
            "kit": kit_adj_list,
        }
        if opt.dataset_name == "sign":
            skeleton_mode = getattr(opt, 'skeleton_mode', '7part')
            _, self.adj_list, _, _ = get_sign_config(skeleton_mode)
        else:
            self.adj_list = adj_lists[opt.dataset_name]
        ## ── end PATCH ──

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