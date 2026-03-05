import torch
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin

import os
import time
import numpy as np
from collections import OrderedDict, defaultdict

from utils.eval_t2m import evaluation_vae, test_vae
from utils.utils import print_current_loss

def def_value():
    return 0.0

# ── Part-wise loss config ──
# Each tuple: (name, start_idx, end_idx, weight)
# Weights reflect importance: hand > face > body

PART_LOSS_CONFIG_133 = [
    ('root',    0,   3,   0.2),
    ('upper_a', 3,   15,  0.3),
    ('upper_b', 15,  30,  0.5),
    ('lhand',   30,  75,  1.0),
    ('rhand',   75,  120, 1.0),
    ('jaw',     120, 123, 0.5),
    ('expr',    123, 133, 0.8),
]

PART_LOSS_CONFIG_223 = PART_LOSS_CONFIG_133 + [
    ('lhand_vel', 133, 178, 0.8),
    ('rhand_vel', 178, 223, 0.8),
]


class VAETrainer:
    def __init__(self, opt, vae):
        self.opt = opt
        self.vae = vae

        if opt.is_train:
            self.logger = SummaryWriter(opt.log_dir)
            if opt.recon_loss == "l1":
                self.recon_criterion = torch.nn.L1Loss()
            elif opt.recon_loss == "l1_smooth":
                self.recon_criterion = torch.nn.SmoothL1Loss()

        # Select part config based on pose_dim
        pose_dim = getattr(opt, 'pose_dim', 133)
        if pose_dim == 223:
            self._part_config = PART_LOSS_CONFIG_223
        else:
            self._part_config = PART_LOSS_CONFIG_133


    def train_forward(self, batch_data):
        motion = batch_data.to(self.opt.device, dtype=torch.float32)

        pred_motion, loss_dict = self.vae.forward(motion)

        if getattr(self.opt, 'dataset_name', 't2m') == 'sign':
            return self._sign_loss(motion, pred_motion, loss_dict)
        else:
            return self._humanml_loss(motion, pred_motion, loss_dict)

    def _kl_weight(self):
        """KL annealing: linearly ramp up from 0 to lambda_kl over kl_anneal_iters."""
        anneal_iters = getattr(self.opt, 'kl_anneal_iters', 0)
        if anneal_iters <= 0:
            return self.opt.lambda_kl
        progress = min(1.0, getattr(self, '_it', 0) / anneal_iters)
        return self.opt.lambda_kl * progress

    def _humanml_loss(self, motion, pred_motion, loss_dict):
        """Original 263D loss: root/ric/rot/vel/contact split."""
        J = self.opt.joints_num
        root, ric, rot, vel, contact = torch.split(
            motion, [4, 3*(J-1), 6*(J-1), 3*J, 4], dim=-1)
        _, pred_ric, _, pred_vel, _ = torch.split(
            pred_motion, [4, 3*(J-1), 6*(J-1), 3*J, 4], dim=-1)

        self.motion = motion
        self.pred_motion = pred_motion

        loss_rec = self.recon_criterion(pred_motion, motion)
        loss_vel = self.recon_criterion(pred_vel, vel)
        loss_pos = self.recon_criterion(pred_ric, ric)
        loss_kl  = loss_dict["loss_kl"]

        loss = (loss_rec
                + loss_vel * self.opt.lambda_vel
                + loss_pos * self.opt.lambda_pos
                + loss_kl  * self.opt.lambda_kl)

        loss_dict["loss_recon"] = loss_rec
        loss_dict["loss_vel"]   = loss_vel
        loss_dict["loss_pos"]   = loss_pos
        return loss, loss_dict

    def _sign_loss(self, motion, pred_motion, loss_dict):
        """Sign loss with part-wise weighting + temporal smoothness.

        Part weights: hand(1.0) > expr(0.8) > upper_b(0.5) > upper_a(0.3) > root(0.2)
        This ensures the model focuses on hand movements (most important for sign).
        """
        self.motion = motion
        self.pred_motion = pred_motion

        # ── Part-wise weighted reconstruction ──
        weighted_sum = 0.0
        weight_total = 0.0
        for name, s, e, w in self._part_config:
            part_loss = self.recon_criterion(pred_motion[..., s:e], motion[..., s:e])
            loss_dict[f'loss_{name}'] = part_loss
            weighted_sum += w * part_loss
            weight_total += w
        loss_rec = weighted_sum / weight_total

        # ── Temporal velocity smoothness ──
        tvel_gt   = motion[:, 1:] - motion[:, :-1]
        tvel_pred = pred_motion[:, 1:] - pred_motion[:, :-1]
        loss_tvel = self.recon_criterion(tvel_pred, tvel_gt)

        # ── KL with annealing ──
        loss_kl = loss_dict["loss_kl"]
        kl_w = self._kl_weight()

        loss = (loss_rec * self.opt.lambda_recon
                + loss_tvel * self.opt.lambda_vel
                + loss_kl * kl_w)

        loss_dict["loss_recon"] = loss_rec
        loss_dict["loss_vel"]   = loss_tvel
        loss_dict["kl_weight"]  = torch.tensor(kl_w)
        return loss, loss_dict


    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):
        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.optim.param_groups:
            param_group["lr"] = current_lr


    def save(self, file_name, epoch, total_iter):
        state = {
            "vae": self.vae.state_dict(),
            "opt_vae": self.optim.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,
            "total_iter": total_iter,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        ckpt = torch.load(pjoin(model_dir, 'latest.tar'), map_location='cpu')
        self.vae.load_state_dict(ckpt['vae'])
        self.optim.load_state_dict(ckpt['opt_vae'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        return ckpt['epoch'], ckpt['total_iter']

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval=None):
        self.vae.to(self.opt.device)

        self.optim = torch.optim.AdamW(
            self.vae.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optim, milestones=self.opt.milestones, gamma=self.opt.gamma)

        epoch = 0
        it = 0
        if self.opt.is_continue:
            epoch, it = self.resume(self.opt.model_dir)

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Training VAE — {total_iters} total iters')

        logs = OrderedDict()

        best_fid, best_div, best_top1, best_matching = 1000, 100, 0, 100

        while epoch < self.opt.max_epoch:
            for i, batch_data in enumerate(train_loader):
                self.vae.train()
                self._it = it   # for KL annealing

                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                loss, loss_dict = self.train_forward(batch_data)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if it >= self.opt.warm_up_iter:
                    self.scheduler.step()

                for k, v in loss_dict.items():
                    if k not in logs:
                        logs[k] = def_value()
                    logs[k] += v.item() if isinstance(v, torch.Tensor) else v
                logs['loss'] = logs.get('loss', 0.0) + loss.item()

                it += 1

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({})
                    for tag, value in logs.items():
                        mean_loss[tag] = value / self.opt.log_every
                        self.logger.add_scalar(f'Train/{tag}', value / self.opt.log_every, it)
                    logs = OrderedDict()
                    print_current_loss(start_time, it, total_iters, mean_loss,
                                       epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            # ── end of epoch ──
            epoch += 1

            # Validation loss
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                self.logger.add_scalar('Val/loss', val_loss, epoch)
                print(f'  [Val] epoch {epoch}, loss: {val_loss:.6f}')

            # HumanML3D eval (not used for sign, eval_val_loader=None)
            if eval_val_loader is not None and epoch % self.opt.eval_every_e == 0:
                # ... (original eval code, skipped for sign)
                pass

            # Save epoch checkpoint
            if epoch % max(1, self.opt.eval_every_e) == 0:
                self.save(pjoin(self.opt.model_dir, f'net_epoch{epoch:04d}.tar'), epoch, it)

        # Final save
        self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
        print(f'Training done. {epoch} epochs, {it} iters.')

    @torch.no_grad()
    def _validate(self, val_loader):
        """Quick validation loss (no eval metrics)."""
        self.vae.eval()
        total_loss = 0.0
        count = 0
        for batch_data in val_loader:
            motion = batch_data.to(self.opt.device, dtype=torch.float32)
            pred_motion, loss_dict = self.vae.forward(motion)

            if getattr(self.opt, 'dataset_name', 't2m') == 'sign':
                loss, _ = self._sign_loss(motion, pred_motion, loss_dict)
            else:
                loss, _ = self._humanml_loss(motion, pred_motion, loss_dict)

            total_loss += loss.item()
            count += 1
        return total_loss / max(count, 1)
