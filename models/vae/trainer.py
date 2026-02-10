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
        

    def train_forward(self, batch_data):
        motion = batch_data.to(self.opt.device, dtype=torch.float32)

        pred_motion, loss_dict = self.vae.forward(motion)

        ## ── PATCH: branch on dataset ──
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
        """133D sign loss: recon + hand emphasis + temporal velocity."""
        self.motion = motion
        self.pred_motion = pred_motion

        # full reconstruction
        loss_rec = self.recon_criterion(pred_motion, motion)

        # hand emphasis: [30:120] = lhand(45) + rhand(45)
        loss_hand = self.recon_criterion(pred_motion[..., 30:120], motion[..., 30:120])

        # temporal velocity (finite difference)
        vel_gt   = motion[:, 1:] - motion[:, :-1]
        vel_pred = pred_motion[:, 1:] - pred_motion[:, :-1]
        loss_vel = self.recon_criterion(vel_pred, vel_gt)

        loss_kl = loss_dict["loss_kl"]
        kl_w = self._kl_weight()

        loss = (loss_rec
                + loss_hand * self.opt.lambda_pos   # reuse lambda_pos for hand
                + loss_vel  * self.opt.lambda_vel
                + loss_kl   * kl_w)

        loss_dict["loss_recon"] = loss_rec
        loss_dict["loss_hand"]  = loss_hand
        loss_dict["loss_vel"]   = loss_vel
        loss_dict["kl_weight"]  = torch.tensor(kl_w)
        return loss, loss_dict
    ## ── end PATCH ──


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
        ckpt = torch.load(pjoin(model_dir, "latest.tar"), map_location="cpu")
        self.vae.load_state_dict(ckpt["vae"])
        self.optim.load_state_dict(ckpt["opt_vae"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        return ckpt["epoch"], ckpt["total_iter"]

    def train(self, train_loader, val_loader, eval_val_loader=None, eval_wrapper=None, plot_eval=None):
        self.vae.to(self.opt.device)

        self.optim = torch.optim.AdamW(
            self.vae.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optim, milestones=self.opt.milestones, gamma=self.opt.gamma)

        epoch = 0
        it = 0
        if self.opt.is_continue:
            epoch, it = self.resume(self.opt.model_dir)
            print(f"Resuming from epoch {epoch}, iter {it}")

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Iters: {total_iters}')
        logs = defaultdict(def_value, OrderedDict())

        best_fid, best_div, best_top1, best_top2, best_top3, best_matching = 1000, 0, 0, 0, 0, 1000

        while epoch < self.opt.max_epoch:
            self.vae.train()
            for i, batch_data in enumerate(train_loader):
                it += 1
                self._it = it   # for KL annealing

                if it < self.opt.warm_up_iter:
                    curr_lr = self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                self.optim.zero_grad()
                loss, loss_dict = self.train_forward(batch_data)
                loss.backward()
                self.optim.step()

                if it >= self.opt.warm_up_iter:
                    self.scheduler.step()
                
                logs["loss"] += loss.item()
                logs["lr"] += self.optim.param_groups[0]['lr']
                for tag, value in loss_dict.items():
                    logs[tag] += value.item()

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            print('Validation time:')
            self.vae.eval()
            val_log = defaultdict(def_value, OrderedDict())
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, loss_dict = self.train_forward(batch_data)
                    val_log["loss"] += loss.item()
                    for tag, value in loss_dict.items():
                        val_log[tag] += value.item()
            
            msg = "Validation loss: "
            for tag, value in val_log.items():
                self.logger.add_scalar('Val/%s'%tag, value / len(val_loader), epoch)
                msg += "%s: %.3f, " % (tag, value / len(val_loader))
            print(msg)
            
            ## ── PATCH: skip eval for sign (no eval_wrapper yet) ──
            if eval_val_loader is not None and eval_wrapper is not None:
                if epoch % self.opt.eval_every_e == 0:
                    best_fid, best_div, best_top1, best_top2, best_top3, best_matching, _ = \
                        evaluation_vae(
                            self.opt.model_dir, eval_val_loader, self.vae,
                            self.logger, epoch, best_fid, best_div,
                            best_top1, best_top2, best_top3, best_matching,
                            eval_wrapper, save=True, draw=True)
            else:
                # For sign: save every eval_every_e epoch (no FID eval yet)
                if epoch % self.opt.eval_every_e == 0:
                    val_loss = val_log["loss"] / max(len(val_loader), 1)
                    self.save(pjoin(self.opt.model_dir, f'net_epoch{epoch:03d}_loss{val_loss:.4f}.tar'), epoch, it)
                    print(f'  [sign] Saved checkpoint at epoch {epoch}, val_loss={val_loss:.4f}')
            ## ── end PATCH ──

    def test(self, eval_wrapper, eval_val_loader, num_repeat, save_dir, cal_mm=True):
        test_vae(eval_val_loader, self.vae, num_repeat, eval_wrapper,
                 self.opt.joints_num, cal_mm=cal_mm)