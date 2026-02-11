from typing import List, Union

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin

import os
import time
import numpy as np
from collections import OrderedDict, defaultdict

from utils.eval_t2m import evaluation_denoiser, test_denoiser
from utils.utils import print_current_loss, attn2img
from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion
from visualization.joints2bvh import Joint2BVHConvertor 

def def_value():
    return 0.0

def lengths_to_mask(lengths: torch.Tensor) -> torch.Tensor:
    max_frames = torch.max(lengths)
    mask = torch.arange(max_frames, device=lengths.device).expand(
        len(lengths), max_frames) < lengths.unsqueeze(1)
    return mask


class DenoiserTrainer:
    def __init__(self, opt, denoiser, vae, scheduler):
        self.opt = opt
        self.denoiser = denoiser.to(opt.device)
        self.vae = vae.to(opt.device)
        self.noise_scheduler = scheduler

        if opt.is_train:
            self.logger = SummaryWriter(opt.log_dir)
            if opt.recon_loss == "l1":
                self.recon_criterion = torch.nn.L1Loss(reduction='none')
            elif opt.recon_loss == "l1_smooth":
                self.recon_criterion = torch.nn.SmoothL1Loss(reduction='none')
            elif opt.recon_loss == "l2":
                self.recon_criterion = torch.nn.MSELoss(reduction='none')
            else:
                raise NotImplementedError(f"Reconstruction loss {opt.recon_loss} not implemented")

            # Part-weight: dynamic based on skeleton_mode
            hw = getattr(opt, 'hand_weight', 1.0)
            skeleton_mode = getattr(opt, 'skeleton_mode', '7part')
            from utils.sign_paramUtil import get_sign_config
            _, _, hand_idx, num_j = get_sign_config(skeleton_mode)
            self.part_weight = torch.ones(num_j, device=opt.device)
            for idx in hand_idx:
                self.part_weight[idx] = hw
            self.part_weight = self.part_weight / self.part_weight.mean()  # normalize so avg=1
            if hw != 1.0:
                print(f"  [Loss] hand_weight={hw}, skeleton_mode={skeleton_mode} → part_weight={self.part_weight.tolist()}")

            # AMP (mixed precision)
            self.use_amp = getattr(opt, 'use_amp', False)
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
            if self.use_amp:
                print("  [AMP] Mixed precision training enabled (bf16)")

            # Recon loss (motion-space)
            self.use_recon_loss = getattr(opt, 'use_recon_loss', False)
            self.lambda_recon = getattr(opt, 'lambda_recon', 0.5)
            if self.use_recon_loss:
                print(f"  [ReconLoss] Enabled — lambda_recon={self.lambda_recon}")

    def _recover_x0(self, pred, noisy_latent, noise, timesteps):
        """Recover x₀ prediction from denoiser output based on prediction_type."""
        if self.opt.prediction_type == "sample":
            return pred
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(pred.device)
        alpha_t = alphas_cumprod[timesteps]
        # reshape for broadcasting: [B] → [B, 1, 1, 1]
        while alpha_t.ndim < pred.ndim:
            alpha_t = alpha_t.unsqueeze(-1)
        if self.opt.prediction_type == "epsilon":
            return (noisy_latent - (1 - alpha_t).sqrt() * pred) / alpha_t.sqrt()
        elif self.opt.prediction_type == "v_prediction":
            return alpha_t.sqrt() * noisy_latent - (1 - alpha_t).sqrt() * pred
        raise NotImplementedError(f"Unknown prediction_type: {self.opt.prediction_type}")
    
    def train_forward(self, batch_data):
        # setup input
        text_or_emb, motion, m_lens = batch_data

        # detect cache mode: tuple of tensors vs list of strings
        if isinstance(text_or_emb, (tuple, list)) and len(text_or_emb) == 3 and torch.is_tensor(text_or_emb[0]):
            # Cached text embeddings — CFG dropout already handled by dataset
            text = None
            text_emb = tuple(t.to(self.opt.device) for t in text_or_emb)
        else:
            # Raw text strings — apply CFG dropout here
            text = [
                "" if np.random.rand(1) < self.opt.cond_drop_prob else t for t in text_or_emb
            ]
            text_emb = None

        # to device
        motion = motion.to(self.opt.device, dtype=torch.float32)
        m_lens = m_lens.to(self.opt.device, dtype=torch.long)
        len_mask = lengths_to_mask(m_lens // 4) # [B, T]

        # latent
        with torch.no_grad():
            latent, _ = self.vae.encode(motion) # [B, T, J, D]
            len_mask = F.pad(len_mask, (0, latent.shape[1] - len_mask.shape[1]), mode="constant", value=False)
            latent = latent * len_mask[..., None, None].float()
        
        # sample diffusion timesteps
        timesteps = torch.randint(
            0,
            self.opt.num_train_timesteps,
            (latent.shape[0],),
            device=latent.device,
        ).long()

        # add noise
        noise = torch.randn_like(latent) # [B, T, J, D]
        noise = noise * len_mask[..., None, None].float()
        noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)

        # predict the noise (AMP autocast)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=self.use_amp):
            pred, attn_list = self.denoiser.forward(noisy_latent, timesteps, text=text, text_emb=text_emb, len_mask=len_mask)
            pred = pred * len_mask[..., None, None].float()
            
            # loss (part-weighted: hands get higher weight)
            loss_dict = {}
            loss = 0
            if self.opt.prediction_type == "sample":
                raw = self.recon_criterion(pred, latent)  # [B, T, J, D]
                loss_sample = (raw * self.part_weight[None, None, :, None]).mean()
                loss += loss_sample
                loss_dict["loss_sample"] = loss_sample

            elif self.opt.prediction_type == "epsilon":
                raw = self.recon_criterion(pred, noise)
                loss_eps = (raw * self.part_weight[None, None, :, None]).mean()
                loss += loss_eps
                loss_dict["loss_eps"] = loss_eps

            elif self.opt.prediction_type == "v_prediction":
                vel = self.noise_scheduler.get_velocity(latent, noise, timesteps)
                raw = self.recon_criterion(pred, vel)  # [B, T, J, D]
                loss_vel = (raw * self.part_weight[None, None, :, None]).mean()
                loss += loss_vel
                loss_dict["loss_vel"] = loss_vel
                
            else:
                raise NotImplementedError(f"Prediction type {self.opt.prediction_type} not implemented")

            # ── recon loss: decode predicted x₀ and compare in motion space ──
            if self.use_recon_loss:
                x0_pred = self._recover_x0(pred, noisy_latent, noise, timesteps)
                motion_pred = self.vae.decode(x0_pred)  # [B, T_full, 133] — VAE frozen, grad flows to denoiser
                # align lengths (VAE may upsample temporally)
                T_min = min(motion_pred.shape[1], motion.shape[1])
                loss_recon = F.l1_loss(motion_pred[:, :T_min], motion[:, :T_min])
                loss += self.lambda_recon * loss_recon
                loss_dict["loss_recon"] = loss_recon

            loss_dict["loss"] = loss

        return loss, attn_list, loss_dict
    

    @torch.no_grad()
    def generate(self, batch_data, need_attn=False):
        self.denoiser.eval()

        # setup input
        text_or_emb, motion, m_lens = batch_data

        # generate always uses text strings (eval datasets should not use cache)
        if isinstance(text_or_emb, (tuple, list)) and len(text_or_emb) == 3 and torch.is_tensor(text_or_emb[0]):
            raise ValueError(
                "generate() requires text strings, not cached embeddings. "
                "Set use_text_cache=False for eval/val datasets."
            )
        text = text_or_emb

        # to device
        motion = motion.to(self.opt.device, dtype=torch.float32)
        m_lens = m_lens.to(self.opt.device, dtype=torch.long) // 4
        len_mask = lengths_to_mask(m_lens) # [B, T]

        input_text = [""] * len(text)
        if self.opt.classifier_free_guidance:
            input_text.extend(text)
        
        # initial noise
        z, _ = self.vae.encode(motion)
        latents = torch.randn_like(z)
        latents = latents * self.noise_scheduler.init_noise_sigma

        len_mask = F.pad(len_mask, (0, latents.shape[1] - len_mask.shape[1]), mode="constant", value=False)
        latents = latents * len_mask[..., None, None].float()

        # set diffusion timesteps
        self.noise_scheduler.set_timesteps(self.opt.num_inference_timesteps)
        timesteps = self.noise_scheduler.timesteps.to(self.opt.device)

        # reverse diffusion
        skel_attn_weights, temp_attn_weights, cross_attn_weights = [], [], []
        for i, timestep in enumerate(timesteps):
            if self.opt.classifier_free_guidance:
                input_latents = torch.cat([latents] * 2, dim=0)
                input_len_mask = torch.cat([len_mask] * 2, dim=0)
            else:
                input_latents = latents
                input_len_mask = len_mask
            
            pred, attn = self.denoiser.forward(input_latents, timestep, text=input_text,
                                               len_mask=input_len_mask, need_attn=need_attn, use_cached_clip=True)

            # classifier-free guidance
            if self.opt.classifier_free_guidance:
                pred_uncond, pred_cond = torch.chunk(pred, 2, dim=0)
                pred = pred_uncond + self.opt.cond_scale * (pred_cond - pred_uncond)
            
            # step
            latents = self.noise_scheduler.step(pred, timestep, latents).prev_sample
            latents = latents * len_mask[..., None, None].float()
            
            # save attention weights
            skel_attn_weights.append(attn[0])
            temp_attn_weights.append(attn[1])
            cross_attn_weights.append(attn[2])

        # decode
        pred_motion = self.vae.decode(latents)
        if isinstance(pred_motion, tuple) or isinstance(pred_motion, list):
            pred_motion = pred_motion[0]

        # stack attention weights
        if need_attn:
            def _safe_stack(wlist):
                valid = [w for w in wlist if w is not None]
                return torch.stack(valid, dim=1) if valid else None
            skel_attn_weights = _safe_stack(skel_attn_weights)
            temp_attn_weights = _safe_stack(temp_attn_weights)
            cross_attn_weights = _safe_stack(cross_attn_weights)
            attn_weights = (skel_attn_weights, temp_attn_weights, cross_attn_weights)
        else:
            attn_weights = (None, None, None)
        
        # remove cached CLIP features
        self.denoiser.remove_clip_cache()

        return pred_motion, attn_weights
    

    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):
        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.optim.param_groups:
            param_group["lr"] = current_lr

        return current_lr
    

    def save(self, file_name, epoch, total_iter):
        state = {
            "denoiser": self.denoiser.state_dict_without_clip(),
            "optim": self.optim.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "total_iter": total_iter,
        }
        torch.save(state, file_name)


    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.opt.device)
        missing_keys, unexpected_keys = self.denoiser.load_state_dict(checkpoint["denoiser"], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith("clip_model.") for k in missing_keys])

        try:
            self.optim.load_state_dict(checkpoint["optim"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            if "scaler" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler"])
        except:
            print("Fail to load optimizer and lr_scheduler")
        return checkpoint["epoch"], checkpoint["total_iter"]


    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval=None):
        self.denoiser.to(self.opt.device)
        self.vae.to(self.opt.device)

        # optimizer
        self.optim = torch.optim.AdamW(self.denoiser.parameters(), lr=self.opt.lr, betas=(0.9, 0.99), weight_decay=self.opt.weight_decay)
        total_iters = self.opt.max_epoch * len(train_loader)
        if getattr(self.opt, 'lr_schedule', 'multistep') == 'cosine':
            eta_min = getattr(self.opt, 'eta_min', 1e-6)
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optim, T_max=total_iters - self.opt.warm_up_iter, eta_min=eta_min)
            print(f"Using CosineAnnealingLR (T_max={total_iters - self.opt.warm_up_iter}, eta_min={eta_min})")
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=self.opt.milestones, gamma=self.opt.gamma)
            print(f"Using MultiStepLR (milestones={self.opt.milestones}, gamma={self.opt.gamma})")

        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, "latest.tar")
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        print(f"Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}")
        print(f"Iters Per Epoch, Training: {len(train_loader)}, Validation: {len(eval_val_loader) if eval_val_loader else 'N/A'}")
        logs = defaultdict(def_value, OrderedDict())

        # eval
        best_fid, best_div, best_top1, best_top2, best_top3, best_matching = 1000, 100, 0, 0, 0, 100
        if eval_val_loader is not None and eval_wrapper is not None:
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer, gt_motion, gen_motion, m_length, cond_list = evaluation_denoiser(
                self.opt.model_dir, eval_val_loader, self.denoiser, self.generate, self.logger, epoch,
                best_fid=best_fid, best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3, best_matching=best_matching,
                eval_wrapper=eval_wrapper, save=True, draw=True, device=self.opt.device
            )
        # else:
        # best_fid, best_div, best_top1, best_top2, best_top3, best_matching = 1000, 100, 0, 0, 0, 100

        # training loop
        while epoch < self.opt.max_epoch:
            torch.cuda.empty_cache()
            self.denoiser.train()
            for i, batch_data in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    curr_lr = self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)
                
                # forward
                loss, attn_list, loss_dict = self.train_forward(batch_data)
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()

                if it >= self.opt.warm_up_iter:
                    self.lr_scheduler.step()
                
                # log
                logs["lr"] += self.optim.param_groups[0]["lr"]
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
                    self.save(pjoin(self.opt.model_dir, "latest.tar"), epoch, it)

            self.save(pjoin(self.opt.model_dir, "latest.tar"), epoch, it)

            epoch += 1
            print("Validation time:")
            self.denoiser.eval()
            val_log = defaultdict(def_value, OrderedDict())
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, attn_list, loss_dict = self.train_forward(batch_data)
                    for tag, value in loss_dict.items():
                        val_log[tag] += value.item()

            msg = "Validation loss:"
            for tag, value in val_log.items():
                self.logger.add_scalar("Val/%s"%tag, value / len(val_loader), epoch)
                msg += f" {tag}: {value / len(val_loader):.4f}"
            print(msg)
            
            # evaluation
            if epoch % self.opt.eval_every_e == 0:
                if eval_val_loader is not None and eval_wrapper is not None:
                    best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer, gt_motion, gen_motion, m_length, cond_list = evaluation_denoiser(
                        self.opt.model_dir, eval_val_loader, self.denoiser, self.generate, self.logger, epoch,
                        best_fid=best_fid, best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3, best_matching=best_matching,
                        eval_wrapper=eval_wrapper, save=True, draw=True, device=self.opt.device
                    )

                    data = np.concatenate([gt_motion[:4], gen_motion[:4]], axis=0)
                    length = np.concatenate([m_length[:4], m_length[:4]], axis=0)
                    cond_list = cond_list[:4] + cond_list[:4]
                    save_dir = pjoin(self.opt.eval_dir, "E%04d" % (epoch))
                    os.makedirs(save_dir, exist_ok=True)
                    plot_eval(data, save_dir, cond_list, length)
                else:
                    # sign: no eval_wrapper, save checkpoint periodically
                    val_loss = val_log.get("loss", 0) / max(len(val_loader), 1)
                    self.save(pjoin(self.opt.model_dir, f'net_epoch{epoch:03d}_loss{val_loss:.4f}.tar'), epoch, it)
    
    
    @torch.no_grad()
    def test(self, eval_wrapper, eval_val_loader, repeat_time, save_dir, cal_mm=True, save_motion=True):
        os.makedirs(save_dir, exist_ok=True)
        f = open(pjoin(save_dir, f"eval_steps{self.opt.num_inference_timesteps}_scale{self.opt.cond_scale}.log"), "w")

        self.denoiser.eval()
        self.vae.eval()
        self.noise_scheduler.set_timesteps(self.opt.num_inference_timesteps)
        metrics = {
            "fid": [],
            "div": [],
            "top1": [],
            "top2": [],
            "top3": [],
            "matching": [],
            "mm": []
        }
        for i in range(repeat_time):
            msg, fid, div, R_precision, matching, l1_dist, mm, pred_motion, caption_list = test_denoiser(
                eval_val_loader, self.generate, i, eval_wrapper, self.opt.joints_num, cal_mm=cal_mm
            )
            print(msg, file=f, flush=True)
            metrics["fid"].append(fid)
            metrics["div"].append(div)
            metrics["top1"].append(R_precision[0])
            metrics["top2"].append(R_precision[1])
            metrics["top3"].append(R_precision[2])
            metrics["matching"].append(matching)
            metrics["mm"].append(mm)

            if save_motion:
                converter = Joint2BVHConvertor()
                motion_save_dir = pjoin(save_dir, f"motion-steps{self.opt.num_inference_timesteps}-{i:02d}")
                os.makedirs(motion_save_dir, exist_ok=True)
                for i, (motion, caption) in enumerate(zip(pred_motion, caption_list)):
                    _, ik_joint = converter.convert(motion, pjoin(motion_save_dir, f"{i:06d}_ik.bvh"), foot_ik=True)
                    plot_3d_motion(pjoin(motion_save_dir, f"{i:06d}.mp4"), self.opt.kinematic_chain, motion, title=caption, fps=self.opt.fps)
                    np.savez(pjoin(motion_save_dir, f"{i:06d}.npz"), motion=motion, caption=caption)

        fid = np.array(metrics["fid"])
        div = np.array(metrics["div"])
        top1 = np.array(metrics["top1"])
        top2 = np.array(metrics["top2"])
        top3 = np.array(metrics["top3"])
        matching = np.array(metrics["matching"])
        mm = np.array(metrics["mm"])
        
        msg_final = f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}\n" \
                    f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}\n" \
                    f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}\n" \
                    f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}\n" \
                    f"\tMultimodality: {np.mean(mm):.3f}, conf. {np.std(mm)*1.96/np.sqrt(repeat_time):.3f}\n\n"
        print(msg_final)
        print(msg_final, file=f, flush=True)

        f.close()