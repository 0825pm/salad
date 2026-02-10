import torch
import torch.nn as nn

from models.skeleton.linear import MultiLinear
from models.vae.encdec import (
    MotionEncoder, MotionDecoder,
    SignMotionEncoder, SignMotionDecoder,  ## PATCH
    STConvEncoder, STConvDecoder,
)

class VAE(nn.Module):
    def __init__(self, opt):
        super(VAE, self).__init__()
        self.opt = opt

        ## ── PATCH: branch on dataset ──
        if getattr(opt, 'dataset_name', 't2m') == 'sign':
            self.motion_enc = SignMotionEncoder(opt)
            self.motion_dec = SignMotionDecoder(opt)
            from utils.sign_paramUtil import get_sign_config
            _, _, _, num_parts = get_sign_config(getattr(opt, 'skeleton_mode', '7part'))
        else:
            self.motion_enc = MotionEncoder(opt)
            self.motion_dec = MotionDecoder(opt)
            num_parts = 7   # original SALAD pooled parts
        ## ── end PATCH ──

        self.conv_enc = STConvEncoder(opt)
        self.conv_dec = STConvDecoder(opt, self.conv_enc)

        self.dist = MultiLinear(opt.latent_dim, opt.latent_dim * 2, num_parts)
    
    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def encode(self, x):
        x = self.motion_enc(x)
        x = self.conv_enc(x)
        x = self.dist(x)
        mu, logvar = x.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        loss_kl = 0.5 * torch.mean(torch.pow(mu, 2) + torch.exp(logvar) - logvar - 1.0)
        return z, {"loss_kl": loss_kl}
    
    def decode(self, x):
        x = self.conv_dec(x)
        x = self.motion_dec(x)
        return x
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        x = x.detach().float()
        z, loss_dict = self.encode(x)
        out = self.decode(z)
        return out, loss_dict