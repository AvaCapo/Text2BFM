import torch
import mediapy as media
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from metamotivo.fb_cpr.huggingface import FBcprModel


class FBActionLoss(nn.Module):
    """KL Divergence Loss between two distributions"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fb_model = FBcprModel.from_pretrained("facebook/metamotivo-S-1", local_files_only=True)

    def forward(self, z1, z2, obs, mask=None):
        obs = obs.reshape(-1, obs.shape[-1])
        z1 = z1.reshape(-1, z1.shape[-1])
        z2 = z2.reshape(-1, z2.shape[-1])

        d1 = self.fb_model.actor(obs, z1, self.fb_model.cfg.actor_std)
        d2 = self.fb_model.actor(obs, z2, self.fb_model.cfg.actor_std)

        kl = self.kl_gaussian_gaussian(d1.mean, d1.variance.log(), d2.mean, d2.variance.log())

        if mask is None:
            return kl.mean()

        mask_exp = mask.reshape(-1, 1).to(dtype=kl.dtype, device=kl.device)
        denom = mask_exp.sum() * kl.shape[-1] + 1e-8
        return (kl * mask_exp).sum() / denom

    
    def kl_gaussian_gaussian(self, mean1, log_var1, mean2, log_var2):
        
        if log_var2 is None:
            log_var2 = torch.zeros_like(log_var1)
        
        var1 = log_var1.exp()
        var2 = log_var2.exp()
        
        kl = 0.5 * (log_var2 - log_var1 + (var1 + (mean1 - mean2).pow(2)) / var2 - 1)
        # kl = (mean1 - mean2).pow(2)
    
        return kl
    
