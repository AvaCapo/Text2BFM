import dataclasses
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np

from metamotivo.fb_cpr.model import FBcprModel, Config as FBcprConfig
from src.config import MDMAdapterConfig

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # Broadcaster logic:
        # If x is 1D (Batch) -> [Batch, Dim] (Time Embed)
        # If x is 2D (1, Seq) -> [1, Seq, Dim] (Pos Embed)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim, act_layer=nn.SiLU):
        super().__init__()
        self.sequence_pos_encoder = SinusoidalPosEmb(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            act_layer(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, t):
        return self.mlp(self.sequence_pos_encoder(t))

class MDMBlock(nn.Module):
    """
    The Core MDM Transformer Block.
    """
    def __init__(self, hidden_dim, nhead, ff_dim, dropout, activation="gelu"):
        super().__init__()
        
        # Self Attention
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        
        # Cross Attention (Conditioning on Text)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        
        # Feed Forward
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU() if activation == "gelu" else nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, key_padding_mask=None):
        """
        x: [Batch, Seq, Dim]
        context: [Batch, 1, Dim]
        key_padding_mask: [Batch, Seq] (Bool, True = Pad/Ignore)
        """
        # 1. Self Attention (Pre-Norm)
        r = self.norm1(x)
        # Fix: Use key_padding_mask for padding handling, not attn_mask
        attn_out, _ = self.self_attn(r, r, r, key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_out)
        
        # 2. Cross Attention
        r = self.norm2(x)
        attn_out, _ = self.cross_attn(r, context, context)
        x = x + self.dropout(attn_out)
        
        # 3. Feed Forward
        r = self.norm3(x)
        ffn_out = self.ffn(r)
        x = x + self.dropout(ffn_out)
        
        return x

class MDMTransformer(nn.Module):
    def __init__(self, z_dim, cfg: MDMAdapterConfig):
        super().__init__()
        self.hidden_dim = cfg.hidden_dim
        
        # Input Projection
        self.input_proj = nn.Linear(z_dim, self.hidden_dim)
        
        # Time Embedding
        self.time_embed = TimestepEmbedder(self.hidden_dim)
        
        # Positional Embedding 
        self.pos_embed = SinusoidalPosEmb(self.hidden_dim)
        
        # Text Projection
        self.text_proj = nn.Linear(cfg.text_dim, self.hidden_dim)
        
        # Transformer Backbone
        self.blocks = nn.ModuleList([
            MDMBlock(
                hidden_dim=cfg.hidden_dim, 
                nhead=cfg.nhead, 
                ff_dim=cfg.ff_dim, 
                dropout=cfg.dropout,
                activation=cfg.activation
            ) for _ in range(cfg.num_layers)
        ])
        
        # Output Projection
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, z_dim)
        
        # Zero-init output for better convergence
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x, t, text_emb, mask=None):
        """
        x: [Batch, Seq, z_dim]
        t: [Batch]
        text_emb: [Batch, text_dim]
        mask: [Batch, Seq] (Bool, True = Valid, False = Pad) - OPTIONAL
        """
        B, T, _ = x.shape
        
        # 1. Embed Input and Time
        h = self.input_proj(x)                  # [B, T, H]
        time_emb = self.time_embed(t)           # [B, H]
        
        # 2. Add Time and Positional Embeddings
        # Time (Broadcast)
        h = h + time_emb.unsqueeze(1)
        
        # Positional
        pos_indices = torch.arange(T, device=x.device).unsqueeze(0) # [1, T]
        pos_emb = self.pos_embed(pos_indices) # [1, T, H]
        h = h + pos_emb
        
        # 3. Prepare Context
        context = self.text_proj(text_emb).unsqueeze(1) # [B, 1, H]
        
        # 4. Prepare Padding Mask
        # PyTorch MHA expects True for Padding, False for Real data.
        # User mask is True for Real data. We must invert it.
        key_padding_mask = ~mask if mask is not None else None

        # 5. Pass through Transformer
        for block in self.blocks:
            h = block(h, context, key_padding_mask=key_padding_mask)
            
        # 6. Output
        h = self.final_norm(h)
        output = self.output_proj(h)
        return output

class MDMFBcprModel(FBcprModel):
    """
    Based on paper Human Motion Diffusion Model: https://arxiv.org/abs/2209.14916
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.mdm_cfg = kwargs.get('mdm', MDMAdapterConfig())
        if isinstance(self.mdm_cfg, dict):
            self.mdm_cfg = MDMAdapterConfig(**self.mdm_cfg)
            
        z_dim = self.cfg.archi.z_dim
        
        # Scale for normalization (FB-CPR z is on sphere sqrt(d))
        self.z_scale = 1.0 #math.sqrt(float(z_dim))

        # The MDM Backbone
        self.mdm_model = MDMTransformer(z_dim, self.mdm_cfg)

        # Learnable null embedding for Classifier-Free Guidance
        self.null_text_emb = nn.Parameter(torch.randn(1, self.mdm_cfg.text_dim))

        # Freeze Base FB-CPR Model
        self.requires_grad_(True)
        self._forward_map.requires_grad_(False)
        self._backward_map.requires_grad_(False)
        self._actor.requires_grad_(False)
        self._discriminator.requires_grad_(False)
        self._critic.requires_grad_(False)

        # Register Diffusion Schedule
        self.register_schedule()
        
        # Compile
        if self.mdm_cfg.compile:
            print("Compiling MDM Transformer...")
            self.mdm_model = torch.compile(self.mdm_model)

    def register_schedule(self):
        """
        Registers buffers for diffusion (alphas, betas).
        Supports 'linear' and 'cosine' schedules.
        """
        T = self.mdm_cfg.diffusion_steps
        
        if self.mdm_cfg.schedule == 'linear':
            betas = torch.linspace(self.mdm_cfg.beta_start, self.mdm_cfg.beta_end, T)
        elif self.mdm_cfg.schedule == 'cosine':
            # Cosine schedule as proposed by Nichol & Dhariwal (standard in MDM)
            s = 0.008
            steps = T + 1
            x = torch.linspace(0, T, steps)
            alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule {self.mdm_cfg.schedule}")

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    def update(self, text_emb: torch.Tensor, z_seq_gt: torch.Tensor, mask: torch.Tensor,loss:str = 'mse') -> torch.Tensor:
        """
        Main training loop (Corrected for x0-prediction and Masking).
        z_seq_gt: [Batch, Seq_Len, z_dim]
        text_emb: [Batch, text_dim]
        mask: [Batch, Seq_Len] (Bool: True=Valid, False=Pad)
        """
        device = z_seq_gt.device
        batch_size = z_seq_gt.shape[0]
        
        # 1. Normalize Z
        x_start = z_seq_gt / self.z_scale
        
        # 2. Sample Timesteps
        t = torch.randint(0, self.mdm_cfg.diffusion_steps, (batch_size,), device=device).long()
        
        # 3. Add Noise
        noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        x_noisy = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
        
        # 4. CFG Training
        if self.mdm_cfg.cfg_dropout_prob > 0:
            drop_mask = torch.rand(batch_size, 1, device=device) < self.mdm_cfg.cfg_dropout_prob
            text_emb = torch.where(drop_mask, self.null_text_emb, text_emb)

        # 5. Predict x_start
        pred_x0 = self.mdm_model(x_noisy, t, text_emb, mask=mask)
        
        # --- COSINE LOSS (with Masking) ---
        # Normalize: z_scale cancels out in division, so we can just normalize pred_x0
        z_gt_norm = F.normalize(z_seq_gt, dim=-1)
        z_pred_norm = F.normalize(pred_x0, dim=-1)

        # 1. Calculate element-wise cosine loss [Batch, Seq]
        # Dot product of normalized vectors = Cosine Similarity
        cos_sim = (z_pred_norm * z_gt_norm).sum(dim=-1)
        cos_loss_element = 1.0 - cos_sim
        
        # 2. Apply Mask
        # mask is [Batch, Seq]. Sum loss only where mask is True.
        cos_loss = (cos_loss_element * mask).sum() / (mask.sum() + 1e-8)

        # --- MSE LOSS (with Masking) ---
        if loss == 'mse':
            loss_unreduced = F.mse_loss(pred_x0, x_start, reduction='none') # [B, T, D]
            mask_expanded = mask.unsqueeze(-1) # [B, T, 1]
            m_loss = (loss_unreduced * mask_expanded).sum() / (mask_expanded.sum() * x_start.shape[-1] + 1e-8)
        else:
            loss_unreduced = F.l1_loss(pred_x0, x_start, reduction='none')  # [B, T, D]
            mask_expanded = mask.unsqueeze(-1)                              # [B, T, 1]

            m_loss = (loss_unreduced * mask_expanded).sum() / (
                mask_expanded.sum() * x_start.shape[-1] + 1e-8
            )
                    # 7. Total Loss
        return m_loss + cos_loss

    @torch.no_grad()
    def tracking_inference(self, text_emb: torch.Tensor, seq_length: int = 214, guidance_scale: float | None = None) -> torch.Tensor:
        """
        DDIM Sampling adapted for x0-prediction.
        """
        device = text_emb.device
        batch_size = text_emb.shape[0]
        if guidance_scale is None:
            guidance_scale = self.mdm_cfg.guidance_scale
        
        # 1. Start from pure Gaussian Noise
        x_t = torch.randn((batch_size, seq_length, self.cfg.archi.z_dim), device=device)
        
        # 2. CFG Setup
        uncond_emb = self.null_text_emb.expand(batch_size, -1)
        text_emb_cat = torch.cat([text_emb, uncond_emb], dim=0)
        
        inference_steps = self.mdm_cfg.inference_steps
        total_steps = self.mdm_cfg.diffusion_steps
        
        times = torch.linspace(total_steps - 1, 0, inference_steps + 1).long().to(device)
        time_pairs = list(zip(times[:-1], times[1:])) 
        
        # No padding mask needed for inference (generation is fixed length)
        
        for t, t_prev in tqdm(time_pairs, desc="MDM Sampling (x0)", leave=False):
            t_tensor = torch.full((batch_size * 2,), t, device=device, dtype=torch.long)
            x_t_cat = torch.cat([x_t, x_t], dim=0)
            
            # 3. Predict Clean Signal (x0) directly
            pred_x0_cat = self.mdm_model(x_t_cat, t_tensor, text_emb_cat, mask=None)
            pred_x0_cond, pred_x0_uncond = torch.chunk(pred_x0_cat, 2, dim=0)
            
            # 4. Apply Guidance in Signal Space
            # "We guide the signal prediction directly"
            pred_x0 = pred_x0_uncond + guidance_scale * (pred_x0_cond - pred_x0_uncond)
            pred_x0 = F.normalize(pred_x0, dim=-1) * math.sqrt(pred_x0.shape[-1])
            
            # 5. DDIM Update using predicted x0
            alpha_bar_t = self.alphas_cumprod[t]
            alpha_bar_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
            
            # Derive implied noise (epsilon) from the predicted x0
            eps_implied = (x_t - torch.sqrt(alpha_bar_t) * pred_x0) / torch.sqrt(1 - alpha_bar_t)
            
            # Calculate x_{t-1} using standard DDIM formula
            dir_xt = torch.sqrt(1 - alpha_bar_prev) * eps_implied
            x_t = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

        # 6. Final Projection
        z_seq = x_t * self.z_scale
        z_seq_flat = z_seq.view(-1, self.cfg.archi.z_dim)
        z_seq_proj = self.project_z(z_seq_flat)
        return z_seq_proj.view(batch_size, seq_length, -1)

    @torch.no_grad()
    def gt_tracking_inference(self, next_obs: torch.Tensor) -> torch.Tensor:
        """
        Helper for evaluation: Get GT z-sequence from obs.
        """
        z = self.backward_map(next_obs)
        return self.project_z(z)