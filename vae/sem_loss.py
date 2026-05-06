from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mask_fill_value(dtype: torch.dtype) -> float:
    if dtype in (torch.float16, torch.bfloat16):
        return -1e4
    return -1e9


class MotionTextScorer(nn.Module):
    """
    Vectorized motion-text scorer for semantic contrastive training.

    Inputs:
      - motion: [B, Tm, Dm]
      - motion_mask: [B, Tm] bool
      - text: [B, Tt, Dt]
      - text_mask: [B, Tt] bool

    Notes:
      - Transformer and learnable frame-importance head are removed
        to reduce overfitting.
      - Frame importance is computed non-parametrically from frame scores.
    """

    def __init__(
        self,
        motion_dim: int,
        text_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,          # kept for backward compatibility
        num_motion_layers: int = 2,  # kept for backward compatibility
        max_motion_len: int = 256,   # kept for backward compatibility
        dropout: float = 0.15,
        temperature_init: float = 0.07,
        frame_score_mode: str = "logsumexp",
        token_pool_tau: float = 0.1,
        frame_pool_tau: float = 0.2,
    ):
        super().__init__()
        if frame_score_mode not in {"logsumexp", "max"}:
            raise ValueError("frame_score_mode must be either 'logsumexp' or 'max'.")
        if temperature_init <= 0:
            raise ValueError("temperature_init must be > 0.")
        if token_pool_tau <= 0:
            raise ValueError("token_pool_tau must be > 0.")
        if frame_pool_tau <= 0:
            raise ValueError("frame_pool_tau must be > 0.")

        self.motion_dim = int(motion_dim)
        self.text_dim = int(text_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_motion_len = int(max_motion_len)
        self.frame_score_mode = str(frame_score_mode)
        self.token_pool_tau = float(token_pool_tau)
        self.frame_pool_tau = float(frame_pool_tau)

        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature_init)))

        self.motion_proj = nn.Sequential(
            nn.Linear(self.motion_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(dropout),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(dropout),
        )

    def encode_motion(self, motion: torch.Tensor, motion_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if motion.ndim != 3:
            raise ValueError(f"`motion` must be [B, T, D], got {tuple(motion.shape)}.")
        if motion_mask.shape != motion.shape[:2]:
            raise ValueError("`motion` and `motion_mask` shapes are incompatible.")
        if motion.shape[-1] != self.motion_dim:
            raise ValueError(f"Expected motion feature dim {self.motion_dim}, got {motion.shape[-1]}.")
        if motion_mask.dtype != torch.bool:
            motion_mask = motion_mask.bool()

        h = self.motion_proj(motion)

        return {
            "hidden": h,
            "norm": F.normalize(h, dim=-1),
            "mask": motion_mask,
        }

    def encode_text(self, text: torch.Tensor, text_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if text.ndim != 3:
            raise ValueError(f"`text` must be [B, T, D], got {tuple(text.shape)}.")
        if text_mask.shape != text.shape[:2]:
            raise ValueError("`text` and `text_mask` shapes are incompatible.")
        if text.shape[-1] != self.text_dim:
            raise ValueError(f"Expected text feature dim {self.text_dim}, got {text.shape[-1]}.")
        if text_mask.dtype != torch.bool:
            text_mask = text_mask.bool()

        h = self.text_proj(text)
        return {
            "hidden": h,
            "norm": F.normalize(h, dim=-1),
            "mask": text_mask,
        }

    def compute_similarity_matrix(
        self,
        motion_repr_batch: dict[str, torch.Tensor],
        text_repr_batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        motion = motion_repr_batch["norm"]       # [B, Tm, H]
        motion_mask = motion_repr_batch["mask"]  # [B, Tm]
        text = text_repr_batch["norm"]           # [B, Tt, H]
        text_mask = text_repr_batch["mask"]      # [B, Tt]

        sim = torch.einsum("bmh,cth->bcmt", motion, text)  # [B, B, Tm, Tt]
        fill_value = _mask_fill_value(sim.dtype)

        sim = sim.masked_fill(~motion_mask[:, None, :, None], fill_value)
        sim = sim.masked_fill(~text_mask[None, :, None, :], fill_value)

        if self.frame_score_mode == "max":
            frame_scores = sim.max(dim=-1).values
        else:
            frame_scores = self.token_pool_tau * torch.logsumexp(sim / self.token_pool_tau, dim=-1)

        frame_scores = torch.where(
            motion_mask[:, None, :],
            frame_scores,
            torch.zeros_like(frame_scores),
        )

        # Non-parametric frame importance:
        # frames with higher frame-text scores receive larger weights.
        frame_logits = (frame_scores / self.frame_pool_tau).masked_fill(
            ~motion_mask[:, None, :],
            fill_value,
        )
        frame_weights = torch.softmax(frame_logits, dim=-1)
        frame_weights = frame_weights * motion_mask[:, None, :].float()
        frame_weights = frame_weights / frame_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        raw_scores = (frame_scores * frame_weights).sum(dim=-1)  # [B, B]
        return raw_scores, frame_weights


class MotionTextContrastiveLoss(nn.Module):
    def __init__(
        self,
        motion_dim: int,
        text_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_motion_layers: int = 2,
        max_motion_len: int = 256,
        dropout: float = 0.15,
        temperature_init: float = 0.07,
        frame_score_mode: str = "logsumexp",
        token_pool_tau: float = 0.1,
        frame_pool_tau: float = 0.2,
        importance_entropy_weight: float = 0.0,
        max_logit_scale: float = 100.0,
    ):
        super().__init__()
        self.scorer = MotionTextScorer(
            motion_dim=motion_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_motion_layers=num_motion_layers,
            max_motion_len=max_motion_len,
            dropout=dropout,
            temperature_init=temperature_init,
            frame_score_mode=frame_score_mode,
            token_pool_tau=token_pool_tau,
            frame_pool_tau=frame_pool_tau,
        )
        self.importance_entropy_weight = float(importance_entropy_weight)
        self.max_logit_scale = float(max_logit_scale)

    def _importance_entropy_loss(self, importance: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        p = importance.clamp_min(1e-8)
        entropy = -(p * p.log())
        entropy = entropy.masked_fill(~mask, 0.0)
        return entropy.sum(dim=1).mean()

    def forward(
        self,
        motion: torch.Tensor,
        text: torch.Tensor,
        motion_mask: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, Any]]:

        batch_size = motion.shape[0]
        labels = torch.arange(batch_size, device=motion.device)

        if motion_mask.dtype != torch.bool:
            motion_mask = motion_mask.bool()
        if text_mask.dtype != torch.bool:
            text_mask = text_mask.bool()

        motion_repr = self.scorer.encode_motion(motion, motion_mask)
        text_repr = self.scorer.encode_text(text, text_mask)
        raw_sim_matrix, frame_weights = self.scorer.compute_similarity_matrix(motion_repr, text_repr)

        logit_scale = self.scorer.logit_scale.exp().clamp(max=self.max_logit_scale)
        logits = raw_sim_matrix * logit_scale

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        contrastive_loss = 0.5 * (loss_i2t + loss_t2i)

        diag_idx = torch.arange(batch_size, device=motion.device)
        positive_frame_weights = frame_weights[diag_idx, diag_idx]  # [B, Tm]

        if self.importance_entropy_weight > 0:
            importance_reg = self._importance_entropy_loss(
                positive_frame_weights,
                motion_mask,
            )
        else:
            importance_reg = torch.zeros((), device=motion.device)

        total_loss = contrastive_loss + self.importance_entropy_weight * importance_reg

        with torch.no_grad():
            pred_i2t = logits.argmax(dim=1)
            pred_t2i = logits.argmax(dim=0)
            acc_i2t = (pred_i2t == labels).float().mean()
            acc_t2i = (pred_t2i == labels).float().mean()

            metrics = {
                "loss": float(total_loss.item()),
                "contrastive_loss": float(contrastive_loss.item()),
                "importance_reg": float(importance_reg.item()),
                "acc_i2t": float(acc_i2t.item()),
                "acc_t2i": float(acc_t2i.item()),
                "logit_scale": float(logit_scale.item()),
            }

        details = {
            "raw_sim_matrix": raw_sim_matrix.detach(),
            "logits": logits.detach(),
            "motion_importance": positive_frame_weights.detach(),  # name kept for compatibility
        }
        return total_loss, metrics, details