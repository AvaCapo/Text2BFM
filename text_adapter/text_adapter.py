from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class TextAdapterOutput:
    vtxt: torch.Tensor
    ctxt: torch.Tensor
    ctxt_mask: torch.Tensor


class VtxtEncoder(nn.Module):
    """
    Builds a single-vector text embedding from token-level text embeddings.

    Expected input:
      - text_emb: [B, Lt, D_text]
      - text_mask: [B, Lt], where True means a valid token
    """

    def __init__(self, text_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(text_dim)
        self.proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        text_emb: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if text_emb.dim() != 3:
            raise ValueError(f"text_emb must be [B, Lt, D], got {tuple(text_emb.shape)}")

        x = self.norm(text_emb.float())
        if text_mask is None:
            pooled = x.mean(dim=1)
        else:
            mask = text_mask.to(device=x.device, dtype=torch.bool)
            weights = mask.unsqueeze(-1).to(dtype=x.dtype)
            denom = weights.sum(dim=1).clamp_min(1.0)
            pooled = (x * weights).sum(dim=1) / denom
        return self.proj(pooled)


class CtxtEncoder(nn.Module):
    """
    Token-level text adapter for emb outputs.

    """

    def __init__(
        self,
        text_dim: int,
        hidden_dim: int,
        out_dim: int,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.input_proj = nn.Linear(text_dim, hidden_dim)
        if self.num_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.adapter = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        else:
            self.adapter = nn.Identity()
        self.output_proj = nn.Linear(hidden_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        text_emb: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if text_emb.dim() != 3:
            raise ValueError(f"text_emb must be [B, Lt, D], got {tuple(text_emb.shape)}")

        x = self.input_proj(text_emb.float())
        key_padding_mask = None
        if text_mask is not None:
            key_padding_mask = ~text_mask.to(device=x.device, dtype=torch.bool)

        if self.num_layers > 0:
            x = self.adapter(x, src_key_padding_mask=key_padding_mask)
        else:
            x = self.adapter(x)
        x = self.output_proj(x)
        return self.norm(x)


class TextConditionAdapter(nn.Module):
    """
    Converts token-level text embeddings into:
      - vtxt: a global text vector for AdaLN/global conditioning
      - ctxt: token sequence for cross-attention

    This follows the paper's high-level idea:
      1. keep the full T5 token matrix for cross-attention
      2. refine it with a small transformer text adapter
      3. derive a single global text vector alongside token features
    """

    def __init__(
        self,
        text_dim: int,
        hidden_dim: int,
        vtxt_dim: int,
        ctxt_dim: int,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        pool_from: str = "adapted",
    ):
        super().__init__()
        self.pool_from = str(pool_from)
        if self.pool_from not in {"input", "adapted"}:
            raise ValueError(f"pool_from must be 'input' or 'adapted', got {pool_from!r}")

        self.ctxt_encoder = CtxtEncoder(
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            out_dim=ctxt_dim,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.vtxt_encoder = VtxtEncoder(
            text_dim=text_dim if self.pool_from == "input" else ctxt_dim,
            hidden_dim=hidden_dim,
            out_dim=vtxt_dim,
        )

    def forward(
        self,
        text_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> TextAdapterOutput:
        if attention_mask is None:
            attention_mask = torch.ones(
                text_emb.shape[:2],
                device=text_emb.device,
                dtype=torch.bool,
            )
        else:
            attention_mask = attention_mask.to(device=text_emb.device, dtype=torch.bool)

        ctxt = self.ctxt_encoder(text_emb=text_emb, text_mask=attention_mask)
        vtxt_source = text_emb if self.pool_from == "input" else ctxt
        vtxt = self.vtxt_encoder(text_emb=vtxt_source, text_mask=attention_mask)
        return TextAdapterOutput(vtxt=vtxt, ctxt=ctxt, ctxt_mask=attention_mask)


class MultiModalTextConditionAdapter(nn.Module):
    """
    Fuses token-level T5 embeddings with a global CLIP text vector.

    Inputs:
      - ctxt_emb: [B, Lt, D_ctxt] token matrix from T5
      - vtxt_emb: [B, D_vtxt] global text vector from CLIP
      - attention_mask: [B, Lt] bool, True means valid token

    Outputs:
      - adapted ctxt for cross-attention
      - adapted vtxt for global conditioning
    """

    def __init__(
        self,
        ctxt_dim: int,
        vtxt_dim: int,
        hidden_dim: int,
        out_vtxt_dim: int,
        out_ctxt_dim: int,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ctxt_encoder = CtxtEncoder(
            text_dim=ctxt_dim,
            hidden_dim=hidden_dim,
            out_dim=out_ctxt_dim,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.clip_proj = nn.Sequential(
            nn.LayerNorm(vtxt_dim),
            nn.Linear(vtxt_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_ctxt_dim),
        )
        self.vtxt_fuse = nn.Sequential(
            nn.LayerNorm(vtxt_dim + out_ctxt_dim),
            nn.Linear(vtxt_dim + out_ctxt_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_vtxt_dim),
        )
        self.ctxt_gate = nn.Sequential(
            nn.LayerNorm(out_ctxt_dim),
            nn.Linear(out_ctxt_dim, out_ctxt_dim),
            nn.Sigmoid(),
        )
        self.out_ctxt_norm = nn.LayerNorm(out_ctxt_dim)
        self.out_vtxt_norm = nn.LayerNorm(out_vtxt_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        ctxt_emb: torch.Tensor,
        vtxt_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> TextAdapterOutput:
        if ctxt_emb.dim() != 3:
            raise ValueError(f"ctxt_emb must be [B, Lt, D], got {tuple(ctxt_emb.shape)}")
        if vtxt_emb.dim() != 2:
            raise ValueError(f"vtxt_emb must be [B, D], got {tuple(vtxt_emb.shape)}")

        if attention_mask is None:
            attention_mask = torch.ones(
                ctxt_emb.shape[:2],
                device=ctxt_emb.device,
                dtype=torch.bool,
            )
        else:
            attention_mask = attention_mask.to(device=ctxt_emb.device, dtype=torch.bool)

        ctxt = self.ctxt_encoder(text_emb=ctxt_emb, text_mask=attention_mask)
        clip_feat = self.clip_proj(vtxt_emb.float())  # [B, D_ctxt_out]

        # Broadcast global CLIP semantics into token conditioning, but let the model gate it.
        clip_tokens = clip_feat.unsqueeze(1)
        ctxt = ctxt + self.ctxt_gate(ctxt) * clip_tokens
        ctxt = self.out_ctxt_norm(ctxt)

        token_weights = attention_mask.unsqueeze(-1).to(dtype=ctxt.dtype)
        denom = token_weights.sum(dim=1).clamp_min(1.0)
        pooled_ctxt = (ctxt * token_weights).sum(dim=1) / denom
        fused_vtxt = torch.cat([vtxt_emb.float(), pooled_ctxt], dim=-1)
        vtxt = self.out_vtxt_norm(self.vtxt_fuse(fused_vtxt))
        return TextAdapterOutput(vtxt=vtxt, ctxt=ctxt, ctxt_mask=attention_mask)


def _sanity_check() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextConditionAdapter(
        text_dim=1024,
        hidden_dim=768,
        vtxt_dim=768,
        ctxt_dim=768,
        nhead=8,
        num_layers=2,
        dropout=0.1,
    ).to(device)

    batch_size, seq_len, text_dim = 2, 12, 1024
    text_emb = torch.randn(batch_size, seq_len, text_dim, device=device)
    attention_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [1] * seq_len],
        device=device,
        dtype=torch.bool,
    )

    out = model(text_emb, attention_mask=attention_mask)
    assert out.vtxt.shape == (batch_size, 768)
    assert out.ctxt.shape == (batch_size, seq_len, 768)
    assert out.ctxt_mask.shape == (batch_size, seq_len)
    assert torch.isfinite(out.vtxt).all()
    assert torch.isfinite(out.ctxt).all()
    print("TextConditionAdapter sanity check passed")


if __name__ == "__main__":
    _sanity_check()
