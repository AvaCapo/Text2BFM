import json
import math
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hymotion.network.hymotion_mmdit import HunyuanMotionMMDiT
from metamotivo.fb_cpr.model import FBcprModel
from src.config import GeneratorAdapterConfig
from text_adapter.text_adapter import MultiModalTextConditionAdapter
from vae.config import VAEConfig
from vae.vae import VAE


def _load_vae_config(path: str | Path) -> VAEConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"VAE config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return VAEConfig(**payload)


def _extract_vae_state_dict(payload: object) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload).__name__}")

    state_dict = payload.get("model_state", payload)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported state_dict type: {type(state_dict).__name__}")

    vae_prefixed = {
        key[len("vae."):]: value
        for key, value in state_dict.items()
        if isinstance(key, str) and key.startswith("vae.")
    }
    if vae_prefixed:
        return vae_prefixed

    return state_dict


def _extract_backbone_state_dict(payload: object) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload).__name__}")

    state_dict = payload
    for key in ("model_state", "model_state_dict", "state_dict"):
        if key in payload:
            state_dict = payload[key]
            break

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported state_dict type: {type(state_dict).__name__}")

    prefixes = (
        "motion_transformer.",
        "generator_model.",
        "train_modules.generator_model.",
        "module.motion_transformer.",
        "module.generator_model.",
        "module.train_modules.generator_model.",
    )
    for prefix in prefixes:
        stripped = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if isinstance(key, str) and key.startswith(prefix)
        }
        if stripped:
            return stripped

    return state_dict



class EMA:
    """
    Exponential Moving Average (EMA) of model parameters.
    This is not present in MoLingo, which gives us an advantage.
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        self._init_from(model)

    def _init_from(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if p.requires_grad:
                # Keep the shadow weights on the same device as the parameter.
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = p.detach().clone()
            else:
                # Move the shadow weights if the parameter lives on a different device.
                if self.shadow[name].device != p.device:
                    self.shadow[name] = self.shadow[name].to(p.device)
                self.shadow[name].mul_(d).add_(p.detach(), alpha=(1.0 - d))

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        self.backup = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if name in self.shadow
        }

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name].to(p.device).data)

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in getattr(self, "backup", {}):
                p.data.copy_(self.backup[name].data)
        self.backup = {}




class GeneratorFBcprModelVAE(FBcprModel):
    """
    HY-Motion style flow-matching generator operating in the latent space of the local VAE.

    Interface matches the current two-branch text pipeline:
    - `update(...)` for train/val loops
    - `tracking_inference(...)` for videos and benchmark evaluation
    - EMA helpers and gradient clipping for checkpoint compatibility
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.generator_cfg = kwargs.get("generator")
        if self.generator_cfg is None:
            self.generator_cfg = GeneratorAdapterConfig()
        if isinstance(self.generator_cfg, dict):
            self.generator_cfg = GeneratorAdapterConfig(**self.generator_cfg)

        self.eps = float(getattr(self.generator_cfg, "eps", 1e-8))
        self.grad_clip_norm = float(getattr(self.generator_cfg, "grad_clip_norm", 1.0))
        self.use_ema = bool(getattr(self.generator_cfg, "use_ema", True))
        self.ema_decay = float(getattr(self.generator_cfg, "ema_decay", 0.9999))
        self.lambda_flow = float(getattr(self.generator_cfg, "lambda_flow", 1.0))
        self.lambda_m_recon = float(getattr(self.generator_cfg, "lambda_m_recon", 1.0))
        self.lambda_z_recon = float(getattr(self.generator_cfg, "lambda_z_recon", 1.0))
        self.lambda_z_vel = float(getattr(self.generator_cfg, "lambda_z_vel", 0.0))
        self.lambda_policy = float(getattr(self.generator_cfg, "lambda_policy", 0.0))
        self.z_scale = float(getattr(self.generator_cfg, "z_scale", 1.0))
        self.cond_mask_prob = float(
            getattr(
                self.generator_cfg,
                "cond_mask_prob",
                getattr(self.generator_cfg, "cfg_dropout_prob", 0.0),
            )
        )

        self.register_buffer("global_step", torch.zeros((), dtype=torch.long), persistent=True)

        self.vae_cfg = _load_vae_config(self.generator_cfg.vae_config_path)
        self.vae = VAE(
            input_width=int(self.vae_cfg.input_width),
            output_emb_width=int(self.vae_cfg.latent_dim),
            down_t=int(self.vae_cfg.down_t),
            stride_t=int(self.vae_cfg.stride_t),
            width=int(self.vae_cfg.width),
            depth=int(self.vae_cfg.depth),
            dilation_growth_rate=int(self.vae_cfg.dilation_growth_rate),
            activation=str(self.vae_cfg.activation),
            norm=self.vae_cfg.norm,
            pad_mode=str(self.vae_cfg.pad_mode),
            ae=bool(self.vae_cfg.ae),
        )
        self._load_vae_checkpoint(self.generator_cfg.vae_checkpoint_path)
        self.m_dim = int(self.vae.latent_dim)

        self.text_adapter: Optional[MultiModalTextConditionAdapter]
        if bool(getattr(self.generator_cfg, "use_text_adapter", True)):
            self.text_adapter = MultiModalTextConditionAdapter(
                ctxt_dim=int(self.generator_cfg.raw_ctxt_dim),
                vtxt_dim=int(self.generator_cfg.raw_vtxt_dim),
                hidden_dim=int(
                    getattr(
                        self.generator_cfg,
                        "text_adapter_hidden_dim",
                        getattr(self.generator_cfg, "feat_dim", self.generator_cfg.hidden_dim),
                    )
                ),
                out_vtxt_dim=int(self.generator_cfg.vtxt_dim),
                out_ctxt_dim=int(self.generator_cfg.ctxt_dim),
                nhead=int(getattr(self.generator_cfg, "text_adapter_nhead", 8)),
                num_layers=int(getattr(self.generator_cfg, "text_adapter_num_layers", 2)),
                dropout=float(getattr(self.generator_cfg, "text_adapter_dropout", 0.1)),
            )
        else:
            self.text_adapter = None

        self.generator_model = HunyuanMotionMMDiT(
            input_dim=self.m_dim,
            feat_dim=int(getattr(self.generator_cfg, "feat_dim", self.generator_cfg.hidden_dim)),
            output_dim=self.m_dim,
            ctxt_input_dim=int(self.generator_cfg.ctxt_dim),
            vtxt_input_dim=int(self.generator_cfg.vtxt_dim),
            num_layers=int(self.generator_cfg.num_layers),
            num_heads=int(self.generator_cfg.nhead),
            mlp_ratio=float(getattr(self.generator_cfg, "mlp_ratio", 4.0)),
            mlp_act_type=str(getattr(self.generator_cfg, "mlp_act_type", "gelu_tanh")),
            qk_norm_type=getattr(self.generator_cfg, "qk_norm_type", "rms"),
            qkv_bias=bool(getattr(self.generator_cfg, "qkv_bias", True)),
            dropout=float(self.generator_cfg.dropout),
            mask_mode=getattr(self.generator_cfg, "mask_mode", "narrowband"),
            apply_rope_to_single_branch=bool(
                getattr(self.generator_cfg, "apply_rope_to_single_branch", True)
            ),
            insert_start_token=bool(getattr(self.generator_cfg, "insert_start_token", False)),
            with_long_skip_connection=bool(
                getattr(self.generator_cfg, "with_long_skip_connection", False)
            ),
            time_factor=float(getattr(self.generator_cfg, "time_factor", 1.0)),
            narrowband_length=float(getattr(self.generator_cfg, "narrowband_length", 2.0)),
        )

        self.null_vtxt = nn.Parameter(torch.randn(1, 1, int(self.generator_cfg.vtxt_dim)))
        self.null_ctxt = nn.Parameter(
            torch.randn(1, int(self.generator_cfg.null_ctxt_len), int(self.generator_cfg.ctxt_dim))
        )

        self.requires_grad_(True)
        self._forward_map.requires_grad_(False)
        self._backward_map.requires_grad_(False)
        self._actor.requires_grad_(False)
        self._discriminator.requires_grad_(False)
        self._critic.requires_grad_(False)

        self.freeze_vae_encoder = bool(getattr(self.generator_cfg, "freeze_vae_encoder", True))
        self.freeze_vae_decoder = bool(getattr(self.generator_cfg, "freeze_vae_decoder", True))
        self._freeze_vae_parts()

        backbone_checkpoint_path = str(getattr(self.generator_cfg, "backbone_checkpoint_path", "")).strip()
        if backbone_checkpoint_path:
            self._load_backbone_checkpoint(
                backbone_checkpoint_path,
                strict=bool(getattr(self.generator_cfg, "backbone_checkpoint_strict", False)),
            )

        if bool(getattr(self.generator_cfg, "compile", False)):
            print("Compiling HY-Motion generator backbone...")
            self.generator_model = torch.compile(self.generator_model)

        self.train_modules = nn.ModuleDict({"generator_model": self.generator_model})
        if self.text_adapter is not None:
            self.train_modules["text_adapter"] = self.text_adapter

        self.ema: Optional[EMA] = EMA(self.train_modules, decay=self.ema_decay) if self.use_ema else None
        self.latest_metrics: dict[str, float] = {}

        self.train(True)

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "vae"):
            if self.freeze_vae_encoder and self.freeze_vae_decoder:
                self.vae.eval()
            else:
                self.vae.train(mode)

        for module_name in (
            "_forward_map",
            "_backward_map",
            "_actor",
            "_discriminator",
            "_critic",
            "_obs_normalizer",
        ):
            if hasattr(self, module_name):
                getattr(self, module_name).eval()
        return self

    def _load_vae_checkpoint(self, ckpt_path: str | Path) -> None:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"VAE checkpoint not found: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = _extract_vae_state_dict(payload)
        missing, unexpected = self.vae.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[GeneratorVAE] Missing VAE keys: {missing}")
        if unexpected:
            print(f"[GeneratorVAE] Unexpected VAE keys: {unexpected}")

    def _load_backbone_checkpoint(self, ckpt_path: str | Path, strict: bool = False) -> None:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Generator backbone checkpoint not found: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw_state_dict = _extract_backbone_state_dict(payload)
        state_dict = raw_state_dict
        skipped_shape_keys: list[str] = []
        if not strict:
            model_state = self.generator_model.state_dict()
            filtered_state_dict = {}
            for key, value in raw_state_dict.items():
                if key not in model_state:
                    continue
                target = model_state[key]
                if torch.is_tensor(value) and torch.is_tensor(target) and value.shape != target.shape:
                    skipped_shape_keys.append(key)
                    continue
                filtered_state_dict[key] = value
            state_dict = filtered_state_dict
        incompatible = self.generator_model.load_state_dict(state_dict, strict=strict)
        print(f"[GeneratorVAE] Loaded backbone init checkpoint: {ckpt_path}")
        print(
            f"  missing_keys={len(incompatible.missing_keys)}  "
            f"unexpected_keys={len(incompatible.unexpected_keys)}"
        )
        if skipped_shape_keys:
            print(f"  skipped_shape_mismatch={len(skipped_shape_keys)}")

    def _freeze_vae_parts(self) -> None:
        if self.freeze_vae_encoder:
            self.vae.encoder.requires_grad_(False)
            self.vae.post_proj.requires_grad_(False)
        if self.freeze_vae_decoder:
            self.vae.decoder.requires_grad_(False)
        if self.freeze_vae_encoder and self.freeze_vae_decoder:
            self.vae.eval()

    @staticmethod
    def _normalize_vtxt(vtxt_input: torch.Tensor) -> torch.Tensor:
        if vtxt_input.dim() == 2:
            return vtxt_input
        if vtxt_input.dim() == 3:
            return vtxt_input[:, 0, :] if vtxt_input.shape[1] == 1 else vtxt_input.mean(dim=1)
        raise ValueError(f"vtxt_input must be 2D or 3D, got {tuple(vtxt_input.shape)}")

    @staticmethod
    def _normalize_ctxt(ctxt_input: torch.Tensor) -> torch.Tensor:
        if ctxt_input.dim() == 3:
            return ctxt_input
        if ctxt_input.dim() == 2:
            return ctxt_input.unsqueeze(1)
        raise ValueError(f"ctxt_input must be 2D or 3D, got {tuple(ctxt_input.shape)}")

    @staticmethod
    def _prepare_vtxt_tokens(vtxt_input: torch.Tensor) -> torch.Tensor:
        if vtxt_input.dim() == 2:
            return vtxt_input.unsqueeze(1)
        if vtxt_input.dim() == 3:
            if vtxt_input.shape[1] == 1:
                return vtxt_input
            return vtxt_input.mean(dim=1, keepdim=True)
        raise ValueError(f"vtxt_input must be 2D or 3D, got {tuple(vtxt_input.shape)}")

    @staticmethod
    def _infer_ctxt_mask(ctxt_input: torch.Tensor) -> torch.Tensor:
        return ctxt_input.abs().sum(dim=-1) > 0

    @staticmethod
    def _make_true_mask(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)

    def _pad_ctxt_to_len(
        self,
        ctxt: torch.Tensor,
        mask: torch.Tensor,
        target_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, dim = ctxt.shape
        if seq_len == target_len:
            return ctxt, mask
        if seq_len > target_len:
            return ctxt[:, :target_len, :], mask[:, :target_len]
        pad_len = target_len - seq_len
        pad_ctxt = torch.zeros((bsz, pad_len, dim), device=ctxt.device, dtype=ctxt.dtype)
        pad_mask = torch.zeros((bsz, pad_len), device=mask.device, dtype=torch.bool)
        return torch.cat([ctxt, pad_ctxt], dim=1), torch.cat([mask, pad_mask], dim=1)

    def _align_ctxt_mask(self, ctxt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.shape[1] == ctxt.shape[1]:
            return mask
        if mask.shape[1] < ctxt.shape[1]:
            pad = torch.zeros((mask.shape[0], ctxt.shape[1] - mask.shape[1]), device=mask.device, dtype=torch.bool)
            return torch.cat([mask, pad], dim=1)
        return mask[:, : ctxt.shape[1]]

    def _mask_to_latent_mask(self, x_mask: torch.Tensor, target_len: int) -> torch.Tensor:
        mask = x_mask.float().unsqueeze(1)
        resized = F.interpolate(mask, size=target_len, mode="nearest")
        return resized.squeeze(1) > 0.5

    def _encode_motion(self, z_seq_gt: torch.Tensor) -> torch.Tensor:
        if self.freeze_vae_encoder:
            with torch.no_grad():
                if self.vae.ae:
                    return self.vae.ae_encode(z_seq_gt)
                _, dist = self.vae.encode(z_seq_gt)
                return dist.loc

        if self.vae.ae:
            return self.vae.ae_encode(z_seq_gt)
        _, dist = self.vae.encode(z_seq_gt)
        return dist.loc

    def _decode_motion(self, m_seq: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(m_seq)

    def _policy_loss(
        self,
        z_pred: torch.Tensor,
        z_gt: torch.Tensor,
        z_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.lambda_policy <= 0:
            return z_pred.new_tensor(0.0)

        batch_size, seq_len, z_dim = z_pred.shape
        flat_pred = z_pred.reshape(batch_size * seq_len, z_dim)
        flat_gt = z_gt.reshape(batch_size * seq_len, z_dim)
        flat_mask = z_mask.reshape(batch_size * seq_len)

        obs = torch.zeros(
            (batch_size * seq_len, int(self.cfg.obs_dim)),
            device=z_pred.device,
            dtype=z_pred.dtype,
        )

        pred_action = self._actor(self._normalize(obs), flat_pred, std=0.0)
        gt_action = self._actor(self._normalize(obs), flat_gt, std=0.0)

        if isinstance(pred_action, tuple):
            pred_action = pred_action[0]
        if isinstance(gt_action, tuple):
            gt_action = gt_action[0]
        if hasattr(pred_action, "mean"):
            pred_action = pred_action.mean if not callable(pred_action.mean) else pred_action.mean
        if hasattr(gt_action, "mean"):
            gt_action = gt_action.mean if not callable(gt_action.mean) else gt_action.mean

        loss = (pred_action - gt_action).pow(2).mean(dim=-1)
        return (loss * flat_mask.float()).sum() / (flat_mask.float().sum() + self.eps)

    def _adapt_text(
        self,
        vtxt_input: torch.Tensor,
        ctxt_input: torch.Tensor,
        ctxt_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.text_adapter is None:
            return vtxt_input, ctxt_input, ctxt_mask
        adapted = self.text_adapter(ctxt_emb=ctxt_input, vtxt_emb=vtxt_input, attention_mask=ctxt_mask)
        return adapted.vtxt, adapted.ctxt, adapted.ctxt_mask

    def _cfg_dropout(
        self,
        vtxt_input: torch.Tensor,
        ctxt_input: torch.Tensor,
        ctxt_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cond_mask_prob <= 0:
            return vtxt_input, ctxt_input, ctxt_mask

        batch_size = vtxt_input.shape[0]
        device = vtxt_input.device
        drop = torch.rand(batch_size, device=device) < self.cond_mask_prob
        if not drop.any():
            return vtxt_input, ctxt_input, ctxt_mask

        vtxt_input = vtxt_input.clone()
        ctxt_input = ctxt_input.clone()
        ctxt_mask = ctxt_mask.clone()

        num_drop = int(drop.sum().item())
        ctxt_len = int(ctxt_input.shape[1])
        vtxt_input[drop] = self.null_vtxt.to(device=device, dtype=vtxt_input.dtype).expand(num_drop, -1, -1)

        null_ctxt = self.null_ctxt.to(device=device, dtype=ctxt_input.dtype).expand(
            num_drop,
            int(self.generator_cfg.null_ctxt_len),
            -1,
        )
        null_mask = self._make_true_mask(num_drop, int(self.generator_cfg.null_ctxt_len), device)
        null_ctxt, null_mask = self._pad_ctxt_to_len(null_ctxt, null_mask, target_len=ctxt_len)
        ctxt_input[drop] = null_ctxt
        ctxt_mask[drop] = null_mask
        return vtxt_input, ctxt_input, ctxt_mask

    @staticmethod
    def _flow_to_clean(x_t: torch.Tensor, flow_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x_t + (1.0 - t).view(-1, 1, 1) * flow_pred

    def update(
        self,
        vtxt_input: torch.Tensor,
        ctxt_input: torch.Tensor,
        ctxt_mask: Optional[torch.Tensor],
        z_seq_gt: torch.Tensor,
        x_mask: torch.Tensor,
        loss: str = "mse",
    ) -> torch.Tensor:
        device = z_seq_gt.device
        batch_size = int(z_seq_gt.shape[0])
        x_mask = x_mask.to(device).bool()

        vtxt_input = self._normalize_vtxt(vtxt_input).to(device)
        ctxt_input = self._normalize_ctxt(ctxt_input).to(device)
        if ctxt_mask is None:
            ctxt_mask = self._infer_ctxt_mask(ctxt_input)
        else:
            if ctxt_mask.dim() == 1:
                ctxt_mask = ctxt_mask.unsqueeze(1)
            ctxt_mask = ctxt_mask.to(device).bool()
        ctxt_mask = self._align_ctxt_mask(ctxt_input, ctxt_mask)

        vtxt_input, ctxt_input, ctxt_mask = self._adapt_text(vtxt_input, ctxt_input, ctxt_mask)
        vtxt_tokens = self._prepare_vtxt_tokens(vtxt_input)

        m_gt = self._encode_motion(z_seq_gt)
        m_mask = self._mask_to_latent_mask(x_mask, target_len=m_gt.shape[1])
        x_start = m_gt / self.z_scale

        noise = torch.randn_like(x_start)
        t = torch.rand(batch_size, device=device, dtype=x_start.dtype)
        x_t = (1.0 - t).view(-1, 1, 1) * noise + t.view(-1, 1, 1) * x_start
        target_flow = x_start - noise

        vtxt_tokens, ctxt_input, ctxt_mask = self._cfg_dropout(vtxt_tokens, ctxt_input, ctxt_mask)
        flow_pred = self.generator_model(
            x=x_t,
            ctxt_input=ctxt_input,
            vtxt_input=vtxt_tokens,
            timesteps=t,
            x_mask_temporal=m_mask,
            ctxt_mask_temporal=ctxt_mask,
        )
        pred_m0 = self._flow_to_clean(x_t=x_t, flow_pred=flow_pred, t=t)

        m_mask_exp = m_mask.unsqueeze(-1).float()
        if str(loss).lower() == "mse":
            flow_loss = ((flow_pred - target_flow).pow(2) * m_mask_exp).sum() / (
                m_mask_exp.sum() * target_flow.shape[-1] + self.eps
            )
            m_pred_loss = ((pred_m0 - x_start).pow(2) * m_mask_exp).sum() / (
                m_mask_exp.sum() * x_start.shape[-1] + self.eps
            )
        else:
            flow_loss = ((flow_pred - target_flow).abs() * m_mask_exp).sum() / (
                m_mask_exp.sum() * target_flow.shape[-1] + self.eps
            )
            m_pred_loss = ((pred_m0 - x_start).abs() * m_mask_exp).sum() / (
                m_mask_exp.sum() * x_start.shape[-1] + self.eps
            )

        m_cos = F.cosine_similarity(
            F.normalize(pred_m0.float(), dim=-1, eps=self.eps),
            F.normalize(x_start.float(), dim=-1, eps=self.eps),
            dim=-1,
        )
        m_cos_loss = ((1.0 - m_cos) * m_mask.float()).sum() / (m_mask.float().sum() + self.eps)

        pred_m = pred_m0 * self.z_scale
        z_pred = self._decode_motion(pred_m)
        seq_len = min(z_pred.shape[1], z_seq_gt.shape[1], x_mask.shape[1])
        z_pred = z_pred[:, :seq_len, :]
        z_gt = z_seq_gt[:, :seq_len, :]
        z_mask = x_mask[:, :seq_len]
        z_mask_exp = z_mask.unsqueeze(-1).float()

        if str(loss).lower() == "mse":
            z_recon_loss = ((z_pred - z_gt).pow(2) * z_mask_exp).sum() / (
                z_mask_exp.sum() * z_gt.shape[-1] + self.eps
            )
        else:
            z_recon_loss = ((z_pred - z_gt).abs() * z_mask_exp).sum() / (
                z_mask_exp.sum() * z_gt.shape[-1] + self.eps
            )

        if self.lambda_z_vel > 0 and seq_len > 1:
            z_vel = z_gt[:, 1:, :] - z_gt[:, :-1, :]
            pred_vel = z_pred[:, 1:, :] - z_pred[:, :-1, :]
            vel_mask = (z_mask[:, 1:] & z_mask[:, :-1]).unsqueeze(-1).float()
            z_vel_loss = ((pred_vel - z_vel).pow(2) * vel_mask).sum() / (
                vel_mask.sum() * z_gt.shape[-1] + self.eps
            )
        else:
            z_vel_loss = torch.tensor(0.0, device=device)

        policy_loss = self._policy_loss(z_pred=z_pred, z_gt=z_gt, z_mask=z_mask)

        total_loss = (
            self.lambda_flow * flow_loss
            + self.lambda_m_recon * (m_pred_loss + m_cos_loss)
            + self.lambda_z_recon * z_recon_loss
        )
        if self.lambda_z_vel > 0:
            total_loss = total_loss + self.lambda_z_vel * z_vel_loss
        if self.lambda_policy > 0:
            total_loss = total_loss + self.lambda_policy * policy_loss

        self.latest_metrics = {
            "total": float(total_loss.detach().item()),
            "flow": float(flow_loss.detach().item()),
            "m_recon": float(m_pred_loss.detach().item()),
            "m_cos": float(m_cos_loss.detach().item()),
            "z_recon": float(z_recon_loss.detach().item()),
            "z_vel": float(z_vel_loss.detach().item()),
            "policy": float(policy_loss.detach().item()),
        }
        if self.training:
            self.global_step += 1
        return total_loss

    def clip_gradients(self, max_norm: Optional[float] = None) -> float:
        if max_norm is None:
            max_norm = self.grad_clip_norm
        if max_norm is None or max_norm <= 0:
            return 0.0
        params = [p for p in self.train_modules.parameters() if p.requires_grad and p.grad is not None]
        if not params:
            return 0.0
        return float(torch.nn.utils.clip_grad_norm_(params, float(max_norm)))

    @torch.no_grad()
    def ema_update(self) -> None:
        if self.ema is not None:
            self.ema.update(self.train_modules)

    @torch.no_grad()
    def ema_store(self) -> None:
        if self.ema is not None:
            self.ema.store(self.train_modules)

    @torch.no_grad()
    def ema_copy_to(self) -> None:
        if self.ema is not None:
            self.ema.copy_to(self.train_modules)

    @torch.no_grad()
    def ema_restore(self) -> None:
        if self.ema is not None:
            self.ema.restore(self.train_modules)

    @torch.no_grad()
    def tracking_inference(
        self,
        vtxt_input: torch.Tensor,
        ctxt_input: torch.Tensor,
        ctxt_mask: Optional[torch.Tensor],
        seq_length: int = 214,
        guidance_scale: Optional[float] = None,
    ) -> torch.Tensor:
        device = vtxt_input.device
        vtxt_input = self._normalize_vtxt(vtxt_input)
        ctxt_input = self._normalize_ctxt(ctxt_input)
        batch_size = int(vtxt_input.shape[0])
        if guidance_scale is None:
            guidance_scale = float(getattr(self.generator_cfg, "guidance_scale", 1.5))

        if ctxt_mask is None:
            ctxt_mask = self._infer_ctxt_mask(ctxt_input)
        else:
            if ctxt_mask.dim() == 1:
                ctxt_mask = ctxt_mask.unsqueeze(1)
            ctxt_mask = ctxt_mask.to(device).bool()
        ctxt_mask = self._align_ctxt_mask(ctxt_input, ctxt_mask)
        vtxt_input, ctxt_input, ctxt_mask = self._adapt_text(vtxt_input, ctxt_input, ctxt_mask)
        vtxt_tokens = self._prepare_vtxt_tokens(vtxt_input)

        latent_scale = int(self.vae_cfg.stride_t) ** int(self.vae_cfg.down_t)
        latent_seq_len = max(1, int(math.ceil(float(seq_length) / float(latent_scale))))
        x_t = torch.randn((batch_size, latent_seq_len, self.m_dim), device=device, dtype=ctxt_input.dtype)
        x_mask = self._make_true_mask(batch_size, latent_seq_len, device)

        do_cfg = guidance_scale > 1.0
        if do_cfg:
            ctxt_len = int(ctxt_input.shape[1])
            uncond_vtxt = self.null_vtxt.to(device=device, dtype=vtxt_tokens.dtype).expand(batch_size, -1, -1)
            uncond_ctxt = self.null_ctxt.to(device=device, dtype=ctxt_input.dtype).expand(
                batch_size,
                int(self.generator_cfg.null_ctxt_len),
                -1,
            )
            uncond_mask = self._make_true_mask(batch_size, int(self.generator_cfg.null_ctxt_len), device)
            uncond_ctxt, uncond_mask = self._pad_ctxt_to_len(uncond_ctxt, uncond_mask, target_len=ctxt_len)

            vtxt_cat = torch.cat([uncond_vtxt, vtxt_tokens], dim=0)
            ctxt_cat = torch.cat([uncond_ctxt, ctxt_input], dim=0)
            ctxt_mask_cat = torch.cat([uncond_mask, ctxt_mask], dim=0)
            x_mask_cat = torch.cat([x_mask, x_mask], dim=0)
        else:
            vtxt_cat = vtxt_tokens
            ctxt_cat = ctxt_input
            ctxt_mask_cat = ctxt_mask
            x_mask_cat = x_mask

        solver = str(getattr(self.generator_cfg, "inference_solver", "euler")).lower()
        if solver != "euler":
            raise ValueError(f"Unsupported inference_solver={solver!r}. Only 'euler' is implemented.")

        inference_steps = max(1, int(getattr(self.generator_cfg, "inference_steps", 50)))
        times = torch.linspace(0.0, 1.0, inference_steps + 1, device=device, dtype=x_t.dtype)
        for t_cur, t_next in zip(times[:-1], times[1:]):
            dt = t_next - t_cur
            t_batch = torch.full((batch_size,), float(t_cur.item()), device=device, dtype=x_t.dtype)
            if do_cfg:
                x_in = torch.cat([x_t, x_t], dim=0)
                t_in = torch.cat([t_batch, t_batch], dim=0)
            else:
                x_in = x_t
                t_in = t_batch

            flow_pred = self.generator_model(
                x=x_in,
                ctxt_input=ctxt_cat,
                vtxt_input=vtxt_cat,
                timesteps=t_in,
                x_mask_temporal=x_mask_cat,
                ctxt_mask_temporal=ctxt_mask_cat,
            )
            if do_cfg:
                flow_uncond, flow_cond = flow_pred.chunk(2, dim=0)
                flow_pred = flow_uncond + guidance_scale * (flow_cond - flow_uncond)

            x_t = x_t + dt * flow_pred

        z_pred = self._decode_motion(x_t * self.z_scale)
        z_pred = z_pred[:, :seq_length, :]
        z_pred_flat = z_pred.reshape(-1, z_pred.shape[-1])
        z_pred_proj = self.project_z(z_pred_flat)
        return z_pred_proj.view(batch_size, z_pred.shape[1], -1)

    @torch.no_grad()
    def gt_tracking_inference(self, next_obs: torch.Tensor) -> torch.Tensor:
        z = self.backward_map(next_obs)
        return self.project_z(z)
