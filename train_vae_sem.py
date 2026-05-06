import math
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import json
import time
import warnings
from pathlib import Path

import hydra
import numpy as np
import rootutils
import torch
import torch.nn as nn
import torch.nn.functional as F
from colorama import Fore, Style
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import Config
from metamotivo.fb_cpr.huggingface import FBcprModel
from text_adapter.text_adapter import MultiModalTextConditionAdapter
from utils import saving_utils
from utils.data_samplers import FixedSubsetMotionDataset, TrainMotionDataset, select_eval_episodes
from utils.train_utils import collate_padded, format_params, get_env
from utils.text_motions import TextMotionBuffer
from metamotivo.wrappers.humenvbench import TrackingWrapper
from vae.config import VAEConfig, VAESemTrainConfig
from vae.fb_loss import FBActionLoss
from vae.sem_loss import MotionTextContrastiveLoss
from vae.vae import VAE

warnings.filterwarnings("ignore")
ROOT = rootutils.setup_root(search_from=__file__, cwd=True, pythonpath=False)

torch.set_float32_matmul_precision("high")
config = Config()

PHASE1_INIT_CKPT_PATH = Path(
    ""
)


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_json_overrides(env_key: str) -> dict | None:
    path_str = os.environ.get(env_key, "").strip()
    if not path_str:
        return None

    path = Path(path_str).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{env_key} points to missing file: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise TypeError(f"{env_key} must point to a JSON object, got {type(payload).__name__}")
    return payload


def _apply_dataclass_overrides(cfg, overrides: dict | None):
    if overrides is None:
        return cfg

    valid_fields = set(cfg.__dataclass_fields__.keys())
    unknown_fields = sorted(set(overrides.keys()) - valid_fields)
    if unknown_fields:
        raise KeyError(
            f"Unknown override fields for {type(cfg).__name__}: {', '.join(unknown_fields)}"
        )

    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _resolve_latent_dim(input_width: int, vae_cfg: VAEConfig) -> int:
    if vae_cfg.latent_dim is not None:
        return int(min(max(1, vae_cfg.latent_dim), input_width))
    raw_dim = int(round(float(input_width) * float(vae_cfg.latent_ratio)))
    raw_dim = max(int(vae_cfg.min_latent_dim), raw_dim)
    raw_dim = min(int(vae_cfg.max_latent_dim), raw_dim, int(input_width))
    aligned = max(8, int(round(raw_dim / 8.0) * 8))
    return int(min(aligned, input_width))


def _mask_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_exp = mask.unsqueeze(-1).to(dtype=pred.dtype)
    denom = mask_exp.sum() * pred.shape[-1] + 1e-8
    return ((pred - target).pow(2) * mask_exp).sum() / denom


def _velocity_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)
    pred_vel = pred[:, 1:, :] - pred[:, :-1, :]
    target_vel = target[:, 1:, :] - target[:, :-1, :]
    vel_mask = (mask[:, 1:] & mask[:, :-1]).unsqueeze(-1).to(dtype=pred.dtype)
    denom = vel_mask.sum() * pred.shape[-1] + 1e-8
    return ((pred_vel - target_vel).pow(2) * vel_mask).sum() / denom


def _kl_loss(dist: torch.distributions.Normal) -> torch.Tensor:
    mu = dist.loc
    std = dist.scale.clamp_min(1e-8)
    log_var = 2.0 * torch.log(std)
    kl = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp())
    return kl.mean()


def _align_reconstruction(z_gt: torch.Tensor, z_rec: torch.Tensor, mask: torch.Tensor):
    seq_len = min(z_gt.shape[1], z_rec.shape[1], mask.shape[1])
    return z_gt[:, :seq_len], z_rec[:, :seq_len], mask[:, :seq_len]


def _align_text_mask(ctxt: torch.Tensor, ctxt_mask: torch.Tensor) -> torch.Tensor:
    target_len = ctxt.shape[1]
    if ctxt_mask.shape[1] == target_len:
        return ctxt_mask
    if ctxt_mask.shape[1] < target_len:
        pad = torch.zeros(
            (ctxt_mask.shape[0], target_len - ctxt_mask.shape[1]),
            device=ctxt_mask.device,
            dtype=torch.bool,
        )
        return torch.cat([ctxt_mask, pad], dim=1)
    return ctxt_mask[:, :target_len]


def _augment_motion_batch(
    z_seq: torch.Tensor,
    z_mask: torch.Tensor,
    train_cfg: VAESemTrainConfig,
) -> torch.Tensor:
    if (not bool(train_cfg.aug_enabled)) or z_seq.numel() == 0:
        return z_seq

    out = z_seq.clone()
    valid = z_mask.unsqueeze(-1).to(dtype=out.dtype)
    bsz, tlen, fdim = out.shape

    # 1) Additive Gaussian noise on valid frames.
    if float(train_cfg.aug_noise_std) > 0.0 and float(train_cfg.aug_noise_prob) > 0.0:
        apply_noise = (
            torch.rand((bsz, 1, 1), device=out.device) < float(train_cfg.aug_noise_prob)
        ).to(dtype=out.dtype)
        noise = torch.randn_like(out) * float(train_cfg.aug_noise_std)
        out = out + noise * apply_noise * valid

    # 2) Global per-sample amplitude scaling.
    if float(train_cfg.aug_scale_prob) > 0.0:
        lo = float(min(train_cfg.aug_scale_min, train_cfg.aug_scale_max))
        hi = float(max(train_cfg.aug_scale_min, train_cfg.aug_scale_max))
        if hi > 0.0 and hi >= lo:
            scale = torch.empty((bsz, 1, 1), device=out.device).uniform_(lo, hi)
            apply_scale = (
                torch.rand((bsz, 1, 1), device=out.device) < float(train_cfg.aug_scale_prob)
            ).to(dtype=out.dtype)
            out = out * (1.0 + apply_scale * (scale - 1.0))

    # 3) Random feature dropout (entire channels for a sample).
    if float(train_cfg.aug_feature_drop_prob) > 0.0:
        keep = (
            torch.rand((bsz, 1, fdim), device=out.device) >= float(train_cfg.aug_feature_drop_prob)
        ).to(dtype=out.dtype)
        out = out * keep

    # 4) Random contiguous time masking on valid frames.
    if float(train_cfg.aug_time_mask_prob) > 0.0 and float(train_cfg.aug_time_mask_max_ratio) > 0.0:
        max_ratio = float(min(max(train_cfg.aug_time_mask_max_ratio, 0.0), 1.0))
        max_span = max(1, int(round(tlen * max_ratio)))
        if max_span > 0:
            for i in range(bsz):
                if torch.rand((), device=out.device) >= float(train_cfg.aug_time_mask_prob):
                    continue
                valid_len = int(z_mask[i].sum().item())
                if valid_len <= 1:
                    continue
                span_hi = min(max_span, valid_len)
                span = int(torch.randint(1, span_hi + 1, (1,), device=out.device).item())
                start_max = max(1, valid_len - span + 1)
                start = int(torch.randint(0, start_max, (1,), device=out.device).item())
                out[i, start : start + span, :] = 0.0

    # Ensure padded frames remain untouched zeros (consistent with masked losses).
    out = out * valid
    return out


def _ramped_weight(
    *,
    base_weight: float,
    step: int,
    warmup_steps: int,
    start_factor: float,
) -> float:
    if base_weight <= 0:
        return 0.0
    if warmup_steps <= 0:
        return float(base_weight)

    progress = min(1.0, max(0.0, float(step + 1) / float(max(1, warmup_steps))))
    scale = float(start_factor) + (1.0 - float(start_factor)) * progress
    return float(base_weight) * scale


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    warmup_start_factor: float,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, min(int(warmup_steps), total_steps - 1))
    warmup_start_factor = float(min(max(warmup_start_factor, 0.0), 1.0))
    min_lr_ratio = float(min(max(min_lr_ratio, 0.0), 1.0))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            progress = float(step + 1) / float(max(1, warmup_steps))
            return warmup_start_factor + (1.0 - warmup_start_factor) * progress

        if total_steps <= warmup_steps + 1:
            return 1.0

        progress = float(step - warmup_steps + 1) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


class GlobalMotionTextContrastiveHead(nn.Module):
    def __init__(
        self,
        *,
        motion_dim: int,
        text_dim: int,
        hidden_dim: int,
        dropout: float,
        temperature_init: float,
        max_logit_scale: float,
    ):
        super().__init__()
        if temperature_init <= 0:
            raise ValueError("temperature_init must be > 0.")

        self.max_logit_scale = float(max_logit_scale)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature_init)))

        self.motion_proj = nn.Sequential(
            nn.LayerNorm(motion_dim),
            nn.Linear(motion_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, text_dim),
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, text_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        *,
        motion: torch.Tensor,
        motion_mask: torch.Tensor,
        text: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        weights = motion_mask.unsqueeze(-1).to(dtype=motion.dtype)
        denom = weights.sum(dim=1).clamp_min(1.0)
        pooled_motion = (motion * weights).sum(dim=1) / denom

        motion_repr = F.normalize(self.motion_proj(pooled_motion.float()), dim=-1)
        text_repr = F.normalize(self.text_proj(text.float()), dim=-1)

        labels = torch.arange(motion.shape[0], device=motion.device)
        logit_scale = self.logit_scale.exp().clamp(max=self.max_logit_scale)
        logits = logit_scale * (motion_repr @ text_repr.T)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        loss = 0.5 * (loss_i2t + loss_t2i)

        with torch.no_grad():
            pred_i2t = logits.argmax(dim=1)
            pred_t2i = logits.argmax(dim=0)
            metrics = {
                "acc_i2t": float((pred_i2t == labels).float().mean().item()),
                "acc_t2i": float((pred_t2i == labels).float().mean().item()),
                "logit_scale": float(logit_scale.item()),
            }

        return loss, metrics


class Phase1SemanticVAE(nn.Module):
    def __init__(
        self,
        *,
        vae: VAE,
        text_adapter: MultiModalTextConditionAdapter,
        sem_loss: MotionTextContrastiveLoss,
        vtxt_loss: GlobalMotionTextContrastiveHead,
        fb_kl_loss: FBActionLoss,
    ):
        super().__init__()
        self.vae = vae
        self.text_adapter = text_adapter
        self.sem_loss = sem_loss
        self.vtxt_loss = vtxt_loss
        self.fb_kl_loss = fb_kl_loss


def _build_phase1_modules(
    *,
    batch: dict,
    vae_cfg: VAEConfig,
    train_cfg: VAESemTrainConfig,
    device: str,
) -> tuple[Phase1SemanticVAE, VAEConfig]:
    z_sample = batch[config.TRAJECTORY_KEY]
    ctxt_sample = batch[config.CTXT_KEY]
    vtxt_sample = batch[config.VTEX_KEY]

    input_width = int(z_sample.shape[-1]) if vae_cfg.input_width is None else int(vae_cfg.input_width)
    latent_dim = _resolve_latent_dim(input_width, vae_cfg)
    vae_cfg.input_width = input_width
    vae_cfg.latent_dim = latent_dim

    vae = VAE(
        input_width=input_width,
        output_emb_width=latent_dim,
        down_t=vae_cfg.down_t,
        stride_t=vae_cfg.stride_t,
        width=vae_cfg.width,
        depth=vae_cfg.depth,
        dilation_growth_rate=vae_cfg.dilation_growth_rate,
        activation=vae_cfg.activation,
        norm=vae_cfg.norm,
        res_dropout=float(vae_cfg.res_dropout),
        pad_mode=vae_cfg.pad_mode,
        ae=vae_cfg.ae,
    ).to(device)

    text_adapter = MultiModalTextConditionAdapter(
        ctxt_dim=int(ctxt_sample.shape[-1]),
        vtxt_dim=int(vtxt_sample.shape[-1]),
        hidden_dim=int(train_cfg.text_adapter_hidden_dim),
        out_vtxt_dim=int(vtxt_sample.shape[-1]),
        out_ctxt_dim=int(ctxt_sample.shape[-1]),
        nhead=int(train_cfg.text_adapter_nhead),
        num_layers=int(train_cfg.text_adapter_num_layers),
        dropout=float(train_cfg.text_adapter_dropout),
    ).to(device)

    latent_seq_len = max(
        1,
        int(math.ceil(float(train_cfg.max_seq_length) / float(int(vae_cfg.stride_t) ** int(vae_cfg.down_t)))),
    )
    sem_loss = MotionTextContrastiveLoss(
        motion_dim=int(latent_dim),
        text_dim=int(ctxt_sample.shape[-1]),
        hidden_dim=int(train_cfg.sem_hidden_dim),
        num_heads=int(train_cfg.sem_num_heads),
        num_motion_layers=int(train_cfg.sem_motion_layers),
        max_motion_len=latent_seq_len,
        dropout=float(train_cfg.sem_dropout),
        temperature_init=float(train_cfg.sem_temperature_init),
        frame_score_mode=str(train_cfg.sem_frame_score_mode),
        token_pool_tau=float(train_cfg.sem_token_pool_tau),
        frame_pool_tau=float(train_cfg.sem_frame_pool_tau),
        importance_entropy_weight=float(train_cfg.sem_importance_entropy_weight),
        max_logit_scale=float(train_cfg.sem_max_logit_scale),
    ).to(device)

    vtxt_loss = GlobalMotionTextContrastiveHead(
        motion_dim=int(latent_dim),
        text_dim=int(vtxt_sample.shape[-1]),
        hidden_dim=int(train_cfg.sem_hidden_dim),
        dropout=float(train_cfg.sem_dropout),
        temperature_init=float(train_cfg.vtxt_temperature_init),
        max_logit_scale=float(train_cfg.vtxt_max_logit_scale),
    ).to(device)

    fb_kl_loss = FBActionLoss().to(device)

    return Phase1SemanticVAE(
        vae=vae,
        text_adapter=text_adapter,
        sem_loss=sem_loss,
        vtxt_loss=vtxt_loss,
        fb_kl_loss=fb_kl_loss,
    ).to(device), vae_cfg


def _compute_phase1_loss(
    model: Phase1SemanticVAE,
    batch: dict,
    train_cfg: VAESemTrainConfig,
    device: str,
    *,
    sem_weight: float | None = None,
    vtxt_weight: float | None = None,
    apply_augment: bool = False,
):
    sem_weight = float(train_cfg.sem_weight if sem_weight is None else sem_weight)
    vtxt_weight = float(train_cfg.vtxt_weight if vtxt_weight is None else vtxt_weight)

    z_seq = batch[config.TRAJECTORY_KEY].to(device, non_blocking=True)
    z_mask = batch["mask"].to(device, non_blocking=True)
    obs_seq = batch["observation"].to(device, non_blocking=True)
    ctxt = batch[config.CTXT_KEY].to(device, non_blocking=True)
    vtxt = batch[config.VTEX_KEY].to(device, non_blocking=True)
    ctxt_mask = batch.get("ctxt_mask", None)
    if ctxt_mask is None:
        ctxt_mask = ctxt.abs().sum(dim=-1) > 0
    else:
        ctxt_mask = ctxt_mask.to(device, non_blocking=True).bool()
    ctxt_mask = _align_text_mask(ctxt, ctxt_mask)

    z_input = _augment_motion_batch(z_seq, z_mask, train_cfg) if apply_augment else z_seq

    if model.vae.ae:
        z_rec, m = model.vae(z_input)
        dist = None
        m_for_sem = m
    else:
        z_rec, m, dist = model.vae(z_input)
        m_for_sem = dist.loc

    print(z_input.shape, m.shape)

    z_gt, z_rec, z_mask = _align_reconstruction(z_seq, z_rec, z_mask)
    seq_len = min(z_gt.shape[1], obs_seq.shape[1], z_mask.shape[1])
    z_gt, z_rec, z_mask, obs_seq = (
        z_gt[:, :seq_len],
        z_rec[:, :seq_len],
        z_mask[:, :seq_len],
        obs_seq[:, :seq_len],
    )

    recon_loss = _mask_mse(z_rec, z_gt, z_mask)
    vel_loss = _velocity_loss(z_rec, z_gt, z_mask)
    kl_loss = z_gt.new_tensor(0.0) if dist is None else _kl_loss(dist)

    adapted = model.text_adapter(ctxt_emb=ctxt, vtxt_emb=vtxt, attention_mask=ctxt_mask)
    m_mask = torch.nn.functional.interpolate(
        z_mask.float().unsqueeze(1),
        size=m_for_sem.shape[1],
        mode="nearest",
    ).squeeze(1) > 0.5

    sem_loss, sem_metrics, _ = model.sem_loss(
        motion=m_for_sem,
        text=adapted.ctxt,
        motion_mask=m_mask,
        text_mask=adapted.ctxt_mask,
    )
    vtxt_loss, vtxt_metrics = model.vtxt_loss(
        motion=m_for_sem,
        motion_mask=m_mask,
        text=adapted.vtxt,
    )

    fb_kl_loss = model.fb_kl_loss(
        z1=z_rec,
        z2=z_gt,
        obs=obs_seq,
        mask=z_mask,
    )

    latent_l2 = m_for_sem.pow(2).mean()

    total_loss = recon_loss + sem_weight * sem_loss + vtxt_weight * vtxt_loss + 0.1 * fb_kl_loss
    if train_cfg.vel_weight > 0:
        total_loss = total_loss + float(train_cfg.vel_weight) * vel_loss
    if (not model.vae.ae) and train_cfg.kl_weight > 0:
        total_loss = total_loss + float(train_cfg.kl_weight) * kl_loss
    if train_cfg.latent_l2_weight > 0:
        total_loss = total_loss + float(train_cfg.latent_l2_weight) * latent_l2

    stats = {
        "loss": total_loss.detach(),
        "recon": recon_loss.detach(),
        "vel": vel_loss.detach(),
        "kl": kl_loss.detach(),
        "fb_kl": fb_kl_loss.detach(),
        "sem": sem_loss.detach(),
        "vtxt": vtxt_loss.detach(),
        "latent_l2": latent_l2.detach(),
        "m_abs_mean": m.detach().abs().mean(),
        "acc_i2t": sem_metrics["acc_i2t"],
        "acc_t2i": sem_metrics["acc_t2i"],
        "vtxt_acc_i2t": vtxt_metrics["acc_i2t"],
        "vtxt_acc_t2i": vtxt_metrics["acc_t2i"],
        "sem_logit_scale": sem_metrics["logit_scale"],
        "vtxt_logit_scale": vtxt_metrics["logit_scale"],
        "sem_weight": sem_weight,
        "vtxt_weight": vtxt_weight,
    }
    return total_loss, stats


@torch.no_grad()
def evaluate_phase1_val_loss(
    model: Phase1SemanticVAE,
    val_loader: DataLoader,
    device: str,
    train_cfg: VAESemTrainConfig,
    max_batches: int,
    *,
    sem_weight: float | None = None,
    vtxt_weight: float | None = None,
):
    model.eval()
    records = []
    for batch_idx, batch in enumerate(val_loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, stats = _compute_phase1_loss(
                model,
                batch,
                train_cfg,
                device,
                sem_weight=sem_weight,
                vtxt_weight=vtxt_weight,
                apply_augment=False,
            )
        records.append(
            {
                "loss": float(stats["loss"].item()),
                "recon": float(stats["recon"].item()),
                "vel": float(stats["vel"].item()),
                "kl": float(stats["kl"].item()),
                "fb_kl": float(stats["fb_kl"].item()),
                "sem": float(stats["sem"].item()),
                "vtxt": float(stats["vtxt"].item()),
                "latent_l2": float(stats["latent_l2"].item()),
                "acc_i2t": float(stats["acc_i2t"]),
                "acc_t2i": float(stats["acc_t2i"]),
                "vtxt_acc_i2t": float(stats["vtxt_acc_i2t"]),
                "vtxt_acc_t2i": float(stats["vtxt_acc_t2i"]),
                "sem_logit_scale": float(stats["sem_logit_scale"]),
                "vtxt_logit_scale": float(stats["vtxt_logit_scale"]),
                "sem_weight": float(stats["sem_weight"]),
                "vtxt_weight": float(stats["vtxt_weight"]),
            }
        )
    model.train()
    if not records:
        return {"loss": float("nan")}
    keys = records[0].keys()
    return {k: float(np.mean([r[k] for r in records])) for k in keys}


@hydra.main(version_base="1.4", config_name="entry.yaml", config_path=str(ROOT) + "/configs")
def main(cfg: DictConfig):
    exp_name = os.environ.get("VAE_SEM_EXP_NAME", "experiments/exp_vae_sem_prec_run/fb_latent_vae_sem").strip()
    if not exp_name:
        exp_name = "experiments/exp_vae_sem_prec_run/fb_latent_vae_sem"
    device = "cuda:0"

    train_cfg = _apply_dataclass_overrides(
        VAESemTrainConfig(),
        _load_json_overrides("VAE_SEM_TRAIN_CFG_JSON"),
    )
    vae_cfg = _apply_dataclass_overrides(
        VAEConfig(),
        _load_json_overrides("VAE_MODEL_CFG_JSON"),
    )

    video_dir = saving_utils.ensure_dir(f"{exp_name}_video_val_{cfg.algo_name}")
    ckpt_dir = saving_utils.ensure_dir(f"{exp_name}_checkpoints_val_{cfg.algo_name}")
    loss_dir = saving_utils.ensure_dir(f"{exp_name}_loss_val_{cfg.algo_name}")
    losses_pkl_path = loss_dir / "losses.pkl"
    val_metrics_path = loss_dir / "val_metrics.jsonl"

    saving_utils.save_config_json(train_cfg, loss_dir / "vae_sem_train_config.json")
    saving_utils.save_config_json(vae_cfg, loss_dir / "vae_model_config.json")

    global_seed = int(cfg.agent.global_seed)
    OmegaConf.set_struct(cfg, False)
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(global_seed)

    raw_buffer = TextMotionBuffer(files=config.PATH_TO_MOTION_FILE, keys=config.KEYS, limit_per_file=None)
    n_total = len(raw_buffer)
    n_val = max(1, int(n_total * config.VAL_RATIO))
    rng = np.random.RandomState(global_seed)
    perm = rng.permutation(n_total)
    val_indices = perm[:n_val]
    train_indices = perm[n_val:]
    if len(train_indices) == 0:
        train_indices = perm[:1]

    print(
        f"{Fore.CYAN}{Style.BRIGHT}Split:{Style.RESET_ALL} "
        f"train={len(train_indices):,}  val={len(val_indices):,} "
        f"(VAL_RATIO={config.VAL_RATIO})"
    )

    train_dataset = TrainMotionDataset(
        raw_buffer,
        subset_indices=train_indices,
        epoch_len=train_cfg.train_steps,
        group_key="group",
        alpha=1.0,
        use_priorities_within=False,
        min_group_size=1,
        verbose=False,
    )
    val_dataset = FixedSubsetMotionDataset(
        raw_buffer,
        subset_indices=val_indices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        collate_fn=lambda x: collate_padded(x, max_len=train_cfg.max_seq_length),
        persistent_workers=train_cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        collate_fn=lambda x: collate_padded(x, max_len=train_cfg.max_seq_length),
        persistent_workers=train_cfg.num_workers > 0,
    )

    first_batch = next(iter(train_loader))
    model, vae_cfg = _build_phase1_modules(
        batch=first_batch,
        vae_cfg=vae_cfg,
        train_cfg=train_cfg,
        device=device,
    )
    if PHASE1_INIT_CKPT_PATH.exists():
        payload = torch.load(PHASE1_INIT_CKPT_PATH, map_location="cpu")
        state_dict = payload.get("model_state", payload)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            f"{Fore.YELLOW}{Style.BRIGHT}Loaded checkpoint:{Style.RESET_ALL} "
            f"{PHASE1_INIT_CKPT_PATH}"
        )
        if missing:
            print(f"{Fore.RED}{Style.BRIGHT}Missing keys:{Style.RESET_ALL} {missing}")
        if unexpected:
            print(f"{Fore.RED}{Style.BRIGHT}Unexpected keys:{Style.RESET_ALL} {unexpected}")
    else:
        print(f"Phase1 init checkpoint not found: {PHASE1_INIT_CKPT_PATH}")

    saving_utils.save_config_json(vae_cfg, loss_dir / "vae_model_config.json")

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{Fore.GREEN}{Style.BRIGHT}--- Parameters Summary ---{Style.RESET_ALL}")
    print(f"Input width:      {vae_cfg.input_width}")
    print(f"Latent dim m:     {vae_cfg.latent_dim}")
    print(f"Trainable params: {format_params(total_trainable_params)} ({total_trainable_params:,})")
    print(f"{Fore.GREEN}{Style.BRIGHT}----------------------------------{Style.RESET_ALL}")

    optimizer = torch.optim.AdamW(
        [
            {"params": model.vae.parameters(), "lr": train_cfg.learning_rate},
            {"params": model.text_adapter.parameters(), "lr": train_cfg.text_learning_rate},
            {"params": model.sem_loss.parameters(), "lr": train_cfg.text_learning_rate},
            {"params": model.vtxt_loss.parameters(), "lr": train_cfg.text_learning_rate},
        ],
        weight_decay=train_cfg.weight_decay,
    )
    total_optim_steps = len(train_loader)
    scheduler = _build_lr_scheduler(
        optimizer,
        total_steps=total_optim_steps,
        warmup_steps=train_cfg.warmup_steps,
        warmup_start_factor=train_cfg.warmup_start_factor,
        min_lr_ratio=train_cfg.min_lr_ratio,
    )
    scaler = torch.amp.GradScaler("cuda" if device.startswith("cuda") else "cpu")

    print(f"Optimizer steps:  {total_optim_steps:,}")
    print(f"Warmup steps:     {min(train_cfg.warmup_steps, total_optim_steps):,}")
    print(f"Min LR ratio:     {train_cfg.min_lr_ratio:.3f}")
    print(f"Sem weight ramp:  {train_cfg.sem_weight_start_factor:.2f} -> 1.00 over {train_cfg.sem_weight_warmup_steps:,} steps")

    eval_episodes, used_fallback = select_eval_episodes(
        raw_buffer=raw_buffer,
        val_indices=val_indices,
        interesting_keywords=config.interesting_keywords,
        max_video_episodes=train_cfg.save_video_episodes,
        text_key="text",
        rng=rng,
    )
    if used_fallback:
        print(f"{Fore.RED}{Style.BRIGHT}WARNING: no interesting VAL eval episodes; using fallback.{Style.RESET_ALL}")

    fb_model = FBcprModel.from_pretrained("facebook/metamotivo-S-1", local_files_only=True).to(device)
    fb_model.eval()
    track_model = TrackingWrapper(model=fb_model)
    eval_env = get_env()

    loss_records = []
    print(f"{Fore.GREEN}{Style.BRIGHT}Started Training!{Style.RESET_ALL}")
    pbar = tqdm(train_loader, dynamic_ncols=True, colour="green", smoothing=0.1)

    for train_step, batch in enumerate(pbar):
        current_sem_weight = _ramped_weight(
            base_weight=train_cfg.sem_weight,
            step=train_step,
            warmup_steps=train_cfg.sem_weight_warmup_steps,
            start_factor=train_cfg.sem_weight_start_factor,
        )
        current_vtxt_weight = _ramped_weight(
            base_weight=train_cfg.vtxt_weight,
            step=train_step,
            warmup_steps=train_cfg.sem_weight_warmup_steps,
            start_factor=train_cfg.sem_weight_start_factor,
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            train_loss, train_stats = _compute_phase1_loss(
                model,
                batch,
                train_cfg,
                device,
                sem_weight=current_sem_weight,
                vtxt_weight=current_vtxt_weight,
                apply_augment=bool(train_cfg.aug_enabled),
            )

        scaler.scale(train_loss).backward()
        scaler.unscale_(optimizer)
        if train_cfg.grad_clip_norm and train_cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.grad_clip_norm))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if train_step % train_cfg.train_log_every == 0:
            loss_val = float(train_stats["loss"].item())
            recon_val = float(train_stats["recon"].item())
            sem_val = float(train_stats["sem"].item())
            vtxt_val = float(train_stats["vtxt"].item())
            vae_lr = float(optimizer.param_groups[0]["lr"])
            text_lr = float(optimizer.param_groups[1]["lr"])
            fb_kl  = float(train_stats["fb_kl"].item())
            pbar.set_description(
                f"loss: {loss_val:.4f} recon: {recon_val:.4f} sem: {sem_val:.4f}"
                f"vtxt: {vtxt_val:.4f} acc_i2t: {float(train_stats['acc_i2t']):.3f} fb_kl: {fb_kl:.4f}"
            )
            loss_records.append(
                {
                    "step": int(train_step),
                    "type": "train",
                    "loss": loss_val,
                    "recon": recon_val,
                    "vel": float(train_stats["vel"].item()),
                    "kl": float(train_stats["kl"].item()),
                    "fb_kl": float(train_stats["fb_kl"].item()),
                    "sem": sem_val,
                    "vtxt": vtxt_val,
                    "latent_l2": float(train_stats["latent_l2"].item()),
                    "m_abs_mean": float(train_stats["m_abs_mean"].item()),
                    "acc_i2t": float(train_stats["acc_i2t"]),
                    "acc_t2i": float(train_stats["acc_t2i"]),
                    "vtxt_acc_i2t": float(train_stats["vtxt_acc_i2t"]),
                    "vtxt_acc_t2i": float(train_stats["vtxt_acc_t2i"]),
                    "sem_logit_scale": float(train_stats["sem_logit_scale"]),
                    "vtxt_logit_scale": float(train_stats["vtxt_logit_scale"]),
                    "sem_weight": float(train_stats["sem_weight"]),
                    "vtxt_weight": float(train_stats["vtxt_weight"]),
                    "lr_vae": vae_lr,
                    "lr_text": text_lr,
                    "time": time.time(),
                }
            )

        if train_step % train_cfg.val_every == 0:
            val_stats = evaluate_phase1_val_loss(
                model=model,
                val_loader=val_loader,
                device=device,
                train_cfg=train_cfg,
                max_batches=train_cfg.val_max_batches,
                sem_weight=current_sem_weight,
                vtxt_weight=current_vtxt_weight,
            )
            print(
                f"{Fore.MAGENTA}{Style.BRIGHT}Val:{Style.RESET_ALL} "
                f"loss={val_stats['loss']:.4f} recon={val_stats['recon']:.4f} "
                f"sem={val_stats['sem']:.4f} vtxt={val_stats['vtxt']:.4f} "
                f"acc_i2t={val_stats['acc_i2t']:.3f} fb_kl={val_stats['fb_kl']:.4f}"
            )
            record = {
                "step": int(train_step),
                "type": "val",
                "time": time.time(),
                "lr_vae": float(optimizer.param_groups[0]["lr"]),
                "lr_text": float(optimizer.param_groups[1]["lr"]),
                **val_stats,
            }
            loss_records.append(record)
            _append_jsonl(val_metrics_path, record)
            saving_utils.save_losses_pkl(loss_records, losses_pkl_path)

        if train_step % train_cfg.video_every == 0:
            with torch.no_grad():
                eval_idx = int(rng.choice(len(eval_episodes)))
                ep = eval_episodes[eval_idx]
                slug = saving_utils.safe_slug(str(ep.get("text", ""))[:50])
                video_path = video_dir / f"step_{train_step:09d}_VAL_eval{eval_idx}_{slug}.mp4"
                saving_utils.save_eval_video_vae_recon(
                    vae=model.vae,
                    fb_model=fb_model,
                    track_model=track_model,
                    env=eval_env,
                    ep=ep,
                    traj_key=config.TRAJECTORY_KEY,
                    out_path=video_path,
                    device=device,
                )
                print(f"{Fore.CYAN}{Style.BRIGHT}Saved video:{Style.RESET_ALL} {video_path}")

        if (train_step > 0) and (train_step % train_cfg.ckpt_every == 0):
            ckpt_path = saving_utils.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=train_step,
                out_dir=ckpt_dir,
            )
            print(f"{Fore.YELLOW}{Style.BRIGHT}Saved checkpoint:{Style.RESET_ALL} {ckpt_path}")

    final_ckpt_path = saving_utils.save_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        step=train_step,
        out_dir=ckpt_dir,
    )
    saving_utils.save_losses_pkl(loss_records, losses_pkl_path)
    print(f"{Fore.YELLOW}{Style.BRIGHT}Saved final checkpoint:{Style.RESET_ALL} {final_ckpt_path}")
    print(f"{Fore.GREEN}{Style.BRIGHT}Saved losses pkl:{Style.RESET_ALL} {losses_pkl_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
