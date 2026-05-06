import json
import os
import time
import warnings
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

import hydra
import numpy as np
import rootutils
import torch
from colorama import Fore, Style
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import Config
from metamotivo.wrappers.humenvbench import TrackingWrapper
from src.config import GeneratorAdapterConfig, GeneratorTrainConfig
from src.huggingface import GeneratorFBcprModelVAE
from utils import saving_utils
from utils.data_samplers import FixedSubsetMotionDataset, TrainMotionDataset, select_eval_episodes
from utils.text_motions import TextMotionBuffer
from utils.text_tracking_evaluation import TextTrackingEvaluation
from utils.train_utils import collate_padded, format_params, get_env

warnings.filterwarnings("ignore")
ROOT = rootutils.setup_root(search_from=__file__, cwd=True, pythonpath=False)

torch.set_float32_matmul_precision("high")
config = Config()
DATASET_FILTER_FIELD = "is_amass"
DATASET_FILTER_VALUE = True


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _fmt(value: torch.Tensor | float) -> str:
    if isinstance(value, torch.Tensor):
        value = float(value.item())
    return f"{value:.3f}"


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
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.no_grad()
def _evaluate_val_metrics(agent, val_loader, device: str, max_batches: int):
    records = []
    agent.eval()
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break

        z_seq_gt = batch[config.TRAJECTORY_KEY].to(device, non_blocking=True)
        x_mask = batch["mask"].to(device, non_blocking=True)
        vtxt = batch[config.VTEX_KEY].to(device, non_blocking=True)
        ctxt = batch[config.CTXT_KEY].to(device, non_blocking=True)
        ctxt_mask = batch.get("ctxt_mask", None)
        if ctxt_mask is None:
            ctxt_mask = ctxt.abs().sum(dim=-1) > 0
        else:
            ctxt_mask = ctxt_mask.to(device, non_blocking=True).bool()

        with torch.amp.autocast(device, dtype=torch.bfloat16):
            loss = agent.update(
                vtxt_input=vtxt,
                ctxt_input=ctxt,
                ctxt_mask=ctxt_mask,
                z_seq_gt=z_seq_gt,
                x_mask=x_mask,
                loss=config.m_loss,
            )

        rec = dict(agent.latest_metrics)
        rec["loss"] = float(loss.item())
        records.append(rec)

    agent.train()
    if not records:
        return {"loss": float("nan")}

    keys = records[0].keys()
    return {key: float(np.mean([r[key] for r in records])) for key in keys}


def _load_training_checkpoint_flexible(
    *,
    ckpt_path: str | os.PathLike,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    strict: bool = False,
    resume_training_state: bool = False,
) -> int:
    path = Path(ckpt_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw_state_dict = payload.get("model_state", payload)
    if not isinstance(raw_state_dict, dict):
        raise TypeError(f"Unsupported checkpoint state type: {type(raw_state_dict).__name__}")

    model_state = model.state_dict()
    state_dict = raw_state_dict
    skipped_shape_keys: list[str] = []

    if not strict:
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

    incompatible = model.load_state_dict(state_dict, strict=strict)

    if getattr(model, "ema", None) is not None and getattr(model, "train_modules", None) is not None:
        model.ema.shadow = {}
        model.ema.backup = {}
        model.ema._init_from(model.train_modules)

    print(f"{Fore.YELLOW}{Style.BRIGHT}Loaded init checkpoint:{Style.RESET_ALL} {path}")
    print(f"  missing_keys={len(incompatible.missing_keys)}  unexpected_keys={len(incompatible.unexpected_keys)}")
    if skipped_shape_keys:
        print(f"  skipped_shape_mismatch={len(skipped_shape_keys)}")
        for key in skipped_shape_keys[:10]:
            print(f"    {key}")
    if getattr(model, "ema", None) is not None and getattr(model, "train_modules", None) is not None:
        print("  synchronized EMA shadow from loaded weights")

    start_step = int(payload.get("step", -1)) + 1 if isinstance(payload, dict) else 0
    if resume_training_state:
        if optimizer is not None and isinstance(payload, dict) and "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
        if scaler is not None and isinstance(payload, dict) and "scaler_state" in payload:
            scaler.load_state_dict(payload["scaler_state"])
        print(f"  resumed optimizer/scaler state from step={max(0, start_step)}")
        return max(0, start_step)

    return 0


@hydra.main(
    version_base="1.4", config_name="entry.yaml", config_path=str(ROOT) + "/configs"
)
def main(cfg: DictConfig):
    exp_name = "experiments/exp_text2m/fb_text_t5_generator"
    device = "cuda:0"

    train_config = GeneratorTrainConfig()
    model_config = GeneratorAdapterConfig()
    model_config.vae_checkpoint_path = config.VAE_CHECKPOINT_PATH
    model_config.vae_config_path = config.VAE_CONFIG_PATH
    model_config.backbone_checkpoint_path = getattr(config, "GENERATOR_BACKBONE_CHECKPOINT_PATH", "")

    if not model_config.vae_checkpoint_path or not model_config.vae_config_path:
        raise ValueError(
            "Set Config.VAE_CHECKPOINT_PATH and Config.VAE_CONFIG_PATH before running generator.py"
        )

    video_dir = saving_utils.ensure_dir(f"{exp_name}_video_val_{cfg.algo_name}_{config.m_loss}")
    ckpt_dir = saving_utils.ensure_dir(f"{exp_name}_checkpoints_val_{cfg.algo_name}_{config.m_loss}")
    loss_dir = saving_utils.ensure_dir(f"{exp_name}_loss_val_{cfg.algo_name}_{config.m_loss}")
    losses_pkl_path = loss_dir / "losses.pkl"
    val_metrics_path = loss_dir / "val_metrics.jsonl"

    saving_utils.save_config_json(train_config, loss_dir / "generator_train_config.json")
    saving_utils.save_config_json(model_config, loss_dir / "generator_model_config.json")

    global_seed = int(cfg.agent.global_seed)
    OmegaConf.set_struct(cfg, False)
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(global_seed)

    raw_buffer = TextMotionBuffer(
        files=config.PATH_TO_MOTION_FILE,
        keys=config.KEYS,
        filter_field=DATASET_FILTER_FIELD,
        filter_value=DATASET_FILTER_VALUE,
    )

    total_items = len(raw_buffer)
    n_val = max(1, int(total_items * config.VAL_RATIO))
    rng = np.random.RandomState(global_seed)
    perm = rng.permutation(total_items)
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
        epoch_len=config.EPOCHS,
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
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=lambda x: collate_padded(x, max_len=train_config.max_seq_length),
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=lambda x: collate_padded(x, max_len=train_config.max_seq_length),
        persistent_workers=True,
    )

    fb_text_agent = GeneratorFBcprModelVAE.from_pretrained(
        "facebook/metamotivo-S-1",
        local_files_only=True,
        generator=model_config,
    ).to(device)

    trainable_params = [p for p in fb_text_agent.parameters() if p.requires_grad]
    total_trainable_params = sum(p.numel() for p in trainable_params)

    print(f"{Fore.GREEN}{Style.BRIGHT}--- Trainable Parameters Summary ---{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}Mode:{Style.RESET_ALL} HY-Motion MMDiT + VAE (two-branch text)")
    print(f"VAE ckpt:  {model_config.vae_checkpoint_path}")
    print(f"HY ckpt:   {model_config.backbone_checkpoint_path or '[random init]'}")
    print(f"m dim:     {fb_text_agent.m_dim}")
    print(f"Trainable: {format_params(total_trainable_params)} ({total_trainable_params:,})")
    print(f"{Fore.GREEN}{Style.BRIGHT}------------------------------------{Style.RESET_ALL}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = _build_lr_scheduler(
        optimizer,
        total_steps=min(len(train_loader), int(train_config.lr_decay_steps)),
        warmup_steps=int(train_config.warmup_steps),
        warmup_start_factor=float(train_config.warmup_start_factor),
        min_lr_ratio=float(train_config.min_lr_ratio),
    )
    scaler = torch.amp.GradScaler(device)

    start_step = 0
    init_checkpoint_path = str(getattr(config, "GENERATOR_INIT_CHECKPOINT_PATH", "")).strip()
    if init_checkpoint_path:
        start_step = _load_training_checkpoint_flexible(
            ckpt_path=init_checkpoint_path,
            model=fb_text_agent,
            optimizer=optimizer,
            scaler=scaler,
            strict=bool(getattr(config, "GENERATOR_INIT_STRICT", False)),
            resume_training_state=bool(getattr(config, "GENERATOR_RESUME_TRAINING_STATE", False)),
        )
        if start_step > 0:
            for _ in range(start_step):
                scheduler.step()

    tracking_eval = TextTrackingEvaluation(
        motions=config.PATH_TO_MOTION_FILE,
        keys=config.KEYS,
        env_kwargs={"state_init": "Default"},
        num_envs=3,
        motion_buffer=raw_buffer,
        filter_field=DATASET_FILTER_FIELD,
        filter_value=DATASET_FILTER_VALUE,
    )

    eval_episodes, used_fallback = select_eval_episodes(
        raw_buffer=raw_buffer,
        val_indices=val_indices,
        interesting_keywords=config.interesting_keywords,
        max_video_episodes=config.MAX_VIDEO_EPISODES,
        text_key="text",
        rng=rng,
    )
    if used_fallback:
        print(f"{Fore.RED}{Style.BRIGHT}WARNING: no interesting VAL eval episodes; using fallback.{Style.RESET_ALL}")

    eval_env = get_env()
    track_model = TrackingWrapper(model=fb_text_agent)

    with torch.no_grad():
        diag_batch = next(iter(train_loader))
        sample = diag_batch[config.TRAJECTORY_KEY].to(device)
        x_mask = diag_batch["mask"].to(device)
        vtxt = diag_batch[config.VTEX_KEY].to(device)
        ctxt = diag_batch[config.CTXT_KEY].to(device)
        ctxt_mask = diag_batch.get("ctxt_mask", None)
        if ctxt_mask is None:
            ctxt_mask = ctxt.abs().sum(dim=-1) > 0
        else:
            ctxt_mask = ctxt_mask.to(device).bool()
        latent = fb_text_agent._encode_motion(sample)
        print(f"vtxt shape:  {tuple(vtxt.shape)}")
        print(f"ctxt shape:  {tuple(ctxt.shape)}")
        print(f"ctxt mask:   {tuple(ctxt_mask.shape)}")
        print(f"z norm mean: {_fmt(sample.norm(dim=-1).mean())}")
        print(f"m abs mean:  {_fmt(latent.abs().mean())}")
        print(f"x mask mean: {_fmt(x_mask.float().mean())}")
        print(f"m shape:     {tuple(latent.shape)}")

    loss_records = []
    best_val_loss = float("inf")
    no_improve_evals = 0
    early_stopped = False
    print(f"{Fore.GREEN}{Style.BRIGHT}Started HY-Motion + VAE training!{Style.RESET_ALL}")
    pbar = tqdm(train_loader, dynamic_ncols=True, colour="green", smoothing=0.1)

    for train_step, batch in enumerate(pbar, start=start_step):
        z_seq_gt = batch[config.TRAJECTORY_KEY].to(device, non_blocking=True)
        x_mask = batch["mask"].to(device, non_blocking=True)
        vtxt = batch[config.VTEX_KEY].to(device, non_blocking=True)
        ctxt = batch[config.CTXT_KEY].to(device, non_blocking=True)
        ctxt_mask = batch.get("ctxt_mask", None)
        if ctxt_mask is None:
            ctxt_mask = ctxt.abs().sum(dim=-1) > 0
        else:
            ctxt_mask = ctxt_mask.to(device, non_blocking=True).bool()

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device, dtype=torch.bfloat16):
            train_loss = fb_text_agent.update(
                vtxt_input=vtxt,
                ctxt_input=ctxt,
                ctxt_mask=ctxt_mask,
                z_seq_gt=z_seq_gt,
                x_mask=x_mask,
                loss=config.m_loss,
            )

        scaler.scale(train_loss).backward()
        scaler.unscale_(optimizer)
        fb_text_agent.clip_gradients()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        fb_text_agent.ema_update()

        if train_step % config.train_val_step == 0:
            metrics = fb_text_agent.latest_metrics
            pbar.set_description(
                "train "
                f"total={metrics.get('total', float(train_loss.item())):.4f} "
                f"flow={metrics.get('flow', 0.0):.4f} "
                f"m={metrics.get('m_recon', 0.0):.4f} "
                f"m_cos={metrics.get('m_cos', 0.0):.4f} "
                f"z={metrics.get('z_recon', 0.0):.4f} "
                f"pi={metrics.get('policy', 0.0):.4f}"
            )
            loss_records.append(
                {
                    "step": int(train_step),
                    "type": "train",
                    "time": time.time(),
                    **metrics,
                }
            )

        if train_step % config.train_val_step_eval == 0:
            fb_text_agent.ema_store()
            fb_text_agent.ema_copy_to()
            val_metrics = _evaluate_val_metrics(
                fb_text_agent,
                val_loader,
                device=device,
                max_batches=config.VAL_MAX_BATCHES,
            )
            fb_text_agent.ema_restore()
            val_loss = float(val_metrics["loss"])

            print(
                f"{Fore.MAGENTA}{Style.BRIGHT}Val:{Style.RESET_ALL} "
                f"loss={val_loss:.4f} flow={val_metrics.get('flow', 0.0):.4f} "
                f"m={val_metrics.get('m_recon', 0.0):.4f} "
                f"m_cos={val_metrics.get('m_cos', 0.0):.4f} "
                f"z={val_metrics.get('z_recon', 0.0):.4f} "
                f"pi={val_metrics.get('policy', 0.0):.4f}"
            )
            record = {
                "step": int(train_step),
                "type": "val",
                "time": time.time(),
                "lr": float(optimizer.param_groups[0]["lr"]),
                **val_metrics,
            }
            loss_records.append(record)
            _append_jsonl(val_metrics_path, record)

            improved = val_loss < (best_val_loss - float(train_config.early_stopping_min_delta))
            if improved:
                best_val_loss = val_loss
                no_improve_evals = 0
                fb_text_agent.ema_store()
                fb_text_agent.ema_copy_to()
                ckpt_path = saving_utils.save_checkpoint(
                    model=fb_text_agent,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=train_step,
                    out_dir=ckpt_dir,
                )
                fb_text_agent.ema_restore()
                print(
                    f"{Fore.YELLOW}{Style.BRIGHT}Saved best checkpoint:{Style.RESET_ALL} "
                    f"{ckpt_path} (val_loss={best_val_loss:.4f})"
                )
            else:
                no_improve_evals += 1
                if bool(train_config.enable_early_stopping):
                    patience = int(train_config.early_stopping_patience_evals)
                    print(
                        f"{Fore.YELLOW}{Style.BRIGHT}Early stopping monitor:{Style.RESET_ALL} "
                        f"no_improve={no_improve_evals}/{patience} "
                        f"(best={best_val_loss:.4f}, current={val_loss:.4f}, "
                        f"min_delta={float(train_config.early_stopping_min_delta):.6f})"
                    )

            with torch.no_grad():
                eval_idx = int(rng.choice(len(eval_episodes)))
                ep = eval_episodes[eval_idx]
                slug = saving_utils.safe_slug(str(ep.get("text", ""))[:50])
                video_path = video_dir / f"step_{train_step:09d}_VAL_eval{eval_idx}_{slug}.mp4"
                saving_utils.save_eval_video_pred_vs_gt(
                    agent=fb_text_agent,
                    track_model=track_model,
                    emb_type=config.EMBEDDING_KEY,
                    env=eval_env,
                    ep=ep,
                    train_config=train_config,
                    text_archi_config=model_config,
                    out_path=video_path,
                    device=device,
                    conditioning_mode="two_branch",
                    vtxt_key=config.VTEX_KEY,
                    ctxt_key=config.CTXT_KEY,
                )
                print(f"{Fore.CYAN}{Style.BRIGHT}Saved video:{Style.RESET_ALL} {video_path}")

            saving_utils.save_losses_pkl(loss_records, losses_pkl_path)

            if bool(train_config.enable_early_stopping):
                patience = int(train_config.early_stopping_patience_evals)
                if patience > 0 and no_improve_evals >= patience:
                    early_stop_record = {
                        "step": int(train_step),
                        "type": "early_stop",
                        "time": time.time(),
                        "best_val_loss": float(best_val_loss),
                        "current_val_loss": float(val_loss),
                        "no_improve_evals": int(no_improve_evals),
                        "patience_evals": int(patience),
                        "min_delta": float(train_config.early_stopping_min_delta),
                    }
                    loss_records.append(early_stop_record)
                    _append_jsonl(val_metrics_path, early_stop_record)
                    saving_utils.save_losses_pkl(loss_records, losses_pkl_path)
                    print(
                        f"{Fore.RED}{Style.BRIGHT}Early stopping triggered:{Style.RESET_ALL} "
                        f"no val improvement for {no_improve_evals} evals. "
                        f"Best val_loss={best_val_loss:.4f}"
                    )
                    early_stopped = True
                    break

        if (train_step > 0) and (
            train_step % train_config.full_bench_eval_step == 0
            or train_step == len(pbar) - 1
        ):
            print(f"{Style.BRIGHT}{Fore.YELLOW}Running Motions Benchmark!{Style.RESET_ALL}")

            fb_text_agent.eval()
            fb_text_agent.ema_store()
            fb_text_agent.ema_copy_to()

            tracking_metrics = tracking_eval.run_text(
                agent=fb_text_agent,
                device=device,
                emb_name=config.EMBEDDING_KEY,
                ctxt_name=config.CTXT_KEY,
                vtxt_name=config.VTEX_KEY,
                conditioning_mode="two_branch",
            )

            fb_text_agent.ema_restore()
            fb_text_agent.train()

            if len(tracking_metrics) == 0:
                print(f"{Fore.RED}{Style.BRIGHT}Benchmark skipped: no motions after filtering.{Style.RESET_ALL}")
                continue

            bench_metrics = {}
            for key in ["success_phc_linf", "emd"]:
                values = np.array([m[key] for m in tracking_metrics.values()])
                bench_metrics[f"gen_mean_{key}"] = float(values.mean())
                bench_metrics[f"GT_{key}"] = float(config.GROUND_TRUTH_EVAL_METRICS[key])

            loss_records.append(
                {
                    "step": int(train_step),
                    "type": "bench",
                    "time": time.time(),
                    **bench_metrics,
                }
            )
            saving_utils.save_losses_pkl(loss_records, losses_pkl_path)

            ckpt_path = saving_utils.save_checkpoint(
                model=fb_text_agent,
                optimizer=optimizer,
                scaler=scaler,
                step=train_step,
                out_dir=ckpt_dir,
            )
            print(f"{Fore.YELLOW}{Style.BRIGHT}Saved checkpoint:{Style.RESET_ALL} {ckpt_path}")

        if early_stopped:
            break

    saving_utils.save_losses_pkl(loss_records, losses_pkl_path)
    print(f"{Fore.GREEN}{Style.BRIGHT}Saved losses pkl:{Style.RESET_ALL} {losses_pkl_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
