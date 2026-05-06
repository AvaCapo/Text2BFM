import os
os.environ['MUJOCO_GL'] = 'egl'

import rootutils
import warnings
import pickle
import json
import dataclasses
import torch

import numpy as np

from pathlib import Path
from utils.train_utils import smooth_z_sequence

torch.set_float32_matmul_precision("high")
warnings.filterwarnings('ignore')
ROOT = rootutils.setup_root(search_from=__file__, cwd=True, pythonpath=False)



def ensure_dir(p: str | Path) -> Path:
    """Ensure that the directory at path `p` exists. If it doesn't, create it."""
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_slug(s: str, max_len: int = 80) -> str:
    """Convert a string into a safe slug for filenames."""
    s = "".join(c if c.isalnum() or c in ("_", "-", " ") else "_" for c in s)
    s = "_".join(s.strip().split())
    return s[:max_len]


def save_losses_pkl(loss_records, out_path: Path):
    """Save loss records (e.g. training curves) to a pickle file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(loss_records, f)


def save_config_json(config, out_path: Path) -> Path:
    """Save a dataclass config (or dict) to a JSON file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        payload = dataclasses.asdict(config)
    elif isinstance(config, dict):
        payload = config
    else:
        raise TypeError(
            f"config must be a dataclass instance or dict, got {type(config).__name__}"
        )

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)

    return out_path


def save_generator_configs(train_config, adapter_config, out_dir: Path) -> dict[str, Path]:
    """Save generator train/model configs as JSON files in out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = save_config_json(train_config, out_dir / "generator_train_config.json")
    adapter_path = save_config_json(adapter_config, out_dir / "generator_model_config.json")

    return {
        "train_config": train_path,
        "adapter_config": adapter_path,
    }


def save_video_mp4(frames, out_path: Path, fps: int = 30):
    """Save a sequence of frames as an MP4 video.
    frames: list[np.ndarray] with shape (H, W, 3) uint8 (or convertible)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
    except Exception as e:
        raise RuntimeError(
            "imageio is required to save mp4. Install: pip install imageio imageio-ffmpeg"
        ) from e

    processed = []
    for f in frames:
        if f.dtype != np.uint8:
            f = np.clip(f, 0, 255).astype(np.uint8)
        processed.append(f)

    imageio.mimsave(str(out_path), processed, fps=fps)


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scaler: torch.amp.GradScaler,
                    step: int,
                    out_dir: Path):
    """Save a training checkpoint including model state, optimizer state, and scaler state."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"ckpt_step_{step:09d}.pt"
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
        },
        ckpt_path,
    )
    return ckpt_path


@torch.no_grad()
def save_eval_video_vae_recon(
    *,
    vae,
    fb_model,
    track_model,
    env,
    ep,
    traj_key: str,
    out_path: Path,
    device: str,
    smooth_sigma: float = 1.0,
):
    """Save a side-by-side video for VAE reconstruction vs GT latent trajectory."""
    was_training = vae.training
    z_gt = torch.from_numpy(np.asarray(ep[traj_key], dtype=np.float32)).to(device, non_blocking=True)
    if z_gt.dim() == 2:
        z_in = z_gt.unsqueeze(0)
    else:
        raise ValueError(f"Expected 2D trajectory for {traj_key}, got shape={tuple(z_gt.shape)}")

    vae.eval()
    if getattr(vae, "ae", False):
        z_rec, _ = vae(z_in)
    else:
        z_rec, _, _ = vae(z_in)

    z_rec = z_rec.squeeze(0)
    z_gt = z_gt[: z_rec.shape[0]]

    if smooth_sigma and smooth_sigma > 0:
        z_rec = smooth_z_sequence(z_rec, sigma=smooth_sigma)

    z_rec = fb_model.project_z(z_rec)
    z_gt = fb_model.project_z(z_gt)

    qpos_np = np.asarray(ep["qpos"], dtype=np.float32)
    qvel_np = np.asarray(ep["qvel"], dtype=np.float32)
    qpos_init = qpos_np[0]
    qvel_init = qvel_np[0]

    observation, _ = env.reset(options={"qpos": qpos_init, "qvel": qvel_init})
    frames_pred = [env.render()]

    for t in range(len(z_rec)):
        obs = torch.as_tensor(observation.reshape(1, -1), dtype=torch.float32, device=device)
        action = track_model.act(obs=obs, z=z_rec[t]).ravel()
        observation, _, _, _, _ = env.step(action)
        frames_pred.append(env.render())

    observation, _ = env.reset(options={"qpos": qpos_init, "qvel": qvel_init})
    frames_gt = [env.render()]

    for t in range(len(z_gt)):
        obs = torch.as_tensor(observation.reshape(1, -1), dtype=torch.float32, device=device)
        action = track_model.act(obs=obs, z=z_gt[t]).ravel()
        observation, _, _, _, _ = env.step(action)
        frames_gt.append(env.render())

    min_frames = min(len(frames_pred), len(frames_gt))
    combined_frames = [
        np.concatenate([fp, fg], axis=1)
        for fp, fg in zip(frames_pred[:min_frames], frames_gt[:min_frames])
    ]
    save_video_mp4(combined_frames, out_path, fps=30)
    if was_training:
        vae.train()


@torch.no_grad()
def save_eval_video_pred_vs_gt(
    *,
    agent,
    track_model,
    emb_type,
    env,
    ep,
    train_config,
    text_archi_config,
    out_path: Path,
    device: str,
    conditioning_mode: str = "single",
    vtxt_key: str = "vtxt",
    ctxt_key: str = "ctxt",
):
    """Given an episode, roll out the predicted motion from text and the GT motion from next_obs, and save a side-by-side video."""
    if conditioning_mode == "single":
        eval_text_emb = torch.from_numpy(np.asarray(ep[emb_type], dtype=np.float32)).to(device, non_blocking=True)

        text_ctx = agent.tracking_inference(
            eval_text_emb,
            train_config.max_seq_length,
            guidance_scale=text_archi_config.guidance_scale
        ).squeeze(0)
    
    elif conditioning_mode == "two_branch":
        vtxt = torch.from_numpy(np.asarray(ep[vtxt_key], dtype=np.float32)).to(device, non_blocking=True)
        if vtxt.dim() == 1:
            vtxt = vtxt.unsqueeze(0)
        elif vtxt.dim() == 3 and vtxt.shape[1] == 1:
            vtxt = vtxt.squeeze(1)

        ctxt = torch.from_numpy(np.asarray(ep[ctxt_key], dtype=np.float32)).to(device, non_blocking=True)
        if ctxt.dim() == 1:
            ctxt = ctxt.unsqueeze(0).unsqueeze(0)
        elif ctxt.dim() == 2:
            ctxt = ctxt.unsqueeze(0)

        ctxt_mask_arr = ep.get("ctxt_mask", None)
        if ctxt_mask_arr is None:
            ctxt_mask = (ctxt.abs().sum(dim=-1) > 0)
        else:
            ctxt_mask = torch.from_numpy(np.asarray(ctxt_mask_arr, dtype=np.bool_)).to(device, non_blocking=True)
            if ctxt_mask.dim() == 1:
                ctxt_mask = ctxt_mask.unsqueeze(0)

        text_ctx = agent.tracking_inference(
            vtxt_input=vtxt,
            ctxt_input=ctxt,
            ctxt_mask=ctxt_mask,
            seq_length=train_config.max_seq_length,
            guidance_scale=text_archi_config.guidance_scale,
        ).squeeze(0)
    else:
        raise ValueError(
            f"Unknown conditioning_mode={conditioning_mode}. Expected 'single' or 'two_branch'."
        )

    text_ctx_smoothed = smooth_z_sequence(text_ctx, sigma=1.0)
    text_ctx_smoothed = agent.project_z(text_ctx_smoothed)

    # init state
    qpos_np = np.asarray(ep["qpos"], dtype=np.float32)
    qvel_np = np.asarray(ep["qvel"], dtype=np.float32)
    qpos_init = qpos_np[0]
    qvel_init = qvel_np[0]

    # rollout predicted
    observation, _ = env.reset(options={"qpos": qpos_init, "qvel": qvel_init})
    frames_pred = [env.render()]

    for t in range(len(text_ctx_smoothed)):
        obs = torch.as_tensor(observation.reshape(1, -1), dtype=torch.float32, device=device)
        action = track_model.act(obs=obs, z=text_ctx_smoothed[t]).ravel()
        observation, _, _, _, _ = env.step(action)
        frames_pred.append(env.render())

    # GT latent from next_obs
    gt_obs_seq = torch.from_numpy(np.asarray(ep["observation"], dtype=np.float32)).to(device, non_blocking=True)[1:]
    gt_ctx = agent.gt_tracking_inference(next_obs=gt_obs_seq)

    observation, _ = env.reset(options={"qpos": qpos_init, "qvel": qvel_init})
    frames_gt = [env.render()]

    run_len = min(len(gt_ctx), len(text_ctx_smoothed))
    for t in range(run_len):
        obs = torch.as_tensor(observation.reshape(1, -1), dtype=torch.float32, device=device)
        action = track_model.act(obs=obs, z=gt_ctx[t]).ravel()
        observation, _, _, _, _ = env.step(action)
        frames_gt.append(env.render())

    min_frames = min(len(frames_pred), len(frames_gt))
    combined_frames = [
        np.concatenate([fp, fg], axis=1)
        for fp, fg in zip(frames_pred[:min_frames], frames_gt[:min_frames])
    ]

    save_video_mp4(combined_frames, out_path, fps=30)
