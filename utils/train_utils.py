import torch
import scipy
import random
import numpy as np
from humenv import make_humenv
from gymnasium.wrappers import FlattenObservation, TransformObservation
import numpy as np

def format_params(num):
    if num >= 1e9:
        # Billions
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        # Millions
        return f"{num / 1e6:.2f}M"
    else:
        # Standard format
        return f"{num:,}"

def collate_padded(batch, max_len=128):
    batch_out = {}
    lengths = [
        next((v.shape[0] for v in item.values() if isinstance(v, np.ndarray) and v.ndim > 1 and v.shape[0] > 1), 1)
        for item in batch
    ]
    current_max = min(max(lengths), max_len)
    
    keys = batch[0].keys()
    for k in keys:
        first_val = batch[0][k]
        if isinstance(first_val, np.ndarray) and (first_val.ndim == 1 or first_val.shape[0] == 1):
            arrs = [item[k].flatten() for item in batch]
            batch_out[k] = torch.from_numpy(np.stack(arrs))
        elif isinstance(first_val, np.ndarray) and first_val.ndim > 1:
            dim = first_val.shape[1]
            padded = torch.zeros(len(batch), current_max, dim, dtype=torch.float32)
            for i, item in enumerate(batch):
                val = item[k]
                length = min(val.shape[0], current_max)
                padded[i, :length, :] = torch.from_numpy(val[:length])
            batch_out[k] = padded
        elif isinstance(first_val, (str, np.str_)):
            batch_out[k] = [item[k] for item in batch]
    
    mask = torch.zeros(len(batch), current_max, dtype=torch.bool)
    for i, L in enumerate(lengths):
        valid_len = min(L, current_max)
        mask[i, :valid_len] = True
    batch_out["mask"] = mask

    return batch_out

def lr_lambda_schedule(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + np.cos(np.pi * progress))

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_env():
    device = "cuda:0"
    transform_obs_wrapper = lambda env: TransformObservation(
            env, lambda obs: torch.as_tensor(obs.reshape(1, -1), dtype=torch.float32, device=device), env.observation_space
        )
    env, _ = make_humenv(
        num_envs=1,
        wrappers=[FlattenObservation, transform_obs_wrapper],
        state_init="Default",
    )
    return env

def smooth_z_sequence(z_seq, sigma=2.0):
    """
    z_seq: (T, D) tensor or numpy array
    sigma: kernel size for Gaussian smoothing
    """
    is_tensor = torch.is_tensor(z_seq)
    if is_tensor:
        device = z_seq.device
        z_seq_np = z_seq.detach().cpu().numpy()
    else:
        z_seq_np = z_seq
    z_smoothed = scipy.ndimage.gaussian_filter1d(z_seq_np, sigma=sigma, axis=0)

    if is_tensor:
        return torch.from_numpy(z_smoothed).to(device).float()
    return z_smoothed