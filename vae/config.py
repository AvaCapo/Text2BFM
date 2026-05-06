from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass
class VAEConfig:
    input_width: Optional[int] = 256
    latent_dim: Optional[int] = 48
    latent_ratio: float = 0.25
    min_latent_dim: int = 32
    max_latent_dim: int = 48
    down_t: int = 3
    stride_t: int = 2
    width: int = 256
    depth: int = 3
    dilation_growth_rate: int = 3
    activation: str = "relu"
    norm: Optional[str] = "GN"
    pad_mode: str = "replicate"
    res_dropout: float = 0.10
    ae: bool = False


@dataclasses.dataclass
class VAETrainConfig:
    batch_size: int = 256
    max_seq_length: int = 256
    learning_rate: float = 6e-5
    weight_decay: float = 5e-4
    kl_weight: float = 1e-3
    vel_weight: float = 0.2
    grad_clip_norm: float = 1.0
    num_workers: int = 8
    warmup_steps: int = 2000
    warmup_start_factor: float = 0.1
    min_lr_ratio: float = 0.2
    train_steps: int = 21_560_000
    val_max_batches: int = 50
    train_log_every: int = 50
    val_every: int = 500
    video_every: int = 5000
    ckpt_every: int = 5000
    save_video_episodes: int = 10
    amp_dtype: str = "bfloat16"


@dataclasses.dataclass
class VAESemTrainConfig(VAETrainConfig):
    learning_rate: float = 6e-5
    text_learning_rate: float = 3e-5
    # sem_weight: float = 0.0
    sem_weight: float = 0.3
    sem_weight_warmup_steps: int = 6000
    # sem_weight_start_factor: float = 0.0
    sem_weight_start_factor: float = 0.1
    vtxt_weight: float = 0.0
    vtxt_temperature_init: float = 0.07
    sem_max_logit_scale: float = 30.0
    vtxt_max_logit_scale: float = 30.0
    sem_hidden_dim: int = 32
    sem_num_heads: int = 1
    sem_motion_layers: int = 1
    sem_dropout: float = 0.1
    sem_temperature_init: float = 0.07
    sem_frame_score_mode: str = "logsumexp"
    sem_token_pool_tau: float = 0.1
    # Added for the lighter semantic loss variant with non-parametric frame pooling.
    sem_frame_pool_tau: float = 0.3
    sem_importance_entropy_weight: float = 1e-3
    latent_l2_weight: float = 1e-4
    text_adapter_hidden_dim: int = 768
    text_adapter_nhead: int = 1
    text_adapter_num_layers: int = 1
    text_adapter_dropout: float = 0.1

    # Motion augmentations (train only) to reduce overfitting.
    aug_enabled: bool = True
    aug_noise_std: float = 0.1
    aug_noise_prob: float = 0.7
    aug_scale_min: float = 0.98
    aug_scale_max: float = 1.02
    aug_scale_prob: float = 0.5
    aug_feature_drop_prob: float = 0.01
    aug_time_mask_prob: float = 0.2
    aug_time_mask_max_ratio: float = 0.1
