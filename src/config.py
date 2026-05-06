import dataclasses


@dataclasses.dataclass
class MDMTrainConfig:
    batch_size: int = 256
    max_seq_length: int = 256
    eval_step: int = 5000
    learning_rate: float = 5e-5#0.000102310906537332
    weight_decay: float = 6.080313359530695e-05
    full_bench_eval_step: int = 50_000
    
@dataclasses.dataclass
class MDMAdapterConfig:
    """
    Configuration for the MDM-style Transformer.
    Default values are tuned for high-quality complex motion generation.
    """
    vtxt_dim: int = 768
    ctxt_dim: int = 1024#4096
    
    text_dim: int = 1024       
    hidden_dim: int = 768     # Model width
    num_layers: int = 4       # Slightly smaller backbone for better stability
    nhead: int = 8#6            # Attention heads
    ff_dim: int = 2048#4096        # FFN expansion
    dropout: float = 0.1      # Regularization
    activation: str = "gelu"
    
    # Diffusion Parameters
    diffusion_steps: int = 1000
    schedule: str = "cosine"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    
    # Inference / Sampling
    guidance_scale: float = 2.5   # MDM usually benefits from 2.5 - 7.5
    inference_steps: int = 50     # DDIM steps
    
    # Training
    cfg_dropout_prob: float = 0.1 # Disable CFG dropout for a cleaner optimization signal
    compile: bool = False
    null_ctxt_len: int = 1

    eps: float = 1e-8
    lambda_cos: float = 0.1
    lambda_cos_warmup_steps: int = 0
    min_snr_gamma: float = 5.0
    inference_norm_mode: str = "none"  # none|clamp|sphere
    inference_norm_clamp_min: float = 0.5
    inference_norm_clamp_max: float = 3.0
    grad_clip_norm: float = 1.0
    use_ema: bool = True
    ema_decay: float = 0.9999
    use_adaln_zero: bool = True
    z_scale: float = 1.0


@dataclasses.dataclass
class MDMVAETrainConfig:
    batch_size: int = 512
    max_seq_length: int = 256
    eval_step: int = 5000
    learning_rate: float = 2e-5
    weight_decay: float = 6.080313359530695e-05
    full_bench_eval_step: int = 25_000
    warmup_steps: int = 2_000
    warmup_start_factor: float = 0.1
    min_lr_ratio: float = 0.1#0.2
    lr_decay_steps: int = 15000#20_000
    enable_early_stopping: bool = True
    early_stopping_patience_evals: int = 5
    early_stopping_min_delta: float = 1e-4


@dataclasses.dataclass
class MDMVAEAdapterConfig(MDMAdapterConfig):
    vae_checkpoint_path: str = ""
    vae_config_path: str = ""
    masked_training: bool = False
    ctxt_encoder_layers: int = 2
    lambda_m_recon: float = 1.0
    lambda_z_recon: float = 1.0
    lambda_z_vel: float = 0.2
    freeze_vae_encoder: bool = True
    freeze_vae_decoder: bool = True
    use_text_adapter: bool = True
    raw_vtxt_dim: int = 768
    raw_ctxt_dim: int = 1024
    text_adapter_hidden_dim: int = 768
    text_adapter_num_layers: int = 2
    text_adapter_nhead: int = 8
    text_adapter_dropout: float = 0.1
    lambda_policy: float = 0.0
    policy_use_obs: bool = True



@dataclasses.dataclass
class GeneratorTrainConfig(MDMVAETrainConfig):
    batch_size: int = 128
    learning_rate: float = 3e-5
    weight_decay: float = 5e-4
    enable_early_stopping: bool = True
    early_stopping_patience_evals: int = 12
    early_stopping_min_delta: float = 1e-4


@dataclasses.dataclass
class GeneratorAdapterConfig(MDMVAEAdapterConfig):
    num_layers: int = 8
    nhead: int = 8
    dropout: float = 0.1
    guidance_scale: float = 1.5
    feat_dim: int = 512
    mlp_ratio: float = 4.0
    mlp_act_type: str = "gelu_tanh"
    qk_norm_type: str = "rms"
    qkv_bias: bool = True
    mask_mode: str | None = "narrowband"
    apply_rope_to_single_branch: bool = True
    insert_start_token: bool = False
    with_long_skip_connection: bool = False
    time_factor: float = 1.0
    narrowband_length: float = 2.0
    cond_mask_prob: float = 0.2
    lambda_flow: float = 1.0
    inference_solver: str = "euler"
    backbone_checkpoint_path: str = ""
    backbone_checkpoint_strict: bool = False
