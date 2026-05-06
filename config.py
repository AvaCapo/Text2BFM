from typing import Literal


class Config:
    VAL_RATIO: float = 0.2
    VAL_MAX_BATCHES: int = 50
    MAX_VIDEO_EPISODES: int = 10
    PATH_TO_MOTION_FILE: str = (
        "/home/jovyan/Metamotivo/TextMetamotivo/DATASET_CTXT.hdf5"
    )
    EMBEDDING_KEY: str = "t5_emb"
    CTXT_KEY: str = "ctxt"
    VTEX_KEY: str = "vtxt"
    TRAJECTORY_KEY: str = "fb_traj_embedding"
    KEYS: list = [
        "observation",
        "rewritten_text",
        "text",
        "qpos",
        "qvel",
        "motion_id",
        "fused",
        "t5_emb",
        "ctxt",
        "vtxt",
        "fb_traj_embedding",
        "is_amass",
        "group",
        "ctxt_mask"
    ]
    GROUND_TRUTH_EVAL_METRICS: dict = {"success_phc_linf": 0.85, "emd": 1.33}
    interesting_keywords: list = [
        "cartwheel",
        "flip",
        "backflip",
        "handstand",
        "jump",
        "run",
        "stand",
        "walk",
        "squat",
    ]
    train_val_step: int = 50
    train_val_step_eval: int = 100
    m_loss: str = "mse"
    full_bench_eval_step: int = 5000
    conditioning_mode: Literal["single", "two_branch"] = "two_branch"
    EPOCHS: int = 80_000_000
    VAE_CHECKPOINT_PATH: str = "/home/jovyan/Metamotivo/TextMetamotivo/experiments/exp_vae_sem_final/fb_latent_vae_sem_checkpoints_val_MDM/ckpt_step_000030000.pt"
    VAE_CONFIG_PATH: str = "/home/jovyan/Metamotivo/TextMetamotivo/experiments/exp_vae_sem_final/fb_latent_vae_sem_loss_val_MDM/vae_model_config.json"
    GENERATOR_BACKBONE_CHECKPOINT_PATH: str = ""
    GENERATOR_INIT_CHECKPOINT_PATH: str = ""
    GENERATOR_INIT_STRICT: bool = False
    GENERATOR_RESUME_TRAINING_STATE: bool = False
