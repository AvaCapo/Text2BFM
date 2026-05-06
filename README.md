# Text2BFM

Text2BFM is a research codebase for learning text-conditioned motion representations that can be executed by the Metamotivo humanoid tracking policy.

The project does not generate raw joint trajectories directly. Instead, it learns to predict sequences of `fb_traj_embedding` vectors from text, and those vectors are then consumed by the Metamotivo FB-CPR policy to produce actions in HumEnv. In practice, this gives you a pipeline that looks like:

1. Text features in.
2. A motion-representation sequence out.
3. Metamotivo tracking policy converts that sequence into humanoid behavior.

The current repository is organized around a two-stage training setup:

1. A semantic VAE compresses and reconstructs `fb_traj_embedding` trajectories while aligning them with text.
2. A text-to-latent generator learns to produce VAE latents from a two-branch text condition.

`main.py` currently launches the stage-2 generator training path.

## What This Repository Learns

The learning target throughout the repo is the sequence stored under `fb_traj_embedding` in the dataset.

Conceptually:

1. `fb_traj_embedding` is a compact motion representation already compatible with Metamotivo.
2. The VAE learns a smoother and lower-dimensional latent motion space `m`.
3. The generator learns to map text into that latent motion space.
4. The latent motion is decoded back into `fb_traj_embedding`.
5. The tracking policy turns decoded embeddings into actions that animate the humanoid.

This means the repo is not a generic text-to-motion system in joint-angle space. It is specifically a text-to-Metamotivo-trajectory system.

## Conditioning Setup

The active generator uses a two-branch text interface:

1. `ctxt`: token-level text features, typically a T5-style contextual embedding matrix.
2. `vtxt`: a single global text embedding, typically a CLIP-like sentence vector.
3. `ctxt_mask`: an optional token mask for `ctxt`.

The generator adapts these text features with `text_adapter/MultiModalTextConditionAdapter` before they are passed into the motion backbone.

## Training Pipeline

### Stage 1: Semantic VAE

The stage-1 script is `train_vae_sem.py`.

Its job is to learn a latent motion code `m` from `fb_traj_embedding` trajectories while keeping that latent space both reconstructive and semantically aligned with text.

The stage-1 model contains:

1. A temporal VAE from `vae/vae.py`.
2. A multimodal text adapter from `text_adapter/text_adapter.py`.
3. A motion-text contrastive loss from `vae/sem_loss.py`.
4. An optional global-text contrastive head for `vtxt`.
5. A behavior-preserving KL-style action loss from `vae/fb_loss.py`.

The main losses used in stage 1 are:

1. Reconstruction loss on `fb_traj_embedding`.
2. Velocity reconstruction loss.
3. KL loss for the latent posterior when the model is not in AE mode.
4. Motion-to-text semantic contrastive loss using token-level text.
5. Optional `vtxt` contrastive loss using the global text vector.
6. FB action KL loss that encourages reconstructed trajectories to induce similar policy behavior.
7. Optional latent L2 regularization.

What stage 1 learns in practice:

1. A compressed latent representation of the motion trajectory.
2. A decoder that can reconstruct trajectory embeddings well enough for downstream tracking.
3. A latent space that is better aligned with language than a plain reconstruction-only VAE.

### Stage 2: Text-to-BFM Generator

The stage-2 script is `train_text2bfm.py`, and `main.py` is a small wrapper around it.

Stage 2 loads a pretrained VAE and trains `src/generator.py`, which is a flow-matching latent generator built around a HY-Motion style MMDiT backbone.

The stage-2 model contains:

1. A frozen or partially frozen VAE encoder/decoder.
2. A multimodal text adapter.
3. A Hunyuan-style MMDiT motion backbone from `hymotion/network/hymotion_mmdit.py`.
4. EMA tracking for the trainable generator modules.

The generator is trained in VAE latent space:

1. The ground-truth `fb_traj_embedding` sequence is encoded into latent motion `m`.
2. Noise is mixed with the latent motion.
3. The transformer predicts flow toward the clean latent sequence.
4. The predicted clean latent is decoded back into `fb_traj_embedding`.

The main stage-2 losses are:

1. Flow matching loss in latent space.
2. Latent reconstruction loss on the predicted clean latent.
3. Cosine similarity loss on the latent reconstruction.
4. Reconstruction loss after decoding back to `fb_traj_embedding`.
5. Optional velocity loss in decoded trajectory space.
6. Optional policy consistency loss against the frozen Metamotivo actor.

What stage 2 learns in practice:

1. How to turn text into a motion latent trajectory.
2. How to preserve behavior after decoding through the VAE.
3. How to produce motion that can be tracked by the downstream policy.

## Dataset Format

The repository expects an HDF5 dataset where each root group corresponds to one motion episode or task example.

Important fields used by the current training code are:

| Key | Type / shape | Purpose |
| --- | --- | --- |
| `fb_traj_embedding` | `[T, D]` | Main supervision target for both VAE and generator |
| `observation` | `[T, obs_dim]` | Humanoid observations used in evaluation and FB action loss |
| `ctxt` | `[Lt, D_ctxt]` | Token-level text embeddings |
| `vtxt` | `[D_vtxt]` or `[1, D_vtxt]` | Global text embedding |
| `ctxt_mask` | `[Lt]` | Optional boolean mask for valid text tokens |
| `text` / `rewritten_text` | string | Human-readable text prompt |
| `qpos` | `[T, ...]` | Initial state and tracking rollout reset state |
| `qvel` | `[T, ...]` | Initial state and tracking rollout reset state |
| `motion_id` | scalar | Motion identifier |
| `group` | scalar or string | Sampling group for balanced training |
| `is_amass` | bool | Optional filter used by the active generator path |

Notes about loading:

1. `utils/text_motions.py` loads root groups from HDF5 and converts them into in-memory episode dicts.
2. One-dimensional static arrays are reshaped so the samplers can treat them consistently.
3. `utils/train_utils.collate_padded` pads variable-length trajectories and creates a boolean `mask`.
4. The current generator path filters the dataset with `is_amass == True`.

## Repository Layout

| Path | Role |
| --- | --- |
| `main.py` | Default entrypoint for stage-2 generator training |
| `train_text2bfm.py` | Main text-to-BFM generator training script |
| `train_vae_sem.py` | Stage-1 semantic VAE training script |
| `src/generator.py` | Generator model, EMA, latent flow training, inference |
| `src/config.py` | Active generator-only config dataclasses |
| `vae/` | VAE architecture, VAE configs, semantic losses, FB action loss |
| `text_adapter/` | Adapters for token-level and global text conditioning |
| `utils/` | Data loading, padding, evaluation, checkpoint/video helpers |
| `hymotion/` | HY-Motion backbone components used by the generator |
| `metamotivo/` | Vendored Metamotivo code used for tracking and FB-CPR models |
| `configs/` | Hydra config entrypoint and agent seed config |

## Installation

The project targets Python 3.11+.

With `uv`:

```bash
uv sync
```

With `pip`:

```bash
pip install -e .
```

Important external requirements:

1. A working PyTorch CUDA installation if you want to train on GPU.
2. MuJoCo and EGL support for HumEnv rendering and benchmark rollouts.
3. Access to the Metamotivo model weights `facebook/metamotivo-S-1`.

## Metamotivo Weights

Several scripts load the Metamotivo base policy with `local_files_only=True`. That means the code expects the checkpoint to already exist in your local Hugging Face cache.

If the model is not already cached, the relevant scripts will fail instead of downloading it automatically.

## Configuration

There are three main config locations:

1. `config.py`
2. `src/config.py`
3. `vae/config.py`

### `config.py`

This file contains project-level runtime settings such as:

1. Dataset path.
2. Dataset keys.
3. Validation ratio.
4. Text key names.
5. VAE checkpoint path used by stage 2.
6. Generator init checkpoint path.

You will almost certainly need to edit this file before running on a new machine.

### `src/config.py`

This file contains the active stage-2 dataclasses:

1. `GeneratorTrainConfig`
2. `GeneratorAdapterConfig`

These control:

1. Batch size and learning rate.
2. Warmup and LR schedule.
3. Backbone depth, heads, feature sizes.
4. Classifier-free style conditioning dropout.
5. EMA settings.
6. VAE freeze policy.
7. Loss weights.
8. Inference steps and guidance scale.

### `vae/config.py`

This file defines:

1. `VAEConfig`
2. `VAETrainConfig`
3. `VAESemTrainConfig`

These control:

1. VAE width, depth, downsampling, latent size.
2. Reconstruction and KL weighting.
3. Text-learning rates and semantic loss weights.
4. Augmentation settings for stage-1 training.

## How To Run

### Current default path: stage 2 generator training

```bash
python main.py
```

This calls `train_text2bfm.py`.

You can also run it directly:

```bash
python train_text2bfm.py
```

Because the script is wrapped with Hydra, extra CLI overrides are forwarded as usual. Example:

```bash
python main.py agent.global_seed=123
```

Before running stage 2, make sure:

1. `Config.PATH_TO_MOTION_FILE` points to your HDF5 dataset.
2. `Config.VAE_CHECKPOINT_PATH` points to a trained stage-1 VAE checkpoint.
3. `Config.VAE_CONFIG_PATH` points to the matching VAE config JSON.
4. Metamotivo base weights are already cached locally.

### Stage 1 semantic VAE training

The research script for stage 1 lives in:

```bash
python train_vae_sem.py
```

Useful environment variables supported by the script:

1. `VAE_SEM_EXP_NAME`
2. `VAE_SEM_TRAIN_CFG_JSON`
3. `VAE_MODEL_CFG_JSON`

These allow you to override the experiment directory name and load JSON overrides for dataclass configs.

## Training Outputs

Both training stages save artifacts under experiment-specific folders.

Typical outputs include:

1. Checkpoints as `ckpt_step_XXXXXXXXX.pt`.
2. Saved config JSON files used for the run.
3. `losses.pkl` for offline analysis.
4. `val_metrics.jsonl` for validation history.
5. Side-by-side MP4 videos for qualitative inspection.

Stage 2 additionally saves benchmark rollouts against the tracking evaluator.

## Evaluation

The main evaluation path uses `utils/text_tracking_evaluation.py`.

For the generator:

1. The model predicts a sequence of `fb_traj_embedding` vectors from text.
2. The Metamotivo tracking wrapper executes those vectors in HumEnv.
3. Tracking metrics are computed against the stored target motion.

The code logs benchmark values such as:

1. `success_phc_linf`
2. `emd`

Here is why these two metrics matter:

1. `success_phc_linf` is the benchmark's thresholded tracking-success score. In HumEnv naming, it is tied to a PHC-style `L_inf` error criterion, so it answers the practical question: did the generated motion stay close enough to the target rollout to count as a successful track. Higher is better.
2. `emd` is the benchmark's continuous distance score between the generated and target motion rollout. Lower is better. It is useful because it still tells you how close the motion was even when the sequence does not cross the success threshold.

We report both because they capture different failure modes:

1. `success_phc_linf` tells you whether the motion is good enough to be considered solved.
2. `emd` tells you how far the model is from the target in a graded way.

Using both avoids a misleading picture where a model looks decent under a soft distance metric but rarely achieves real tracking success, or where a model passes a threshold sometimes but is unstable in overall motion fidelity.

It also saves side-by-side videos comparing generated tracking against ground truth tracking.

## Important Practical Notes

1. `config.py` contains machine-specific absolute paths in the current version of the repo.
2. Several scripts hardcode `MUJOCO_GL` and `CUDA_VISIBLE_DEVICES`; adjust them for your machine or cluster environment.
3. The repo assumes precomputed text features already exist in the dataset. It does not currently compute T5 or CLIP embeddings on the fly during training.
4. The active code under `src/` is now generator-focused. Some older research scripts may still reflect earlier naming or wrapper assumptions.

## Summary

Text2BFM is a text-conditioned motion-representation learning project built around Metamotivo.

The central idea is:

1. Compress trajectory embeddings with a semantic VAE.
2. Learn a text-conditioned latent generator in that compressed space.
3. Decode back into `fb_traj_embedding`.
4. Execute the result with the Metamotivo tracking policy.

If you are trying to understand the repo quickly, start with these files:

1. `train_text2bfm.py`
2. `src/generator.py`
3. `train_vae_sem.py`
4. `vae/config.py`
5. `utils/text_motions.py`
