import os
from utils.text_motions import TextMotionBuffer
from humenv.bench.tracking_evaluation import TrackingEvaluation, _calc_metrics
from typing import Any, Dict, List, Literal
import dataclasses
import multiprocessing
import numpy as np
import functools
from humenv.misc.motionlib import MotionBuffer
from humenv.bench.gym_utils.episodes import Episode
from humenv import make_humenv, CustomManager
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import torch

CustomManager.register("TextMotionBuffer", TextMotionBuffer)

@dataclasses.dataclass(kw_only=True)
class TextTrackingEvaluation(TrackingEvaluation):
    motions: List[str] | str
    keys: list[str]
    motion_buffer: TextMotionBuffer | None = None
    filter_field: str | None = None
    filter_value: Any = True
    
    def __post_init__(self):
        if self.num_envs > 1:
            self.mp_manager = CustomManager()
            self.mp_manager.start()
            self.motion_buffer = self.mp_manager.TextMotionBuffer(
                files=self.motions,
                keys=self.keys,
                filter_field=self.filter_field,
                filter_value=self.filter_value,
            )
        else:
            self.mp_manager = None
            if self.motion_buffer is None:
                self.motion_buffer = TextMotionBuffer(
                    files=self.motions,
                    keys=self.keys,
                    filter_field=self.filter_field,
                    filter_value=self.filter_value,
                )
    
                
    def run_text(
        self,
        agent: Any,
        device: str = 'cuda',
        emb_name: str = 'fused',
        conditioning_mode: Literal['single', 'two_branch'] = 'single',
        vtxt_name: str = 'vtxt',
        ctxt_name: str = 'ctxt',
        ctxt_mask_name: str = 'ctxt_mask',
        filter_field: str | None = 'is_amass',
        filter_value: Any = True,
    ) -> Dict[str, Any]:
        if self.motion_buffer is None:
            raise RuntimeError("motion_buffer is not initialized")

        ids = self.motion_buffer.get_motion_ids()
        if filter_field is not None:
            filtered_ids = []
            for i in ids:
                episode_filter_value = self.motion_buffer.get(i).get(filter_field)
                if isinstance(episode_filter_value, np.ndarray):
                    if episode_filter_value.size == 0:
                        continue
                    episode_filter_value = episode_filter_value.reshape(-1)[0]
                    if hasattr(episode_filter_value, "item"):
                        episode_filter_value = episode_filter_value.item()

                if episode_filter_value == filter_value:
                    filtered_ids.append(i)
            ids = filtered_ids

        if len(ids) == 0:
            return {}

        np.random.shuffle(ids)  # shuffle the ids to evenly distribute the motions, as different datasets have different motion length
        num_workers = min(self.num_envs, len(ids))
        motions_per_worker = np.array_split(ids, num_workers)
        f = functools.partial(
            _async_tracking_worker_text,
            wrappers=self.wrappers,
            env_kwargs=self.env_kwargs,
            motion_buffer=self.motion_buffer,
            device=device,
            emb_name=emb_name,
            conditioning_mode=conditioning_mode,
            vtxt_name=vtxt_name,
            ctxt_name=ctxt_name,
            ctxt_mask_name=ctxt_mask_name,

        )
        if num_workers == 1:
            metrics = f((motions_per_worker[0], 0, agent))
        else:
            prev_omp_num_th = os.environ.get("OMP_NUM_THREADS", None)
            os.environ["OMP_NUM_THREADS"] = "1"
            try:
                with ProcessPoolExecutor(
                    max_workers=num_workers,
                    mp_context=multiprocessing.get_context(self.mp_context),
                ) as pool:
                    inputs = [(x, y, agent) for x, y in zip(motions_per_worker, range(len(motions_per_worker)))]
                    list_res = pool.map(f, inputs)
                    metrics = {}
                    for el in list_res:
                        metrics.update(el)
            finally:
                if prev_omp_num_th is None:
                    del os.environ["OMP_NUM_THREADS"]
                else:
                    os.environ["OMP_NUM_THREADS"] = prev_omp_num_th
        return metrics
    
def _to_1batch_tensor(arr: np.ndarray, device: str) -> torch.Tensor:
    t = torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(device, non_blocking=True)
    if t.dim() == 1:
        return t.unsqueeze(0)
    if t.dim() == 2:
        return t.unsqueeze(0)
    if t.dim() == 3:
        return t
    raise ValueError(f"Unexpected tensor dim={t.dim()} shape={tuple(t.shape)}")


def _infer_ctxt_mask_from_ctxt(ctxt: torch.Tensor) -> torch.Tensor:
    if ctxt.dim() == 1:
        ctxt = ctxt.unsqueeze(0).unsqueeze(0)
    elif ctxt.dim() == 2:
        ctxt = ctxt.unsqueeze(0)
    elif ctxt.dim() != 3:
        raise ValueError(f"Unexpected ctxt dim={ctxt.dim()} shape={tuple(ctxt.shape)}")

    return (ctxt.abs().sum(dim=-1) > 0).bool()


def _async_tracking_worker_text(
    inputs,
    wrappers,
    env_kwargs,
    motion_buffer: MotionBuffer,
    device: str = 'cuda',
    emb_name: str = 'fused',
    conditioning_mode: Literal['single', 'two_branch'] = 'single',
    vtxt_name: str = 'vtxt',
    ctxt_name: str = 'ctxt',
    ctxt_mask_name: str = 'ctxt_mask',
):
    motion_ids, pos, agent = inputs
    env = make_humenv(num_envs=1, wrappers=wrappers, **env_kwargs)[0]
    metrics = {}
    for m_id in tqdm(motion_ids, position=pos, leave=False, disable=pos > 0):
        ep_ = motion_buffer.get(m_id)
        # we ignore the first state since we need to pass the next observation
        tracking_target = ep_["observation"][1:1+214]

        if conditioning_mode == 'single':
            # tracking_text_emb = _to_1batch_tensor(ep_[emb_name], device=device)
            tracking_text_emb = torch.from_numpy(
                    np.asarray(ep_[emb_name], dtype=np.float32)
                ).to(device, non_blocking=True)
            ctx = agent.tracking_inference(tracking_text_emb, tracking_target.shape[0]).squeeze(0)
        elif conditioning_mode == 'two_branch':
            if vtxt_name not in ep_ or ctxt_name not in ep_:
                raise KeyError(
                    f"Episode must contain '{vtxt_name}' and '{ctxt_name}' for two_branch conditioning. "
                    f"Got keys: {list(ep_.keys())}"
                )

            vtxt = _to_1batch_tensor(ep_[vtxt_name], device=device)
            ctxt = _to_1batch_tensor(ep_[ctxt_name], device=device)

            if ctxt_mask_name in ep_ and ep_[ctxt_mask_name] is not None:
                ctxt_mask_np = np.asarray(ep_[ctxt_mask_name])
                ctxt_mask = torch.from_numpy(ctxt_mask_np.astype(np.bool_)).to(device, non_blocking=True)
                if ctxt_mask.dim() == 1:
                    ctxt_mask = ctxt_mask.unsqueeze(0)
                elif ctxt_mask.dim() != 2:
                    raise ValueError(f"Unexpected {ctxt_mask_name} shape={ctxt_mask.shape}")
            else:
                ctxt_mask = _infer_ctxt_mask_from_ctxt(ctxt)

            ctx = agent.tracking_inference(
                vtxt_input=vtxt,
                ctxt_input=ctxt,
                ctxt_mask=ctxt_mask,
                seq_length=tracking_target.shape[0],
                guidance_scale=None,
            ).squeeze(0)
        else:
            raise ValueError(f"Unknown conditioning_mode: {conditioning_mode}")

        observation, info = env.reset(options={"qpos": ep_["qpos"][0], "qvel": ep_["qvel"][0]})
        _episode = Episode()
        _episode.initialise(observation, info)
        for i in range(min(len(ctx), 214)):
            observation = torch.as_tensor(
                        observation.reshape(1, -1),
                        dtype=torch.float32,
                        device=device
            )
            action = agent.act(obs=observation, z=ctx[i][None]).ravel()
            observation, reward, terminated, truncated, info = env.step(action.cpu())
            _episode.add(observation, reward, action.cpu().numpy(), terminated, truncated, info)
        tmp = _episode.get()
        tmp["tracking_target"] = tracking_target
        tmp["motion_id"] = m_id
        tmp["motion_file"] = motion_buffer.get_name(m_id)
        metrics.update(_calc_metrics(tmp))
    env.close()
    return metrics
