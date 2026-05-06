import rootutils
import warnings
import torch
import numpy as np
from typing import Sequence

from torch.utils.data import Dataset
from utils.text_motions import TextMotionBuffer

warnings.filterwarnings('ignore')
ROOT = rootutils.setup_root(search_from=__file__, cwd=True, pythonpath=False)
torch.set_float32_matmul_precision("high")


class TrainMotionDataset(Dataset):
    """
    Dataset that samples motions from a TextMotionBuffer according to a mixture distribution over motion types.
    Each motion is associated with a "group" (e.g. dataset name, motion type
    , etc.) specified by `group_key` in the motion metadata. The sampling distribution is defined as:
    1) Sample a group with probability p(group) ∝ count(group)^(-alpha)
    2) Sample a motion within the group, either uniformly or according to priorities if available.

    """
    def __init__(
        self,
        motion_buffer: TextMotionBuffer,
        subset_indices,
        epoch_len: int = 20_000_000,
        group_key: str = "group",
        alpha: float = 1.0,              
        use_priorities_within: bool = True,
        min_group_size: int = 1,        
        unknown_group: str = "unknown",
        verbose: bool = True,
    ):
        self.mb = motion_buffer
        self.epoch_len = epoch_len
        self.indices = np.asarray(subset_indices, dtype=np.int64)
        self.group_key = group_key

        # priorities
        prios = getattr(self.mb, "priorities", None)
        if prios is None:
            prios = np.ones(len(self.mb.storage), dtype=np.float64)
        else:
            prios = np.asarray(prios, dtype=np.float64)

        # ---- group indices by motion type ----
        by_group = {}
        for idx in self.indices:
            ep = self.mb.storage[int(idx)]
            g = ep.get(group_key, None)
            if g is None:
                g = unknown_group
            g = str(g)

            by_group.setdefault(g, []).append(int(idx))

        # min_group_size filtering
        by_group = {g: v for g, v in by_group.items() if len(v) >= min_group_size}
        if len(by_group) == 0:
            raise RuntimeError("No groups left after filtering. Check group_key/min_group_size.")

        self.groups = sorted(by_group.keys())
        self.by_group = {g: np.asarray(by_group[g], dtype=np.int64) for g in self.groups}

        # ---- p(group) ∝ count(group)^(-alpha) ----
        counts = np.array([len(self.by_group[g]) for g in self.groups], dtype=np.float64)
        w = np.power(np.clip(counts, 1.0, None), -alpha)
        self.p_group = w / w.sum()

        # ---- inside group probabilities or uniform ----
        self.p_within = {}
        for g in self.groups:
            idxs = self.by_group[g]
            if use_priorities_within:
                p = prios[idxs]
                p = np.clip(p, 1e-12, None)
                p = p / p.sum()
            else:
                p = np.ones(len(idxs), dtype=np.float64) / len(idxs)
            self.p_within[g] = p

        if verbose:
            # print top groups by size and p_group
            pairs = [(g, len(self.by_group[g]), float(self.p_group[i])) for i, g in enumerate(self.groups)]
            pairs.sort(key=lambda x: x[1], reverse=True)
            print(f"[Mixture-by-group] num_groups={len(self.groups)} alpha={alpha} use_priorities_within={use_priorities_within}")
            print("[Mixture-by-group] Top groups by size:")
            for g, c, pg in pairs[:15]:
                print(f"  {g:>20s}: count={c:>8d}  p_group={pg:.4f}")

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, _):
        # 1) sample group
        g = np.random.choice(self.groups, p=self.p_group)

        # 2) sample episode within group
        idxs = self.by_group[g]
        p = self.p_within[g]
        j = int(np.random.choice(len(idxs), p=p))
        ep_idx = int(idxs[j])

        return self.mb.storage[ep_idx]


class FixedSubsetMotionDataset(Dataset):
    """
    Deterministic dataset over a fixed subset of the motion buffer.

    This is intended for validation, where we want stable metrics across
    evaluations instead of re-sampling different examples every time.
    """

    def __init__(self, motion_buffer: TextMotionBuffer, subset_indices):
        self.mb = motion_buffer
        self.indices = np.asarray(subset_indices, dtype=np.int64)
        if self.indices.size == 0:
            raise RuntimeError("FixedSubsetMotionDataset received an empty subset.")

    def __len__(self):
        return int(self.indices.size)

    def __getitem__(self, index):
        ep_idx = int(self.indices[int(index)])
        return self.mb.storage[ep_idx]


def select_eval_episodes(
    raw_buffer: TextMotionBuffer,
    val_indices: Sequence[int] | np.ndarray,
    interesting_keywords: Sequence[str],
    max_video_episodes: int,
    text_key: str = "rewritten_text",
    rng: np.random.RandomState | np.random.Generator | None = None,
):
    """
    Select fixed validation episodes for qualitative video evaluation.

    Strategy:
    1) Scan val_indices in order and keep episodes whose text contains any keyword.
    2) If none found, fallback to random episodes from val_indices.
    """
    eval_episodes = []
    keywords_lc = [str(kw).lower() for kw in interesting_keywords]

    for idx in val_indices:
        ep = raw_buffer.storage[int(idx)]
        txt = str(ep.get(text_key, "")).lower()
        if any(kw in txt for kw in keywords_lc):
            eval_episodes.append(ep)
            if len(eval_episodes) >= max_video_episodes:
                break

    used_fallback = len(eval_episodes) == 0
    if used_fallback:
        if rng is None:
            for _ in range(max_video_episodes):
                eval_rand_ind = int(np.random.choice(val_indices))
                eval_episodes.append(raw_buffer.storage[eval_rand_ind])
        else:
            for _ in range(max_video_episodes):
                eval_rand_ind = int(rng.choice(val_indices))
                eval_episodes.append(raw_buffer.storage[eval_rand_ind])

    return eval_episodes, used_fallback
