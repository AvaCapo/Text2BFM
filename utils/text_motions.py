import h5py
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Any
import collections
import numbers

from tqdm.auto import tqdm
from humenv.misc.motionlib import MotionBuffer, canonicalize

def safe_decode_h5(val: Any) -> Any:
    """
    Helper to safely decode h5py data to numpy/python types.
    Handles Datasets, Arrays, Bytes, and Scalars.
    """
    if isinstance(val, h5py.Dataset):
        val = val[()]
    
    if isinstance(val, (np.ndarray, list)):
        if isinstance(val, np.ndarray):
            if val.size > 1:
                val = val.flatten()[0]
            elif val.size == 1:
                val = val.flat[0]
        elif isinstance(val, list) and len(val) > 0:
            val = val[0]

    if isinstance(val, (bytes, np.bytes_)):
        return val.decode('utf-8', errors='ignore')
    
    if hasattr(val, 'item') and (np.isscalar(val) or val.ndim == 0):
        return val.item()
        
    return val

def _normalize_filter_value(val: Any) -> Any:
    val = safe_decode_h5(val)
    if isinstance(val, np.ndarray):
        if val.size == 0:
            return None
        val = val.flatten()[0]
        if hasattr(val, "item"):
            val = val.item()
    return val


def load_task_based_h5(
    file: str,
    keys: List[str] | None = None,
    use_tqdm: bool = False,
    filter_field: str | None = None,
    filter_value: Any = True,
):
    """
    Loads HDF5 where root keys are Task Names/Motion Names rather than 'ep_0'.
    """
    hf = h5py.File(file, "r")
    data = []
    
    group_keys = list(hf.keys())####### [:5] limit the number of loaded samples
    tqdm_bar = tqdm(group_keys, leave=False) if use_tqdm else None
    
    for key in group_keys:
        group = hf[key]
        if filter_field is not None:
            if filter_field not in group:
                if tqdm_bar is not None:
                    tqdm_bar.update()
                continue

            episode_filter_value = _normalize_filter_value(group[filter_field])
            if episode_filter_value != filter_value:
                if tqdm_bar is not None:
                    tqdm_bar.update()
                continue

        fields_to_load = keys if keys is not None else group.keys()
        ep = {}
        
        # --- 1. Load Data Fields ---
        for k in fields_to_load:
            if k in group:
                val = group[k]
                if isinstance(val, h5py.Dataset):
                    arr = val[()]
                    if isinstance(arr, (bytes, np.bytes_)):
                         arr = safe_decode_h5(arr)
                    elif isinstance(arr, np.ndarray) and arr.dtype.kind == 'S':
                         if arr.size == 1:
                             arr = safe_decode_h5(arr)

                    # Reshape 1D static embeddings (D,) to (1, D) for the sampler logic
                    if isinstance(arr, np.ndarray) and arr.ndim == 1 and k != 'motion_id' and not np.issubdtype(arr.dtype, np.str_):
                        arr = arr[None, :] 
                        
                    ep[k] = arr

        # --- 2. Load Motion ID ---
        if 'motion_id' in group:
            mid = safe_decode_h5(group['motion_id'])
            ep['motion_id'] = np.array([mid])
        else:
            ep['motion_id'] = np.array([-1])

        ep["file_name"] = file
        ep["task_name"] = key
        data.append(ep)
        
        if tqdm_bar is not None:
            tqdm_bar.update()
            
    return data

class TextMotionBuffer(MotionBuffer):
    def __init__(
        self,
        files: List[str] | str,
        base_path: str | None = None,
        keys: list[str] = ["fb_traj_embedding", "text_embedding", "text"],
        filter_field: str | None = None,
        filter_value: Any = True,
    ) -> None:
        self.filter_field = filter_field
        self.filter_value = filter_value
        self.storage, motion_ids, self.file_names = [], [], []
        files = [files] if isinstance(files, str) else files
        
        if len(files) == 0:
            raise ValueError("MotionBuffer received no files to load.")
            
        for f in files:
            if f.endswith("txt"):
                with open(f, "r") as txtf:
                    h5files = [el.strip().replace(" ", "") for el in txtf.readlines()]
                episodes = []
                for h5 in tqdm(h5files, leave=False, colour='green', position=0):
                    h5 = canonicalize(h5, base_path=base_path)
                    episodes.extend(
                        load_task_based_h5(
                            h5,
                            keys=keys,
                            filter_field=filter_field,
                            filter_value=filter_value,
                        )
                    )
            else:
                h5 = canonicalize(f, base_path=base_path)
                episodes = load_task_based_h5(
                    h5,
                    keys=keys,
                    use_tqdm=True,
                    filter_field=filter_field,
                    filter_value=filter_value,
                )
            
            for ep in episodes:
                if "motion_id" in ep and ep["motion_id"].size > 0:
                    _mid = ep["motion_id"].item()
                else:
                    _mid = -1

                _e = {}
                for k in keys:
                    if k in ep:
                        _e[k] = ep[k]

                _e["motion_id"] = _mid
                _e["task_name"] = ep.get("task_name", "Unknown")

                self.storage.append(_e)
                motion_ids.append(_mid)
                self.file_names.append(str(ep["task_name"]))

        if len(self.storage) == 0:
            filter_desc = ""
            if filter_field is not None:
                filter_desc = f" for {filter_field} == {filter_value!r}"
            raise ValueError(f"MotionBuffer loaded no episodes{filter_desc}.")

        self.motion_ids = np.array(motion_ids)
        self.priorities = np.ones_like(self.motion_ids, dtype=np.float64) / len(self.motion_ids)
    
    def sample(self, batch_size: int = 1, max_len: int = -1) -> Dict[str, np.ndarray]:
        """
        Samples full trajectories, padding them to the max length in the batch.
        
        Args:
            batch_size: Number of trajectories to sample.
            max_len: If > 0, truncates trajectories to this length and pads shorter ones 
                     to this length. If -1, pads to the maximum length found in the batch.
                     
        Returns:
            Dict containing:
            - Temporal fields: (Batch, Max_Len, D)
            - Static fields: (Batch, D)
            - mask: (Batch, Max_Len) boolean mask where True indicates valid data
        """
        # 1. Select Episodes
        self.ep_ind = np.random.choice(len(self), p=self.priorities, size=batch_size, replace=True)
        
        batch_entries = []
        max_found_len = 0
        
        # 2. Determine Max Length in this Batch
        for ep_idx in self.ep_ind:
            data = self.storage[ep_idx]
            curr_len = 1
            # Scan for temporal arrays to determine length
            for v in data.values():
                if isinstance(v, np.ndarray) and v.ndim > 1 and v.shape[0] > 1:
                     curr_len = max(curr_len, v.shape[0])
            
            max_found_len = max(max_found_len, curr_len)
            batch_entries.append((data, curr_len))

        # Determine target dimension size
        if max_len > 0:
            target_timesteps = max_len
        else:
            target_timesteps = max_found_len

        # 3. Construct Output Arrays
        result = {}
        if not batch_entries:
            return result
            
        # Use keys from first entry as template
        keys = batch_entries[0][0].keys()
        
        for k in keys:
            first_val = batch_entries[0][0][k]
            
            # --- Type A: Text / Strings ---
            if isinstance(first_val, (str, np.str_)):
                vals = [entry[0][k] for entry in batch_entries]
                result[k] = np.array(vals)
                
            # --- Type B: Scalars (Numbers) ---
            elif isinstance(first_val, (numbers.Number, np.number)):
                vals = [entry[0][k] for entry in batch_entries]
                result[k] = np.array(vals).reshape(batch_size, -1)
                
            # --- Type C: Numpy Arrays (Embeddings vs Trajectories) ---
            elif isinstance(first_val, np.ndarray):
                # Is it Temporal (T > 1)?
                if first_val.ndim > 1 and first_val.shape[0] > 1:
                    # Initialize Padded Array: (Batch, Target_Time, Dim)
                    dim = first_val.shape[1]
                    padded_arr = np.zeros((batch_size, target_timesteps, dim), dtype=first_val.dtype)
                    
                    for i, (entry, length) in enumerate(batch_entries):
                        val = entry[k]
                        
                        # Determine how much to copy (Truncate if val is too long, copy all if short)
                        src_len = val.shape[0]
                        copy_len = min(src_len, target_timesteps)
                        
                        padded_arr[i, :copy_len, :] = val[:copy_len]
                    
                    result[k] = padded_arr
                else:
                    # It is Static (1, D) -> Stack to (Batch, D)
                    dim = first_val.shape[-1]
                    stacked_arr = np.zeros((batch_size, dim), dtype=first_val.dtype)
                    
                    for i, (entry, _) in enumerate(batch_entries):
                        val = entry[k]
                        # Flatten to ensure shape is (D,) before assigning to row i
                        stacked_arr[i, :] = val.flatten()
                        
                    result[k] = stacked_arr

        # 4. Create Mask
        # (Batch, Target_Time) - True for real data, False for padding
        mask = np.zeros((batch_size, target_timesteps), dtype=bool)
        for i, (_, length) in enumerate(batch_entries):
            # Mark valid regions (respecting truncation)
            valid_len = min(length, target_timesteps)
            mask[i, :valid_len] = True
        
        result['mask'] = mask
        
        return result
