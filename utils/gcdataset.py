import torch
import numpy as np
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Dict, List, Union
from metamotivo.buffers.buffers import DictBuffer, extract_values

class GCDictBuffer(DictBuffer):
    value_p_curgoal: float = 0.0 #Probability of using the current state as the value goal.
    value_p_trajgoal: float = 1.0 #Probability of using a future state in the same trajectory as the value goal.
    value_p_randomgoal: float = 0.0 #Probability of using a random state as the value goal.
    actor_p_curgoal: float = 0.5 #Probability of using the current state as the actor goal.
    actor_p_trajgoal: float = 0.5#Probability of using a future state in the same trajectory as the actor goal.
    actor_p_randomgoal: float = 0.0 #Probability of using a random state as the actor goal.
    discount: float = 0.99
    gc_negative: bool = True
    actor_geom_sample: bool = False
    value_geom_sample: bool = True
    
    
    def _init_(self):
        self.terminal_locs, _ = torch.where(self.storage['terminated'] | self.storage['truncated'])
        self.initial_locs = torch.concatenate([torch.tensor([0]), self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == len(self) - 1
        
        assert np.isclose(
            self.value_p_curgoal + self.value_p_trajgoal + self.value_p_randomgoal, 1.0
        )
        assert np.isclose(
            self.actor_p_curgoal + self.actor_p_trajgoal + self.actor_p_randomgoal, 1.0
        )
        
    @torch.no_grad
    def sample(self, batch_size, idxs=None) -> Dict[str, torch.Tensor]:
        if idxs is None:
            self.ind = torch.randint(0, len(self), (batch_size,))
            
        batch = extract_values(self.storage, self.ind)
        value_goal_idxs = self.sample_goals(self.ind, self.value_p_curgoal, self.value_p_trajgoal,
                                            self.value_p_randomgoal, self.value_geom_sample)
        actor_goal_idxs = self.sample_goals(self.ind, self.actor_p_curgoal, self.actor_p_trajgoal,
                                            self.actor_p_randomgoal, self.actor_geom_sample)
        batch['value_goals'] = self.storage['observation'][value_goal_idxs]
        batch['actor_goals'] = self.storage['observation'][actor_goal_idxs]
        batch['value_goal_idxs'] = value_goal_idxs
        batch['actor_goal_idxs'] = actor_goal_idxs
        batch['obs_idxs'] = self.ind
        successes = torch.tensor(self.ind == value_goal_idxs, dtype=torch.float32)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.gc_negative else 0.0)
        
        return batch
        
    def sample_goals(self, indxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample):
        batch_size = len(indxs)
        random_goal_idxs = torch.randint(0, len(self), (batch_size, ))
        final_state_idxs = self.terminal_locs[torch.searchsorted(self.terminal_locs, indxs)]
        
        if geom_sample:
            # Geometric sampling.
            offsets = torch.distributions.Geometric(probs=1 - self.discount).sample((batch_size,)) + 1
            traj_goal_idxs = np.minimum(indxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = torch.rand(batch_size)  # in [0, 1)
            traj_goal_idxs = np.round(
                (np.minimum(indxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            )
        
        if p_curgoal == 1.0:
            goal_idxs = indxs
        else:
            goal_idxs = torch.where(
                torch.randn(batch_size) < p_trajgoal / (1.0 - p_curgoal), traj_goal_idxs, random_goal_idxs
            )

            # Goals at the current state.
            goal_idxs = torch.where(torch.randn(batch_size) < p_curgoal, indxs, goal_idxs)

        return goal_idxs.to(torch.int32)