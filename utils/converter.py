import numpy as np

from utils.skeleton_structure import body_ids

class HumenvSMPLConverter:
    def __init__(self):
        pass

    @staticmethod
    def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiplies two quaternions q1 and q2. Both q1 and q2 should have shape (..., 4) 
            where the last dimension represents (w, x, y, z).
            Returns the product quaternion with the same shape.
        
        """
        # MuJoCo quats are (w, x, y, z)
        w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
        w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return np.stack([w, x, y, z], axis=-1)
    
    @staticmethod
    def quat_conj(q: np.ndarray) -> np.ndarray:
        """Returns the conjugate of a quaternion q. The input q should have shape (..., 4)
            where the last dimension represents (w, x, y, z)."""
        qc = q.copy()
        qc[..., 1:] *= -1.0
        return qc

    @staticmethod
    def quat_inv(q: np.ndarray) -> np.ndarray:
        """Returns the inverse of a unit quaternion q. The input q should have shape (..., 4)
            where the last dimension represents (w, x, y, z)."""
        # unit quaternion inverse = conjugate
        return HumenvSMPLConverter.quat_conj(q)
    
    @staticmethod
    def quat_to_axis_angle(q, eps=1e-8):
        """Converts a quaternion q to axis-angle representation. The input q should have shape (..., 4)
            where the last dimension represents (w, x, y, z). The output is the axis-angle representation with shape (..., 3).
            The axis is normalized and multiplied by the rotation angle in radians.
        """
        # q: (...,4) in (w,x,y,z)
        q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + eps)
        w = np.clip(q[..., 0], -1.0, 1.0)
        angle = 2.0 * np.arccos(w)  # (...,)
        s = np.sqrt(1.0 - w*w)      # (...,)
        axis = np.zeros(q.shape[:-1] + (3,), dtype=np.float64)
        mask = s > eps
        axis[mask] = q[mask, 1:] / s[mask][..., None]
        # axis-angle = axis * angle
        aa = axis * angle[..., None]
        return aa
    
    @staticmethod
    def mujoco_bodies_to_smpl_pose72(model,
                                     data:np.ndarray) -> np.ndarray:
        """Converts MuJoCo body orientations to SMPL pose parameters.
            The input data should have shape (T, num_bodies, 4) where the last dimension represents quaternions (w, x, y, z).
            The output is the SMPL pose parameters with shape (T, 72).
        """
       
        gq = data.xquat[body_ids].copy() 
        parent = model.body_parentid[body_ids]
        id_to_local_index = {bid: i for i, bid in enumerate(body_ids)}

        lq = np.zeros_like(gq)
        for i, bid in enumerate(body_ids):
            pid = parent[i]
            if pid in id_to_local_index:
                p_i = id_to_local_index[pid]
                lq[i] = HumenvSMPLConverter.quat_mul(
                    HumenvSMPLConverter.quat_inv(gq[p_i]), gq[i]
                    )
            else:
                # root relative to world
                lq[i] = gq[i]
        
        aa = HumenvSMPLConverter.quat_to_axis_angle(lq)
        pose72 = aa.reshape(-1)
        return pose72
    


# USAGE: with Env and Tracker

# def rollout_to_smpl_npz(env, tracker, z_seq, out_path="motion_smpl.npz", fps=30):
#     u = env.unwrapped

   
#     print(body_ids)

#     poses = []
#     trans = []
    

#     obs, info = env.reset()
#     frames = [env.render()]
#     with torch.no_grad():
#         for t in range(len(z_seq)):
#             obs_t = torch.as_tensor(obs.reshape(1, -1), dtype=torch.float32, device=z_seq.device)
#             action = tracker.act(obs=obs_t, z=z_seq[t]).ravel()
#             obs, _, _, _, info = env.step(action)
#             frames.append(env.render())

#             qpos = info["qpos"] 
#             trans.append(qpos[:3].copy())

#             pose72 = mujoco_bodies_to_smpl_pose72(u.model, u.data, body_ids)
#             poses.append(pose72.astype(np.float32))

#     poses = np.stack(poses, axis=0) 
#     trans = np.stack(trans, axis=0).astype(np.float32) 

#     betas = np.zeros(10, dtype=np.float32)  

#     np.savez(
#         out_path,
#         poses=poses,
#         trans=trans,
#         betas=betas,
#         mocap_framerate=fps,
#         gender = 'neutral'
#     )
#     return out_path,frames
