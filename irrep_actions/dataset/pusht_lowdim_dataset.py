import numpy as np
from irrep_actions.dataset.base_dataset import BaseDataset

class PushTLowdimDataset(BaseDataset):
    def __init__(
        self,
        path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        harmonic_action: bool = False,
        seed=0,
        val_ratio=0.0,
        max_train_episodes=None,
    ):
        super().__init__(
            path,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            buffer_keys=['keypoint', 'state' , 'action'],
            harmonic_action=harmonic_action,
            seed=seed,
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes
        )

    def _sample_to_data(self, sample):
        keypoint = sample['keypoint']
        state = sample['state']
        agent_pos = state[:,:2]
        obs = np.concatenate([
            keypoint.reshape(keypoint.shape[0], -1),
            agent_pos], axis=-1
        )

        T = obs.shape[0]
        Do = obs.shape[-1] // 2
        x_obs = (obs.reshape(-1,Do,2)[:,:,0] - 255.0)
        y_obs = (obs.reshape(-1,Do,2)[:,:,1] - 255.0) * -1
        obs = np.concatenate((x_obs[..., np.newaxis], y_obs[..., np.newaxis]), axis=-1).reshape(T, -1)

        x_act = sample['action'][:,0]
        y_act = sample['action'][:,1] * -1
        action = np.concatenate((x_act[..., np.newaxis], y_act[..., np.newaxis]), axis=-1).reshape(T, 2)

        data = {
            'obs': obs,
            "action": action,  # T, D_a
        }
        return data
