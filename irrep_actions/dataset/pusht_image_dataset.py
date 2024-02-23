import numpy as np
from irrep_actions.dataset.base_dataset import BaseDataset

class PushTImageDataset(BaseDataset):
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
            buffer_keys=['img', 'state' , 'action'],
            harmonic_action=harmonic_action,
            seed=seed,
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes
        )

    def _sample_to_data(self, sample):
        obs = np.moveaxis(sample['img'],-1,1) / 255

        T = obs.shape[0]

        x_act = sample['action'][:,0]
        y_act = sample['action'][:,1] * -1
        action = np.concatenate((x_act[..., np.newaxis], y_act[..., np.newaxis]), axis=-1).reshape(T, 2)

        data = {
            'obs': obs, # T, D_o
            "action": action,  # T, D_a
        }
        return data
