import numpy as np
from irrep_actions.dataset.base_dataset import BaseDataset

class BlockPushingLowdimDataset(BaseDataset):
    def __init__(
        self,
        path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        obs_eef_target: bool = True,
        harmonic_action: bool = False,
        seed=0,
        val_ratio=0.0,
        max_train_episodes=None,
    ):
        self.obs_eef_target = obs_eef_target

        super().__init__(
            path,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            harmonic_action=harmonic_action,
            seed=seed,
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes
        )

    def _sample_to_data(self, sample):
        obs = sample['obs']
        if not self.obs_eef_target:
            obs[:, 8:10] = 0

        data = {
            'obs': obs, # T, D_o
            "action": sample["action"],  # T, D_a
        }
        return data
