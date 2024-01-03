import numpy as np
from irrep_actions.dataset.base_dataset import BaseDataset


class LowdimDataset(BaseDataset):
    def __init__(
        self,
        path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=0,
        val_ratio=0.0,
    ):
        super().__init__(
            path,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            seed=seed,
            val_ratio=val_ratio,
        )

    def _sample_to_data(self, sample):
        keypoint = sample['keypoint']
        state = sample['state']
        agent_pos = state[:,:2]
        obs = np.concatenate([
            keypoint.reshape(keypoint.shape[0], -1),
            agent_pos], axis=-1
        )

        data = {
            'obs': obs, # T, D_o
            "action": sample["action"],  # T, D_a
        }
        return data
