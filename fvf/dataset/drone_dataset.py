import numpy as np

from fvf.dataset.base_dataset import BaseDataset
from fvf.utils import action_utils
from fvf.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


class DroneDataset(BaseDataset):
    def __init__(
        self,
        path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        action_coords: str = "rectangular",
        seed=0,
        val_ratio=0.0,
        max_train_episodes=None,
    ):
        super().__init__(
            path,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            buffer_keys=["obs", "action"],
            action_coords=action_coords,
            seed=seed,
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes,
        )

    def get_normalizer(self, mode="limits", **kwargs):
        sample_data = self._sample_to_data(self.replay_buffer)
        data = {
            "keypoints": sample_data["obs"]["keypoints"],
            "action": sample_data["action"],
        }
        data["action"] = action_utils.convert_action_coords(
            data["action"], self.action_coords
        )

        normalizer = super().get_normalizer(data, mode=mode, **kwargs)

        # if self.action_coords == "spherical":
        #    act_norm = SingleFieldLinearNormalizer()
        #    act_norm.fit(data=data["action"], output_min=0.1, output_max=1)
        #    normalizer["action"] = act_norm

        return normalizer

    def _sample_to_data(self, sample):
        obs = sample["obs"]

        T = obs.shape[0]
        Do = obs.shape[-1] // 2
        action = sample["action"].reshape(T, 3)

        data = {
            "obs": {"keypoints": obs},
            "action": action,  # T, D_a
        }
        return data
