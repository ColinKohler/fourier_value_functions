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

        if self.action_coords == "spherical":
            normalizer["action"] = act_normalizer(data["action"])
            normalizer["keypoints"] = ws_normalizer(data["keypoints"])

        return normalizer

    def _sample_to_data(self, sample):
        obs = sample["obs"]

        T = obs.shape[0]
        action = sample["action"].reshape(T, 3)

        obs = obs.reshape(T, 2, 3)
        # obs = obs[:, :, [1, 0, 2]]
        # obs[:, :, 1] = -obs[:, :, 1]
        obs = obs.reshape(T, 6)

        # action = action[:, [1, 0, 2]]
        # action[:, 1] = -action[:, 1]

        data = {
            "obs": {"keypoints": obs},
            "action": action,  # T, D_a
        }
        return data


def ws_normalizer(arr, nmin=-1.0, nmax=1.0):
    stat = {
        "min": np.array([-1.5, -1.5, 0.0, -1.5, -1.5, 0.0], dtype=np.float32),
        "max": np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5], dtype=np.float32),
        "mean": np.mean(arr, axis=0),
        "std": np.std(arr, axis=0),
    }
    scale = (nmax - nmin) / (stat["max"] - stat["min"])
    offset = nmin - scale * stat["min"]
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def act_normalizer(arr, nmin=0.0, nmax=1.0):
    stat = {
        "min": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "max": np.array([arr.max(0)[0], np.pi, 2 * np.pi], dtype=np.float32),
        "mean": np.mean(arr, axis=0),
        "std": np.std(arr, axis=0),
    }
    scale = (nmax - nmin) / (stat["max"] - stat["min"])
    offset = nmin - scale * stat["min"]
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )
