import numpy as np

from irrep_actions.dataset.base_dataset import BaseDataset
from irrep_actions.utils import normalize_utils, data_augmentation

class PushTImageDataset(BaseDataset):
    def __init__(
        self,
        path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        harmonic_action: bool = False,
        crop_image_size: int=84,
        seed=0,
        val_ratio=0.0,
        max_train_episodes=None,
    ):
        self.crop_image_size = crop_image_size

        super().__init__(
            path,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            buffer_keys=['img', 'action'],
            harmonic_action=harmonic_action,
            seed=seed,
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes
        )

    def get_normalizer(self, mode="limits", **kwargs):
        normalizer = super().get_normalizer(mode=mode, **kwargs)
        normalizer['image'] = normalize_utils.get_image_range_normalizer()

        return normalizer

    def _sample_to_data(self, sample, rand_crop=True):
        obs = np.moveaxis(sample['img'],-1,1) / 255
        if rand_crop:
            obs = data_augmentation.random_crop(obs, self.crop_image_size)

        T = obs.shape[0]
        x_act = sample['action'][:,0]
        y_act = sample['action'][:,1] * -1
        action = np.concatenate((x_act[..., np.newaxis], y_act[..., np.newaxis]), axis=-1).reshape(T, 2)

        data = {
            'obs': obs, # T, D_o
            "action": action,  # T, D_a
        }
        return data
