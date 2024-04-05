import numpy as np

from imitation_learning.dataset.base_dataset import BaseDataset
from imitation_learning.utils import normalize_utils, data_augmentation

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
            buffer_keys=['img', 'state', 'action'],
            harmonic_action=harmonic_action,
            seed=seed,
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes
        )

    def get_normalizer(self, mode="limits", **kwargs):
        sample_data = self._sample_to_data(self.replay_buffer)
        data = {
            'action': sample_data['action'],
            'agent_pos': sample_data['obs']['agent_pos']
        }
        normalizer = super().get_normalizer(data, mode=mode, **kwargs)
        normalizer['image'] = normalize_utils.get_image_range_normalizer()

        return normalizer

    def _sample_to_data(self, sample, rand_crop=True):
        T = sample['img'].shape[0]

        x_pos = (sample['state'][:,0] - 255.0)
        y_pos = (sample['state'][:,1] - 255.0) * -1
        agent_pos = np.concatenate((x_pos[..., np.newaxis], y_pos[..., np.newaxis]), axis=-1).reshape(T, 2)

        obs = np.moveaxis(sample['img'],-1,1) / 255
        if rand_crop:
            obs = data_augmentation.random_crop(obs, self.crop_image_size)

        x_act = sample['action'][:,0]
        y_act = sample['action'][:,1] * -1
        action = np.concatenate((x_act[..., np.newaxis], y_act[..., np.newaxis]), axis=-1).reshape(T, 2)

        data = {
            'obs' : {
                'image': obs, # T, C, H, W
                'agent_pos' : agent_pos # T, 2
            },
            "action": action,  # T, D_a
        }
        return data