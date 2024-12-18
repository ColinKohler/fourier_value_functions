import copy
from typing import Dict
import torch
import numpy as np
import numpy.random as npr
import pickle

from fvf.dataset.replay_buffer import ReplayBuffer
from fvf.utils import torch_utils, action_utils
from fvf.model.common.normalizer import LinearNormalizer
from fvf.utils.sampler import SequenceSampler, get_val_mask, downsample_mask


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        buffer_keys: list = ['obs', 'action'],
        action_coords: str = "rectangular",
        seed: int = 0,
        val_ratio: float = 0.0,
        max_train_episodes: int = None,
    ):
        super().__init__()

        self.replay_buffer = ReplayBuffer.copy_from_path(path, keys=buffer_keys)
        self.augment = True

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed
        )
        self.train_mask = ~val_mask
        self.train_mask = downsample_mask(
            mask=self.train_mask,
            max_n=max_train_episodes,
            seed=seed
        )

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=self.train_mask,
        )

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.action_coords = action_coords

        self.normalizer = self.get_normalizer()

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, data=None, mode="limits", **kwargs):
        if data is None:
            data = self._sample_to_data(self.replay_buffer)

        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        data["action"] = action_utils.convert_action_coords(data["action"], self.action_coords)

        torch_data = torch_utils.dict_apply(data, torch.from_numpy)

        return torch_data

    def _sample_to_data(self, sample):
        raise NotImplementedError()
