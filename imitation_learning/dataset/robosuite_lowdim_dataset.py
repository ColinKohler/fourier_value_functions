import os
import copy
from typing import Dict
import torch
import numpy as np
import numpy.random as npr
import pickle
import h5py
import tqdm

from imitation_learning.dataset.replay_buffer import ReplayBuffer
from imitation_learning.utils import torch_utils
from imitation_learning.utils import harmonics
from imitation_learning.model.common.normalizer import LinearNormalizer
from imitation_learning.utils.sampler import SequenceSampler, get_val_mask, downsample_mask

class RobosuiteLowdimDataset(torch.utils.data.Dataset):
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
        super().__init__()

        self.obs_eef_target = obs_eef_target

        self.replay_buffer = ReplayBuffer.create_empty_numpy()
        pbar = tqdm.tqdm(total=len(os.listdir(path)), desc="Loading hdf5 to ReplayBuffer")
        for demo_dir in os.listdir(path):
            demo_dir = os.path.join(path, demo_dir)
            episode = _data_to_obs(demo_dir)
            self.replay_buffer.add_episode(episode)
            pbar.update(1)

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
        self.harmonic_action = harmonic_action

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

        if self.harmonic_action:
            xy = data["action"][:,:2]
            zg = data["action"][:,2:]
            polar_xy = harmonics.convert_to_polar(xy)
            data["action"] = np.concatenate([polar_xy, zg], axis=-1)
        data = {
            'keypoints': data['obs']['keypoints'],
            'action': data['action']
        }

        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        if self.harmonic_action:
            xy = data["action"][:,:2]
            zg = data["action"][:,2:]
            polar_xy = harmonics.convert_to_polar(xy)
            data["action"] = np.concatenate([polar_xy, zg], axis=-1)

        torch_data = torch_utils.dict_apply(data, torch.from_numpy)

        return torch_data

    def _sample_to_data(self, sample):
        data = {
            'obs': {
                'keypoints' : sample['obs'], # T, D_o
            },
            "action": sample["actions"],  # T, D_a
        }
        return data

def _data_to_obs(demo_dir: str) -> dict:
    # Read object pose
    with h5py.File(os.path.join(demo_dir, 'object.hdf5'), 'r') as f:
        obj_pose = np.array(f['cube_pose'][:])

    # Read end effector pose
    with h5py.File(os.path.join(demo_dir, 'robot.hdf5'), 'r') as f:
        eef_pose = np.array(f['eef_pose'][:])
        gripper_q = np.array(f['gripper_q'][:])

    # Read actions
    with h5py.File(os.path.join(demo_dir, 'actions.hdf5'), 'r') as f:
        actions = np.array(f['actions'][:])

    obj_pos = obj_pose.reshape(-1,4,4)[:,:3,-1].reshape(-1,3)
    eef_pos = eef_pose.reshape(-1,4,4)[:,:3,-1].reshape(-1,3)
    gripper_q = gripper_q[:,0].reshape(-1,1)
    obs = np.concatenate([obj_pos, eef_pos, gripper_q], axis=-1)

    data = {
        'obs': obs,
        'actions': actions
    }
    return data
