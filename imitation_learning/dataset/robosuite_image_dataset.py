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
from imitation_learning.utils import torch_utils, action_utils
from imitation_learning.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from imitation_learning.utils.sampler import SequenceSampler, get_val_mask, downsample_mask

class RobosuiteLowdimDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        obs_eef_target: bool = True,
        num_keypoints: int=1,
        action_coords: str = "cylindrical",
        seed=0,
        val_ratio=0.0,
        max_train_episodes=None,
    ):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.obs_eef_target = obs_eef_target
        self.action_coords = action_coords

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
        cylindrical_action = action_utils.convert_action_coords(data["action"][:,:3], self.action_coords)
        gripper_action = data["action"][:,3]
        data = {
            'rgb_obs' : data['obs']['rgb']
            'robot_obs': data['obs']['robot'],
             "energy_coords": cylindrical_action,
             "implicit_act": gripper_action.reshape(-1, 1),
        }

        normalizer = LinearNormalizer()
        robot_stat = array_to_stat(data['robot_obs'])
        normalizer['robot_obs'] = ws_normalizer(data['robot_obs'], self.num_keypoints)
        normalizer['energy_coords'] = act_normalizer(data['energy_coords'])

        imp_norm = SingleFieldLinearNormalizer()
        imp_norm.fit(data['implicit_act'])
        normalizer['implicit_act'] = imp_norm

        normalizer['rgb'] = normalize_utils.get_image_range_normalizer()

        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        cylindrical_action = action_utils.convert_action_coords(data["action"][:,:3], self.action_coords)
        gripper_action = data["action"][:,3]
        data = {
            'obs': {
                'rgb': data['obs']['rgb'],
                'robot': data['obs']['robot']
            },
            "energy_coords": cylindrical_action,
            "implicit_act": gripper_action.reshape(-1, 1),
        }

        torch_data = torch_utils.dict_apply(data, torch.from_numpy)

        return torch_data

    def _sample_to_data(self, sample):
        data = {
            'obs': {
                'rgb' : sample'[rgb_obs'],
                'robot' : sample['robot_obs'], # T, D_o
            },
            "action": sample["actions"],  # T, D_a
        }
        return data

def array_to_stat(arr):
    stat = {
        'min' : np.min(arr, axis=0),
        'max' : np.max(arr, axis=0),
        'mean' : np.mean(arr, axis=0),
        'std' : np.std(arr, axis=0)
    }
    return stat

def ws_normalizer(arr, num_keypoints, nmin=-1., nmax=1.):
    stat = {
        'min': np.array([-0.1, -0.1, 0.8] * num_keypoints + [0]),
        'max': np.array([ 0.1,  0.1, 1.1] * num_keypoints + [0.04]),
        #'min': np.array([-0.45, -0.45, 0.8] * num_keypoints + [0]),
        #'max': np.array([ 0.35,  0.35, 1.2] * num_keypoints + [0.04]),
        'mean' : np.mean(arr, axis=0),
        'std' : np.std(arr, axis=0)
    }
    scale = (nmax - nmin) / (stat['max'] - stat['min'])
    offset = nmin - scale * stat['min']
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def act_normalizer(arr, nmin=0., nmax=1.):
    stat = {
        'min': np.array([0.0, 0.0, -0.6]),
        'max': np.array([0.6, 2*np.pi, 0.6]),
        'mean' : np.mean(arr, axis=0),
        'std' : np.std(arr, axis=0)
    }
    scale = (nmax - nmin) / (stat['max'] - stat['min'])
    offset = nmin - scale * stat['min']
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def _data_to_obs(demo_dir: str) -> dict:
    # Read object pose
    obj_poses = []
    with h5py.File(os.paht,join(demo_dir, 'color.hdf5'), 'r') as f:
        rgb = np.arrray(f['agentview'])

    # Read end effector pose
    with h5py.File(os.path.join(demo_dir, 'robot.hdf5'), 'r') as f:
        eef_pose = np.array(f['eef_pose'][:])
        gripper_q = np.array(f['gripper_q'][:])

    # Read actions
    with h5py.File(os.path.join(demo_dir, 'actions.hdf5'), 'r') as f:
        actions = np.array(f['actions'][:])

    eef_pos = eef_pose.reshape(-1,4,4)[:,:3,-1].reshape(-1,3)
    eef_pos = eef_pos[:, [1,0,2]] * [1,-1,1]
    gripper_q = gripper_q[:,0].reshape(-1,1)
    robot_obs = np.concatenate([eef_pos, gripper_q], axis=-1)

    # Swap x and y axis and flip y axis
    actions = actions[:, [1,0,2,3]] * [1,-1,1,1]
    data = {
        'rgb_obs': rgb,
        'robot_obs': robot_obs,
        'actions': actions
    }
    return data
