import os
import copy
from typing import Dict
import torch
import numpy as np
import numpy.random as npr
import pickle
import h5py
import tqdm
from pytorch3d.transforms import (
    euler_angles_to_matrix,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    quaternion_to_matrix,
)

from fvf.dataset.replay_buffer import ReplayBuffer
from fvf.utils import torch_utils, action_utils, robosuite_utils
from fvf.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from fvf.utils.sampler import SequenceSampler, get_val_mask, downsample_mask


class RobosuiteLowdimDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        obs_eef_target: bool = True,
        num_keypoints: int = 1,
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
        pbar = tqdm.tqdm(
            total=len(os.listdir(path)), desc="Loading hdf5 to ReplayBuffer"
        )
        for demo_dir in os.listdir(path):
            demo_dir = os.path.join(path, demo_dir)
            episode = _data_to_obs(demo_dir)
            self.replay_buffer.add_episode(episode)
            pbar.update(1)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        self.train_mask = ~val_mask
        self.train_mask = downsample_mask(
            mask=self.train_mask, max_n=max_train_episodes, seed=seed
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
        cylindrical_action = action_utils.convert_action_coords(
            data["action"][:, :3], self.action_coords
        )
        so3_action = data["action"][:, 3:-1]
        pose_action = np.concatenate((cylindrical_action, so3_action), axis=-1)
        gripper_action = data["action"][:, -1]
        data = {
            "keypoints": data["obs"]["keypoints"],
            "pose_act": pose_action,
            "gripper_act": gripper_action.reshape(-1, 1),
        }

        normalizer = LinearNormalizer()
        obs_stat = array_to_stat(data["keypoints"])
        normalizer["keypoints"] = ws_normalizer(data["keypoints"], self.num_keypoints)
        normalizer["pose_act"] = act_normalizer(data["pose_act"])

        imp_norm = SingleFieldLinearNormalizer()
        imp_norm.fit(data["gripper_act"], output_min=0, output_max=1)
        normalizer["gripper_act"] = imp_norm

        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        cylindrical_action = action_utils.convert_action_coords(
            data["action"][:, :3], self.action_coords
        )
        so3_action = data["action"][:, 3:-1]
        pose_action = np.concatenate((cylindrical_action, so3_action), axis=-1)
        gripper_action = data["action"][:, -1]
        data = {
            "obs": {"keypoints": data["obs"]["keypoints"]},
            "pose_act": pose_action,
            "gripper_act": gripper_action.reshape(-1, 1),
        }

        torch_data = torch_utils.dict_apply(data, torch.from_numpy)

        return torch_data

    def _sample_to_data(self, sample):
        data = {
            "obs": {
                "keypoints": sample["obs"],  # T, D_o
            },
            "action": sample["actions"],  # T, D_a
        }
        return data


def array_to_stat(arr):
    stat = {
        "min": np.min(arr, axis=0),
        "max": np.max(arr, axis=0),
        "mean": np.mean(arr, axis=0),
        "std": np.std(arr, axis=0),
    }
    return stat


def ws_normalizer(arr, num_keypoints, nmin=-1.0, nmax=1.0):
    stat = {
        "min": np.array(
            [-0.3, -0.3, 0.8, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0] * num_keypoints + [0]
        ),
        "max": np.array(
            [0.3, 0.3, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] * num_keypoints + [0.05]
        ),
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
        "min": np.array([0.0, 0.0, -0.6, -np.pi, -np.pi, -np.pi]),
        "max": np.array([0.6, 2 * np.pi, 0.6, np.pi, np.pi, np.pi]),
        "mean": np.mean(arr, axis=0),
        "std": np.std(arr, axis=0),
    }
    scale = (nmax - nmin) / (stat["max"] - stat["min"])
    offset = nmin - scale * stat["min"]
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def normalizer_from_stat(stat):
    max_abs = np.maximum(stat["max"].max(), np.abs(stat["min"]).max())
    scale = np.full_like(stat["max"], fill_value=1 / max_abs)
    offset = np.zeros_like(stat["max"])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def _data_to_obs(demo_dir: str) -> dict:
    # Read object pose
    obj_poses = []
    with h5py.File(os.path.join(demo_dir, "object.hdf5"), "r") as f:
        for obj_name, obj_pose in f.items():
            obj_poses.append(np.array(obj_pose[:]))

    # Read end effector pose
    with h5py.File(os.path.join(demo_dir, "robot.hdf5"), "r") as f:
        eef_pose = np.array(f["eef_pose"][:])
        gripper_q = np.array(f["gripper_q"][:])

    # Read actions
    with h5py.File(os.path.join(demo_dir, "actions.hdf5"), "r") as f:
        actions = np.array(f["actions"][:])

    obj_positions = []
    obj_rotations = []
    for obj_pose in obj_poses:
        obj_pose = obj_pose.reshape(-1, 4, 4)
        obj_rot = matrix_to_quaternion(torch.from_numpy(obj_pose[:, :3, :3]))
        obj_rot = quaternion_to_matrix(obj_rot[:, [3, 0, 1, 2]])
        obj_pose[:, :3, :3] = obj_rot.numpy()
        obj_pos, obj_rot = robosuite_utils.preprocess_pose(obj_pose.reshape(-1, 16))

        obj_positions.append(obj_pos)
        obj_rotations.append(obj_rot.reshape(-1, 6).numpy())
    obj_positions = np.concatenate(obj_positions, axis=-1)
    obj_rotations = np.concatenate(obj_rotations, axis=-1)

    eef_pose = eef_pose.reshape(-1, 4, 4)
    eef_rot = matrix_to_quaternion(torch.from_numpy(eef_pose[:, :3, :3]))
    eef_rot = quaternion_to_matrix(eef_rot[:, [3, 0, 1, 2]])
    eef_pose[:, :3, :3] = eef_rot.numpy()
    eef_pos, eef_rot = robosuite_utils.preprocess_pose(eef_pose)
    eef_rot = eef_rot.reshape(-1, 6).numpy()

    gripper_q = gripper_q[:, 0].reshape(-1, 1)

    obs = np.concatenate(
        [
            obj_positions,
            obj_rotations[:, 0].reshape(-1, 1),
            obj_rotations[:, 3].reshape(-1, 1),
            obj_rotations[:, 1].reshape(-1, 1),
            obj_rotations[:, 4].reshape(-1, 1),
            obj_rotations[:, 2].reshape(-1, 1),
            obj_rotations[:, 5].reshape(-1, 1),
            eef_pos,
            eef_rot[:, 0].reshape(-1, 1),
            eef_rot[:, 3].reshape(-1, 1),
            eef_rot[:, 1].reshape(-1, 1),
            eef_rot[:, 4].reshape(-1, 1),
            eef_rot[:, 2].reshape(-1, 1),
            eef_rot[:, 5].reshape(-1, 1),
            gripper_q,
        ],
        axis=-1,
    )

    # Swap x and y axis and flip y axis
    action_matrix = np.eye(4).reshape(1, 4, 4).repeat(actions.shape[0], axis=0)
    action_matrix[:, :3, :3] = axis_angle_to_matrix(
        torch.from_numpy(actions[:, 3:6])
    ).numpy()
    action_matrix[:, :3, -1] = actions[:, :3]

    action_pos, action_rot = robosuite_utils.preprocess_pose(
        action_matrix, rot_type="zyz"
    )

    actions = np.hstack([action_pos, action_rot, actions[:, -1].reshape(-1, 1)])
    data = {"obs": obs, "actions": actions}
    return data
