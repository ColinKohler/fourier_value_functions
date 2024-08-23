""" robosuite_utils.py """

import numpy as np
import torch
from pytorch3d.transforms import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    matrix_to_axis_angle,
)


def preprocess_pose(pose, rot_type="6d"):
    """Preprocess robsuite 4x4 pose into pos and euler rot with x and y flipped and y reversed."""
    pose = pose.reshape(-1, 4, 4)
    pos = pose[:, :3, -1].reshape(-1, 3)
    pos = pos[:, [1, 0, 2]]
    pos[:, 1] = -pos[:, 1]

    rot = pose[:, :3, :3].reshape(-1, 3, 3)
    rot = matrix_to_euler_angles(torch.from_numpy(rot), "XYZ")
    rot = rot[:, [1, 0, 2]]
    rot[:, 1] = -rot[:, 1]
    if rot_type == "6d":
        rot = euler_angles_to_matrix(rot, "XYZ")[:, :2, :3]
    elif rot_type == "matrix":
        rot = euler_angles_to_matrix(rot, "XYZ")
    elif rot_type == "zyz":
        rot = matrix_to_euler_angles(euler_angles_to_matrix(rot, "XYZ"), "ZYZ")
    elif rot_type == "axis":
        rot = matrix_to_axis_angle(euler_angles_to_matrix(rot, "XYZ"))

    return pos, rot


def process_action(action):
    """Convert actions from model output to robosuite input."""
    B = action.shape[0]

    action_pose = np.eye(4).reshape(1, 4, 4).repeat(B, axis=0)
    action_pose[:, :3, :3] = euler_angles_to_matrix(
        torch.from_numpy(action[:, 0, 3:6]), "ZYZ"
    ).numpy()
    action_pose[:, :3, -1] = action[:, 0, :3]

    action_pos = action_pose[:, :3, -1]
    action_pos[:, 1] = -action_pos[:, 1]
    action_pos = action_pos[:, [1, 0, 2]]

    action_rot = matrix_to_euler_angles(torch.from_numpy(action_pose[:, :3, :3]), "XYZ")
    action_rot[:, 1] = -action_rot[:, 1]
    action_rot = action_rot[:, [1, 0, 2]]
    action_rot = matrix_to_axis_angle(euler_angles_to_matrix(action_rot, "XYZ")).numpy()
    zaction_rot = np.zeros_like(action_rot)

    gripper_act = action[:, 0, -1].reshape(-1, 1)

    action = np.hstack([action_pos, action_rot, gripper_act]).reshape(B, 1, -1)

    return action
