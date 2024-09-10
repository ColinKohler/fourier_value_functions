import numpy as np
import copy


def convert_action_coords(action, coord_type):
    if coord_type == "rectangular":
        return action
    elif coord_type == "polar":
        return convert_to_polar(action)
    elif coord_type == "cylindrical":
        return convert_to_cylindrical(action)
    elif coord_type == "spherical":
        return convert_to_spherical(action)
    else:
        raise ValueError("Invalid action coordindates specified.")


def convert_to_polar(action):
    r = np.sqrt(action[:, 0] ** 2 + action[:, 1] ** 2)
    theta = np.arctan2(action[:, 1], action[:, 0])
    theta[np.where(theta < 0)] += 2 * np.pi

    return np.concatenate((r[:, None], theta[:, None]), axis=1)


def convert_to_cylindrical(action):
    r = np.sqrt(action[:, 0] ** 2 + action[:, 1] ** 2)
    theta = np.arctan2(action[:, 1], action[:, 0])
    theta[np.where(theta < 0)] += 2 * np.pi
    z = action[:, 2]

    return np.concatenate((r[:, None], theta[:, None], z[:, None]), axis=1)


def convert_to_spherical(action):
    r = np.sqrt(action[:, 0] ** 2 + action[:, 1] ** 2 + action[:, 2] ** 2)
    theta = np.arctan2(action[:, 1], action[:, 0])
    theta[np.where(theta < 0)] += 2 * np.pi
    phi = np.arccos(action[:, 2] / r)

    return np.concatenate((r[:, None], theta[:, None], phi[:, None]), axis=1)
