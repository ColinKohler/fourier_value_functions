import numpy as np

def convert_action_coords(action, coord_type):
    if coord_type == "rectangular":
        return action
    elif coord_type == "polar":
        return convert_to_polar(action)
    elif coord_type == "cylinderical":
        return convert_to_cylinderical(action)
    else:
        raise ValueError("Invalid action coordindates specified.")

def convert_to_polar(action):
    r = np.sqrt(action[:, 0] ** 2 + action[:, 1] ** 2)
    theta = np.arctan2(action[:, 1], (action[:, 0]))
    theta[np.where(theta < 0)] += 2 * np.pi

    return np.concatenate((r[:,None], theta[:,None]), axis=1)

def convert_to_cylinderical(action):
    r = np.sqrt(action[:, 0] ** 2 + action[:, 1] ** 2)
    theta = np.arctan2(action[:, 1], (action[:, 0]))
    theta[np.where(theta < 0)] += 2 * np.pi
    z = action[:,2]

    return np.concatenate((r[:,None], theta[:,None]), axis=1)
