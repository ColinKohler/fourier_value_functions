import os
import sys

sys.path.insert(0, os.path.abspath("."))

import click
import time
import numpy as np
from fvf.dataset.replay_buffer import ReplayBuffer
from fvf.env.drone.go_to_target_env import GoToTargetEnv

@click.command()
@click.option('-o', '--output', required=True)
def main(output):
    # create replay buffer in read-write mode
    replay_buffer = ReplayBuffer.create_from_path(output, mode='a')

    env = GoToTargetEnv(gui=False)

    for _ in range(10):
        episode: list = []
        seed = replay_buffer.n_episodes
        print(f'starting seed {seed}')

        obs, info = env.reset(seed=seed)
        done = False

        while not done:
            action = np.clip(obs[3:] - obs[:3], -0.1, 0.1)
            data = {
                'obs': np.float32(obs),
                'action': np.float32(action),
            }
            episode.append(data)

            obs, reward, done, terminated, info = env.step(action)

        data_dict = dict()
        for key in episode[0].keys():
            data_dict[key] = np.stack(
                [x[key] for x in episode])
        replay_buffer.add_episode(data_dict, compressors='disk')
        print(f'saved seed {seed}')

    env.close()

if __name__ == "__main__":
    main()
