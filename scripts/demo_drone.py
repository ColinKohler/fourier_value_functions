import os
import sys

sys.path.insert(0, os.path.abspath("."))

import click
import time
import numpy as np
from fvf.dataset.replay_buffer import ReplayBuffer
from fvf.env.drone.go_to_target_env import GoToTargetEnv
from fvf.env.drone.fly_through_gate_env import FlyThroughGateEnv


@click.command()
@click.option("-o", "--output", required=True)
def main(output):
    # create replay buffer in read-write mode
    replay_buffer = ReplayBuffer.create_from_path(output, mode="a")

    env = FlyThroughGateEnv(gui=False)

    success = 0
    seed = 0
    while success < 50:
        episode: list = []
        print(f"starting seed {seed}")

        obs = env.reset(seed=seed)
        done = False
        terminated = False

        i = 0
        while not done and not terminated:
            i += 1
            target_pos = obs[3:].reshape(2, 3).mean(0)
            target_pos[2] += 0.1
            action = np.clip(target_pos - obs[:3], -0.1, 0.1)
            data = {
                "obs": np.float32(obs),
                "action": np.float32(action),
            }
            episode.append(data)

            obs, reward, done, terminated, info = env.step(action)

        if reward == 1:
            data_dict = dict()
            for key in episode[0].keys():
                data_dict[key] = np.stack([x[key] for x in episode])
            replay_buffer.add_episode(data_dict, compressors="disk")
            print(f"saved seed {seed}...")
            success += 1
        else:
            print(f"seed {seed} failed, not saving...")
        seed += 1

    env.close()


if __name__ == "__main__":
    main()
