import os
import numpy as np
import click
import pygame

from fvf.dataset.replay_buffer import ReplayBuffer
from fvf.env.two_path.four_path_env import FourPathEnv

@click.command()
@click.option("-o", "--output", required=True)
@click.option("-rs", "--render_size", default=96, type=int)
@click.option("-hz", "--control_hz", default=10, type=int)
def main(output, render_size, control_hz):
    """
    Collect demonstration for the Two Path task.

    Usage: python demo_two_path.py -o data/two_path_demo.zarr

    This script is compatible with both Linux and MacOS.
    Hover mouse close to the blue circle to start.
    Move the agent into the green area.
    The episode will automatically terminate if the task is succeeded.
    Press "Q" to exit.
    Press "R" to retry.
    Hold "Space" to pause.
    """

    # create replay buffer in read-write mode
    replay_buffer = ReplayBuffer.create_from_path(output, mode="a")

    # create env
    env = FourPathEnv(
        render_size=render_size,
        render_action=False,
        render_mode="human",
    )
    agent = env.teleop_agent()
    clock = pygame.time.Clock()

    # episode-level while loop
    while True:
        episode = list()
        # record in seed order, starting with 0
        seed = replay_buffer.n_episodes
        print(f"starting seed {seed}")

        # set seed for env
        env.seed(seed)

        # reset env and get observations (including info and render for recording)
        obs = env.reset(seed, None)
        info = env._get_info()
        img = env.render()

        # loop state
        retry = False
        pause = False
        done = False
        plan_idx = 0
        pygame.display.set_caption(f"plan_idx:{plan_idx}")
        # step-level while loop
        while not done:
            # process keypress events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # hold Space to pause
                        plan_idx += 1
                        pygame.display.set_caption(f"plan_idx:{plan_idx}")
                        pause = True
                    elif event.key == pygame.K_r:
                        # press "R" to retry
                        retry = True
                    elif event.key == pygame.K_q:
                        # press "Q" to exit
                        exit(0)
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE:
                        pause = False

            # handle control flow
            if retry:
                break
            if pause:
                continue

            # get action from mouse
            # None if mouse is not close to the agent
            act = agent.act(obs)
            if not act is None:
                # teleop started
                state = np.concatenate([info["pos_agent"], info["wall_pose"]])
                #keypoint = obs.reshape(2,-1)[0].reshape(-1,2)[:18]
                keypoint = np.array([0])
                data = {
                    "img": img,
                    "state": np.float32(state),
                    "keypoint": np.float32(keypoint),
                    "action": np.float32(act),
                    "n_contacts": np.float32([info["n_contacts"]]),
                }
                episode.append(data)

            # step env and render
            obs, reward, done, terminated, info = env.step(act)
            img = env.render()

            # regulate control frequency
            clock.tick(control_hz)
        if not retry:
            # save episode buffer to replay buffer (on disk)
            data_dict = dict()
            for key in episode[0].keys():
                data_dict[key] = np.stack([x[key] for x in episode])
            replay_buffer.add_episode(data_dict, compressors="disk")
            print(f"saved seed {seed}")
        else:
            print(f"retry seed {seed}")


if __name__ == "__main__":
    main()
