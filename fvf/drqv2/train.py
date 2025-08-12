# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
import matplotlib.pyplot as plt
from torch.autograd.grad_mode import enable_grad
from drqv2 import cartesian_to_polar
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

#os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
#os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs
from escnn import gspaces

import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, task, cfg):
    cfg.obs_shape = obs_spec.shape
    if cfg.action_space=="polar":
        cfg.action_shape = [2]
    else:
        cfg.action_shape = action_spec.shape
    if "Equi" in cfg._target_:
        if "reacher" in task:
            gspace = gspaces.flipRot2dOnR2(N=2)

        elif any(t in task for t in ["acrobot", "pendulum", "cartpole", "cup"]):
            gspace = gspaces.flip2dOnR2()

        return hydra.utils.instantiate(
            cfg,
            gspace=gspace,
        )

    else:
        return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            cfg.task_name,
            self.cfg.agent,
        )
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        self.train_env = dmc.make(
            self.cfg.task_name,
            self.cfg.frame_stack,
            self.cfg.action_repeat,
            self.cfg.seed,
        )
        self.eval_env = dmc.make(
            self.cfg.task_name,
            self.cfg.frame_stack,
            self.cfg.action_repeat,
            self.cfg.seed,
        )
        # create replay buffer
        data_specs = (
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            specs.Array((1,), np.float32, "reward"),
            specs.Array((1,), np.float32, "discount"),
        )

        self.replay_storage = ReplayBufferStorage(data_specs, self.work_dir / "buffer")

        self.replay_loader = make_replay_loader(
            self.work_dir / "buffer",
            self.cfg.replay_buffer_size,
            self.cfg.batch_size,
            self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot,
            self.cfg.nstep,
            self.cfg.discount,
        )
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None
        )
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None
        )

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def plot_rollout(self, qs, actions, rewards, episode):
        """
        Plots 4 line plots (qs, actions, rewards, steps) in the same figure.

        Args:
            qs (list): List of Q-values.
            actions (list): List of actions.
            rewards (list): List of rewards.
            steps (list): List of steps or any other metric.
        """
        fig, axs = plt.subplots(5, 1, sharex=True)

        # Plot Q-values
        axs[0].plot(qs, label="Q-values")
        axs[0].set_ylabel("Q-values")
        axs[0].grid(True)

        # Plot Actions
        axs[1].plot(actions, label="Cartesion")
        axs[1].set_ylabel("Cartesion")
        axs[1].grid(True)

        polar_actions = cartesian_to_polar(torch.tensor(actions)[:,0], torch.tensor(actions)[:,1])
        # Plot Actions
        axs[2].plot(polar_actions, label="Polar")
        axs[2].set_ylabel("Polar")
        axs[2].grid(True)

        bins = torch.stack([self.agent.critic.Q1.ph.r2d, self.agent.critic.Q1.ph.p2d], -1).to(polar_actions.device)
        def subtract(x, y):
            return x - y
        batched_subtract = torch.vmap(subtract, (None, 0))
        indices = batched_subtract(bins, polar_actions).abs().mean(-1).view(500,-1).argmin(-1)
        binned_actions = bins.view(-1,2)[indices.cpu()]
        axs[3].plot((binned_actions - polar_actions).abs(), label="Binned")
        axs[3].set_ylabel("Binned")
        axs[3].grid(True)
        # Plot Rewards
        axs[4].plot(rewards, label="Rewards")
        axs[4].set_ylabel("Rewards")
        axs[4].grid(True)

        plt.tight_layout()
        plt.savefig(f"rollout_{self._global_step}_{episode}.png")

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        qs = []
        actions = []
        rewards = []
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(
                        time_step.observation, self.global_step, eval_mode=True
                    )
                e = self.agent.encoder(torch.as_tensor(time_step.observation, device=self.device).unsqueeze(0))
                q1, q2 = self.agent.critic(e, torch.as_tensor(action, device=self.device).unsqueeze(0))
                q = torch.min(q1, q2)
                actions.append(action)
                qs.append(q[0].item())
                rewards.append(time_step.reward)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1
            self.plot_rollout(qs, actions, rewards, episode)
            qs = []
            actions = []
            rewards = []
            episode += 1
            self.video_recorder.save(f"{self.global_frame}.mp4")

        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            log("episode_reward", total_reward / episode)
            log("episode_length", step * self.cfg.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(
            self.cfg.num_train_frames, self.cfg.action_repeat
        )
        seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
        eval_every_step = utils.Every(
            self.cfg.eval_every_frames, self.cfg.action_repeat
        )

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f"{self.global_frame}.mp4")
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(
                        self.global_frame, ty="train"
                    ) as log:
                        log("fps", episode_frame / elapsed_time)
                        log("total_time", total_time)
                        log("episode_reward", episode_reward)
                        log("episode_length", episode_frame)
                        log("episode", self.global_episode)
                        log("buffer_size", len(self.replay_storage))
                        log("step", self.global_step)

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log(
                    "eval_total_time", self.timer.total_time(), self.global_frame
                )
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(
                    time_step.observation, self.global_step, eval_mode=False
                )

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty="train")

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        keys_to_save = ["timer", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        # Save agent separately
        agent_dict = self.agent.save()
        payload = {**payload, **agent_dict}
        with snapshot.open("wb") as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        with snapshot.open("rb") as f:
            payload = torch.load(f, map_location="cpu")
        for k, v in payload.items():
            if not k.startswith("agent"):
                self.__dict__[k] = v

        # Load agent separately
        self.agent.load(payload)


@hydra.main(config_path="cfgs", config_name="config")
def main(cfg):
    from train import Workspace as W

    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / "snapshot.pt"
    '''if snapshot.exists():
        print(f"resuming: {snapshot}")
        workspace.load_snapshot()'''
    workspace.train()


if __name__ == "__main__":
    main()
