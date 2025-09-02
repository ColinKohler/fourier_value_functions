# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, task, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = [2]
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
        # create replay buffer
        data_specs = (
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            specs.Array((1,), np.float32, "reward"),
            specs.Array((1,), np.float32, "discount"),
        )

        self.replay_storage = ReplayBufferStorage(data_specs, self.work_dir / "buffer")

        self.train_replay_loader = make_replay_loader(
            Path("/Users/aagha/Playground/fourier_value_functions/exp_local_cartesian/buffer/50_percent"),
            self.cfg.replay_buffer_size,
            self.cfg.batch_size,
            self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot,
            self.cfg.nstep,
            self.cfg.discount,
        )
        self.test_replay_loader = make_replay_loader(
            Path("/Users/aagha/Playground/fourier_value_functions/exp_local_cartesian/buffer/test"),
            self.cfg.replay_buffer_size,
            self.cfg.batch_size,
            self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot,
            self.cfg.nstep,
            self.cfg.discount,
        )
        self._train_replay_iter = None
        self._test_replay_iter = None

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
    def train_replay_iter(self):
        if self._train_replay_iter is None:
            self._train_replay_iter = iter(self.train_replay_loader)
        return self._train_replay_iter
    
    @property
    def test_replay_iter(self):
        if self._test_replay_iter is None:
            self._test_replay_iter = iter(self.test_replay_loader)
        return self._test_replay_iter

    def train(self):
        train_until_step = utils.Until(1e4, 1)
        metrics = None
        self._global_step = 0
        while train_until_step(self.global_step):
            metrics = self.agent.update(self.train_replay_iter, self.test_replay_iter, self.global_step)
            self.logger.log_metrics(metrics, self.global_frame, ty="train")
            self._global_step += 1
            with self.logger.log_and_dump_ctx(
                        self.global_frame, ty="train"
                ) as log:
                    log("step", self.global_step)
    def load_snapshot(self):
        snapshot = Path("/Users/aagha/Playground/fourier_value_functions/exp_local_cartesian/snapshot.pt")
        with snapshot.open("rb") as f:
            payload = torch.load(f, map_location="cpu", weights_only=False)
        for k, v in payload.items():
            if not k.startswith("agent"):
                self.__dict__[k] = v

        # Load agent separately
        self.agent.load(payload)


@hydra.main(config_path="cfgs", config_name="config")
def main(cfg):
    root_dir = Path.cwd()
    workspace = Workspace(cfg)
    snapshot = "/Users/aagha/Playground/fourier_value_functions/exp_local_cartesian/snapshot.pt"
    print(f"resuming: {snapshot}")
    workspace.load_snapshot()
    workspace.train()


if __name__ == "__main__":
    main()
