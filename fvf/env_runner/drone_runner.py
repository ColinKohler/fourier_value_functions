""" drone_runner.py  """

import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
import imageio

from fvf.env.drone.go_to_target_env import GoToTargetEnv
from fvf.gym_util.async_vector_env import AsyncVectorEnv
from fvf.gym_util.multistep_wrapper import MultiStepWrapper
from fvf.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from fvf.policy.base_policy import BasePolicy
from fvf.env_runner.base_runner import BaseRunner
from fvf.utils.torch_utils import dict_apply


class DroneRunner(BaseRunner):
    """Drone domain runner class."""

    def __init__(
        self,
        output_dir,
        keypoint_visible_rate=1.0,
        num_train=10,
        num_train_vis=3,
        train_start_seed=0,
        num_test=22,
        num_test_vis=6,
        test_start_seed=10000,
        max_steps=200,
        num_obs_steps=8,
        num_action_steps=8,
        num_latency_steps=0,
        fps=10,
        crf=22,
        agent_keypoints=False,
        past_action=False,
        tqdm_interval_sec=5.0,
        num_envs=None,
        action_coords="rectangular",
    ):
        super().__init__(output_dir)
        num_envs = num_train + num_test if num_envs is None else num_envs
        self.action_coords = action_coords

        env_num_obs_steps = num_obs_steps + num_latency_steps
        env_num_action_steps = num_action_steps

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    GoToTargetEnv(),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec="h264",
                        input_pix_fmt="rgb24",
                        crf=crf,
                        thread_type="FRAME",
                        thread_count=1,
                    ),
                    file_path=None,
                ),
                n_obs_steps=env_num_obs_steps,
                n_action_steps=env_num_action_steps,
                max_episode_steps=max_steps,
            )

        env_fns = [env_fn] * num_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # Training
        for i in range(num_train):
            seed = train_start_seed + i
            enable_render = i < num_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None

                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        "media", wv.util.generate_id() + ".mp4"
                    )
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                assert isinstance(env, MultiStepWrapper)
                env.set_seed(seed)

            env_seeds.append(seed)
            env_prefixs.append("train/")
            env_init_fn_dills.append(dill.dumps(init_fn))

        # Testing
        for i in range(num_test):
            seed = test_start_seed + i
            enable_render = i < num_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None

                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        "media", wv.util.generate_id() + ".mp4"
                    )
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                assert isinstance(env, MultiStepWrapper)
                env.set_seed(seed)

            env_seeds.append(seed)
            env_prefixs.append("test/")
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.agent_keypoints = agent_keypoints
        self.num_obs_steps = num_obs_steps
        self.num_action_steps = num_action_steps
        self.num_latency_steps = num_latency_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(
        self,
        policy: BasePolicy,
        plot_energy_fn: bool = False,
        plot_weights_basis_fns: bool = False,
    ):
        device = policy.device
        dtype = policy.dtype

        env = self.env

        num_envs = len(self.env_fns)
        num_inits = len(self.env_init_fn_dills)
        num_chunks = math.ceil(num_inits / num_envs)

        all_video_paths = [None] * num_inits
        all_rewards = [None] * num_inits
        energy_fn_plots = [list() for _ in range(num_inits)]

        for chunk_idx in range(num_chunks):
            start = chunk_idx * num_envs
            end = min(num_inits, start + num_envs)
            this_global_slice = slice(start, end)
            this_num_active_envs = end - start
            this_local_slice = slice(0, this_num_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            num_diff = num_envs - len(this_init_fns)
            if num_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * num_diff)
            assert len(this_init_fns) == num_envs

            # Initialize envs
            env.call_each("run_dill_function", args_list=[(x,) for x in this_init_fns])

            obs = env.reset()
            past_action = None
            policy.reset()

            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval DroneRunner {chunk_idx+1} / {num_chunks}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )

            done = False
            while not done:
                B = obs.shape[0]

                # obs = obs.reshape(B, -1, 2, 3)
                # obs = obs[:, :, :, [1, 0, 2]]
                # obs[:, :, :, 1] = -obs[:, :, :, 1]
                # obs = obs.reshape(B, -1, 6)

                obs_dict = {
                    "keypoints": obs[..., : self.num_obs_steps, :].astype(np.float32),
                }
                # print(obs_dict["keypoints"].round(3))

                if self.past_action and (past_action is not None):
                    obs["past_action"] = past_action[
                        :, -(self.num_obs_steps - 1) :
                    ].astype(np.float32)

                obs_dict = dict_apply(
                    obs_dict, lambda x: torch.from_numpy(x).to(device)
                )

                with torch.no_grad():
                    action_dict = policy.get_action(obs_dict, device)
                # print(action_dict["action"])

                if plot_energy_fn:
                    for i, env_id in enumerate(range(start, end)):
                        img = env.call_each("render2")[0]
                        h, w, c = img.shape
                        img = rgba2rgb(img)
                        img = img.reshape(1, h, w, 3).transpose(0, 3, 1, 2)
                        energy_fn_plots[env_id].append(
                            policy.plot_energy_fn(img, action_dict["energy"][i])
                        )

                action_dict = dict_apply(action_dict, lambda x: x.to("cpu").numpy())
                action = action_dict["action"][:, self.num_latency_steps :]

                # action[:, :, 1] = -action[:, :, 1]
                # action = action[:, :, [1, 0, 2]]

                # Step env
                obs, reward, done, timeout, info = env.step(action)
                done = np.all(done)
                past_action = action

                pbar.update(action.shape[1])
            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call("get_attr", "reward")[
                this_local_slice
            ]

        # Logging
        max_rewards = collections.defaultdict(list)
        log_data = dict()

        for i in range(num_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix + f"sim_max_reward_{seed}"] = max_reward

            # Visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix + f"sim_video_{seed}"] = sim_video
                if plot_energy_fn:
                    media_path = video_path.rpartition(".")[0]
                    energy_fn_plot_path = f"{media_path}_energy_fn.mp4"
                    # log_data[prefix+f'energy_fn_{seed}'] = energy_fn_plot_path
                    imageio.mimwrite(energy_fn_plot_path, energy_fn_plots[i])

        # Log aggergate metrics
        for prefix, v in max_rewards.items():
            log_data[prefix + "mean_score"] = np.mean(v)

        return log_data


def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, "RGBA image has 4 channels."

    rgb = np.zeros((row, col, 3), dtype="float32")
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    a = np.asarray(a, dtype="float32") / 255.0

    R, G, B = background

    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype="uint8")
