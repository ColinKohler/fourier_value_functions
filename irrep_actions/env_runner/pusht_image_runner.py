import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv

from irrep_actions.env.pusht.pusht_image_env import PushTImageEnv
from irrep_actions.gym_util.async_vector_env import AsyncVectorEnv
from irrep_actions.gym_util.multistep_wrapper import MultiStepWrapper
from irrep_actions.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from irrep_actions.policy.base_policy import BasePolicy
from irrep_actions.env_runner.base_runner import BaseRunner
from irrep_actions.utils.torch_utils import dict_apply

class PushTImageRunner(BaseRunner):
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
        render_size=96,
        fps=10,
        crf=22,
        past_action=False,
        tqdm_interval_sec=5.0,
        num_envs = None,
        random_goal_pose=False,
    ):
        num_test_vis=50
        super().__init__(output_dir)
        num_envs = num_train + num_test if num_envs is None else num_envs

        env_num_obs_steps = num_obs_steps + num_latency_steps
        env_num_action_steps = num_action_steps

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTImageEnv(
                        legacy=False,
                        render_size=render_size,
                        random_goal_pose=random_goal_pose,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                ),
                n_obs_steps=env_num_obs_steps,
                n_action_steps=env_num_action_steps,
                max_episode_steps=max_steps
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
                    filename = pathlib.Path(output_dir).joinpath('media', wv.util.generate_id() + '.mp4')
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('train/')
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
                    filename = pathlib.Path(output_dir).joinpath('media', wv.util.generate_id() + '.mp4')
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.num_obs_steps = num_obs_steps
        self.num_action_steps = num_action_steps
        self.num_latency_steps = num_latency_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype

        env = self.env

        num_envs = len(self.env_fns)
        num_inits = len(self.env_init_fn_dills)
        num_chunks = math.ceil(num_inits / num_envs)

        all_video_paths = [None] * num_inits
        all_rewards = [None] * num_inits

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
            env.call_each('run_dill_function', args_list=[(x,) for x in this_init_fns])

            obs = env.reset()
            past_action = None
            policy.reset()

            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f'Eval PushtImageRunner {chunk_idx+1} / {num_chunks}',
                leave=False,
                mininterval=self.tqdm_interval_sec
            )

            done = False
            while not done:
                B, T, C, H, W = obs['image'].shape
                #cropped_image = obs['image'][:,:,:,6:-6, 6:-6]
                from torchvision.transforms.functional import resize
                cropped_image = resize(torch.tensor(obs['image']).view(-1, 3, 96,96), (84, 84)).view(-1, 2, 3, 84, 84).numpy()
                #from torchvision.transforms.functional import resize

                x_pos = (obs['agent_pos'][:,:,0] - 255.0)
                y_pos = (obs['agent_pos'][:,:,1] - 255.0) * -1
                agent_pos = np.concatenate((x_pos[..., np.newaxis], y_pos[..., np.newaxis]), axis=-1).reshape(B, T, 2)

                obs_dict = {
                    'image': cropped_image,
                    'agent_pos': agent_pos
                }
                if self.past_action and (past_action is not None):
                    obs['past_action'] = past_action[:, -(self.num_obs_steps-1):].astype(np.float32)
                obs_dict = dict_apply(obs_dict, lambda x: torch.from_numpy(x).to(device))

                with torch.no_grad():
                    action_dict = policy.get_action(obs_dict, device)

                x_act = action_dict['action'][:,:,0]
                y_act = action_dict['action'][:,:,1] * -1
                action_dict['action'] =  torch.concatenate((x_act, y_act), dim=-1).view(B,1,2)
                action_dict = dict_apply(action_dict, lambda x: x.to('cpu').numpy())
                action = action_dict['action'][:, self.num_latency_steps:]

                # Step env
                obs, reward, done, info = env.step(action)
                done = np.all(done)
                past_action = action

                pbar.update(action.shape[1])
            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]

        # Logging
        max_rewards = collections.defaultdict(list)
        log_data = dict()

        for i in range(num_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # Visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # Log aggergate metrics
        for prefix, v in max_rewards.items():
            log_data[prefix+'mean_score'] = np.mean(v)

        return log_data
