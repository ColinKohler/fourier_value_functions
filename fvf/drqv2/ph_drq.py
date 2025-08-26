# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
import random
from fvf.model.implicit import energy_mlp
import utils

def polar_to_cartesian(coords: torch.Tensor) -> torch.Tensor:
    """
    Converts polar coordinates to Cartesian coordinates.

    Args:
        coords (torch.Tensor): A tensor of shape (N, 2), where each row is (r, θ).

    Returns:
        torch.Tensor: A tensor of shape (N, 2), where each row is (x, y).
    """
    r = coords[:, 0]
    theta = coords[:, 1]
    
    # Compute x and y
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    
    # Combine x and y into a single tensor
    cartesian_coords = torch.stack((x, y), dim=-1)
    
    return cartesian_coords

def cartesian_to_polar(x, z):
    """Convert Cartesian coordinates (x, z) to polar coordinates (r, theta).
    Args:
        x (torch.Tensor): x-coordinates.
        z (torch.Tensor): z-coordinates.
    Returns:
        torch.Tensor: Polar coordinates (r, theta) where r is the radius and theta is the angle.
    """
    assert x.shape == z.shape, "x and z must have the same shape"
    r = torch.sqrt(x**2 + z**2)
    theta = torch.atan2(z, x) + torch.pi  # Shift to [0, 2*pi]
    return torch.stack([r, theta], -1)

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, _, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape, hidden_dim, out_dim):
        super().__init__()

        assert len(obs_shape) == 3
        # self.repr_dim = out_dim * 35 * 35
        self.repr_dim = out_dim * 7 * 7

        self.convnet = nn.Sequential(
            # 85x85
            nn.Conv2d(obs_shape[0], hidden_dim, 3, stride=2),
            nn.ReLU(),
            # 42x42
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1),
            nn.ReLU(),
            # 40x40
            nn.MaxPool2d(2),
            # 20x20
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1),
            nn.ReLU(),
            # 18x18
            nn.MaxPool2d(2),
            # 9x9
            nn.Conv2d(hidden_dim, out_dim, 3, stride=1),
            nn.ReLU(),
            # 7x7
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h

class PolarHarmonicsCritic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, action_space, 
                 num_radii=30, num_phi=90, radial_freq=1, angular_freq=1):
        super().__init__()
        self.n_act_dims = action_shape[0]
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )
        self.Q1 = energy_mlp.PolarEnergyMLP(
            feature_dim,
            hidden_dim,
            num_layers=2,
            dropout=0,
            spec_norm=False,
            radial_freq=radial_freq,
            angular_freq=angular_freq,
            min_radius=0.1,
            max_radius=1.0,
            num_radii=num_radii,
            num_phi=num_phi,
        )
        self.Q2 = energy_mlp.PolarEnergyMLP(
            feature_dim,
            hidden_dim,
            num_layers=2,
            dropout=0,
            spec_norm=False,
            radial_freq=radial_freq,
            angular_freq=angular_freq,
            min_radius=0.1,
            max_radius=1.0,
            num_radii=num_radii,
            num_phi=num_phi,
        )
        self.action_space = action_space
        self.apply(utils.weight_init)

    def forward(self, obs, action=None, bin=False):
        h = self.trunk(obs)
        if action is not None:
            d = action.shape[-1]
            action = action.view(-1, 1, d)
        q1 = self.Q1(h, action, bin=bin)
        q2 = self.Q2(h, action, bin=bin)

        return q1, q2


class DrQV2Agent:
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        lr,
        feature_dim,
        hidden_dim,
        critic_target_tau,
        num_expl_steps,
        update_every_steps,
        stddev_schedule,
        stddev_clip,
        use_tb,
        encoder_hidden_dim,
        encoder_out_dim,
        mixed_precision,
        action_space,
        num_radii=30,
        num_phi=90,
        radial_freq=1,
        angular_freq=1,
    ):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        # models
        self.encoder = Encoder(obs_shape, encoder_hidden_dim, encoder_out_dim).to(
            device
        )

        self.critic = PolarHarmonicsCritic(
            self.encoder.repr_dim, action_shape, feature_dim, hidden_dim, action_space, num_radii=num_radii, 
            num_phi=num_phi, radial_freq=radial_freq, angular_freq=angular_freq
        ).to(device)
        self.critic_target = PolarHarmonicsCritic(
            self.encoder.repr_dim, action_shape, feature_dim, hidden_dim, action_space, num_radii=num_radii,
            num_phi=num_phi, radial_freq=radial_freq, angular_freq=angular_freq
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # scaler for mixed precision
        self.scaler = GradScaler()
        self.mixed_precision = mixed_precision
        self.binned_actions = torch.stack([self.critic.Q1.ph.r2d, 
                                        self.critic.Q1.ph.p2d], -1).view(-1,2).to(device)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.critic.train(training)

    def eval(self):
        self.training = False
        self.encoder.eval()
        self.critic.eval()

    def get_mode(self, obs):
        Q1, Q2 = self.critic(obs)
        Q = torch.min(Q1, Q2)
        flat_index = torch.argmax(Q)
        _, _, cols = Q.shape
        x_coord = flat_index // cols
        y_coord = flat_index % cols
        r = self.critic.Q1.ph.r2d[x_coord, y_coord]
        theta = self.critic.Q1.ph.p2d[x_coord, y_coord]
        action = torch.stack([r,theta],-1)
        return action

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        if eval_mode:
            action = self.get_mode(obs)
            action = polar_to_cartesian(action.unsqueeze(0))
        else:
            if random.random() < stddev or step < self.num_expl_steps:
                action = torch.zeros(1,2)
                action.uniform_(-1.0, 1.0)
            else:
                action = self.get_mode(obs)
                action = polar_to_cartesian(action.unsqueeze(0))
        return action.cpu().numpy()[0]

    def update_critic(self, obs, reward, discount, next_obs):
        metrics = dict()

        with torch.no_grad():
            target_Q1, target_Q2 = self.critic_target(next_obs)
            target_V = torch.min(target_Q1, target_Q2)
            target_V = target_V.max(-1).values.max(-1, keepdim=True).values
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs)
        Q1 = Q1.max(-1).values.max(-1, keepdim=True).values
        Q2 = Q2.max(-1).values.max(-1, keepdim=True).values
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics["critic_target_q"] = target_Q.mean().item()
            metrics["critic_q1"] = Q1.mean().item()
            metrics["critic_q2"] = Q2.mean().item()
            metrics["critic_loss"] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)

        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()
        
        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, _, reward, discount, next_obs, _ = utils.to_torch(batch, self.device)

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics["batch_reward"] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, reward, discount, next_obs)
        )

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        # self.scaler.update()

        return metrics

    def save(self):
        save_dict = dict()
        self.eval()
        self.critic_target.eval()

        # model
        save_dict["agent.encoder"] = self.encoder.state_dict()
        save_dict["agent.critic"] = self.critic.state_dict()
        save_dict["agent.critic_target"] = self.critic_target.state_dict()

        # optimizers
        save_dict["agent.encoder_opt"] = self.encoder_opt.state_dict()
        save_dict["agent.critic_opt"] = self.critic_opt.state_dict()

        self.train()
        self.critic_target.train()

        return save_dict

    def load(self, state_dict):
        self.eval()
        self.critic_target.eval()

        # model
        self.encoder.load_state_dict(state_dict["agent.encoder"])
        self.critic.load_state_dict(state_dict["agent.critic"])
        self.critic_target.load_state_dict(state_dict["agent.critic_target"])

        # optimizers
        self.encoder_opt.load_state_dict(state_dict["agent.encoder_opt"])
        self.critic_opt.load_state_dict(state_dict["agent.critic_opt"])

        self.encoder.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

        self.train()
        self.critic_target.train()
