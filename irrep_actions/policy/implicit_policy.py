import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from irrep_actions.model.layers import MLP
from irrep_actions.utils.normalizer import LinearNormalizer
from irrep_actions.utils import torch_utils
from irrep_actions.policy.base_policy import BasePolicy


class ImplicitPolicy(BasePolicy):
    def __init__(
        self,
        obs_dim,
        action_dim,
        num_obs_steps,
        num_action_steps,
        horizon,
        z_dim,
        num_neg_act_samples,
        pred_n_iter,
        pred_n_samples,
        dropout,
        #encoder,
    ):
        super().__init__(obs_dim, action_dim, num_obs_steps, num_action_steps, horizon, z_dim)
        self.num_neg_act_samples = num_neg_act_samples
        self.pred_n_iter = pred_n_iter
        self.pred_n_samples = pred_n_samples

        in_action_channels = action_dim * num_action_steps
        in_obs_channels = obs_dim * num_obs_steps
        in_channels = in_action_channels + in_obs_channels
        mid_channels = 1024
        out_channels = 1

        #self.encoder = encoder
        self.energy_mlp = MLP(
            [in_channels, mid_channels, mid_channels, mid_channels, out_channels], dropout=dropout, act_out=False
        )

        self.apply(torch_utils.init_weights)

    def forward(self, obs, action):
        B, N, Ta, Da = action.shape
        B, To, Do = obs.shape

        s = obs.reshape(B, 1, -1).expand(-1, N, -1)
        s_a = torch.cat([s, action.reshape(B, N, -1)], dim=-1).reshape(B*N, -1)
        out = self.energy_mlp(s_a)

        return out.reshape(B, N)

    def get_action(self, obs, device):
        nobs = self.normalizer['obs'].normalize(obs['obs'])
        Do = self.obs_dim
        Da = self.action_dim
        To = self.num_obs_steps
        Ta = self.num_action_steps
        B = nobs.shape[0]

        # Sample actions: (B, num_samples, Ta, Da)
        action_stats = self.get_action_stats()
        action_dist = torch.distributions.Uniform(
            low=action_stats["min"], high=action_stats["max"]
        )
        samples = action_dist.sample((B, self.pred_n_samples, Ta)).to(
            dtype=nobs.dtype
        )

        zero = torch.tensor(0, device=device)
        resample_std = torch.tensor(3e-1, device=device)
        for i in range(self.pred_n_iter):
            logits = self.forward(nobs, samples)

            prob = torch.softmax(logits, dim=-1)

            if i < (self.pred_n_iter - 1):
                idxs = torch.multinomial(prob, self.pred_n_samples, replacement=True)
                samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]
                samples += torch.normal(
                    zero, resample_std, size=samples.shape, device=device
                )
                resample_std *= 0.5

        idxs = torch.multinomial(prob, num_samples=1, replacement=True)
        acts_n = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs].squeeze(1)
        action = self.normalizer["action"].unnormalize(acts_n)

        return {'action' : action}

    def compute_loss(self, batch):
        # Load batch
        nobs = batch["obs"].float()
        naction = batch["action"].float()

        Do = self.obs_dim
        Da = self.action_dim
        To = self.num_obs_steps
        Ta = self.num_action_steps
        B = naction.shape[0]

        nobs = nobs[:, :To]
        start = To - 1
        end = start + Ta
        naction = naction[:, start:end]

        # Add noise to positive samples
        action_noise = torch.normal(
            mean=0,
            std=1e-4,
            size=naction.shape,
            dtype=naction.dtype,
            device=naction.device,
        )
        noisy_actions = naction + action_noise

        # Sample negatives: (B, train_n_neg, Da)
        action_stats = self.get_action_stats()
        action_dist = torch.distributions.Uniform(
            low=action_stats["min"], high=action_stats["max"]
        )
        negatives = action_dist.sample((B, self.num_neg_act_samples, Ta)).to(
            dtype=naction.dtype
        )

        # Combine pos and neg samples: (B, train_n_neg+1, Ta, Da)
        targets = torch.cat([noisy_actions.unsqueeze(1), negatives], dim=1)

        # Randomly permute the positive and negative samples
        permutation = torch.rand(targets.size(0), targets.size(1)).argsort(dim=1)
        targets = targets[torch.arange(targets.size(0)).unsqueeze(-1), permutation]
        ground_truth = (permutation == 0).nonzero()[:, 1].to(naction.device)

        energy = self.forward(nobs, targets)
        loss = F.cross_entropy(energy, ground_truth)

        return loss

    def get_action_stats(self):
        action_stats = self.normalizer["action"].get_output_stats()

        repeated_stats = dict()
        for key, value in action_stats.items():
            n_repeats = self.action_dim // value.shape[0]
            repeated_stats[key] = value.repeat(n_repeats)
        return repeated_stats
