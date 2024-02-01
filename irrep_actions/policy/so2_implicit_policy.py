import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from escnn import gspaces
from escnn import nn as enn
from escnn import group

from irrep_actions.model.layers import SO2MLP
from irrep_actions.utils.normalizer import LinearNormalizer
from irrep_actions.utils import torch_utils
from irrep_actions.policy.base_policy import BasePolicy
from irrep_actions.utils import mcmc


class SO2ImplicitPolicy(BasePolicy):
    def __init__(
        self,
        obs_dim,
        action_dim,
        num_obs_steps,
        num_action_steps,
        horizon,
        lmax,
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


    def get_action(self, obs, device):
        obs['obs'] -= 255
        nobs = self.normalizer["obs"].normalize(obs['obs'])
        Do = self.obs_dim
        Da = self.action_dim
        To = self.num_obs_steps
        Ta = self.num_action_steps
        B = nobs.shape[0]

        # Sample actions: (1, num_samples, Da)
        action_stats = self.get_action_stats()
        action_dist = torch.distributions.Uniform(
            low=action_stats["min"], high=action_stats["max"]
        )
        actions = action_dist.sample((B, self.pred_n_samples, Ta)).to(
            dtype=nobs.dtype
        )

        # Optimize actions
        if False:
            action_probs, actions = mcmc.iterative_dfo(
                self,
                nobs,
                actions,
                [action_stats['min'], action_stats['max']],
            )
        else:
            action_probs, actions = mcmc.langevin_actions(
                self,
                nobs,
                actions,
                [action_stats['min'], action_stats['max']],
            )

        idxs = torch.multinomial(action_probs, num_samples=1, replacement=True)
        actions = actions[torch.arange(B).unsqueeze(-1), idxs].squeeze(1)
        actions = self.normalizer["action"].unnormalize(actions)

        return {'action' : actions}

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
        if True:
            _, negatives = mcmc.langevin_actions(
                self,
                nobs,
                negatives,
                [action_stats['min'], action_stats['max']],
            )

        # Combine pos and neg samples: (B, train_n_neg+1, Ta, Da)
        targets = torch.cat([noisy_actions.unsqueeze(1), negatives], dim=1)

        # Randomly permute the positive and negative samples
        permutation = torch.rand(targets.size(0), targets.size(1)).argsort(dim=1)
        targets = targets[torch.arange(targets.size(0)).unsqueeze(-1), permutation]
        ground_truth = (permutation == 0).nonzero()[:, 1].to(naction.device)

        energy = self.forward(nobs, targets)
        ebm_loss = F.cross_entropy(energy, ground_truth)

        de_dact, _ = mcmc.gradient_wrt_action(
            self,
            nobs,
            targets.detach(),
            False,
        )
        grad_norm = mcmc.compute_grad_norm(de_dact).view(B,-1)
        grad_norm = grad_norm - 1.0
        grad_norm = torch.clamp(grad_norm, 0., 1e10)
        grad_norm = grad_norm ** 2
        grad_loss = torch.mean(grad_norm)
        loss = ebm_loss + grad_loss

        return loss, ebm_loss, grad_loss

    def get_action_stats(self):
        action_stats = self.normalizer["action"].get_output_stats()

        repeated_stats = dict()
        for key, value in action_stats.items():
            n_repeats = self.action_dim // value.shape[0]
            repeated_stats[key] = value.repeat(n_repeats)
        return repeated_stats
