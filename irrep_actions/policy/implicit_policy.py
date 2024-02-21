import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from irrep_actions.utils.normalizer import LinearNormalizer
from irrep_actions.utils import torch_utils
from irrep_actions.policy.base_policy import BasePolicy
from irrep_actions.utils import mcmc


class ImplicitPolicy(BasePolicy):
    def __init__(
        self,
        energy_model,
        obs_dim,
        action_dim,
        num_obs_steps,
        num_action_steps,
        action_sampling,
        num_neg_act_samples,
        pred_n_iter,
        pred_n_samples,
        harmonic_actions,
        optimize_negatives=False,
        sample_actions=False,
        temperature=1.0,
        grad_pen=False,
    ):
        super().__init__(obs_dim, action_dim, num_obs_steps, num_action_steps)
        self.action_sampling = action_sampling
        self.num_neg_act_samples = num_neg_act_samples
        self.pred_n_iter = pred_n_iter
        self.pred_n_samples = pred_n_samples
        self.harmonic_actions = harmonic_actions
        self.optimize_negatives = optimize_negatives
        self.sample_actions = sample_actions
        self.temperature = temperature
        self.grad_pen = grad_pen

        self.energy_model = energy_model
        self.apply(torch_utils.init_weights)

    def get_action(self, obs, device):
        B = obs['obs'].shape[0]

        nobs = self.normalizer['obs'].normalize(obs['obs'])
        Do = self.obs_dim
        Da = self.action_dim
        To = self.num_obs_steps
        Ta = self.num_action_steps

        action_stats = self.get_action_stats()
        action_dist = torch.distributions.Uniform(
            low=action_stats["min"], high=action_stats["max"]
        )

        # Optimize actions
        if self.harmonic_actions:
            mag = torch.linspace(action_stats['min'][0].item(), action_stats['max'][0].item(), self.pred_n_samples)
            mag = mag.view(1, -1, 1).repeat(B, 1, 1).view(B, -1, 1, 1).to(device)
            theta = torch.linspace(0, 2*np.pi, self.energy_model.num_rot).view(1, -1).repeat(B, 1).to(device)
            logits = self.energy_model.get_energy_ball(nobs, mag).view(B, -1)
            action_probs = torch.softmax(logits/self.temperature, dim=-1).view(B, self.pred_n_samples, self.energy_model.num_rot)

            if self.sample_actions:
                flat_indexes = torch.multinomial(action_probs.flatten(start_dim=-2), num_samples=1, replacement=True).squeeze()
            else:
                flat_indexes = torch.argmax(action_probs.flatten(start_dim=-2), dim=-1)
            mag_idx = flat_indexes.div(action_probs.shape[-1], rounding_mode='floor')
            theta_idx = torch.remainder(flat_indexes, action_probs.shape[-1])
            actions = torch.vstack([mag[torch.arange(B),mag_idx,0,0], theta[torch.arange(B), theta_idx]]).permute(1,0).view(B,1,2)
        else:
            # Sample actions: (B, num_samples, Ta, Da)
            actions = action_dist.sample((B, self.pred_n_samples, Ta)).to(
                dtype=nobs.dtype
            )

            if self.action_sampling == 'dfo':
                action_probs, actions = mcmc.iterative_dfo(
                    self.energy_model,
                    nobs,
                    actions,
                    [action_stats['min'], action_stats['max']],
                    harmonic_actions=self.harmonic_actions,
                    normalizer=self.normalizer
                )
            elif self.action_sampling == 'langevin':
                action_probs, actions = mcmc.langevin_actions(
                    self.energy_model,
                    nobs,
                    actions,
                    [action_stats['min'], action_stats['max']],
                    num_iterations=100,
                    harmonic_actions=self.harmonic_actions,
                    normalizer=self.normalizer
                )
            else:
                raise ValueError('Invalid action sampling suggested.')

            if self.sample_actions:
                idxs = torch.multinomial(action_probs, num_samples=1, replacement=True)
            else:
                idxs = torch.argmax(action_probs, dim=-1)
            actions = actions[torch.arange(B).unsqueeze(-1), idxs].squeeze(1)

        if self.harmonic_actions:
            mag = self.normalizer["action"].unnormalize(actions)[:,:,0]
            theta = actions[:,:,1]
            x = mag * torch.cos(theta)
            y = mag * torch.sin(theta)
            actions = torch.concat([x.view(B,1), y.view(B,1)], dim=1).unsqueeze(1)
        else:
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

        # Add noise to positive samples and observations
        action_noise = torch.normal(
            mean=0,
            std=1e-4,
            size=naction.shape,
            dtype=naction.dtype,
            device=naction.device,
        )
        noisy_actions = naction + action_noise

        # Add the same noise to all the points in the observation
        #obs_noise = torch.normal(
        #    mean=0,
        #    std=1e-3,
        #    size=(B, 2),
        #    dtype=nobs.dtype,
        #    device=nobs.device
        #)
        #obs_noise = obs_noise.view(B,1,1,2).repeat(1,To,Do//2,1).view(B,To,-1)
        #nobs = nobs + obs_noise

        # Sample negatives: (B, train_n_neg, Da)
        action_stats = self.get_action_stats()
        action_dist = torch.distributions.Uniform(
            low=action_stats["min"], high=action_stats["max"]
        )

        if self.optimize_negatives and self.harmonic_actions:
            mag = torch.linspace(action_stats['min'][0].item(), action_stats['max'][0].item(), 1000)
            mag = mag.view(1, -1, 1).repeat(B, 1, 1).view(B, -1, 1, 1).to(nobs.device)
            theta = torch.linspace(0, 2*np.pi, self.energy_model.num_rot).view(1, -1).repeat(B, 1).to(nobs.device)
            with torch.no_grad():
                logits = self.energy_model.get_energy_ball(nobs, mag).view(B, -1)
            action_probs = torch.softmax(logits/2., dim=-1).view(B, 1000, self.energy_model.num_rot)

            flat_indexes = torch.multinomial(action_probs.flatten(start_dim=-2), num_samples=self.num_neg_act_samples, replacement=True)
            mag_idx = flat_indexes.view(-1).div(action_probs.shape[-1], rounding_mode='floor')
            theta_idx = torch.remainder(flat_indexes.view(-1), action_probs.shape[-1])
            negatives = torch.vstack([
                mag.repeat(self.num_neg_act_samples, 1, 1, 1)[torch.arange(B*self.num_neg_act_samples),mag_idx,0,0],
                theta.repeat(self.num_neg_act_samples, 1)[torch.arange(B*self.num_neg_act_samples), theta_idx]
            ]).permute(1,0).view(B,self.num_neg_act_samples,Ta,2)
        elif self.optimize_negatives and self.action_sampling == 'langevin':
            negatives = action_dist.sample((B, self.num_neg_act_samples, Ta)).to(
                dtype=naction.dtype
            )

            self.energy_model.eval()
            _, negatives = mcmc.langevin_actions(
                self.energy_model,
                nobs,
                negatives,
                [action_stats['min'], action_stats['max']],
                num_iterations=100,
                harmonic_actions=self.harmonic_actions,
                normalizer=self.normalizer
            )
            self.energy_model.train()
        elif self.optimize_negatives and self.action_sampling == 'dfo':
            negatives = action_dist.sample((B, self.num_neg_act_samples, Ta)).to(
                dtype=naction.dtype
            )
            _, negatives = mcmc.iterative_dfo(
                    self.energy_model,
                    nobs,
                    negatives,
                    [action_stats['min'], action_stats['max']],
                    harmonic_actions=self.harmonic_actions,
                    normalizer=self.normalizer
                )
        else:
            negatives = action_dist.sample((B, self.num_neg_act_samples, Ta)).to(
                dtype=naction.dtype
            )

        # Combine pos and neg samples: (B, train_n_neg+1, Ta, Da)
        targets = torch.cat([noisy_actions.unsqueeze(1), negatives], dim=1)

        # Randomly permute the positive and negative samples
        permutation = torch.rand(targets.size(0), targets.size(1)).argsort(dim=1)
        targets = targets[torch.arange(targets.size(0)).unsqueeze(-1), permutation]
        ground_truth = (permutation == 0).nonzero()[:, 1].to(naction.device)
        one_hot = F.one_hot(ground_truth, num_classes=self.num_neg_act_samples+1).float()

        if self.harmonic_actions:
            mag = targets[:,:,:,0].unsqueeze(3)
            theta = self.normalizer["action"].unnormalize(targets)[:,:,0,1]
            energy = self.energy_model(nobs, mag, theta)
        else:
            energy = self.energy_model(nobs, targets)

        probs = F.log_softmax(energy, dim=1)
        ebm_loss = F.kl_div(probs, one_hot, reduction='batchmean')

        #energy_data = energy[:,0]
        #energy_samp = energy[:,1:]
        #cd_per_example_loss = torch.mean(energy_sampl, axis=1) - torch.mean(energy_data, axis=1)

        #dist = torch.sum()
        #entropy_temp = 1e-1
        #entropy = -torch.exp(-entropy_temp * dist)
        #kl_per_example_loss = torch.mean(-entropy_samp_copy[..., None] - entropy)

        #per_example_loss = cd_per_example_loss + kl_per_example_loss
        #ebm_loss = F.cross_entropy(energy, ground_truth)

        if self.grad_pen:
            de_dact, _ = mcmc.gradient_wrt_action(
                self.energy_model,
                nobs,
                targets.detach(),
                harmonic_actions=self.harmonic_actions,
                normalizer=self.normalizer
            )
            grad_norm = mcmc.compute_grad_norm(de_dact).view(B,-1)
            grad_norm = grad_norm - 1.0
            grad_norm = torch.clamp(grad_norm, 0., 1e3)
            grad_norm = grad_norm ** 2
            grad_loss = torch.mean(grad_norm)
            loss = ebm_loss + grad_loss
        else:
            grad_loss = torch.Tensor([0])
            loss = ebm_loss

        return loss, ebm_loss, grad_loss

    def get_action_stats(self):
        action_stats = self.normalizer["action"].get_output_stats()

        repeated_stats = dict()
        for key, value in action_stats.items():
            n_repeats = self.action_dim // value.shape[0]
            repeated_stats[key] = value.repeat(n_repeats)
        return repeated_stats
