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
    ):
        super().__init__(obs_dim, action_dim, num_obs_steps, num_action_steps)
        self.action_sampling = action_sampling
        self.num_neg_act_samples = num_neg_act_samples
        self.pred_n_iter = pred_n_iter
        self.pred_n_samples = pred_n_samples
        self.harmonic_actions = harmonic_actions

        self.energy_model = energy_model
        self.apply(torch_utils.init_weights)

    def get_action(self, obs, device):
        B = obs['obs'].shape[0]

        x_obs = (obs['obs'].reshape(B,19*2,2)[:,:,0] - 255.0)
        y_obs = (obs['obs'].reshape(B,19*2,2)[:,:,1] - 255.0) * -1.
        new_d = torch.concatenate((x_obs.unsqueeze(-1), y_obs.unsqueeze(-1)), dim=-1).view(B, -1).view(B,2,19*2)
        obs['obs'] = new_d
        #obs['obs'] = new_d - 255

        nobs = self.normalizer['obs'].normalize(obs['obs'])
        Do = self.obs_dim
        Da = self.action_dim
        To = self.num_obs_steps
        Ta = self.num_action_steps

        # Sample actions: (B, num_samples, Ta, Da)
        action_stats = self.get_action_stats()
        action_dist = torch.distributions.Uniform(
            low=action_stats["min"], high=action_stats["max"]
        )
        actions = action_dist.sample((B, self.pred_n_samples, Ta)).to(
            dtype=nobs.dtype
        )

        #num_disp = 100
        #num_rot = 45
        #mag = torch.linspace(action_stats['min'][0].item(), action_stats['max'][0].item(), num_disp)
        #mag = (
        #    mag.view(1, -1, 1)
        #    .repeat(
        #        B,
        #        1,
        #        num_rot,
        #    )
        #    .view(B, -1, 1, 1)
        #).to(device)
        #theta = torch.linspace(action_stats['min'][1].item(), action_stats['max'][1].item(), num_rot)
        #theta = theta.view(1, 1, -1).repeat(B, num_disp, 1).view(-1, 1).to(device)
        #actions = torch.concatenate((mag, theta.view(B, -1, 1, 1)), dim=-1)

        # Optimize actions
        if False:#self.harmonic_actions:
            num_disp = 100
            num_rot = 45
            mag = torch.linspace(action_stats['min'][0].item(), action_stats['max'][0].item(), num_disp)
            mag = (
                mag.view(1, -1, 1)
                .repeat(
                    B,
                    1,
                    num_rot,
                )
                .view(B, -1, 1, 1)
            ).to(device)
            theta = torch.linspace(0, 2*np.pi, num_rot)
            theta = theta.view(1, 1, -1).repeat(B, num_disp, 1).view(-1, 1).to(device)
            logits = self.energy_model(nobs, mag, theta)
            action_probs = torch.softmax(logits, dim=-1)
            actions = torch.concatenate((mag, theta.view(B, -1, 1, 1)), dim=-1)
        else:
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

        #idxs = torch.multinomial(action_probs, num_samples=1, replacement=True)
        idxs = torch.argmax(action_probs, dim=-1).unsqueeze(-1)
        #from irrep_actions.utils import harmonics
        #breakpoint()
        actions = actions[torch.arange(B).unsqueeze(-1), idxs].squeeze(1)

        if self.harmonic_actions:
            mag = self.normalizer["action"].unnormalize(actions)[:,:,0]
            theta = self.normalizer["action"].unnormalize(actions)[:,:,1]
            #theta = actions[:,:,1]
            x = mag * torch.cos(theta)
            y = mag * torch.sin(theta)
            actions = torch.concat([x.view(B,1), y.view(B,1)], dim=1).unsqueeze(1)
        else:
            actions = self.normalizer["action"].unnormalize(actions)

        x_act = actions[:,:,0]
        y_act = actions[:,:,1] * -1
        new_act = torch.concatenate((x_act, y_act), dim=-1).view(B,1,2)
        return {'action' : new_act}

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
            std=1e-3,
            size=naction.shape,
            dtype=naction.dtype,
            device=naction.device,
        )
        noisy_actions = naction + action_noise

        obs_noise = torch.normal(
            mean=0,
            std=1e-3,
            size=nobs.shape,
            dtype=nobs.dtype,
            device=nobs.device
        )
        nobs = nobs + obs_noise

        # Sample negatives: (B, train_n_neg, Da)
        action_stats = self.get_action_stats()
        action_dist = torch.distributions.Uniform(
            low=action_stats["min"], high=action_stats["max"]
        )
        negatives = action_dist.sample((B, self.num_neg_act_samples, Ta)).to(
            dtype=naction.dtype
        )
        if self.action_sampling == 'langevin':
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

        if self.action_sampling == 'langevin':
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
