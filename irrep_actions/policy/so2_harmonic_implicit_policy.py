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
from irrep_actions.utils import harmonics
from irrep_actions.policy.base_policy import BasePolicy


class SO2HarmonicImplicitPolicy(BasePolicy):
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
        self.Lmax = lmax
        self.num_neg_act_samples = num_neg_act_samples
        self.pred_n_iter = pred_n_iter
        self.pred_n_samples = pred_n_samples

        self.G = group.so2_group()
        self.gspace = gspaces.no_base_space(self.G)
        self.in_type = self.gspace.type(
            *[self.G.standard_representation()] * 20 + [self.G.trivial_representation]
        )

        out_type = self.gspace.type(*[self.G.bl_regular_representation(L=self.Lmax)])
        #out_type = enn.FieldType(self.gspace, [self.gspace.irrep(l) for l in range(self.Lmax+1)])
        rho = self.G.spectral_regular_representation(*self.G.bl_irreps(L=self.Lmax))
        mid_type = enn.FieldType(self.gspace, z_dim * [rho])
        self.energy_mlp = enn.SequentialModule(
            enn.Linear(self.in_type, mid_type),
            enn.FourierPointwise(self.gspace, z_dim, self.G.bl_irreps(L=self.Lmax), type='regular', N=16),
            enn.Linear(mid_type, mid_type),
            enn.FourierPointwise(self.gspace, z_dim, self.G.bl_irreps(L=self.Lmax), type='regular', N=16),
            enn.Linear(mid_type, mid_type),
            enn.FourierPointwise(self.gspace, z_dim, self.G.bl_irreps(L=self.Lmax), type='regular', N=16),
            enn.Linear(mid_type, out_type),
        )

    def forward(self, obs, action):
        B, N, Ta, Da = action.shape
        B, To, Do = obs.shape

        s = obs.reshape(B, 1, -1).expand(-1, N, -1)
        s_a = self.in_type(torch.cat([s, action.reshape(B, N, -1)], dim=-1).reshape(B*N, -1))
        out = self.energy_mlp(s_a)

        return out.tensor.view(B, N, -1)

    def get_energy(self, W, theta):
        B = harmonics.circular_harmonics(self.Lmax, theta)
        return torch.bmm(W.view(-1, 1, self.Lmax * 2 + 1), B)

    def get_action(self, obs, device):
        nobs = self.normalizer["obs"].normalize(obs['obs'])
        Do = self.obs_dim
        Da = self.action_dim
        To = self.num_obs_steps
        Ta = self.num_action_steps
        B = nobs.shape[0]

        # Sample actions: (B, num_samples, Ta, Da)
        #action_stats = self.get_action_stats()
        #action_dist = torch.distributions.Uniform(
        #   low=action_stats["min"], high=action_stats["max"]
        #)
        #samples = action_dist.sample((B, self.pred_n_samples, Ta)).to(
        #   dtype=nobs.dtype
        #)

        #zero = torch.tensor(0, device=device)
        #resample_std = torch.tensor(3e-2, device=device)
        #for i in range(self.pred_n_iter):
        #    W = self.forward(nobs, samples[:,:,:,0].unsqueeze(3))
        #    logits = self.get_energy(W.view(-1, W.size(2)), samples[:,:,:,1].view(-1, 1))
        #    logits = logits.view(B, self.pred_n_samples)

        #    prob = torch.softmax(logits, dim=-1)

        #    if i < (self.pred_n_iter - 1):
        #        idxs = torch.multinomial(prob, self.pred_n_samples, replacement=True)
        #        samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]
        #        samples += torch.normal(
        #            zero, resample_std, size=samples.shape, device=device
        #        )

        #idxs = torch.multinomial(prob, num_samples=1, replacement=True)
        #acts_n = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs].squeeze(1)
        ##action = self.normalizer["action"].unnormalize(acts_n)
        #x = action[:,:,0] * torch.cos(action[:,:,1])
        #y = action[:,:,0] * torch.sin(action[:,:,1])
        #action = self.normalizer["action"].unnormalize(torch.concat([x, y], dim=1))

        num_disp = 10
        num_rot = 360
        radius = torch.linspace(0, 1, num_disp).to(nobs.device)
        radius = radius.view(1, -1, 1).repeat(B, 1, num_rot).view(B, -1, 1, 1)
        theta = torch.linspace(0, 2*np.pi, num_rot).to(nobs.device)
        theta = theta.view(1, -1 , 1).repeat(B, 1, num_disp).view(B, -1, 1, 1)
        W = self.forward(nobs, radius)
        logits = self.get_energy(W.view(-1, W.size(2)), theta.view(-1, 1))
        logits = logits.view(B, -1)
        prob = torch.softmax(logits, dim=-1)
        idxs = torch.argmax(prob, dim=-1)

        r = radius[torch.arange(B), idxs]
        t = theta[torch.arange(B), idxs]
        acts_n = torch.concat([r,t], dim=-1)
        r = self.normalizer["action"].unnormalize(acts_n).cpu().squeeze()[:,0]

        x = r[0] * np.cos(t.cpu().squeeze())
        y = r[0] * np.sin(t.cpu().squeeze())

        return {'action' : torch.concat([x.view(B,1), y.view(B,1)], dim=1).unsqueeze(1)}

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
        negatives = action_dist.sample((B, self.num_neg_act_samples, Ta)).to(dtype=naction.dtype)


        # Combine pos and neg samples: (B, train_n_neg+1, Da)
        targets = torch.cat([noisy_actions.unsqueeze(1), negatives], dim=1)

        # Randomly permute the positive and negative samples
        permutation = torch.rand(targets.size(0), targets.size(1)).argsort(dim=1)
        targets = targets[torch.arange(targets.size(0)).unsqueeze(-1), permutation]
        ground_truth = (permutation == 0).nonzero()[:, 1].to(naction.device)

        W = self.forward(nobs, targets[:, :, :, 0].unsqueeze(3))
        theta = self.normalizer['action'].unnormalize(targets)[:,:,0,1]
        energy = self.get_energy(W.view(-1, W.size(2)), theta.view(-1, 1))
        energy = energy.view(B, self.num_neg_act_samples + 1)
        loss = F.cross_entropy(energy, ground_truth)

        return loss

    def get_action_stats(self):
        action_stats = self.normalizer["action"].get_output_stats()

        repeated_stats = dict()
        for key, value in action_stats.items():
            n_repeats = self.action_dim // value.shape[0]
            repeated_stats[key] = value.repeat(n_repeats)
        return repeated_stats
