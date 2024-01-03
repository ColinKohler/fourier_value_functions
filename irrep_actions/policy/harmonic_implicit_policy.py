import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from irrep_actions.model.layers import MLP
from irrep_actions.utils.normalizer import LinearNormalizer
from irrep_actions.utils import torch_utils
from irrep_actions.utils import harmonics
from irrep_actions.policy.base_policy import BasePolicy


class HarmonicImplicitPolicy(BasePolicy):
    def __init__(
        self,
        action_dim,
        seq_len,
        lmax,
        z_dim,
        num_neg_act_samples,
        pred_n_iter,
        pred_n_samples,
        dropout,
        encoder,
    ):
        super().__init__(action_dim, seq_len, z_dim)
        self.Lmax = lmax
        self.num_neg_act_samples = num_neg_act_samples
        self.pred_n_iter = pred_n_iter
        self.pred_n_samples = pred_n_samples

        self.encoder = encoder
        m_dim = z_dim * 2
        self.energy_mlp = MLP(
            [z_dim + 1, m_dim, m_dim, m_dim, 2 * self.Lmax + 1],
            dropout=dropout,
            act_out=False,
        )

        self.apply(torch_utils.init_weights)

    def forward(self, x, y, a):
        batch_size = x.size(0)
        z = self.encoder(x, y)

        z_a = torch.cat([z.unsqueeze(1).expand(-1, a.size(1), -1), a], dim=-1)
        B, N, D = z_a.shape
        z_a.reshape(B * N, D)

        out = self.energy_mlp(z_a)

        return out.view(B, N, -1)

    def get_energy(self, W, theta):
        B = harmonics.circular_harmonics(self.Lmax, theta)
        return torch.bmm(W.view(-1, 1, self.Lmax * 2 + 1), B)

    def get_action(self, obs, goal, device):
        ngoal = self.normalizer["goal"].normalize(goal)
        nobs = self.normalizer["obs"].normalize(np.stack(obs))
        # goal_noise = npr.uniform([-0.010, -0.010, 0.0], [0.010, 0.010, 0])
        goal_noise = 0

        policy_obs = nobs.unsqueeze(0).flatten(1, 2)
        # policy_obs = torch.concat((ngoal.view(1,1,3).repeat(1,20,1), policy_obs), dim=-1)
        policy_obs[:, :, :3] = ngoal.view(1, 1, 3).repeat(1, self.seq_len, 1) - (
            policy_obs[:, :, :3] + goal_noise
        )
        policy_obs = policy_obs.to(device)

        # Sample actions: (1, num_samples, Da)
        action_stats = self.get_action_stats()
        # action_dist = torch.distributions.Uniform(
        #    low=action_stats["min"], high=action_stats["max"]
        # )
        # samples = action_dist.sample((1, self.pred_n_samples)).to(
        #    dtype=policy_obs.dtype
        # )

        # zero = torch.tensor(0, device=device)
        # resample_std = torch.tensor(3e-2, device=device)
        # for i in range(self.pred_n_iter):
        #    W = self.forward(policy_obs, samples[:,:,0].unsqueeze(2))
        #    logits = self.get_energy(W.view(-1, W.size(2)), samples[:,:,1].view(-1, 1))
        #    logits = logits.view(1, self.pred_n_samples)

        #    # attn = self.encoder.transformer.getAttnMaps(policy_obs)
        #    # torch_utils.plotAttnMaps(torch.arange(9).view(1,9), attn)

        #    prob = torch.softmax(logits, dim=-1)

        #    if i < (self.pred_n_iter - 1):
        #        idxs = torch.multinomial(prob, self.pred_n_samples, replacement=True)
        #        samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]
        #        samples += torch.normal(
        #            zero, resample_std, size=samples.shape, device=device
        #        )

        num_disp = 1
        num_rot = 360
        # radius = torch.linspace(action_stats['min'][0].item(), action_stats['max'][0].item(), num_disp)
        radius = torch.tensor([1]).view(1, 1).float()
        radius = (
            radius.view(1, -1, 1)
            .repeat(
                1,
                1,
                num_rot,
            )
            .view(1, -1, 1)
        )
        theta = torch.linspace(
            action_stats["min"][1].item(), action_stats["max"][1].item(), num_rot
        )
        theta = theta.view(1, -1, 1).repeat(1, 1, num_disp).view(-1, 1)
        W = self.forward(policy_obs, radius)
        logits = self.get_energy(W.view(-1, W.size(2)), theta)
        logits = logits.view(1, -1)
        prob = torch.softmax(logits, dim=-1)

        idxs = torch.multinomial(prob, num_samples=1, replacement=True)
        # acts_n = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs].squeeze(1)

        print(idxs)
        idxs = torch.argmax(prob)
        print(idxs)
        acts_n = torch.tensor([radius[0, idxs.item()], theta[idxs.item(), 0]])
        action = self.normalizer["action"].unnormalize(acts_n).cpu().squeeze()
        # action[0] = 0.02
        # action[1] = np.pi

        x = action[0] * np.cos(action[1])
        y = action[0] * np.sin(action[1])

        print(acts_n)
        print(action)
        print([x.item(), y.item()])
        harmonics.plot_energy_circle(prob[0, :].detach().numpy())

        return [x, y]

    def compute_loss(self, batch):
        # Load batch
        nobs = batch["obs"].float()
        naction = batch["action"].float()
        ngoal = batch["goal"].float()

        B = nobs.shape[0]
        obs = nobs.flatten(1, 2)
        # obs = torch.concat((ngoal[:,0,:].unsqueeze(1).repeat(1,20,1), obs), dim=-1)
        obj_state = (
            ngoal[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1) - obs[:, :, :3]
        )

        # Add noise to positive samples
        batch_size = naction.size(0)
        action_noise = torch.normal(
            mean=0,
            std=1e-4,
            size=naction[:, -1].shape,
            dtype=naction.dtype,
            device=naction.device,
        )
        noisy_actions = naction[:, -1] + action_noise

        # Sample negatives: (B, train_n_neg, Da)
        action_stats = self.get_action_stats()
        action_dist = torch.distributions.Uniform(
            low=action_stats["min"], high=action_stats["max"]
        )
        negatives = (
            action_dist.sample((batch_size, self.num_neg_act_samples))
            .to(dtype=naction.dtype)
            .view(B, -1, 2)
        )

        # Combine pos and neg samples: (B, train_n_neg+1, Da)
        targets = torch.cat([noisy_actions.view(B, 1, 2), negatives], dim=1)

        # Randomly permute the positive and negative samples
        permutation = torch.rand(targets.size(0), targets.size(1)).argsort(dim=1)
        targets = targets[torch.arange(targets.size(0)).unsqueeze(-1), permutation]
        ground_truth = (permutation == 0).nonzero()[:, 1].to(naction.device)

        W = self.forward(obs, obj_state, targets[:, :, 0].unsqueeze(2))
        energy = self.get_energy(W.view(-1, W.size(2)), targets[:, :, 1].view(-1, 1))
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
