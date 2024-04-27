import io
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt

from imitation_learning.model.common.normalizer import LinearNormalizer
from imitation_learning.utils import torch_utils
from imitation_learning.policy.base_policy import BasePolicy
from imitation_learning.utils import mcmc


class DiskImplicitPolicy(BasePolicy):
    def __init__(
        self,
        obs_encoder: nn.Module,
        energy_head: nn.Module,
        obs_dim: int,
        action_dim: int,
        num_obs_steps: int,
        num_action_steps: int,
        num_neg_act_samples: int,
        pred_n_samples: int,
        optimize_negatives: bool=False,
        sample_actions: bool=False,
        temperature: float=1.0,
        grad_pen: bool=False,
    ):
        super().__init__(obs_dim, action_dim, num_obs_steps, num_action_steps)
        self.num_neg_act_samples = num_neg_act_samples
        self.pred_n_samples = pred_n_samples
        self.optimize_negatives = optimize_negatives
        self.sample_actions = sample_actions
        #self.sample_actions = False
        self.temperature = temperature
        #self.temperature = 0.5
        self.grad_pen = grad_pen

        self.obs_encoder = obs_encoder
        self.energy_head = energy_head
        self.apply(torch_utils.init_weights)

    def get_action(self, obs, device):

        nobs = self.normalizer.normalize(obs)
        B = list(obs.values())[0].shape[0]

        Do = self.obs_dim
        Da = self.action_dim
        To = self.num_obs_steps
        Ta = self.num_action_steps

        action_stats = self.get_action_stats()
        action_dist = torch.distributions.Uniform(
            low=action_stats["min"], high=action_stats["max"]
        )

        # Optimize actions
        #r = torch.linspace(action_stats['min'][0].item(), action_stats['max'][0].item(), self.energy_head.num_radii).view(1,-1).repeat(B,1).to(device)
        r = torch.linspace(action_stats['min'][0].item(), action_stats['max'][0].item(), self.energy_head.num_radii).view(1,-1).repeat(B,1).to(device)
        phi = torch.linspace(0, 2*np.pi, self.energy_head.num_phi).view(1, -1).repeat(B, 1).to(device)
        obs_feat = self.obs_encoder(nobs)
        logits = self.energy_head(obs_feat).view(B, -1)
        action_probs = torch.softmax(logits/self.temperature, dim=-1).view(B, self.energy_head.num_radii, self.energy_head.num_phi)

        if self.sample_actions:
            flat_indexes = torch.multinomial(action_probs.flatten(start_dim=-2), num_samples=1, replacement=True).squeeze()
        else:
            flat_indexes = torch.argmax(action_probs.flatten(start_dim=-2), dim=-1)

        r_idx = flat_indexes.div(action_probs.shape[-1], rounding_mode='floor')
        phi_idx = torch.remainder(flat_indexes, action_probs.shape[-1])
        actions = torch.vstack([r[torch.arange(B),r_idx], phi[torch.arange(B), phi_idx]]).permute(1,0).view(B,1,2)

        r = self.normalizer["action"].unnormalize(actions)[:,:,0]
        phi = actions[:,:,1]
        x = r * torch.cos(phi)
        y = r * torch.sin(phi)
        actions = torch.concat([x.view(B,1), y.view(B,1)], dim=1).unsqueeze(1)

        return {'action' : actions, 'energy' : action_probs}

    def compute_loss(self, batch):
        # Load batch
        nobs = self.normalizer.normalize(batch['obs'])
        naction = self.normalizer['action'].normalize(batch["action"]).float()

        Do = self.obs_dim
        Da = self.action_dim
        To = self.num_obs_steps
        Ta = self.num_action_steps
        B = naction.shape[0]

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

        # Sample negatives: (B, train_n_neg, Da)
        action_stats = self.get_action_stats()
        action_dist = torch.distributions.Uniform(
            low=action_stats["min"], high=action_stats["max"]
        )

        if self.optimize_negatives:
            r = torch.linspace(action_stats['min'][0].item(), action_stats['max'][0].item(), self.energy_head.num_radii).view(1,-1).repeat(B*self.num_neg_act_samples,1).to(naction.device)
            phi = torch.linspace(0, 2*np.pi, self.energy_head.num_phi).view(1, -1).repeat(B*self.num_neg_act_samples, 1).to(naction.device)
            with torch.no_grad():
                obs_feat = self.obs_encoder(nobs)
                logits = self.energy_head(obs_feat).view(B, self.energy_head.num_radii, self.energy_head.num_phi)
            action_probs = torch.softmax(logits/2., dim=-1).view(B, self.energy_head.num_radii, self.energy_head.num_phi)
            flat_indexes = torch.multinomial(action_probs.flatten(start_dim=-2), num_samples=self.num_neg_act_samples, replacement=False).squeeze()

            r_idx = flat_indexes.div(action_probs.shape[-1], rounding_mode='floor')
            phi_idx = torch.remainder(flat_indexes, action_probs.shape[-1])
            negatives = torch.vstack([r[torch.arange(B*self.num_neg_act_samples),r_idx.view(-1)], phi[torch.arange(B*self.num_neg_act_samples), phi_idx.view(-1)]]).permute(1,0).view(B,self.num_neg_act_samples, 1, 2)
        else:
            negatives = action_dist.sample((B, self.num_neg_act_samples, Ta)).to(
                dtype=naction.dtype
            )

        # Combine pos and neg samples: (B, train_n_neg+1, Ta, Da)
        targets = torch.cat([noisy_actions.unsqueeze(1), negatives], dim=1)
        N = targets.size(1)

        # Randomly permute the positive and negative samples
        permutation = torch.rand(B, N).argsort(dim=1)
        targets = targets[torch.arange(B).unsqueeze(-1), permutation]
        ground_truth = (permutation == 0).nonzero()[:, 1].to(naction.device)
        one_hot = F.one_hot(ground_truth, num_classes=self.num_neg_act_samples+1).float()

        # Compute ciruclar energy function for the given obs and action magnitudes
        r = targets[:,:,0,0]
        #rs = torch.linspace(action_stats["min"][0], 1.0, self.energy_head.num_radii).unsqueeze(0).repeat(B*N, 1).to(r.device)
        #r_idxs = torch.argmin((r.view(-1,1) - rs).abs(), dim=1)
        phi = self.normalizer["action"].unnormalize(targets)[:,:,0,1]
        #phis = torch.linspace(0, 2*np.pi, self.energy_head.num_radii).unsqueeze(0).repeat(B*N, 1).to(r.device)
        #phi_idxs = torch.argmin((phi.view(-1,1) - phis).abs(), dim=1)
        #polar_act = torch.concatenate([rs[np.arange(B*N), r_idxs].view(B,N,1), phis[np.arange(B*N), phi_idxs].view(B,N,1)], axis=2)
        polar_act = torch.concatenate([r.view(B,N,1), phi.view(B,N,1)], axis=2)

        obs_feat = self.obs_encoder(nobs)
        energy = self.energy_head(obs_feat, polar_act)

        # Compute InfoNCE loss, i.e. try to predict the expert action from the randomly sampled actions
        probs = F.log_softmax(energy, dim=1)
        ebm_loss = F.kl_div(probs, one_hot, reduction='batchmean')
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

    def plot_energy_fn(self, img, energy):
        action_stats = self.get_action_stats()
        r = torch.linspace(action_stats['min'][0].item(), action_stats['max'][0].item(), self.energy_head.num_radii)
        phi = torch.linspace(0, 2*np.pi, self.energy_head.num_phi)

        flat_indexes = torch.argmax(energy.flatten())
        r_idx = flat_indexes.div(energy.shape[-1], rounding_mode='floor')
        phi_idx = torch.remainder(flat_indexes, energy.shape[-1])

        f = plt.figure(figsize=(10,3))
        ax1 = f.add_subplot(111)
        ax2 = f.add_subplot(141, projection='polar')

        if img is not None:
            ax1.imshow(img[-1].transpose(1,2,0))
        ax2.pcolormesh(phi,r,energy.cpu())
        ax2.scatter(phi[phi_idx], r[r_idx], c='red', s=5)
        ax2.grid(False)
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])

        io_buf = io.BytesIO()
        f.savefig(io_buf, format='raw')
        io_buf.seek(0)
        img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                     newshape=(int(f.bbox.bounds[3]), int(f.bbox.bounds[2]), -1))
        io_buf.close()
        plt.close()

        return img_arr
