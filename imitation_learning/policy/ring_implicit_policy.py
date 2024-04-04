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


class RingImplicitPolicy(BasePolicy):
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
        self.temperature = temperature
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
        r = torch.linspace(action_stats['min'][0].item(), action_stats['max'][0].item(), self.energy_head.num_radii).view(1,-1).repeat(B,1).to(device)
        phi = torch.linspace(0, 2*np.pi, self.energy_head.num_phi).view(1, -1).repeat(B, 1).to(device)
        obs_feat = self.obs_encoder(nobs)
        logits = self.energy_head(obs_feat).view(B, self.energy_head.num_radii, self.energy_head.num_phi)
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

        if self.optimize_negatives:
            mag = torch.linspace(action_stats['min'][0].item(), action_stats['max'][0].item(), 1000)
            mag = mag.view(1, -1, 1).repeat(B, 1, 1).view(B, -1, 1, 1).to(nobs.device)
            theta = torch.linspace(0, 2*np.pi, self.energy_head.num_rot).view(1, -1).repeat(B, 1).to(nobs.device)
            with torch.no_grad():
                logits = self.get_energy_ball(nobs, mag).view(B, -1)
            action_probs = torch.softmax(logits/2., dim=-1).view(B, 1000, self.energy_head.num_rot)

            theta_idx = torch.remainder(flat_indexes.view(-1), action_probs.shape[-1])
            negatives = torch.vstack([
                mag.repeat(self.num_neg_act_samples, 1, 1, 1)[torch.arange(B*self.num_neg_act_samples),mag_idx,0,0],
                theta.repeat(self.num_neg_act_samples, 1)[torch.arange(B*self.num_neg_act_samples), theta_idx]
            ]).permute(1,0).view(B,self.num_neg_act_samples,Ta,2)
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
        #r = self.normalizer["action"].unnormalize(targets)[:,:,0,0]
        r = targets[:,:,0,0]
        phi = self.normalizer["action"].unnormalize(targets)[:,:,0,1]
        obs_feat = self.obs_encoder(nobs)
        energy = self.energy_head(obs_feat).view(B, 1, self.energy_head.num_radii, self.energy_head.num_phi)
        energy = energy.repeat(1,N,1,1).view(B*N, self.energy_head.num_radii, self.energy_head.num_phi)

        # Find closest theta for all actions and index the energy function at the theta
        max_r = action_stats["max"][0]
        rs = torch.linspace(action_stats["min"][0], max_r, self.energy_head.num_radii).unsqueeze(0).repeat(B*N, 1).to(r.device)
        r_idxs = torch.argmin((r.view(-1,1) - rs).abs(), dim=1)
        phis = torch.linspace(0, 2*np.pi, self.energy_head.num_phi).unsqueeze(0).repeat(B*N, 1).to(phi.device)
        phi_idxs = torch.argmin((phi.view(-1,1) - phis).abs(), dim=1)
        energy = energy[torch.arange(B*N), r_idxs, phi_idxs].view(B, N)

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
        max_disp = torch.max(energy, dim=-1)[0]
        E = energy[torch.argmax(max_disp).item()].cpu().numpy()

        f = plt.figure(figsize=(10,3))
        ax1 = f.add_subplot(111)
        ax2 = f.add_subplot(141, projection='polar')

        if img is not None:
            ax1.imshow(img[-1].transpose(1,2,0))
        ax2.plot(np.linspace(0, 2*np.pi, E.shape[0]), E)
        ax2.set_rticks(list())
        ax2.grid(True)
        ax2.set_title(f"R={torch.max(max_disp).item():.3f}", va="bottom")

        io_buf = io.BytesIO()
        f.savefig(io_buf, format='raw')
        io_buf.seek(0)
        img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                     newshape=(int(f.bbox.bounds[3]), int(f.bbox.bounds[2]), -1))
        io_buf.close()
        plt.close()

        return img_arr
