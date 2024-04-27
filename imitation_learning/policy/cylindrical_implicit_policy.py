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


class CylindricalImplicitPolicy(BasePolicy):
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

        # Optimize actions
        implicit_stats, energy_stats = self.get_action_stats()
        r = torch.linspace(energy_stats['min'][0].item(), energy_stats['max'][0].item(), self.energy_head.num_radii).view(1,-1).repeat(B,1).to(device)
        phi = torch.linspace(0, 2*np.pi, self.energy_head.num_phi).view(1, -1).repeat(B, 1).to(device)
        z = torch.linspace(energy_stats['min'][2].item(), energy_stats['max'][2].item(), self.energy_head.num_height).view(1,-1).repeat(B,1).to(device)
        gripper = torch.linspace(implicit_stats['min'][0].item(), implicit_stats['max'][0].item(), 10).view(1,-1,1,1).repeat(B,1,1,1).to(device)

        cylindrical_act = torch.concatenate([r, phi, z], axis=1)
        obs_feat = self.obs_encoder(nobs)
        logits = self.energy_head(obs_feat, gripper).view(B, -1)
        action_probs = torch.softmax(logits/self.temperature, dim=-1).view(B, 10, self.energy_head.num_radii, self.energy_head.num_phi, self.energy_head.num_height)

        if self.sample_actions:
            flat_indexes = torch.multinomial(action_probs.flatten(start_dim=1), num_samples=1, replacement=True).squeeze()
        else:
            flat_indexes = torch.argmax(action_probs.flatten(start_dim=1), dim=-1)

        # TODO: This can be a torch unravel if Torch is updated
        idxs = np.unravel_index(flat_indexes.cpu(), action_probs.shape[1:])
        nactions = torch.vstack([
            r[torch.arange(B), idxs[1]],
            phi[torch.arange(B), idxs[2]],
            z[torch.arange(B), idxs[3]],
        ]).permute(1,0).view(B,1,3)
        ngripper_act = gripper[torch.arange(B), idxs[0], 0, 0]

        r = self.normalizer["energy_coords"].unnormalize(nactions)[:,:,0]
        phi = nactions[:,:,1]
        x = r * torch.cos(phi)
        y = r * torch.sin(phi)
        z = self.normalizer["energy_coords"].unnormalize(nactions)[:,:,2]
        gripper_act = self.normalizer["implicit_act"].unnormalize(ngripper_act)
        actions = torch.concat([x.view(B,1), y.view(B,1), z.view(B,1), gripper_act.view(B,1)], dim=1).unsqueeze(1)
        breakpoint()

        return {'action' : actions, 'energy' : action_probs}

    def compute_loss(self, batch):
        # Load batch
        nobs = self.normalizer.normalize(batch['obs'])
        nenergy_coords = self.normalizer['energy_coords'].normalize(batch["energy_coords"]).float()
        nimplicit_act = self.normalizer['implicit_act'].normalize(batch["implicit_act"]).float()

        Do = self.obs_dim
        Da = self.action_dim
        To = self.num_obs_steps
        Ta = self.num_action_steps
        B = nimplicit_act.shape[0]

        nimplicit_act, nenergy_coords = self.augment_action(nimplicit_act, nenergy_coords)

        # Sample negatives: (B, train_n_neg, Da)
        implicit_stats, energy_stats = self.get_action_stats()
        implicit_dist = torch.distributions.Uniform(
            low=implicit_stats["min"], high=implicit_stats["max"]
        )
        energy_dist = torch.distributions.Uniform(
            low=energy_stats["min"], high=energy_stats["max"]
        )

        if self.optimize_negatives:
            pass
        else:
            implicit_negatives = implicit_dist.sample((B, self.num_neg_act_samples, Ta)).to(
                dtype=nimplicit_act.dtype
            )
            energy_negatives = energy_dist.sample((B, self.num_neg_act_samples, Ta)).to(
                dtype=nenergy_coords.dtype
            )

        # Combine pos and neg samples: (B, train_n_neg+1, Ta, Da)
        implicit_targets = torch.cat([nimplicit_act.unsqueeze(1), implicit_negatives], dim=1)
        energy_targets = torch.cat([nenergy_coords.unsqueeze(1), energy_negatives], dim=1)
        N = implicit_targets.size(1)

        # Randomly permute the positive and negative samples
        permutation = torch.rand(B, N).argsort(dim=1)
        implicit_targets = implicit_targets[torch.arange(B).unsqueeze(-1), permutation]
        energy_targets = energy_targets[torch.arange(B).unsqueeze(-1), permutation]
        ground_truth = (permutation == 0).nonzero()[:, 1].to(nimplicit_act.device)
        one_hot = F.one_hot(ground_truth, num_classes=self.num_neg_act_samples+1).float()

        # Compute ciruclar energy function for the given obs and action magnitudes
        obs_feat = self.obs_encoder(nobs)
        energy = self.energy_head(obs_feat, implicit_targets, energy_targets)

        # Compute InfoNCE loss, i.e. try to predict the expert action from the randomly sampled actions
        probs = F.log_softmax(energy, dim=1)
        ebm_loss = F.kl_div(probs, one_hot, reduction='batchmean')
        grad_loss = torch.Tensor([0])
        loss = ebm_loss

        return loss, ebm_loss, grad_loss

    def augment_action(self, implicit_act, energy_coords, noise=1e-3):
        start = self.num_obs_steps - 1
        end = start + self.num_action_steps
        implicit_act = implicit_act[:, start:end]
        energy_coords = energy_coords[:, start:end]

        # Add noise
        implicit_act += torch.normal(
            mean=0,
            std=noise,
            size=implicit_act.shape,
            dtype=implicit_act.dtype,
            device=implicit_act.device,
        )
        energy_coords += torch.normal(
            mean=0,
            std=noise,
            size=energy_coords.shape,
            dtype=energy_coords.dtype,
            device=energy_coords.device,
        )

        return implicit_act, energy_coords


    def get_action_stats(self):
        energy_stats = self.normalizer["energy_coords"].get_output_stats()
        implicit_stats = self.normalizer["implicit_act"].get_output_stats()

        return implicit_stats, energy_stats

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
