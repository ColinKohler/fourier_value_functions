import io
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
from skimage.transform import resize

from fvf.model.common.normalizer import LinearNormalizer
from fvf.utils import torch_utils
from fvf.policy.base_policy import BasePolicy
from fvf.utils import mcmc
from fvf.model.modules.harmonics import grid


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
        nobs['keypoints'] = nobs['keypoints'].float()
        B = list(obs.values())[0].shape[0]

        Do = self.obs_dim
        Da = self.action_dim
        To = self.num_obs_steps
        Ta = self.num_action_steps

        # Optimize actions
        implicit_stats, energy_stats = self.get_action_stats()
        redges, r = grid.grid1D(energy_stats['max'][0].item(), self.energy_head.num_radii, origin=energy_stats['min'][0].item())
        r = r.view(1,-1).repeat(B,1).to(device)
        pedges, phi = grid.grid1D(2.0 * torch.pi, self.energy_head.num_phi)
        phi = phi.view(1,-1).repeat(B,1).to(device)
        zedges, z = grid.grid1D(energy_stats['max'][2].item(), self.energy_head.num_height, origin=energy_stats['min'][2].item())
        z = z.view(1,-1).repeat(B,1).to(device)

        obs_feat = self.obs_encoder(nobs)
        logits, gripper = self.energy_head(obs_feat)
        logits = logits.view(B, -1)
        action_probs = torch.softmax(logits/self.temperature, dim=-1).view(B, self.energy_head.num_radii, self.energy_head.num_phi, self.energy_head.num_height)

        if self.sample_actions:
            flat_indexes = torch.multinomial(action_probs.flatten(start_dim=1), num_samples=1, replacement=True).squeeze()
        else:
            flat_indexes = torch.argmax(action_probs.flatten(start_dim=1), dim=-1)

        # TODO: This can be a torch unravel if Torch is updated
        idxs = np.unravel_index(flat_indexes.cpu(), action_probs.shape[1:])
        nactions = torch.vstack([
            r[torch.arange(B), idxs[0]],
            phi[torch.arange(B), idxs[1]],
            z[torch.arange(B), idxs[2]],
        ]).permute(1,0).view(B,1,3)

        r = self.normalizer["energy_coords"].unnormalize(nactions)[:,:,0]
        phi = nactions[:,:,1]
        x = r * torch.cos(phi)
        y = r * torch.sin(phi)
        z = self.normalizer["energy_coords"].unnormalize(nactions)[:,:,2]
        gripper_act = self.normalizer["implicit_act"].unnormalize(torch.bernoulli(gripper))
        actions = torch.concat([x.view(B,1), y.view(B,1), z.view(B,1), gripper_act.view(B,1)], dim=1).unsqueeze(1)

        return {'action' : actions, 'action_idxs' : np.stack(idxs).transpose(1,0), 'energy' : action_probs, 'gripper': gripper}

    def compute_loss(self, batch):
        # Load batch
        nobs = self.normalizer.normalize(batch['obs'])
        nobs['keypoints'] = nobs['keypoints'].float()
        nenergy_coords = self.normalizer['energy_coords'].normalize(batch["energy_coords"]).float()
        nimplicit_act = self.normalizer['implicit_act'].normalize(batch["implicit_act"]).float()

        Do = self.obs_dim
        Da = self.action_dim
        To = self.num_obs_steps
        Ta = self.num_action_steps
        B = nimplicit_act.shape[0]

        nimplicit_act, nenergy_coords = self.augment_action(nimplicit_act, nenergy_coords, noise=1e-3)

        # Sample negatives: (B, train_n_neg, Da)
        implicit_stats, energy_stats = self.get_action_stats()
        energy_dist = torch.distributions.Uniform(
            low=energy_stats["min"], high=energy_stats["max"]
        )

        if self.optimize_negatives:
            pass
        else:
            a = torch.tensor([-1, 1])
            p = torch.ones(2) / 2
            idxs = p.multinomial(num_samples=B*self.num_neg_act_samples, replacement=True)
            #implicit_negatives = a[idxs].view(B,self.num_neg_act_samples, Ta, 1).to(nimplicit_act.device)
            energy_negatives = energy_dist.sample((B, self.num_neg_act_samples, Ta)).to(
                dtype=nenergy_coords.dtype
            )

        # Combine pos and neg samples: (B, train_n_neg+1, Ta, Da)
        #implicit_targets = torch.cat([nimplicit_act.unsqueeze(1), implicit_negatives], dim=1)
        energy_targets = torch.cat([nenergy_coords.unsqueeze(1), energy_negatives], dim=1)
        N = energy_targets.size(1)

        # Randomly permute the positive and negative samples
        permutation = torch.rand(B, N).argsort(dim=1)
        #implicit_targets = implicit_targets[torch.arange(B).unsqueeze(-1), permutation]
        energy_targets = energy_targets[torch.arange(B).unsqueeze(-1), permutation]
        ground_truth = (permutation == 0).nonzero()[:, 1].to(nenergy_coords.device)
        one_hot = F.one_hot(ground_truth, num_classes=self.num_neg_act_samples+1).float()

        r = energy_targets[:,:,0,0]
        phi = self.normalizer["energy_coords"].unnormalize(energy_targets)[:,:,0,1]
        z = energy_targets[:,:,0,2]
        energy_targets = torch.concatenate([r.view(B,N,1), phi.view(B,N,1), z.view(B,N,1)], axis=2).view(B,N,1,-1)

        # Compute ciruclar energy function for the given obs and action magnitudes
        obs_feat = self.obs_encoder(nobs)
        energy, gripper_pred = self.energy_head(obs_feat, energy_targets)

        # Compute InfoNCE loss, i.e. try to predict the expert action from the randomly sampled actions
        probs = F.log_softmax(energy, dim=1)
        ebm_loss = F.kl_div(probs, one_hot, reduction='batchmean')
        gripper_loss = F.binary_cross_entropy(gripper_pred, nimplicit_act.view(B,1))
        loss = 0.01 * ebm_loss + gripper_loss

        return loss, ebm_loss, gripper_loss

    def augment_action(self, implicit_act, energy_coords, noise=1e-4):
        start = self.num_obs_steps - 1
        end = start + self.num_action_steps
        implicit_act = implicit_act[:, start:end]
        energy_coords = energy_coords[:, start:end]

        # Add noise
        #implicit_act += torch.normal(
        #    mean=0,
        #    std=noise,
        #    size=implicit_act.shape,
        #    dtype=implicit_act.dtype,
        #    device=implicit_act.device,
        #)
        energy_coords += torch.normal(
            mean=0,
            std=noise,
            size=energy_coords.shape,
            dtype=energy_coords.dtype,
            device=energy_coords.device,
        )

        return implicit_act, energy_coords


    def get_action_stats(self):
        implicit_stats = self.normalizer["implicit_act"].get_output_stats()
        energy_stats = self.normalizer["energy_coords"].get_output_stats()

        return implicit_stats, energy_stats

    def plot_energy_fn(self, img, idxs, energy, gripper):
        _, action_stats = self.get_action_stats()
        zenergy = energy[:,:,idxs[-1]].cpu().numpy()
        renergy = energy[idxs[0],:,:].cpu().numpy()[:,::-1]
        r = torch.linspace(action_stats['min'][0].item(), action_stats['max'][0].item(), self.energy_head.num_radii)
        phi = torch.linspace(0, 2*np.pi, self.energy_head.num_phi)

        f = plt.figure(figsize=(10,3))
        ax1 = f.add_subplot(133)
        ax2 = f.add_subplot(131, projection='polar')
        ax3 = f.add_subplot(132)

        if img is not None:
            ax1.imshow(img[-1].transpose(1,2,0))
            ax1.axes.get_xaxis().set_ticks([])
            ax1.axes.get_yaxis().set_ticks([])
        ax2.pcolormesh(phi,r,zenergy)
        ax2.grid(False)
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
        ax2.set_title(f"z={idxs[-1]}", va="bottom")
        ax3.imshow(resize(renergy.T, (renergy.T.shape[0]*4, renergy.T.shape[1]*4)))
        ax3.set_title(f"g={gripper.item():.3f}", va="bottom")

        io_buf = io.BytesIO()
        f.savefig(io_buf, format='raw')
        io_buf.seek(0)
        img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                     newshape=(int(f.bbox.bounds[3]), int(f.bbox.bounds[2]), -1))
        io_buf.close()
        plt.close()

        return img_arr
