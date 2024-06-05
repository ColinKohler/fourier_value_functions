import io
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
from skimage.transform import resize
from escnn.group.groups.so3_utils import _grid
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_euler_angles, euler_angles_to_matrix, matrix_to_axis_angle

from imitation_learning.model.common.normalizer import LinearNormalizer
from imitation_learning.utils import torch_utils
from imitation_learning.policy.base_policy import BasePolicy
from imitation_learning.utils import mcmc
from imitation_learning.model.modules.harmonics import grid

class SO3CylindricalImplicitPolicy(BasePolicy):
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
        gripper_stats, pose_stats = self.get_action_stats()
        redges, r = grid.grid1D(pose_stats['max'][0].item(), self.energy_head.num_radii, origin=pose_stats['min'][0].item())
        r = r.view(1,-1).repeat(B,1).to(device)
        pedges, phi = grid.grid1D(2.0 * torch.pi, self.energy_head.num_phi)
        phi = phi.view(1,-1).repeat(B,1).to(device)
        zedges, z = grid.grid1D(pose_stats['max'][2].item(), self.energy_head.num_height, origin=pose_stats['min'][2].item())
        z = z.view(1,-1).repeat(B,1).to(device)
        so3_grid = torch.from_numpy(_grid('thomson', N=self.energy_head.num_so3, parametrization='ZYZ'))
        so3_grid = so3_grid.view(1,self.energy_head.num_so3,3).repeat(B,1,1).to(device)

        obs_feat = self.obs_encoder(nobs)
        pos_logits, rot_logits, gripper = self.energy_head(obs_feat)
        pos_logits = pos_logits.view(B, -1)
        rot_logits = rot_logits.view(B, -1)
        pos_probs = torch.softmax(pos_logits/self.temperature, dim=-1).view(B, self.energy_head.num_radii, self.energy_head.num_phi, self.energy_head.num_height)
        rot_probs= torch.softmax(rot_logits/self.temperature, dim=-1).view(B, self.energy_head.num_so3)

        if self.sample_actions:
            pos_flat_indexes = torch.multinomial(pos_probs.flatten(start_dim=1), num_samples=1, replacement=True).squeeze()
            rot_flat_indexes = torch.multinomial(rot_probs.flatten(start_dim=1), num_samples=1, replacement=True).squeeze()
        else:
            pos_flat_indexes = torch.argmax(pos_probs.flatten(start_dim=1), dim=-1)
            rot_flat_indexes = torch.argmax(rot_probs.flatten(start_dim=1), dim=-1)

        # TODO: This can be a torch unravel if Torch is updated
        pos_idxs = np.unravel_index(pos_flat_indexes.cpu(), pos_probs.shape[1:])
        npos_actions = torch.vstack([
            r[torch.arange(B), pos_idxs[0]],
            phi[torch.arange(B), pos_idxs[1]],
            z[torch.arange(B), pos_idxs[2]],
        ]).permute(1,0).view(B,1,3)

        rot_idxs = np.unravel_index(rot_flat_indexes.cpu(), rot_probs.shape[1:])
        nrot_actions = so3_grid[torch.arange(B), rot_idxs[0]].view(B,1,3)

        nactions = torch.concat([npos_actions, nrot_actions], dim=2)

        r = self.normalizer["pose_act"].unnormalize(nactions)[:,:,0]
        phi = nactions[:,:,1]
        x = r * torch.cos(phi)
        y = r * torch.sin(phi)
        z = self.normalizer["pose_act"].unnormalize(nactions)[:,:,2]
        r_axis_angle = matrix_to_axis_angle(euler_angles_to_matrix(nrot_actions, 'ZYZ'))
        gripper_act = self.normalizer["gripper_act"].unnormalize(torch.bernoulli(gripper))
        actions = torch.concat([x.view(B,1), y.view(B,1), z.view(B,1), r_axis_angle.view(B,3), gripper_act.view(B,1)], dim=1).unsqueeze(1)

        return {'action' : actions, 'action_idxs' : np.stack(pos_idxs).transpose(1,0), 'energy' : pos_probs, 'gripper': gripper}

    def compute_loss(self, batch):
        # Load batch
        nobs = self.normalizer.normalize(batch['obs'])
        nobs['keypoints'] = nobs['keypoints'].float()
        npose_act = self.normalizer['pose_act'].normalize(batch["pose_act"]).float()
        ngripper_act = self.normalizer['gripper_act'].normalize(batch["gripper_act"]).float()

        Do = self.obs_dim
        Da = self.action_dim
        To = self.num_obs_steps
        Ta = self.num_action_steps
        B = ngripper_act.shape[0]

        ngripper_act, npose_act = self.augment_action(ngripper_act, npose_act, noise=1e-3)

        # Sample negatives: (B, train_n_neg, Da)
        gripper_stats, pose_stats = self.get_action_stats()
        pose_dist = torch.distributions.Uniform(
            low=pose_stats["min"], high=pose_stats["max"]
        )

        if self.optimize_negatives:
            pass
        else:
            a = torch.tensor([-1, 1])
            p = torch.ones(2) / 2
            idxs = p.multinomial(num_samples=B*self.num_neg_act_samples, replacement=True)
            pose_negatives = pose_dist.sample((B, self.num_neg_act_samples, Ta)).to(
                dtype=npose_act.dtype
            )

        # Combine pos and neg samples: (B, train_n_neg+1, Ta, Da)
        pose_targets = torch.cat([npose_act.unsqueeze(1), pose_negatives], dim=1)
        N = pose_targets.size(1)

        # Randomly permute the positive and negative samples
        permutation = torch.rand(B, N).argsort(dim=1)
        pose_targets = pose_targets[torch.arange(B).unsqueeze(-1), permutation]
        ground_truth = (permutation == 0).nonzero()[:, 1].to(npose_act.device)
        one_hot = F.one_hot(ground_truth, num_classes=self.num_neg_act_samples+1).float()

        r = pose_targets[:,:,0,0]
        phi = self.normalizer["pose_act"].unnormalize(pose_targets)[:,:,0,1]
        z = pose_targets[:,:,0,2]

        rot_axis_angle = self.normalizer["pose_act"].unnormalize(pose_targets[:,:,0,3:])
        rot_zyz = matrix_to_euler_angles(axis_angle_to_matrix(rot_axis_angle.view(-1,3)), 'ZYZ').view(B,N,1,3)
        pose_targets = torch.concatenate([r.view(B,N,1,1), phi.view(B,N,1,1), z.view(B,N,1,1), rot_zyz], axis=3).view(B,N,1,6)

        # Compute energy function
        obs_feat = self.obs_encoder(nobs)
        pos_energy, rot_energy, gripper_pred = self.energy_head(obs_feat, pose_targets)

        # Compute InfoNCE loss, i.e. try to predict the expert action from the randomly sampled actions
        pos_probs = F.log_softmax(pos_energy, dim=1)
        rot_probs = F.log_softmax(rot_energy, dim=1)

        pos_ebm_loss = F.kl_div(pos_probs, one_hot, reduction='batchmean')
        rot_ebm_loss = F.kl_div(rot_probs, one_hot, reduction='batchmean')
        gripper_loss = F.binary_cross_entropy(gripper_pred, ngripper_act.view(B,1))
        loss = 1e-1 * pos_ebm_loss + 1e-1 * rot_ebm_loss + gripper_loss

        return loss, pos_ebm_loss, rot_ebm_loss, gripper_loss

    def augment_action(self, gripper_act, pose_act, noise=1e-4):
        start = self.num_obs_steps - 1
        end = start + self.num_action_steps
        gripper_act = gripper_act[:, start:end]
        pose_act = pose_act[:, start:end]

        # Add noise
        pose_act += torch.normal(
            mean=0,
            std=noise,
            size=pose_act.shape,
            dtype=pose_act.dtype,
            device=pose_act.device,
        )

        return gripper_act, pose_act


    def get_action_stats(self):
        gripper_stats = self.normalizer["gripper_act"].get_output_stats()
        pose_stats = self.normalizer["pose_act"].get_output_stats()

        return gripper_stats, pose_stats

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
