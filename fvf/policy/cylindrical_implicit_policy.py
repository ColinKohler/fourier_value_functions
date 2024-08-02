import io
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

from fvf.model.common.normalizer import LinearNormalizer
from fvf.utils import torch_utils
from fvf.policy.base_policy import BasePolicy
from fvf.utils import mcmc
from fvf.model.modules.harmonics import grid

from torch_harmonics import grid, plotting


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
        optimize_negatives: bool = False,
        sample_actions: bool = False,
        temperature: float = 1.0,
        grad_pen: bool = False,
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
        nobs["keypoints"] = nobs["keypoints"].float()
        B = list(obs.values())[0].shape[0]

        Do = self.obs_dim
        Da = self.action_dim
        To = self.num_obs_steps
        Ta = self.num_action_steps

        # Optimize actions
        implicit_stats, energy_stats = self.get_action_stats()
        redges, r = grid.grid1D(
            energy_stats["max"][0].item(),
            self.energy_head.num_radii,
            origin=energy_stats["min"][0].item(),
        )
        r = r.view(1, -1).repeat(B, 1).to(device)
        pedges, phi = grid.grid1D(2.0 * torch.pi, self.energy_head.num_phi)
        phi = phi.view(1, -1).repeat(B, 1).to(device)
        zedges, z = grid.grid1D(
            energy_stats["max"][2].item(),
            self.energy_head.num_height,
            origin=energy_stats["min"][2].item(),
        )
        z = z.view(1, -1).repeat(B, 1).to(device)

        obs_feat = self.obs_encoder(nobs)
        logits, gripper, coeffs = self.energy_head(obs_feat, return_coeffs=True)
        logits = logits.view(B, -1)
        action_probs = torch.softmax(logits / self.temperature, dim=-1).view(
            B,
            self.energy_head.num_radii,
            self.energy_head.num_phi,
            self.energy_head.num_height,
        )

        if self.sample_actions:
            flat_indexes = torch.multinomial(
                action_probs.flatten(start_dim=1), num_samples=1, replacement=True
            ).squeeze()
        else:
            flat_indexes = torch.argmax(action_probs.flatten(start_dim=1), dim=-1)

        # TODO: This can be a torch unravel if Torch is updated
        idxs = np.unravel_index(flat_indexes.cpu(), action_probs.shape[1:])
        nactions = (
            torch.vstack(
                [
                    r[torch.arange(B), idxs[0]],
                    phi[torch.arange(B), idxs[1]],
                    z[torch.arange(B), idxs[2]],
                    torch.zeros(B).to(r.device),
                    torch.zeros(B).to(r.device),
                    torch.zeros(B).to(r.device),
                ]
            )
            .permute(1, 0)
            .view(B, 1, 6)
        )

        r = self.normalizer["pose_act"].unnormalize(nactions)[:, :, 0]
        phi = nactions[:, :, 1]
        x = r * torch.cos(phi)
        y = r * torch.sin(phi)
        z = self.normalizer["pose_act"].unnormalize(nactions)[:, :, 2]
        gripper_act = self.normalizer["gripper_act"].unnormalize(
            torch.bernoulli(gripper)
        )
        actions = torch.concat(
            [
                x.view(B, 1),
                y.view(B, 1),
                z.view(B, 1),
                torch.zeros(B, 3).to(x.device),
                gripper_act.view(B, 1),
            ],
            dim=1,
        ).unsqueeze(1)

        return {
            "action": actions,
            "gripper": gripper,
            "energy": logits.view(
                B,
                self.energy_head.ch.num_radii,
                self.energy_head.ch.num_phi,
                self.energy_head.ch.num_height,
            ),
            "fourier_coeffs": coeffs.cpu(),
        }

    def compute_loss(self, batch):
        # Load batch
        nobs = self.normalizer.normalize(batch["obs"])
        nobs["keypoints"] = nobs["keypoints"].float()
        npose_act = self.normalizer["pose_act"].normalize(batch["pose_act"]).float()
        ngripper_act = (
            self.normalizer["gripper_act"].normalize(batch["gripper_act"]).float()
        )

        Do = self.obs_dim
        Da = self.action_dim
        To = self.num_obs_steps
        Ta = self.num_action_steps
        B = ngripper_act.shape[0]

        ngripper_act, npose_act = self.augment_action(
            ngripper_act, npose_act, noise=1e-3
        )

        # Sample negatives: (B, train_n_neg, Da)
        gripper_stats, pose_stats = self.get_action_stats()
        pose_dist = torch.distributions.Uniform(
            low=pose_stats["min"], high=pose_stats["max"]
        )

        if self.optimize_negatives:
            pass
        else:
            pos_negatives = pose_dist.sample((B, self.num_neg_act_samples, Ta)).to(
                dtype=npose_act.dtype
            )

        # Combine pos and neg samples: (B, train_n_neg+1, Ta, Da)
        pos_targets = torch.cat([npose_act.unsqueeze(1), pos_negatives], dim=1)
        N = pos_targets.size(1)

        # Randomly permute the positive and negative samples
        permutation = torch.rand(B, N).argsort(dim=1)
        pos_targets = pos_targets[torch.arange(B).unsqueeze(-1), permutation]
        ground_truth = (permutation == 0).nonzero()[:, 1].to(npose_act.device)
        one_hot = F.one_hot(
            ground_truth, num_classes=self.num_neg_act_samples + 1
        ).float()

        r = pos_targets[:, :, 0, 0]
        phi = self.normalizer["pose_act"].unnormalize(pos_targets)[:, :, 0, 1]
        z = pos_targets[:, :, 0, 2]
        pos_targets = torch.concatenate(
            [r.view(B, N, 1), phi.view(B, N, 1), z.view(B, N, 1)], axis=2
        ).view(B, N, 1, -1)

        # Compute ciruclar energy function for the given obs and action magnitudes
        obs_feat = self.obs_encoder(nobs)
        energy, gripper_pred = self.energy_head(obs_feat, pos_targets)

        # Compute InfoNCE loss, i.e. try to predict the expert action from the randomly sampled actions
        probs = F.log_softmax(energy, dim=1)
        ebm_loss = F.kl_div(probs, one_hot, reduction="batchmean")
        gripper_loss = F.binary_cross_entropy(gripper_pred, ngripper_act.view(B, 1))
        loss = ebm_loss + gripper_loss

        return loss, ebm_loss, torch.tensor(0.0), gripper_loss

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

    def plot_energy_fn(self, img, energy, gripper):
        probs = torch.softmax(energy.view(1, -1) / self.temperature, dim=-1).view(
            self.energy_head.ch.num_radii,
            self.energy_head.ch.num_phi,
            self.energy_head.ch.num_height,
        )

        _, action_stats = self.get_action_stats()
        r = torch.linspace(
            action_stats["min"][0].item(),
            action_stats["max"][0].item(),
            self.energy_head.num_radii,
        )
        phi = torch.linspace(0, 2 * np.pi, self.energy_head.num_phi)
        z = torch.linspace(
            action_stats["min"][2].item(),
            action_stats["max"][2].item(),
            self.energy_head.num_radii,
        )

        fig = plt.figure()
        subfigs = fig.subfigures(1, 3)
        ax1 = subfigs[2].add_subplot()

        if img is not None:
            ax1.imshow(img[-1].transpose(1, 2, 0))
            ax1.set_title("Rollouts", va="bottom")
            ax1.set_axis_off()

        plotting.plot_cylinder_prob(energy.cpu(), fig=subfigs[0], title="Energy")
        plotting.plot_cylinder_prob(probs.cpu(), fig=subfigs[1], title="Softmax")

        io_buf = io.BytesIO()
        fig.savefig(io_buf, format="raw")
        io_buf.seek(0)
        img_arr = np.reshape(
            np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
            newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
        )
        io_buf.close()
        plt.close()

        return img_arr
