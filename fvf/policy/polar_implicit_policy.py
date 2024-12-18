""" polar_implicit_policy.py """

import io
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

from fvf.utils import torch_utils
from fvf.policy.base_policy import BasePolicy

from eharmony import grid, plotting


class PolarImplicitPolicy(BasePolicy):
    """
    Polar implicit policy.
    """

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
        self.i = 0

    def get_action(self, obs, device):
        """Get the action for the observation."""
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
        redges, r = grid.grid1D(
            action_stats["max"][0].item(),
            self.energy_head.ph.num_radii,
            origin=action_stats["min"][0].item(),
        )
        r = r.view(1, -1).repeat(B, 1).to(device)
        pedges, phi = grid.grid1D(2.0 * torch.pi, self.energy_head.ph.num_phi)
        phi = phi.view(1, -1).repeat(B, 1).to(device)
        obs_feat = self.obs_encoder(nobs)
        logits, coeffs = self.energy_head(obs_feat, return_coeffs=True)
        logits = logits.view(B, -1)
        # if self.i <= 5:
        #    self.temperature = 10
        # else:
        #    self.temperature = 1
        # self.i += 1

        action_probs = torch.softmax(logits / self.temperature, dim=-1).view(
            B, self.energy_head.ph.num_radii, self.energy_head.ph.num_phi
        )

        if self.sample_actions:
            flat_indexes = torch.multinomial(
                action_probs.flatten(start_dim=-2), num_samples=1, replacement=True
            ).squeeze()
        else:
            flat_indexes = torch.argmax(action_probs.flatten(start_dim=-2), dim=-1)

        r_idx = flat_indexes.div(action_probs.shape[-1], rounding_mode="floor")
        phi_idx = torch.remainder(flat_indexes, action_probs.shape[-1])
        actions = (
            torch.vstack([r[torch.arange(B), r_idx], phi[torch.arange(B), phi_idx]])
            .permute(1, 0)
            .view(B, 1, 2)
        )

        r = self.normalizer["action"].unnormalize(actions)[:, :, 0]
        phi = actions[:, :, 1]
        x = r * torch.cos(phi)
        y = r * torch.sin(phi)
        actions = torch.concat([x.view(B, 1), y.view(B, 1)], dim=1).unsqueeze(1)

        return {
            "action": actions,
            "energy": logits.view(
                B, self.energy_head.ph.num_radii, self.energy_head.ph.num_phi
            ),
            "fourier_coeffs": coeffs.cpu(),
        }

    def compute_loss(self, batch):
        # Load batch
        nobs = self.normalizer.normalize(batch["obs"])
        naction = self.normalizer["action"].normalize(batch["action"]).float()

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
            r = (
                torch.linspace(
                    action_stats["min"][0].item(),
                    action_stats["max"][0].item(),
                    self.energy_head.num_radii,
                )
                .view(1, -1)
                .repeat(B * self.num_neg_act_samples, 1)
                .to(naction.device)
            )
            phi = (
                torch.linspace(0, 2 * np.pi, self.energy_head.num_phi)
                .view(1, -1)
                .repeat(B * self.num_neg_act_samples, 1)
                .to(naction.device)
            )
            with torch.no_grad():
                obs_feat = self.obs_encoder(nobs)
                logits = self.energy_head(obs_feat).view(
                    B, self.energy_head.num_radii, self.energy_head.num_phi
                )
            action_probs = torch.softmax(logits / 2.0, dim=-1).view(
                B, self.energy_head.num_radii, self.energy_head.num_phi
            )
            flat_indexes = torch.multinomial(
                action_probs.flatten(start_dim=-2),
                num_samples=self.num_neg_act_samples,
                replacement=False,
            ).squeeze()

            r_idx = flat_indexes.div(action_probs.shape[-1], rounding_mode="floor")
            phi_idx = torch.remainder(flat_indexes, action_probs.shape[-1])
            negatives = (
                torch.vstack(
                    [
                        r[torch.arange(B * self.num_neg_act_samples), r_idx.view(-1)],
                        phi[
                            torch.arange(B * self.num_neg_act_samples), phi_idx.view(-1)
                        ],
                    ]
                )
                .permute(1, 0)
                .view(B, self.num_neg_act_samples, 1, 2)
            )
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
        one_hot = F.one_hot(
            ground_truth, num_classes=self.num_neg_act_samples + 1
        ).float()

        # Compute ciruclar energy function for the given obs and action magnitudes
        r = targets[:, :, 0, 0]
        phi = self.normalizer["action"].unnormalize(targets)[:, :, 0, 1]
        polar_act = torch.concatenate([r.view(B, N, 1), phi.view(B, N, 1)], axis=2)

        obs_feat = self.obs_encoder(nobs)
        energy = self.energy_head(obs_feat, polar_act)

        # Compute InfoNCE loss, i.e. try to predict the expert action from the randomly sampled actions
        probs = F.log_softmax(energy, dim=1)
        ebm_loss = F.kl_div(probs, one_hot, reduction="batchmean")
        loss = ebm_loss

        return loss, ebm_loss, torch.tensor(0.0), torch.tensor(0.0)

    def get_action_stats(self):
        action_stats = self.normalizer["action"].get_output_stats()

        repeated_stats = dict()
        for key, value in action_stats.items():
            n_repeats = self.action_dim // value.shape[0]
            repeated_stats[key] = value.repeat(n_repeats)
        return repeated_stats

    def plot_energy_fn(self, img, energy):
        probs = torch.softmax(energy.view(1, -1) / self.temperature, dim=-1).view(
            self.energy_head.ph.num_radii, self.energy_head.ph.num_phi
        )

        action_stats = self.get_action_stats()
        r = torch.linspace(
            action_stats["min"][0].item(),
            action_stats["max"][0].item(),
            self.energy_head.ph.num_radii,
        )
        r0_rmin = torch.linspace(0, action_stats["min"][0], 10)
        r = torch.concat([r0_rmin, r])
        energy = torch.concat(
            [torch.zeros(10, self.energy_head.ph.num_phi), energy.cpu()]
        )
        phi = torch.linspace(0, 2 * np.pi, self.energy_head.ph.num_phi)

        probs = torch.concat(
            [torch.zeros(10, self.energy_head.ph.num_phi), probs.cpu()]
        )

        fig = plt.figure()
        subfigs = fig.subfigures(1, 3)
        ax1 = subfigs[2].add_subplot()

        if img is not None:
            ax1.imshow(img[-1].transpose(1, 2, 0))
            ax1.set_title("Rollouts", va="bottom")
            ax1.set_axis_off()

        plotting.plot_polar_fn(energy, r=r, phi=phi, fig=subfigs[0], title="Energy")
        plotting.plot_polar_fn(probs, r=r, phi=phi, fig=subfigs[1], title="Softmax")

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

    def plot_weighted_basis_fns(self, w):
        w_ph = torch.einsum("bwrp,bw->bwrp", self.energy_head.ph.Psi, w).view(
            self.energy_head.ph.K,
            2 * self.energy_head.ph.L + 1,
            self.energy_head.ph.num_radii,
            self.energy_head.ph.num_phi,
        )
        w = w.view(
            self.energy_head.ph.K,
            2 * self.energy_head.ph.L + 1,
        )

        fig = plt.figure(figsize=(4 * self.energy_head.ph.L, 2 * self.energy_head.ph.K))
        subfigs = fig.subfigures(self.energy_head.ph.K, self.energy_head.ph.L * 2 + 1)
        for k in range(self.energy_head.ph.K):
            for l in range(self.energy_head.ph.L * 2 + 1):
                plotting.plot_polar_fn(
                    w_ph[k, l].numpy(),
                    title=f"w: {w[k,l]:.1f}",
                    fig=subfigs[k, l],
                    vmin=w_ph.min(),
                    vmax=w_ph.max(),
                )

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
