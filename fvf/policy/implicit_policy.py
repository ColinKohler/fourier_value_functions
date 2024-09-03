import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from fvf.model.common.normalizer import LinearNormalizer
from fvf.utils import torch_utils
from fvf.policy.base_policy import BasePolicy
from fvf.utils import mcmc


class ImplicitPolicy(BasePolicy):
    def __init__(
        self,
        obs_encoder: nn.Module,
        energy_head: nn.Module,
        obs_dim: int,
        action_dim: int,
        num_obs_steps: int,
        num_action_steps: int,
        action_sampling: str,
        num_neg_act_samples: int,
        pred_n_iter: int,
        pred_n_samples: int,
        optimize_negatives: bool = False,
        sample_actions: bool = False,
        temperature: float = 1.0,
        grad_pen: bool = False,
    ):
        super().__init__(obs_dim, action_dim, num_obs_steps, num_action_steps)
        self.action_sampling = action_sampling
        self.num_neg_act_samples = num_neg_act_samples
        self.pred_n_iter = pred_n_iter
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
        actions = action_dist.sample((B, self.pred_n_samples, Ta))

        obs_feat = self.obs_encoder(nobs)
        # self.action_sampling = 'dense'
        # self.pred_n_samples = 600**2
        if self.action_sampling == "dfo":
            action_probs, actions = mcmc.iterative_dfo(
                self.energy_head,
                obs_feat,
                actions,
                [action_stats["min"], action_stats["max"]],
                normalizer=self.normalizer,
            )
        elif self.action_sampling == "langevin":
            action_probs, actions = mcmc.langevin_actions(
                self.energy_head,
                obs_feat,
                actions,
                [action_stats["min"], action_stats["max"]],
                num_iterations=100,
                normalizer=self.normalizer,
            )
        elif self.action_sampling == "dense":
            x = np.linspace(-1, 1, int(np.sqrt(self.pred_n_samples)))
            y = np.linspace(-1, 1, int(np.sqrt(self.pred_n_samples)))
            xv, yv = np.meshgrid(x, y)
            actions = torch.tensor(np.dstack([xv, yv]).reshape(-1, 2)).to(device)
            actions = actions.view(1, 600**2, 1, 2).repeat(B, 1, 1, 1).float()
            logits = self.energy_head(obs_feat, actions)
            action_probs = torch.softmax(logits, dim=-1)
        else:
            raise ValueError("Invalid action sampling suggested.")

        if self.sample_actions:
            idxs = torch.multinomial(action_probs, num_samples=1, replacement=True)
        else:
            idxs = torch.argmax(action_probs, dim=-1)
        actions = actions[torch.arange(B).unsqueeze(-1), idxs].squeeze(1)
        actions = self.normalizer["action"].unnormalize(actions)

        return {"action": actions}

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

        if self.optimize_negatives and self.action_sampling == "langevin":
            negatives = action_dist.sample((B, self.num_neg_act_samples, Ta)).to(
                dtype=naction.dtype
            )

            self.eval()
            _, negatives = mcmc.langevin_actions(
                self,
                nobs,
                negatives,
                [action_stats["min"], action_stats["max"]],
                num_iterations=100,
                normalizer=self.normalizer,
            )
            self.train()
        elif self.optimize_negatives and self.action_sampling == "dfo":
            negatives = action_dist.sample((B, self.num_neg_act_samples, Ta)).to(
                dtype=naction.dtype
            )
            _, negatives = mcmc.iterative_dfo(
                self,
                nobs,
                negatives,
                [action_stats["min"], action_stats["max"]],
                normalizer=self.normalizer,
            )
        else:
            negatives = action_dist.sample((B, self.num_neg_act_samples, Ta)).to(
                dtype=naction.dtype
            )

        # Combine pos and neg samples: (B, train_n_neg+1, Ta, Da)
        targets = torch.cat([noisy_actions.unsqueeze(1), negatives], dim=1)

        # Randomly permute the positive and negative samples
        permutation = torch.rand(targets.size(0), targets.size(1)).argsort(dim=1)
        targets = targets[torch.arange(targets.size(0)).unsqueeze(-1), permutation]
        ground_truth = (permutation == 0).nonzero()[:, 1].to(naction.device)
        one_hot = F.one_hot(
            ground_truth, num_classes=self.num_neg_act_samples + 1
        ).float()

        obs_feat = self.obs_encoder(nobs)
        energy = self.energy_head(obs_feat, targets)

        probs = F.log_softmax(energy, dim=1)
        ebm_loss = F.kl_div(probs, one_hot, reduction="batchmean")

        # energy_data = energy[:,0]
        # energy_samp = energy[:,1:]
        # cd_per_example_loss = torch.mean(energy_sampl, axis=1) - torch.mean(energy_data, axis=1)

        # dist = torch.sum()
        # entropy_temp = 1e-1
        # entropy = -torch.exp(-entropy_temp * dist)
        # kl_per_example_loss = torch.mean(-entropy_samp_copy[..., None] - entropy)

        # per_example_loss = cd_per_example_loss + kl_per_example_loss
        # ebm_loss = F.cross_entropy(energy, ground_truth)

        if self.grad_pen:
            de_dact, _ = mcmc.gradient_wrt_action(
                self, nobs, targets.detach(), normalizer=self.normalizer
            )
            grad_norm = mcmc.compute_grad_norm(de_dact).view(B, -1)
            grad_norm = grad_norm - 1.0
            grad_norm = torch.clamp(grad_norm, 0.0, 1e3)
            grad_norm = grad_norm**2
            grad_loss = torch.mean(grad_norm)
            loss = ebm_loss + grad_loss
        else:
            grad_loss = torch.Tensor([0])
            loss = ebm_loss

        return loss, ebm_loss, torch.tensor(0.0), torch.tensor(0.0)

    def get_action_stats(self):
        action_stats = self.normalizer["action"].get_output_stats()

        repeated_stats = dict()
        for key, value in action_stats.items():
            n_repeats = self.action_dim // value.shape[0]
            repeated_stats[key] = value.repeat(n_repeats)
        return repeated_stats
