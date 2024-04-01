import os
import pathlib
from tqdm import tqdm
import hydra
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
from typing import Optional
from omegaconf import OmegaConf
import wandb
import copy

from imitation_learning.dataset.base_dataset import BaseDataset
from imitation_learning.workflow.base_workflow import BaseWorkflow
from imitation_learning.env_runner.base_runner import BaseRunner
from imitation_learning.policy.diffusion_unet_policy import DiffusionUnetPolicy
from imitation_learning.utils import torch_utils
from imitation_learning.utils.json_logger import JsonLogger
from imitation_learning.utils.checkpoint_manager import TopKCheckpointManager

from diffusers.training_utils import EMAModel

OmegaConf.register_new_resolver("eval", eval, replace=True)


class DiffusionWorkflow(BaseWorkflow):
    include_keys = ["global_step", "epoch"]

    def __init__(self, config: OmegaConf, output_dir: Optional[str] = None):
        super().__init__(config, output_dir=output_dir)

        # Set random seed
        seed = config.training.seed
        torch.manual_seed(seed)
        npr.seed(seed)
        random.seed(seed)

        self.model: DiffusionUnetPolicy
        self.model = hydra.utils.instantiate(config.policy)

        self.ema_model: DiffusionUnetPolicy
        if config.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        self.optimizer = hydra.utils.instantiate(
            config.optimizer, params=self.model.parameters()
        )

        self.global_step = 0
        self.epoch = 0

    def run(self):
        # Resume training
        if self.config.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # Datasets
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(self.config.task.dataset)
        train_dataloader = DataLoader(dataset, **self.config.dataloader)
        normalizer = dataset.get_normalizer()

        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **self.config.val_dataloader)

        self.model.set_normalizer(normalizer)
        if self.config.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # LR scheduler
        lr_scheduler = torch_utils.CosineWarmupScheduler(
            self.optimizer,
            self.config.training.lr_warmup_steps,
            len(train_dataloader) * self.config.training.num_epochs,
        )

        device = torch.device(self.config.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)

        # Configure EMA
        ema: EMAModel = None
        if self.config.training.use_ema:
            ema = hydra.utils.instantiate(
                self.config.ema,
                model=self.ema_model)

        # Env runner
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(self.config.task.env_runner, output_dir=self.output_dir)

        # Setup logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(self.config, resolve=True),
            **self.config.logging,
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )
        log_path = os.path.join(self.output_dir, "logs.json.txt")

        # Checkpointer
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **self.config.checkpoint.topk
        )

        # Save batch for diffusion sampling
        train_sampling_batch = None

        # Training
        with JsonLogger(log_path) as json_logger:
            for epoch in range(self.config.training.num_epochs):
                step_log = dict()
                train_losses = list()
                self.model.train()
                with tqdm(
                    train_dataloader,
                    desc=f"Training Epoch {self.epoch}",
                    leave=False,
                    mininterval=self.config.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = torch_utils.dict_apply(
                            batch, lambda x: x.to(device, non_blocking=True)
                        )
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # Compute loss
                        loss = self.model.compute_loss(batch)
                        loss.backward()

                        # Optimization
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()

                        # Update EMA
                        if self.config.training.use_ema:
                            ema.step(self.model)

                        # Logging
                        loss_cpu = loss.item()
                        tepoch.set_postfix(loss=loss_cpu, refresh=False)
                        train_losses.append(loss_cpu)
                        step_log = {
                            "train_loss": loss.item(),
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0],
                        }

                        is_last_batch = batch_idx == len(train_dataloader) - 1
                        if not is_last_batch:
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                    tepoch.set_postfix(loss=np.mean(train_losses))

                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                # Validation
                self.model.eval()

                if self.epoch % self.config.training.rollout_every == 0:
                    runner_log = env_runner.run(self.model)
                    step_log.update(runner_log)

                if self.epoch % self.config.training.val_every == 0:
                    val_losses = list()
                    with torch.no_grad():
                        with tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=self.config.training.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = torch_utils.dict_apply(
                                    batch, lambda x: x.to(device, non_blocking=True)
                                )

                                # Compute loss
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)

                                if len(val_losses) > 0:
                                    step_log["val_loss"] = loss

                # Diffusion sampling on a training batch
                if (self.epoch % self.config.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = train_sampling_batch
                        gt_action = batch['action']

                        result = self.model.get_action(batch['obs'], device)
                        if self.config.pred_action_steps_only:
                            pred_action = result['action']
                            start = self.config.n_obs_steps - 1
                            end = start + self.config.n_action_steps
                            gt_action = gt_action[:,start:end]
                        else:
                            pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()

                        # Free memory allocated to batch
                        del batch
                        del gt_action
                        del result
                        del pred_action
                        del mse

                # Checkpoint
                if (self.epoch % self.config.training.checkpoint_every) == 0:
                    if self.config.checkpoint.save_last_checkpoint:
                        self.save_checkpoint()

                    metric_dict = dict()
                    for k, v in step_log.items():
                        metric_dict[k.replace('/', '_')] = v
                    topk_checkpoint_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_checkpoint_path is not None:
                        self.save_checkpoint(path=topk_checkpoint_path)

                # Bookkeeping
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(config):
    workflow = DiffusionWorkflow(config)
    workflow.run()


if __name__ == "__main__":
    main()
