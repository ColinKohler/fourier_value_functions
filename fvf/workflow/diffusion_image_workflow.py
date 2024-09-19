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

from fvf.dataset.pusht_image_dataset import PushTImageDataset
from fvf.workflow.base_workflow import BaseWorkflow
from fvf.env_runner.pusht_image_runner import PushTImageRunner
from fvf.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from fvf.utils import torch_utils
from fvf.utils.json_logger import JsonLogger
from fvf.utils.checkpoint_manager import TopKCheckpointManager

from diffusers.training_utils import EMAModel
from fvf.utils.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


class DiffusionImageWorkflow(BaseWorkflow):
    include_keys = ["global_step", "epoch"]

    def __init__(
        self, config: OmegaConf, output_dir=None
    ):
        super().__init__(config, output_dir=output_dir)

        # Set random seed
        seed = config.training.seed
        torch.manual_seed(seed)
        npr.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(config.policy)

        self.ema_model: DiffusionUnetImagePolicy = None
        if config.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            config.optimizer, params=self.model.parameters()
        )

        # configure training state
        self.global_step = 0
        self.epoch = 0
    
    def run(self):
        config = copy.deepcopy(self.config)
        # Resume training
        if self.config.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: PushTImageDataset
        dataset = hydra.utils.instantiate(self.config.task.dataset)
        assert isinstance(dataset, PushTImageDataset)
        train_dataloader = DataLoader(dataset, **self.config.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **self.config.val_dataloader)

        self.model.set_normalizer(normalizer)
        if self.config.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # LR scheduler
        lr_scheduler = get_scheduler(
            self.config.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * self.config.training.num_epochs) \
                    // self.config.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # device transfer
        device = torch.device(self.config.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)

        # Configure EMA
        ema: EMAModel = None
        if self.config.training.use_ema:
            ema = hydra.utils.instantiate(self.config.ema, model=self.ema_model)

        # configure env
        env_runner: PushTImageRunner
        env_runner = hydra.utils.instantiate(
            self.config.task.env_runner, output_dir=self.output_dir
        )
        assert isinstance(env_runner, PushTImageRunner)

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
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **self.config.checkpoint.topk,
        )
        torch_utils.optimizer_to(self.optimizer, device)

        # Save batch for diffusion sampling
        train_sampling_batch = None

        if config.training.debug:
            config.training.num_epochs = 2
            config.training.max_train_steps = 3
            config.training.max_val_steps = 3
            config.training.rollout_every = 1
            config.training.checkpoint_every = 1
            config.training.val_every = 1
            config.training.sample_every = 1
        # Training
        with JsonLogger(log_path) as json_logger:
            for epoch in range(self.config.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                if config.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm(
                    train_dataloader,
                    desc=f"Training Epoch {self.epoch}",
                    leave=False,
                    mininterval=self.config.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = torch_utils.dict_apply(
                            batch, lambda x: x.to(device, non_blocking=True)
                        )
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # Compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / config.training.gradient_accumulate_every
                        loss.backward()

                        # Optimization
                        if self.global_step % self.config.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        # Update EMA
                        if self.config.training.use_ema:
                            ema.step(self.model)

                        # Logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (config.training.max_train_steps is not None) \
                            and batch_idx >= (config.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if config.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if self.epoch % self.config.training.rollout_every == 0:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
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
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (self.config.training.max_val_steps is not None) \
                                    and batch_idx >= (config.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log["val_loss"] = val_loss

                # Diffusion sampling on a training batch
                if (self.epoch % self.config.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = torch_utils.dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch["obs"]
                        gt_action = batch["action"]
                        
                        result = policy.get_action(obs_dict, device)
                        pred_action = result["action_pred"]
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log["train_action_mse_error"] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                # Checkpoint
                if (self.epoch % self.config.training.checkpoint_every) == 0:
                    if self.config.checkpoint.save_last_checkpoint:
                        self.save_checkpoint()
                    if self.config.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for k, v in step_log.items():
                        metric_dict[k.replace("/", "_")] = v
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_checkpoint_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_checkpoint_path is not None:
                        self.save_checkpoint(path=topk_checkpoint_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
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
    workflow = DiffusionImageWorkflow(config)
    workflow.run()


if __name__ == "__main__":
    main()
