import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil

from fvf.utils.torch_utils import dict_apply, optimizer_to
from fvf.workflow.base_workflow import BaseWorkflow
from fvf.policy.diffusion_transformer_lowdim_policy import (
    DiffusionTransformerLowdimPolicy,
)
from fvf.dataset.base_dataset import BaseDataset
from fvf.env_runner.base_runner import BaseRunner
from fvf.utils.lr_scheduler import get_scheduler
from fvf.utils.checkpoint_manager import TopKCheckpointManager
from fvf.utils.json_logger import JsonLogger

from diffusers.training_utils import EMAModel

OmegaConf.register_new_resolver("eval", eval, replace=True)


# %%
class DiffusionTransformerLowdimWorkflow(BaseWorkflow):
    include_keys = ["global_step", "epoch"]

    def __init__(self, config: OmegaConf):
        super().__init__(config)

        # set seed
        seed = config.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionTransformerLowdimPolicy
        self.model = hydra.utils.instantiate(config.policy)

        self.ema_model: DiffusionTransformerLowdimPolicy = None
        if config.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = self.model.get_optimizer(**config.optimizer)

        self.global_step = 0
        self.epoch = 0

    def run(self):
        config = copy.deepcopy(self.config)

        # resume training
        if config.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(config.task.dataset)
        assert isinstance(dataset, BaseDataset)
        train_dataloader = DataLoader(dataset, **config.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **config.val_dataloader)

        self.model.set_normalizer(normalizer)
        if config.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            config.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=config.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * config.training.num_epochs)
            // config.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1,
        )

        # configure ema
        ema: EMAModel = None
        if config.training.use_ema:
            ema = hydra.utils.instantiate(config.ema, model=self.ema_model)

        # configure env runner
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            config.task.env_runner, output_dir=self.output_dir
        )
        assert isinstance(env_runner, BaseRunner)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(config, resolve=True),
            **config.logging,
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **config.checkpoint.topk,
        )

        # device transfer
        device = torch.device(config.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if config.training.debug:
            config.training.num_epochs = 2
            config.training.max_train_steps = 3
            config.training.max_val_steps = 3
            config.training.rollout_every = 1
            config.training.checkpoint_every = 1
            config.training.val_every = 1
            config.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(config.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=config.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(
                            batch, lambda x: x.to(device, non_blocking=True)
                        )
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / config.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if (
                            self.global_step % config.training.gradient_accumulate_every
                            == 0
                        ):
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # update ema
                        if config.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (
                            config.training.max_train_steps is not None
                        ) and batch_idx >= (config.training.max_train_steps - 1):
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
                if (self.epoch % config.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                if (self.epoch % config.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=config.training.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(
                                    batch, lambda x: x.to(device, non_blocking=True)
                                )
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (
                                    config.training.max_val_steps is not None
                                ) and batch_idx >= (config.training.max_val_steps - 1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log["val_loss"] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % config.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(
                            train_sampling_batch,
                            lambda x: x.to(device, non_blocking=True),
                        )
                        gt_action = batch["action"]

                        result = policy.get_action(batch["obs"], device)
                        if config.pred_action_steps_only:
                            pred_action = result["action"]
                            start = config.n_obs_steps - 1
                            end = start + config.n_action_steps
                            gt_action = gt_action[:, start:end]
                        else:
                            pred_action = result["action_pred"]
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log["train_action_mse_error"] = mse.item()
                        del batch
                        del gt_action
                        del result
                        del pred_action
                        del mse

                # checkpoint
                if (self.epoch % config.training.checkpoint_every) == 0:
                    # checkpointing
                    if config.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if config.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace("/", "_")
                        metric_dict[new_key] = value

                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
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
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(config):
    workflow = DiffusionTransformerLowdimWorkflow(config)
    workflow.run()


if __name__ == "__main__":
    main()
