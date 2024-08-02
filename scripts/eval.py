"""
Usage:
python eval.py --checkpoint=data/checkpoint --output=data/output
"""

import os
import sys
import json
import argparse
import pathlib
import hydra
import torch
import dill
import wandb

from fvf.workflow.base_workflow import BaseWorkflow

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)


def evaluate(
    checkpoint: str,
    output_dir: str,
    device: str,
    num_train: int = None,
    num_test: int = None,
    plot_energy_fn: bool = False,
    plot_basis_fn: bool = False,
):
    """Evluate a checkpoint."""
    if os.path.exists(output_dir):
        inp = input(
            "Output directory already exists. Overwrite existing dataset? (y/n)"
        )
        if inp == "y":
            print("Deleting existing dataset")
        else:
            print("Exiting evaluation...")
            sys.exit()
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["config"]
    if num_train is not None:
        cfg["task"]["env_runner"]["num_train"] = num_train
    if num_test is not None:
        cfg["task"]["env_runner"]["num_test"] = num_test
        cfg["task"]["env_runner"]["num_envs"] = None
    cls = hydra.utils.get_class(cfg._target_)
    workflow = cls(cfg, output_dir=output_dir, eval=True)
    workflow: BaseWorkflow
    workflow.load_payload(payload, exclude_keys=None, include_keys=None)

    # Load policy
    policy = workflow.model
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # Run evaluation
    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=output_dir)
    runner_log = env_runner.run(
        policy, plot_energy_fn=plot_energy_fn, plot_weights_basis_fns=plot_basis_fn
    )

    # Save logs
    json_log = {}
    for k, v in runner_log.items():
        json_log[k] = v._path if isinstance(v, wandb.sdk.data_types.video.Video) else v
    out_path = os.path.join(output_dir, "eval_log.json")
    json.dump(json_log, open(out_path, "w"), indent=2, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Model checkpoint to evaluate.")
    parser.add_argument(
        "output_dir", type=str, help="Directory to save evaluation output to."
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=None,
        help="Number of environments with training seeds to evaluate on.",
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=None,
        help="Number of environments with testing seeds to evaluate on.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use: (cpu, cuda, cuda:0)"
    )
    parser.add_argument(
        "--plot_energy_fn",
        default=False,
        action="store_true",
        help="Plot the energy function alongside the env images.",
    )
    parser.add_argument(
        "--plot_basis_fn",
        default=False,
        action="store_true",
        help="Plot the weighted basis functions that make up the energy function.",
    )
    args = parser.parse_args()
    evaluate(
        args.checkpoint,
        args.output_dir,
        args.device,
        args.num_train,
        args.num_test,
        args.plot_energy_fn,
        args.plot_basis_fn,
    )
