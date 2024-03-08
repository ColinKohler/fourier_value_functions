"""
Usage:
python eval.py --checkpoint=data/checkpoint --output=data/output
"""

import os
import sys

sys.path.insert(0, os.path.abspath("."))

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import os
import pathlib
import hydra
import torch
import dill
import wandb
import json
import argparse
from irrep_actions.workflow.base_workflow import BaseWorkflow

def evaluate(checkpoint: str, output_dir: str, device: str, plot_energy_fn: bool=False):
    if os.path.exists(output_dir):
        inp = input("Output directory already exists. Overwrite existing dataset? (y/n)")
        if inp == "y":
            print("Deleting existing dataset")
        else:
            print("Exiting evaluation...")
            sys.exit()
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['config']
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
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir
    )
    runner_log = env_runner.run(policy, plot_energy_fn=plot_energy_fn)

    # Save logs
    json_log = dict()
    for k, v in runner_log.items():
        json_log[k] = v._path if isinstance(v, wandb.sdk.data_types.video.Video) else v
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Model checkpoint to evaluate.'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Directory to save evaluation output to.'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use: (cpu, cuda, cuda:0)'
    )
    parser.add_argument(
        '--plot_energy_fn',
        default=False,
        action='store_true',
        help='Plot the energy function alongside the env images. Only works on circular models.'
    )
    args = parser.parse_args()
    evaluate(args.checkpoint, args.output_dir, args.device, args.plot_energy_fn)
