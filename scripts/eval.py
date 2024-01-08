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

def evaluate(checkpoint: str, output_dir: str, device: str):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument()
    args = parser.parse_args()
    evaluate(args.checkpoint, args.output_dir, args.device)
