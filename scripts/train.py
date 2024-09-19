"""
Usage:
Training:
python train.py --config-name=train_explicit_policy
"""

import os
import sys

import hydra
from omegaconf import OmegaConf
from fvf.workflow.base_workflow import BaseWorkflow

sys.path.insert(0, os.path.abspath("."))

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path="/home/harbor07/fourier_value_functions/config/",
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workflow: BaseWorkflow = cls(cfg)
    workflow.run()


if __name__ == "__main__":
    main()
