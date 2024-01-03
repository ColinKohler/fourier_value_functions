import copy
from typing import Dict, Callable, List
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


def plotAttnMaps(input_data, attn_maps, idx=0):
    if input_data is not None:
        input_data = input_data[idx].detach().cpu().numpy()
    else:
        input_data = np.arange(attn_maps[0][idx].shape[-1])
    attn_maps = [m[idx].detach().cpu().numpy() for m in attn_maps]

    num_heads = attn_maps[0].shape[0]
    num_layers = len(attn_maps)
    seq_len = input_data.shape[0]
    fig_size = 4 if num_heads == 1 else 3
    fig, ax = plt.subplots(
        num_layers, num_heads, figsize=(num_heads * fig_size, num_layers * fig_size)
    )
    if num_layers == 1:
        ax = [ax]
    if num_heads == 1:
        ax = [[a] for a in ax]
    for row in range(num_layers):
        for column in range(num_heads):
            ax[row][column].imshow(attn_maps[row][column], origin="lower", vmin=0)
            ax[row][column].set_xticks(list(range(seq_len)))
            ax[row][column].set_xticklabels(
                np.round(input_data.tolist(), 3), rotation=45
            )
            ax[row][column].set_yticks(list(range(seq_len)))
            ax[row][column].set_yticklabels(np.round(input_data.tolist(), 3))
            ax[row][column].set_title("Layer %i, Head %i" % (row + 1, column + 1))
    fig.subplots_adjust(hspace=0.5)
    plt.show()


def normalize(x, x_min, x_max):
    r = x_max - x_min
    x01 = (x - x_min) / r
    return 2 * x01 - 1


def unnormalize(x, x_min, x_max):
    r = x_max - x_min
    return 0.5 * (x + 1) * r + x_min


def dict_apply(
    x: Dict[str, torch.Tensor], func: Callable[[torch.Tensor], torch.Tensor]
) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


def copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu")
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)
