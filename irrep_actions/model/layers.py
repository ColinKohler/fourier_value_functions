import torch
from torch.nn.utils import spectral_norm
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(
        self,
        hiddens: list[int],
        dropout: float=0.0,
        act_out: bool=True,
        spec_norm: bool=False
    ):
        super().__init__()

        layers = list()
        for i, (h, h_) in enumerate(zip(hiddens, hiddens[1:])):
            if spec_norm:
                layers.append(spectral_norm(nn.Linear(h, h_)))
            else:
                layers.append(nn.Linear(h, h_))
            is_last_layer = i == len(hiddens) - 2
            if not is_last_layer or act_out:
                layers.append(nn.LeakyReLU(0.01, inplace=True))
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int=3,
        stride: int=1,
    ):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=stride
            ),
            nn.ReLU(inplace=True)
        )

        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=stride
            ),
        )

        self.upscale = None
        if in_channels != out_channels or stride != 1:
            self.upscale = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.conv1(x)
        out = self.conv2(x)
        if self.upscale is not None:
            out += self.upscale(res)
        else:
            out += res
        out = self.act(out)

        return out
