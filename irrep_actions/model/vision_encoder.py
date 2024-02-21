import torch
import torch.nn as nn

from escnn import gspaces
from escnn import nn as enn
from escnn import group

from irrep_actions.model.layers import ResNetBlock, SO2ResNetBlock

class ResNetEncoder(nn.Module):
    def __init__(self, in_channels, z_dim):
        self.conv = nn.Sequential(
            ResNetBlock(),
            nn.Conv2d(),
            nn.ReLU(inplace=True),
        )

    def forward(x: nn.Tensor) -> nn.Tensor:
        pass

class SO2ResNetEncoder(nn.Module):
    def __init__(self, in_channels, z_dim):
        self.conv = enn.SequentialModule(
            SO2ResNetBlock(),
            enn.R2Conv(),
            enn.ReLU(, inplace=True)
        )

    def forward(x: enn.GeometricTensor) -> enn.GeometricTensor:
        pass
