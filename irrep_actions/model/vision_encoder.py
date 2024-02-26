import torch
import torch.nn as nn

from escnn import gspaces
from escnn import nn as enn
from escnn import group

from irrep_actions.model.layers import ResNetBlock
from irrep_actions.model.equiv_layers import SO2ResNetBlock

class ImageEncoder(nn.Module):
    def __init__(self, in_channels, z_dim, dropout):
        super().__init__()

        self.conv = nn.Sequential(
            # 96x96
            ResNetBlock(in_channels, z_dim // 8),
            # 48x48
            ResNetBlock(z_dim // 8, z_dim // 4),
            # 24x24
            ResNetBlock(z_dim // 4, z_dim // 2),
            # 12x12
            ResNetBlock(z_dim // 2, z_dim),
            # 6x6
            ResNetBlock(z_dim, z_dim),
            # 3x3
            nn.Conv2d(z_dim, z_dim, kernel_size=3, padding=0),
            nn.ReLU(inplace=True)
            # 1x1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class SO2ImageEncoder(nn.Module):
    def __init__(self, in_channels, z_dim, dropout):
        pass

    def forward(x: enn.GeometricTensor) -> enn.GeometricTensor:
        pass
