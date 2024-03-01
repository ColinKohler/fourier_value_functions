import torch
import torch.nn as nn

from escnn import gspaces
from escnn import nn as enn
from escnn import group
from escnn.gspaces.r2 import GSpace2D

from irrep_actions.model.layers import ResNetBlock
from irrep_actions.model.equiv_layers import CyclicResNetBlock, SO2ResNetBlock
from irrep_actions.model.fourier import Fourier

class ImageEncoder(nn.Module):
    def __init__(self, in_channels, z_dim, dropout):
        super().__init__()

        self.conv = nn.Sequential(
            # 84x84
            nn.Conv2d(in_channels, z_dim // 8, kernel_size=5, padding=0),
            nn.ReLU(inplace=True),
            # 80x80
            ResNetBlock(z_dim // 8, z_dim // 8),
            ResNetBlock(z_dim // 8, z_dim // 8),
            nn.MaxPool2d(2),
            # 40x40
            ResNetBlock(z_dim // 8, z_dim // 4),
            ResNetBlock(z_dim // 4, z_dim // 4),
            nn.MaxPool2d(2),
            # 20x20
            ResNetBlock(z_dim // 4, z_dim // 2),
            ResNetBlock(z_dim // 2, z_dim // 2),
            nn.MaxPool2d(2),
            # 10x10
            ResNetBlock(z_dim // 2, z_dim),
            ResNetBlock(z_dim, z_dim),
            nn.MaxPool2d(2),
            # 5x5
            nn.Conv2d(z_dim, z_dim, kernel_size=5, padding=0),
            nn.ReLU(inplace=True)
            # 1x1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class CyclicImageEncoder(nn.Module):
    def __init__(self, in_channels, z_dim, dropout, lmax=3, N=16, initialize=True):
        super().__init__()
        self.cyclic = gspaces.rot2dOnR2(N)

        self.G = group.so2_group()
        self.gspace = gspaces.no_base_space(self.G)
        self.z_dim = z_dim

        self.in_type = enn.FieldType(
            self.cyclic,
            [self.cyclic.trivial_repr] * in_channels
        )
        self.out_type = enn.FieldType(
            self.cyclic,
            z_dim * [self.cyclic.regular_repr]
        )

        layers = list()
        # 84x84
        layers.append(
            enn.R2Conv(
                self.in_type,
                enn.FieldType(self.cyclic, z_dim // 8 * [self.cyclic.regular_repr]),
                kernel_size=5,
                padding=0,
                initialize=initialize
            )
        )
        layers.append(
            enn.ReLU(
                enn.FieldType(self.cyclic, z_dim // 8 * [self.cyclic.regular_repr]),
                inplace=True
            )
        )
        # 80x80
        layers.append(CyclicResNetBlock(layers[-1].out_type, z_dim // 8, N=N, initialize=initialize))
        layers.append(CyclicResNetBlock(layers[-1].out_type, z_dim // 8, N=N, initialize=initialize))
        layers.append(enn.PointwiseMaxPool(layers[-1].out_type, 2))
        # 40x40
        layers.append(CyclicResNetBlock(layers[-1].out_type, z_dim // 4, N=N, initialize=initialize))
        layers.append(CyclicResNetBlock(layers[-1].out_type, z_dim // 4, N=N, initialize=initialize))
        layers.append(enn.PointwiseMaxPool(layers[-1].out_type, 2))
        # 20x20
        layers.append(CyclicResNetBlock(layers[-1].out_type, z_dim // 2, N=N, initialize=initialize))
        layers.append(CyclicResNetBlock(layers[-1].out_type, z_dim // 2, N=N, initialize=initialize))
        layers.append(enn.PointwiseMaxPool(layers[-1].out_type, 2))
        # 10x10
        layers.append(CyclicResNetBlock(layers[-1].out_type, z_dim, N=N, initialize=initialize))
        layers.append(CyclicResNetBlock(layers[-1].out_type, z_dim, N=N, initialize=initialize))
        layers.append(enn.PointwiseMaxPool(layers[-1].out_type, 2))
        # 5x5
        layers.append(
            enn.R2Conv(
                layers[-1].out_type,
                self.out_type,
                kernel_size=5,
                padding=0,
                initialize=initialize
            )
        )
        layers.append(enn.ReLU(self.out_type, inplace=True))
        # 1x1

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        B = x.size(0)
        x = enn.GeometricTensor(x, self.in_type)
        out = self.conv(x).tensor.view(B, self.z_dim, -1)

        return out

class SO2ImageEncoder(nn.Module):
    def __init__(self, in_channels, z_dim, dropout, lmax=3, N=16, initialize=True):
        super().__init__()
        self.G = group.so2_group()
        self.gspace = GSpace2D((None, -1), lmax)

        self.in_type = enn.FieldType(
            self.gspace,
            [self.gspace.trivial_repr] * in_channels
        )

        layers = list()
        # 96x96
        layers.append(SO2ResNetBlock(self.in_type, z_dim // 8, lmax=lmax, N=N, initialize=initialize))
        layers.append(enn.NormMaxPool(layers[-1].out_type, 2))
        # 48x48
        layers.append(SO2ResNetBlock(layers[-1].out_type, z_dim // 4, lmax=lmax, N=N, initialize=initialize))
        layers.append(enn.NormMaxPool(layers[-1].out_type, 2))
        # 24x24
        layers.append(SO2ResNetBlock(layers[-1].out_type, z_dim // 2, lmax=lmax, N=N, initialize=initialize))
        layers.append(enn.NormMaxPool(layers[-1].out_type, 2))
        # 12x12
        layers.append(SO2ResNetBlock(layers[-1].out_type, z_dim, lmax=lmax, N=N, initialize=initialize))
        layers.append(enn.NormMaxPool(layers[-1].out_type, 2))
        # 6x6
        layers.append(SO2ResNetBlock(layers[-1].out_type, z_dim, lmax=lmax, N=N, initialize=initialize))
        layers.append(enn.NormMaxPool(layers[-1].out_type, 2))
        # 3x3
        act = enn.FourierELU(
            self.gspace,
            channels=z_dim,
            irreps=self.G.bl_irreps(L=lmax),
            inplace=True,
            type="regular",
            N=N,
        )
        layers.append(
            enn.R2Conv(
                layers[-1].out_type,
                act.in_type,
                kernel_size=3,
                padding=0,
                initialize=initialize
            )
        )
        layers.append(act)
        # 1x1

        self.out_type = layers[-1].out_type
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        B = x.size(0)
        x = enn.GeometricTensor(x, self.in_type)
        out = self.conv(x)

        return out.tensor
