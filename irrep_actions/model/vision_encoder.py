import torch
import torch.nn as nn

from escnn import gspaces
from escnn import nn as enn
from escnn import group

from irrep_actions.model.layers import ResNetBlock
from irrep_actions.model.equiv_layers import CyclicResNetBlock, SO2ResNetBlock
from irrep_actions.model.fourier import Fourier

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
        # 96x96
        layers.append(CyclicResNetBlock(self.in_type, z_dim // 8, N=N))
        layers.append(enn.PointwiseMaxPool(layers[-1].out_type, 2))
        # 48x48
        layers.append(CyclicResNetBlock(layers[-1].out_type, z_dim // 4, N=N))
        layers.append(enn.PointwiseMaxPool(layers[-1].out_type, 2))
        # 24x24
        layers.append(CyclicResNetBlock(layers[-1].out_type, z_dim // 2, N=N))
        layers.append(enn.PointwiseMaxPool(layers[-1].out_type, 2))
        # 12x12
        layers.append(CyclicResNetBlock(layers[-1].out_type, z_dim, N=N))
        layers.append(enn.PointwiseMaxPool(layers[-1].out_type, 2))
        # 6x6
        layers.append(CyclicResNetBlock(layers[-1].out_type, z_dim, N=N))
        layers.append(enn.PointwiseMaxPool(layers[-1].out_type, 2))
        # 3x3
        layers.append(
            enn.R2Conv(
                layers[-1].out_type,
                self.out_type,
                kernel_size=3,
                padding=0,
                initialize=initialize
            )
        )
        layers.append(enn.ReLU(self.out_type, inplace=True))
        # 1x1

        self.conv = nn.Sequential(*layers)
        self.fourier = Fourier(
            self.gspace,
            z_dim,
            self.G.bl_regular_representation(L=3).irreps,
            N=N
        )

    def forward(self, x):
        B = x.size(0)
        x = enn.GeometricTensor(x, self.in_type)
        cyclic_out = self.conv(x).tensor.view(B, self.z_dim, -1)
        out = self.fourier(cyclic_out)

        return out.tensor

class SO2ImageEncoder(nn.Module):
    def __init__(self, in_channels, z_dim, dropout, lmax=3, N=16, initialize=True):
        super().__init__()
        self.G = group.so2_group()
        self.gspace = gspaces.no_base_space(self.G)

        self.in_type = enn.FieldType(
            self.gspace,
            [self.gspace.trivial_repr] * in_channels
        )

        layers = list()
        # 96x96
        layers.append(SO2ResNetBlock(self.in_type, z_dim // 8, lmax=lmax, N=N))
        # 48x48
        layers.append(SO2ResNetBlockLayers(layers[-1].out_type, z_dim // 4, lmax=lmax, N=N))
        # 24x24
        layers.append(SO2ResNetBlockLayers(layers[-1].out_type, z_dim // 2, lmax=lmax, N=N))
        # 12x12
        layers.append(SO2ResNetBlockLayers(layers[-1].out_type, z_dim, lmax=lmax, N=N))
        # 6x6
        layers.append(SO2ResNetBlockLayers(layers[-1].out_type, z_dim, lmax=lmax, N=N))
        # 3x3
        act = enn.FourierELU(
            self.gspace,
            channels=channels,
            irreps=self.G.bl_regular_representation(L=lmax).irreps,
            inplace=True,
            type="regular",
            N=N,
        )
        layers.append(
            enn.R2Conv(
                layers[-1].out_type,
                act.in_type,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=stride,
                initialize=initialize
            )
        )
        layers.append(act)
        # 1x1

        self.conv = enn.SequentialDict(*layers)

    def forward(x: enn.GeometricTensor) -> enn.GeometricTensor:
       return self.conv(x)
