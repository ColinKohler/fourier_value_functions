import torch
import torch.nn as nn
import torch.nn.functional as F

from escnn import gspaces
from escnn import nn as enn
from escnn import group

class SO2MLP(nn.Module):
    def __init__(
        self,
        in_type: enn.FieldType,
        out_type: enn.FieldType,
        channels: list[int],
        lmaxs: list[int],
        N: int=8,
        dropout: float=0.0,
        act_out: bool=True
    ):
        super().__init__()

        self.G = group.so2_group()
        self.gspace = gspaces.no_base_space(self.G)
        self.in_type = in_type
        self.out_type = out_type

        blocks = list()
        in_type = self.in_type
        for i, (c, l) in enumerate(zip(channels, lmaxs)):
            act = enn.FourierELU(
                self.gspace,
                channels=c,
                irreps=self.G.bl_regular_representation(L=l).irreps,
                inplace=True,
                type="regular",
                N=N,
            )
            is_last_layer = i == len(channels) - 1
            if not is_last_layer or act_out:
                blocks.append(enn.Linear(in_type, act.in_type))
                blocks.append(enn.FieldDropout(act.in_type, dropout))
                blocks.append(act)
            else:
                blocks.append(enn.Linear(in_type, out_type))
            in_type = act.out_type
        self.so2_mlp = enn.SequentialModule(*blocks)

    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        return self.so2_mlp(x)


class CyclicResNetBlock(nn.Module):
    def __init__(
        self,
        in_type: enn.FieldType,
        channels: int,
        kernel_size: int=3,
        stride: int=1,
        N: int=8,
        initialize: bool=True,
    ):
        super().__init__()

        self.G = gspaces.rot2dOnR2(N)
        self.gspace = gspaces.no_base_space(self.G)
        self.in_type = in_type
        self.out_type = enn.FieldType(self.G, channels * [self.G.regular_repr])

        self.conv1 = enn.SequentialModule(
            enn.R2Conv(
                in_type,
                self.out_type,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=stride,
                initialize=initialize
            ),
            enn.ReLU(self.out_type, inplace=True)
        )

        self.act = enn.ReLU(self.out_type, inplace=True)
        self.conv2 = enn.R2Conv(
            self.out_type,
            self.out_type,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            stride=stride,
            initialize=initialize
        )

        self.upscale = None
        if len(self.in_type) != channels or stride != 1:
            self.upscale = enn.R2Conv(
                self.in_type,
                self.out_type,
                kernel_size=1,
                stride=stride,
                bias=False,
                initialize=initialize
            )

    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.upscale is not None:
            out += self.upscale(res)
        else:
            out += res
        out = self.act(out)

        return out


class SO2ResNetBlock(nn.Module):
    def __init__(
        self,
        in_type: enn.FieldType,
        channels: int,
        kernel_size: int=3,
        stride: int=2,
        lmax: int=3,
        N: int=8,
        initialize: bool=True,
    ):
        super().__init__()

        self.G = group.so2_group()
        self.gspace = gspaces.no_base_space(self.G)
        self.in_type = in_type

        act1 = enn.FourierELU(
            self.gspace,
            channels=channels,
            irreps=self.G.bl_regular_representation(L=lmax).irreps,
            inplace=True,
            type="regular",
            N=N,
        )
        self.conv1 = enn.SequentialModule(
            enn.R2Conv(
                in_type,
                act1.in_type,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=stride,
                initialize=initialize
            ),
            act1
        )

        self.act = enn.FourierELU(
            self.gspace,
            channels=channels,
            irreps=self.G.bl_regular_representation(L=lmax).irreps,
            inplace=True,
            type="regular",
            N=N,
        )
        self.conv2 = enn.R2Conv(
            act1.out_type,
            self.act.in_type,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            stride=stride,
            initialize=initialize
        )

        self.upscale = None
        if len(self.in_type) != channels or stride != 1:
            self.upscale = nn.R2Conv(
                self.in_type,
                self.act.out_type,
                kernel_size=1,
                stride=stride,
                bias=False,
                initialize=initialize
            )


    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        res = x
        out = self.conv1(x)
        out = self.conv2(x)
        if self.upscale is not None:
            out += self.upscale(res)
        else:
            out += res
        out = self.act(out)

        return out
