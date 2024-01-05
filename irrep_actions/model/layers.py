import torch
import torch.nn as nn
import torch.nn.functional as F

from escnn import gspaces
from escnn import nn as enn
from escnn import group


class MLP(nn.Module):
    def __init__(self, hiddens, dropout=0.0, act_out=True):
        super().__init__()

        layers = list()
        for i, (h, h_) in enumerate(zip(hiddens, hiddens[1:])):
            layers.append(nn.Linear(h, h_))
            is_last_layer = i == len(hiddens) - 2
            if not is_last_layer or act_out:
                layers.append(nn.LeakyReLU(0.01, inplace=True))
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class ResNet(nn.Module):
    def __init__(self, hiddens):
        super().__init__()

        layers = list()
        for h, h_ in zip(hiddens, hiddens[1:]):
            layers.append(nn.Conv2d(h, h_, kernel_size=3, padding=1, stride=2))
            layers.appen(nn.MaxPool2d(2))
            layers.append(nn.ReLU(inplace=True))
        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn(x)


class SO2MLP(nn.Module):
    def __init__(
        self, in_type, out_type, channels, lmaxs, N=8, dropout=0.0, act_out=True
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
        #self.out_type = blocks[-1].out_type
        self.so2_mlp = enn.SequentialModule(*blocks)

    def forward(self, x):
        return self.so2_mlp(x)
