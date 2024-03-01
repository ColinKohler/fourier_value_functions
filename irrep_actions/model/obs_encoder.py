import torch
import torch.nn as nn

from escnn import gspaces
from escnn import nn as enn
from escnn import group
from escnn.gspaces.r2 import GSpace2D

from irrep_actions.model.vision_encoder import CyclicImageEncoder
from irrep_actions.model.fourier import Fourier
from irrep_actions.model.equiv_layers import SO2MLP

class SO2ObsEncoder(nn.Module):
    def __init__(
        self,
        num_obs: int,
        img_channels: int,
        z_dim: int,
        lmax: int=3,
        N: int=16,
        dropout: float=0.0,
        initialize: bool=True,
    ):
        super().__init__()

        self.G = group.so2_group()
        self.gspace = gspaces.no_base_space(self.G)
        self.z_dim = z_dim

        self.image_encoder = CyclicImageEncoder(
            img_channels,
            z_dim,
            dropout,
            lmax=lmax,
            N=N,
            initialize=initialize
        )
        self.fourier = Fourier(
            self.gspace,
            z_dim,
            self.G.bl_irreps(L=lmax),
            N=N
        )

        rho = self.G.spectral_regular_representation(*self.G.bl_irreps(L=lmax))
        self.lin_in_type = enn.FieldType(
            self.gspace,
            num_obs * z_dim * [rho] + num_obs * [self.gspace.irrep(1)]
        )
        self.lin = SO2MLP(
            self.lin_in_type,
            channels=[z_dim],
            lmaxs=[lmax],
            N=N,
            dropout=dropout,
            act_out=True
        )
        self.out_type = self.lin.out_type

    def forward(self, obs) -> torch.Tensor:
        B, T, C, H, W = obs['image'].shape

        c8_img_feat = self.image_encoder(obs['image'].view(B*T, C, H, W))
        so2_img_feat = self.fourier(c8_img_feat).tensor.view(B, -1)
        state = obs['agent_pos'].view(B, -1)

        img_feat_state =  self.lin_in_type(torch.cat([so2_img_feat, state], dim=-1))
        obs_feat = self.lin(img_feat_state)

        return obs_feat.tensor
