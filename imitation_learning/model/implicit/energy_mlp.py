import torch
import torch.nn as nn

from escnn import gspaces
from escnn import nn as enn
from escnn import group

from imitation_learning.model.modules.layers import MLP
from imitation_learning.model.modules.equiv_layers import SO2MLP
from imitation_learning.model.modules.harmonics import CircularHarmonics

class EnergyMLP(nn.Module):
    def __init__(self, obs_feat, mlp_dim, dropout, spec_norm, initialize):
        super().__init__()
        self.energy_mlp = MLP(
            [obs_feat + 2] + [mlp_dim] * 4 +  [1],
            dropout=dropout,
            act_out=False,
            spec_norm=spec_norm
        )

    def forward(self, obs, action):
        B, N, Ta, Da = action.shape
        B, Dz = obs.shape

        s = obs.reshape(B, 1, -1).expand(-1, N, -1)
        s_a = torch.cat([s, action.reshape(B, N, -1)], dim=-1).reshape(B*N, -1)
        out = self.energy_mlp(s_a)

        return out.reshape(B, N)

class CyclicEnergyMLP(nn.Module):
    def __init__(self, obs_feat_dim, mlp_dim, lmax, dropout, N=16, initialize=True):
        super().__init__()
        self.Lmax = lmax

        self.G =  group.CyclicGroup(N)
        self.gspace = gspaces.no_base_space(self.G)
        rho = self.gspace.regular_repr
        self.in_type = enn.FieldType(
            self.gspace,
            obs_feat_dim * [rho] + [self.gspace.irrep(1)]
        )

        out_type = enn.FieldType(self.gspace, [self.G.irrep(0)])
        self.energy_mlp = CyclicMLP(
            self.in_type,
            channels=[mlp_dim] * 4,
            lmaxs=[lmax] * 4,
            out_type=out_type,
            N=N,
            dropout=dropout,
            act_out=False,
            initialize=initialize
        )


    def forward(self, obs_feat, action):
        B, N, Ta, Da = action.shape
        B, Dz = obs_feat.shape

        s = obsi_feat.reshape(B, 1, -1).expand(-1, N, -1)
        s_a = self.in_type(torch.cat([s, action.reshape(B, N, -1)], dim=-1).reshape(B*N, -1))
        out = self.energy_mlp(s_a)

        return out.tensor.reshape(B, N)


class SO2EnergyMLP(nn.Module):
    def __init__(self, obs_feat_dim, mlp_dim, lmax, dropout, N=16, initialize=True):
        super().__init__()
        self.Lmax = lmax

        self.G = group.so2_group()
        self.gspace = gspaces.no_base_space(self.G)
        rho = self.G.spectral_regular_representation(*self.G.bl_irreps(L=lmax))

        self.in_type = enn.FieldType(
            self.gspace,
            obs_feat_dim * [rho] + [self.gspace.irrep(1)]
        )
        out_type = enn.FieldType(self.gspace, [self.G.irrep(0)])

        self.energy_mlp = SO2MLP(
            self.in_type,
            channels=[mlp_dim] * 4,
            lmaxs=[lmax] * 4,
            out_type=out_type,
            N=N,
            dropout=dropout,
            act_out=False,
            initialize=initialize
        )

    def forward(self, obs_feat, action):
        B, N, Ta, Da = action.shape
        B, Dz = obs_feat.shape

        s = obs_feat.reshape(B, 1, -1).expand(-1, N, -1)
        s_a = self.in_type(
            torch.cat([s, action.reshape(B, N, -1)], dim=-1).reshape(B*N, -1)
        )
        out = self.energy_mlp(s_a)

        return out.tensor.reshape(B, N)

class CircularEnergyMLP(nn.Module):
    def __init__(self, obs_feat_dim, mlp_dim, lmax, dropout, N=16, num_rot=360, initialize=True):
        super().__init__()
        self.Lmax = lmax
        self.num_rot = num_rot

        self.G = group.so2_group()
        self.gspace = gspaces.no_base_space(self.G)
        rho = self.G.spectral_regular_representation(*self.G.bl_irreps(L=lmax))

        self.in_type = enn.FieldType(
            self.gspace,
            obs_feat_dim * [rho] + [self.gspace.irrep(0)]
        )
        out_type = enn.FieldType(self.gspace, [self.gspace.irrep(l) for l in range(self.Lmax+1)])

        self.energy_mlp = SO2MLP(
            self.in_type,
            channels=[mlp_dim] * 4,
            lmaxs=[lmax] * 4,
            out_type=out_type,
            N=N,
            dropout=dropout,
            act_out = False,
            initialize=initialize
        )
        self.circular_energy_harmonics = CircularHarmonics(lmax, num_rot)

    def forward(self, obs_feat, action_magnitude, action_theta=None):
        ''' Compute the energy function for the desired action.

        '''
        B, N, Ta, Da = action_magnitude.shape
        B, Dz = obs_feat.shape

        s = obs_feat.reshape(B, 1, -1).expand(-1, N, -1)
        s_a = self.in_type(
            torch.cat([s, action_magnitude.reshape(B, N, -1)], dim=-1).reshape(B*N, -1)
        )

        w = self.energy_mlp(s_a).tensor.view(B*N, -1)
        if action_theta is not None:
            return self.circular_energy_harmonics.evaluate(w, action_theta).view(B, N)
        else:
            return self.circular_energy_harmonics.evaluate(w).view(B, N, -1)

class CylindericalEnergyMLP(nn.Module):
    def __init__(self, obs_feat_dim, mlp_dim, lmax, dropout, N=16, num_rot=360, initialize=True):
        super().__init__()
        self.Lmax = lmax
        self.num_rot = num_rot

        self.G = group.so2_group()
        self.gspace = gspaces.no_base_space(self.G)
        rho = self.G.spectral_regular_representation(*self.G.bl_irreps(L=lmax))

        self.in_type = enn.FieldType(
            self.gspace,
            obs_feat_dim * [rho] + [self.gspace.irrep(0), self.gspace.irrep(0)] # [obs_feat_dim, r, z]
        )
        out_type = enn.FieldType(self.gspace, [self.gspace.irrep(l) for l in range(self.Lmax+1)])

        self.energy_mlp = SO2MLP(
            self.in_type,
            channels=[mlp_dim] * 4,
            lmaxs=[lmax] * 4,
            out_type=out_type,
            N=N,
            dropout=dropout,
            act_out = False,
            initialize=initialize
        )
        self.circular_energy_harmonics = CircularHarmonics(lmax, num_rot)

    def forward(self, obs_feat, action_magnitude, action_theta=None):
        ''' Compute the energy function for the desired action.

        '''
        B, N, Ta, Da = action_magnitude.shape
        B, Dz = obs_feat.shape

        s = obs_feat.reshape(B, 1, -1).expand(-1, N, -1)
        s_a = self.in_type(
            torch.cat([s, action_magnitude.reshape(B, N, -1)], dim=-1).reshape(B*N, -1)
        )

        w = self.energy_mlp(s_a).tensor.view(B*N, -1)
        if action_theta is not None:
            return self.circular_energy_harmonics.evaluate(w, action_theta).view(B, N)
        else:
            return self.circular_energy_harmonics.evaluate(w).view(B, N, -1)
