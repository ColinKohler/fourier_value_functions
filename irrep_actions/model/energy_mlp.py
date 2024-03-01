import torch
import torch.nn as nn

from escnn import gspaces
from escnn import nn as enn
from escnn import group

from irrep_actions.model.modules.layers import MLP
from irrep_actions.model.moudles.equiv_layers import SO2MLP
from irrep_actions.model.modules.harmonics import CircularHarmonics
from irrep_actions.utils import harmonics

class EnergyMLP(nn.Module):
    def __init__(self, in_channels, mid_channels, dropout, spec_norm):
        super().__init__()
        self.energy_mlp = MLP(
            [in_channels] + mid_channels +  [1],
            dropout=dropout,
            act_out=False,
            spec_norm=spec_norm
        )

    def forward(self, obs, action):
        B, N, Ta, Da = action.shape
        B, To, Do = obs.shape

        s = obs.reshape(B, 1, -1).expand(-1, N, -1)
        s_a = torch.cat([s, action.reshape(B, N, -1)], dim=-1).reshape(B*N, -1)
        out = self.energy_mlp(s_a)

        return out.reshape(B, N)

class CyclicEnergyMLP(nn.Module):
    def __init__(self, in_channels, mid_channels, lmax, dropout, N=16):
        super().__init__()
        self.Lmax = lmax
        self.G =  group.CyclicGroup(N)
        self.gspace = gspaces.no_base_space(self.G)
        rho = self.gspace.regular_repr
        self.in_type = enn.FieldType(
            self.gspace,
            2 * 256 * [rho] + 2 * [self.gspace.irrep(1)]
        )

        out_type = enn.FieldType(self.gspace, [self.G.irrep(0)])
        mid_type = enn.FieldType(self.gspace, mid_channels * [rho])
        self.energy_mlp = enn.SequentialModule(
            enn.Linear(self.in_type, mid_type),
            enn.ReLU(mid_type, inplace=True),
            enn.FieldDropout(mid_type, dropout),
            enn.Linear(mid_type, mid_type),
            enn.ReLU(mid_type, inplace=True),
            enn.FieldDropout(mid_type, dropout),
            enn.Linear(mid_type, mid_type),
            enn.ReLU(mid_type, inplace=True),
            enn.FieldDropout(mid_type, dropout),
            enn.Linear(mid_type, out_type),
        )

    def forward(self, obs, action):
        B, N, Ta, Da = action.shape
        B, To, Do = obs.shape

        s = obs.reshape(B, 1, -1).expand(-1, N, -1)
        s_a = self.in_type(torch.cat([s, action.reshape(B, N, -1)], dim=-1).reshape(B*N, -1))
        out = self.energy_mlp(s_a)

        return out.tensor.reshape(B, N)


class SO2EnergyMLP(nn.Module):
    def __init__(self, in_channels, mid_channels, lmax, dropout, N=16):
        super().__init__()
        self.Lmax = lmax
        self.G = group.so2_group()
        self.gspace = gspaces.no_base_space(self.G)
        rho = self.G.spectral_regular_representation(*self.G.bl_irreps(L=lmax))
        self.in_type = enn.FieldType(
            self.gspace,
            2 * 256 * [rho] + [self.gspace.irrep(1)]
        )

        out_type = enn.FieldType(self.gspace, [self.G.irrep(0)])
        rho = self.G.spectral_regular_representation(*self.G.bl_irreps(L=self.Lmax), name=None)
        mid_type = enn.FieldType(self.gspace, mid_channels * [rho])
        self.energy_mlp = enn.SequentialModule(
            enn.Linear(self.in_type, mid_type),
            enn.FourierPointwise(self.gspace, mid_channels, self.G.bl_irreps(L=self.Lmax), type='regular', N=N),
            enn.FieldDropout(mid_type, dropout),
            enn.Linear(mid_type, mid_type),
            enn.FourierPointwise(self.gspace, mid_channels, self.G.bl_irreps(L=self.Lmax), type='regular', N=N),
            enn.FieldDropout(mid_type, dropout),
            enn.Linear(mid_type, mid_type),
            enn.FourierPointwise(self.gspace, mid_channels, self.G.bl_irreps(L=self.Lmax), type='regular', N=N),
            enn.FieldDropout(mid_type, dropout),
            enn.Linear(mid_type, out_type),
        )

    def forward(self, obs, action):
        B, N, Ta, Da = action.shape
        B, To, Do = obs.shape

        s = obs.reshape(B, 1, -1).expand(-1, N, -1)
        s_a = self.in_type(torch.cat([s, action.reshape(B, N, -1)], dim=-1).reshape(B*N, -1))
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
        mlp_type = enn.FieldType(self.gspace, mlp_dim * [rho])
        out_type = enn.FieldType(self.gspace, [self.gspace.irrep(l) for l in range(self.Lmax+1)])

        self.energy_mlp = SO2MLP(
            self.in_type,
            channels=[mlp_dim] * 4,
            lmaxs=[lmax] * 4,
            out_type=out_type,
            N=N,
            dropout=dropout,
            act_out = False
        )
        self.circular_energy_harmonics = CircularHarmonics(lmax, N)
        #circle_theta = torch.linspace(0, 2*torch.pi, self.num_rot).view(-1,1)
        #self.B = harmonics.circular_harmonics(lmax, circle_theta).squeeze().permute(1,0)

    def forward(self, obs_feat, action_magnitude, action_theta):
        ''' Compute the energy function for the desired action.

        '''
        B, N, Ta, Da = action_magnitude.shape
        B, Dz = obs_feat.shape

        s = obs_feat.reshape(B, 1, -1).expand(-1, N, -1)
        s_a = self.in_type(
            torch.cat([s, action_magnitude.reshape(B, N, -1)], dim=-1).reshape(B*N, -1)
        )

        w = self.energy_mlp(s_a).tensor.view(B, N, -1)
        return self.circular_energy_harmonics.evaluate(w).view(B, N)
        #Beta = harmonics.circular_harmonics(self.Lmax, action_theta.view(-1,1))
        #return torch.bmm(W.view(-1, 1, self.Lmax * 2 + 1), Beta).view(B,N)

    def get_energy_ball(self, obs_feat, action_magnitude):
        B, N, Ta, Da = action_magnitude.shape
        B, Dz = obs_feat.shape

        s = obs_feat.reshape(B, 1, -1).expand(-1, N, -1)
        s_a = self.in_type(torch.cat([s, action_magnitude.reshape(B, N, -1)], dim=-1).reshape(B*N, -1))
        w = self.energy_mlp(s_a).tensor.view(B*N, 1, self.Lmax*2+1)
        return self.circular_energy_harmonics.evaluate(w).view(B, N, -1)
        #Beta = self.B.view(1,self.Lmax*2+1, self.num_rot).repeat(B*N, 1, 1).to(obs_feat.device)
        #return torch.bmm(W, Beta).view(B, N, -1)
