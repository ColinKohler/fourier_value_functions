import torch
import torch.nn as nn

from escnn import gspaces
from escnn import nn as enn
from escnn import group

from irrep_actions.model.layers import MLP, SO2MLP
from irrep_actions.utils import harmonics

class EnergyMLP(nn.Module):
    def __init__(self, in_channels, mid_channels, dropout, spec_norm):
        super().__init__()
        self.energy_mlp = MLP(
            [in_channels, mid_channels, mid_channels, mid_channels, mid_channels, mid_channels, mid_channels, mid_channels, 1],
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

class SO2EnergyMLP(nn.Module):
    def __init__(self, in_channels, mid_channels, lmax, dropout):
        super().__init__()
        self.Lmax = lmax
        self.G = group.so2_group()
        self.gspace = gspaces.no_base_space(self.G)
        self.in_type = enn.FieldType(
            self.gspace,
            [self.gspace.irrep(1)] * in_channels
        )

        out_type = enn.FieldType(self.gspace, [self.G.irrep(0)])
        rho = self.G.spectral_regular_representation(*self.G.bl_irreps(L=self.Lmax), name=None)
        mid_type = enn.FieldType(self.gspace, mid_channels * [rho])
        self.energy_mlp = enn.SequentialModule(
            enn.Linear(self.in_type, mid_type),
            enn.FourierPointwise(self.gspace, mid_channels, self.G.bl_irreps(L=self.Lmax), type='regular', N=16),
            enn.FieldDropout(mid_type, dropout),
            enn.Linear(mid_type, mid_type),
            enn.FourierPointwise(self.gspace, mid_channels, self.G.bl_irreps(L=self.Lmax), type='regular', N=16),
            enn.FieldDropout(mid_type, dropout),
            enn.Linear(mid_type, mid_type),
            enn.FourierPointwise(self.gspace, mid_channels, self.G.bl_irreps(L=self.Lmax), type='regular', N=16),
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

class SO2EnergySkipMLP(nn.Module):
    def __init__(self, in_channels, mid_channels, lmax, dropout):
        super().__init__()
        self.Lmax = 32
        self.G = group.so2_group()
        self.gspace = gspaces.no_base_space(self.G)
        self.in_type = enn.FieldType(
            self.gspace,
            [self.gspace.irrep(1)] * in_channels
        )

        out_type = enn.FieldType(self.gspace, [self.G.irrep(0)])

        mid_channels_1 = 16
        rho_1 = self.G.spectral_regular_representation(*self.G.bl_irreps(L=32), name=None)
        mid_type_1 = enn.FieldType(self.gspace, mid_channels_1 * [rho_1])
        self.energy_mlp_1 = enn.SequentialModule(
            enn.Linear(self.in_type, mid_type_1),
            enn.FourierPointwise(self.gspace, mid_channels_1, self.G.bl_irreps(L=32), type='regular', N=64),
            enn.FieldDropout(mid_type_1, dropout),
        )

        mid_channels_2 = 32
        rho_2 = self.G.spectral_regular_representation(*self.G.bl_irreps(L=16), name=None)
        mid_type_2 = enn.FieldType(self.gspace, mid_channels_2 * [rho_2])
        self.energy_mlp_2 = enn.SequentialModule(
            enn.Linear(mid_type_1, mid_type_2),
            enn.FourierPointwise(self.gspace, mid_channels_2, self.G.bl_irreps(L=16), type='regular', N=32),
            enn.FieldDropout(mid_type_2, dropout),
        )

        mid_channels_3 = 64
        rho_3 = self.G.spectral_regular_representation(*self.G.bl_irreps(L=8), name=None)
        mid_type_3 = enn.FieldType(self.gspace, mid_channels_3 * [rho_3])
        self.energy_mlp_3 = enn.SequentialModule(
            enn.Linear(mid_type_2, mid_type_3),
            enn.FourierPointwise(self.gspace, mid_channels_3, self.G.bl_irreps(L=8), type='regular', N=16),
            enn.FieldDropout(mid_type_3, dropout),
        )
        self.out_in_type = enn.FieldType(
            self.gspace,
            [self.G.irrep(1)] * in_channels + mid_channels_1 * [rho_1] + mid_channels_2 * [rho_2] + mid_channels_3 * [rho_3]
        )
        self.energy_mlp_out = enn.Linear(self.out_in_type, out_type)

    def forward(self, obs, action):
        B, N, Ta, Da = action.shape
        B, To, Do = obs.shape

        s = obs.reshape(B, 1, -1).expand(-1, N, -1)
        s_a = self.in_type(torch.cat([s, action.reshape(B, N, -1)], dim=-1).reshape(B*N, -1))

        x1 = self.energy_mlp_1(s_a)
        x2 = self.energy_mlp_2(x1)
        x3 = self.energy_mlp_3(x2)
        x_123 = self.out_in_type(torch.cat([s_a.tensor, x1.tensor, x2.tensor, x3.tensor], dim=-1))
        out = self.energy_mlp_out(x_123)

        return out.tensor.reshape(B, N)


class SO2HarmonicEnergyMLP(nn.Module):
    def __init__(self, in_channels, mid_channels, lmax, dropout, num_rot=360):
        super().__init__()
        self.Lmax = lmax
        self.G = group.so2_group()
        self.gspace = gspaces.no_base_space(self.G)
        self.num_rot = num_rot
        self.B = harmonics.circular_harmonics(lmax, torch.linspace(0, 2*torch.pi, self.num_rot).view(-1,1)).squeeze().permute(1,0)
        self.in_type = self.gspace.type(
            *[self.G.standard_representation()] * in_channels + [self.G.trivial_representation]
        )

        #out_type = self.gspace.type(*[self.G.bl_regular_representation(L=self.Lmax)])
        out_type = enn.FieldType(self.gspace, [self.gspace.irrep(l) for l in range(self.Lmax+1)])
        rho = self.G.spectral_regular_representation(*self.G.bl_irreps(L=lmax), name=None)
        mid_type = enn.FieldType(self.gspace, mid_channels * [rho])
        self.energy_mlp = enn.SequentialModule(
            enn.Linear(self.in_type, mid_type),
            enn.FourierPointwise(self.gspace, mid_channels, self.G.bl_irreps(L=lmax), type='regular', N=16),
            enn.FieldDropout(mid_type, dropout),
            enn.Linear(mid_type, mid_type),
            enn.FourierPointwise(self.gspace, mid_channels, self.G.bl_irreps(L=lmax), type='regular', N=16),
            enn.FieldDropout(mid_type, dropout),
            enn.Linear(mid_type, mid_type),
            enn.FourierPointwise(self.gspace, mid_channels, self.G.bl_irreps(L=lmax), type='regular', N=16),
            enn.FieldDropout(mid_type, dropout),
            enn.Linear(mid_type, out_type),
        )

    def forward(self, obs, action_magnitude, action_theta):
        B, N, Ta, Da = action_magnitude.shape
        B, To, Do = obs.shape

        s = obs.reshape(B, 1, -1).expand(-1, N, -1)
        s_a = self.in_type(torch.cat([s, action_magnitude.reshape(B, N, -1)], dim=-1).reshape(B*N, -1))
        W = self.energy_mlp(s_a).tensor.view(B, N, -1)
        Beta = harmonics.circular_harmonics(self.Lmax, action_theta.view(-1,1))
        return torch.bmm(W.view(-1, 1, self.Lmax * 2 + 1), Beta).view(B,N)

    def get_energy_ball(self, obs, action_magnitude):
        B, N, Ta, Da = action_magnitude.shape
        B, To, Do = obs.shape

        s = obs.reshape(B, 1, -1).expand(-1, N, -1)
        s_a = self.in_type(torch.cat([s, action_magnitude.reshape(B, N, -1)], dim=-1).reshape(B*N, -1))
        W = self.energy_mlp(s_a).tensor.view(B*N, 1, self.Lmax*2+1)
        Beta = self.B.view(1,self.Lmax*2+1, self.num_rot).repeat(B*N, 1, 1).to(obs.device)

        return torch.bmm(W, Beta).view(B, N, -1)


class SO2HarmonicEnergySkipMLP(nn.Module):
    def __init__(self, in_channels, mid_channels, lmax, dropout, num_rot=360):
        super().__init__()
        self.Lmax = 3
        self.G = group.so2_group()
        self.gspace = gspaces.no_base_space(self.G)
        self.B = harmonics.circular_harmonics(lmax, torch.linspace(0, 2*torch.pi, num_rot).view(-1,1)).squeeze().permute(1,0)
        self.in_type = self.gspace.type(
            *[self.G.standard_representation()] * in_channels + [self.G.trivial_representation]
        )

        #out_type = self.gspace.type(*[self.G.bl_regular_representation(L=self.Lmax)])
        out_type = enn.FieldType(self.gspace, [self.gspace.irrep(l) for l in range(self.Lmax+1)])

        mid_channels_1 = mid_channels
        L_1, N_1 = 5, 16
        rho_1 = self.G.spectral_regular_representation(*self.G.bl_irreps(L=L_1), name=None)
        mid_type_1 = enn.FieldType(self.gspace, mid_channels_1 * [rho_1])
        self.energy_mlp_1 = enn.SequentialModule(
            enn.Linear(self.in_type, mid_type_1),
            enn.FourierPointwise(self.gspace, mid_channels_1, self.G.bl_irreps(L=L_1), type='regular', N=N_1),
            enn.FieldDropout(mid_type_1, dropout),
        )

        mid_channels_2 = mid_channels
        L_2, N_2 = 4, 16
        rho_2 = self.G.spectral_regular_representation(*self.G.bl_irreps(L=L_2), name=None)
        mid_type_2 = enn.FieldType(self.gspace, mid_channels_2 * [rho_2])
        self.energy_mlp_2 = enn.SequentialModule(
            enn.Linear(mid_type_1, mid_type_2),
            enn.FourierPointwise(self.gspace, mid_channels_2, self.G.bl_irreps(L=L_2), type='regular', N=N_2),
            enn.FieldDropout(mid_type_2, dropout),
        )

        mid_channels_3 = mid_channels
        L_3, N_3 = 3, 16
        rho_3 = self.G.spectral_regular_representation(*self.G.bl_irreps(L=L_3), name=None)
        mid_type_3 = enn.FieldType(self.gspace, mid_channels_3 * [rho_3])
        self.energy_mlp_3 = enn.SequentialModule(
            enn.Linear(mid_type_2, mid_type_3),
            enn.FourierPointwise(self.gspace, mid_channels_3, self.G.bl_irreps(L=L_3), type='regular', N=N_3),
            enn.FieldDropout(mid_type_3, dropout),
        )
        self.out_in_type = enn.FieldType(
            self.gspace,
            mid_channels_1 * [rho_1] + mid_channels_2 * [rho_2] + mid_channels_3 * [rho_3]
        )
        self.energy_mlp_out = enn.Linear(self.out_in_type, out_type)

    def forward(self, obs, action_magnitude, action_theta):
        B, N, Ta, Da = action_magnitude.shape
        B, To, Do = obs.shape

        s = obs.reshape(B, 1, -1).expand(-1, N, -1)
        s_a = self.in_type(torch.cat([s, action_magnitude.reshape(B, N, -1)], dim=-1).reshape(B*N, -1))

        x1 = self.energy_mlp_1(s_a)
        x2 = self.energy_mlp_2(x1)
        x3 = self.energy_mlp_3(x2)
        x_123 = self.out_in_type(torch.cat([x1.tensor, x2.tensor, x3.tensor], dim=-1))
        W = self.energy_mlp_out(x_123).tensor.view(B, N, -1)

        Beta = harmonics.circular_harmonics(self.Lmax, action_theta.view(-1,1))
        return torch.bmm(W.view(-1, 1, self.Lmax * 2 + 1), Beta).view(B,N)

    def get_energy_ball(self, obs, action_magnitude):
        B, N, Ta, Da = action_magnitude.shape
        B, To, Do = obs.shape

        s = obs.reshape(B, 1, -1).expand(-1, N, -1)
        s_a = self.in_type(torch.cat([s, action_magnitude.reshape(B, N, -1)], dim=-1).reshape(B*N, -1))

        x1 = self.energy_mlp_1(s_a)
        x2 = self.energy_mlp_2(x1)
        x3 = self.energy_mlp_3(x2)
        x_123 = self.out_in_type(torch.cat([x1.tensor, x2.tensor, x3.tensor], dim=-1))
        W = self.energy_mlp_out(x_123).tensor.view(B*N, 1, self.Lmax*2+1)

        Beta = self.B.view(1,self.Lmax*2+1, 360).repeat(B*N, 1, 1).to(obs.device)
        return torch.bmm(W, Beta).view(B, N, -1)

