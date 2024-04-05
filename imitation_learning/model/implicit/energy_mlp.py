import torch
import torch.nn as nn

from escnn import gspaces
from escnn import nn as enn
from escnn import group

from imitation_learning.model.modules.layers import MLP
from imitation_learning.model.modules.equiv_layers import SO2MLP
from imitation_learning.model.modules.harmonics import CircularHarmonics, DiskHarmonics

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
    def __init__(self, obs_feat_dim, mlp_dim, lmax, dropout, N=16, num_phi=360, initialize=True):
        super().__init__()
        self.Lmax = lmax
        self.num_phi = num_phi

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
        self.circular_harmonics = CircularHarmonics(lmax, num_phi)

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
            return self.circular_harmonics.evaluate(w, action_theta.view(B*N, 1)).view(B, N)
        else:
            return self.circular_harmonics.evaluate(w).view(B, N, -1)


class RingEnergyMLP(nn.Module):
    def __init__(self, obs_feat_dim, mlp_dim, lmax, dropout, N=16, num_radii=100, num_phi=360, initialize=True):
        super().__init__()
        self.Lmax = lmax
        self.num_phi = num_phi
        self.num_radii = num_radii

        self.G = group.so2_group()
        self.gspace = gspaces.no_base_space(self.G)
        rho = self.G.spectral_regular_representation(*self.G.bl_irreps(L=lmax))

        self.in_type = enn.FieldType(
            self.gspace,
            obs_feat_dim * [rho]
        )
        out_type = enn.FieldType(self.gspace, num_radii * [self.gspace.irrep(l) for l in range(self.Lmax+1)])

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
        self.circular_harmonics = CircularHarmonics(lmax, num_phi)

    def forward(self, obs_feat, polar_actions=None):
        B, Dz = obs_feat.shape

        s = self.in_type(obs_feat)
        w = self.energy_mlp(s).tensor.view(B, self.num_radii, -1)
        if polar_actions is not None:
            B, N, _ = polar_actions.shape
            r = polar_actions[:,:,0].int()
            phi = polar_actions[:,:,1]
            # TODO: Clean up this mess
            w_r = w.view(B, 1, self.num_radii, -1).repeat(1,N,1,1).view(B*N,self.num_radii, -1)[torch.arange(B*N), r.view(B*N)]
            return self.circular_harmonics.evaluate(w_r, phi.view(B*N, 1)).view(B, N)
        else:
            ring_fns = list()
            for r in range(self.num_radii):
                ring_fns.append(self.circular_harmonics.evaluate(w[:,r]).view(B, 1, -1))
            return torch.concatenate(ring_fns, axis=1)


class DiskEnergyMLP(nn.Module):
    def __init__(
        self,
        obs_feat_dim,
        mlp_dim,
        radial_freq,
        angular_freq,
        dropout,
        max_radius,
        N=16,
        num_radii=100,
        num_phi=360,
        initialize=True
    ):
        super().__init__()
        self.radial_freq = radial_freq
        self.angular_freq = angular_freq
        self.max_radius = max_radius
        self.num_phi = num_phi
        self.num_radii = num_radii

        self.G = group.so2_group()
        self.gspace = gspaces.no_base_space(self.G)
        rho = self.G.spectral_regular_representation(*self.G.bl_irreps(L=angular_freq))

        self.in_type = enn.FieldType(
            self.gspace,
            obs_feat_dim * [rho]
        )
        out_type = enn.FieldType(self.gspace, radial_freq * [self.gspace.irrep(l) for l in range(angular_freq+1)])

        self.energy_mlp = SO2MLP(
            self.in_type,
            channels=[mlp_dim] * 4,
            lmaxs=[radial_freq] * 4,
            out_type=out_type,
            N=N,
            dropout=dropout,
            act_out = False,
            initialize=initialize
        )
        self.disk_harmonics = DiskHarmonics(radial_freq, angular_freq, max_radius, num_radii, num_phi)

    def forward(self, obs_feat, polar_actions=None):
        ''' Compute the energy function for the desired action.

        '''
        B, Dz = obs_feat.shape

        s = self.in_type(obs_feat)
        w = self.energy_mlp(s).tensor.view(B, self.radial_freq, self.angular_freq*2+1)
        if polar_actions is not None:
            B, N, _ = polar_actions.shape
            w = w.unsqueeze(1).repeat(1,N,1,1).view(B*N,self.radial_freq, self.angular_freq*2+1)
            return self.disk_harmonics.evaluate(w, polar_actions[:,:,0].view(B*N,-1), polar_actions[:,:,1].view(B*N, -1)).view(B, N)
        else:
            return self.disk_harmonics.evaluate(w)


class CylindericalEnergyMLP(nn.Module):
    def __init__(self, obs_feat_dim, mlp_dim, lmax, dropout, N=16, num_phi=360, initialize=True):
        super().__init__()
        self.Lmax = lmax
        self.num_phi = num_phi

        self.G = group.so2_group()
        self.gspace = gspaces.no_base_space(self.G)
        rho = self.G.spectral_regular_representation(*self.G.bl_irreps(L=lmax))

        self.in_type = enn.FieldType(
            self.gspace,
            obs_feat_dim * [rho] + 3 * [self.gspace.irrep(0)] # [obs_feat_dim, r, z, g]
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
        self.circular_harmonics = CircularHarmonics(lmax, num_phi)

    def forward(self, obs_feat, action, action_theta=None):
        ''' Compute the energy function for the desired action.

        '''
        B, N, Ta, Da = action.shape
        B, Dz = obs_feat.shape

        s = obs_feat.reshape(B, 1, -1).expand(-1, N, -1)
        s_a = self.in_type(
            torch.cat([s, action.reshape(B, N, -1)], dim=-1).reshape(B*N, -1)
        )

        w = self.energy_mlp(s_a).tensor.view(B*N, -1)
        if action_theta is not None:
            return self.circular_harmonics.evaluate(w, action_theta).view(B, N)
        else:
            return self.circular_harmonics.evaluate(w).view(B, N, -1)