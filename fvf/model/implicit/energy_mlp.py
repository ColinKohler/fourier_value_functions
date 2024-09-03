""" energy_mlpy.py """

import torch
from torch import nn

from escnn import gspaces
from escnn import nn as enn
from escnn import group

from fvf.model.modules.layers import MLP
from fvf.model.modules.equiv_layers import CyclicMLP, SO2MLP

from torch_harmonics.circular_harmonics import CircularHarmonics
from torch_harmonics.polar_harmonics import PolarHarmonics
from torch_harmonics.cylindrical_harmonics import CylindricalHarmonics
from torch_harmonics.so3_harmonics import SO3Harmonics


class EnergyMLP(nn.Module):
    """Vanilla IBC energy head."""

    def __init__(
        self,
        obs_feat_dim: int,
        mlp_dim: int,
        dropout: float,
        spec_norm: bool = False,
        initialize: bool = False,
    ):
        super().__init__()
        self.energy_mlp = MLP(
            [obs_feat_dim + 2] + [mlp_dim] * 4 + [1],
            dropout=dropout,
            act_out=False,
            spec_norm=spec_norm,
        )

    def forward(self, obs_feat: torch.Tensor, action: torch.Tensor):
        """Compute energy for observation and action pairs."""
        B, N, _, _ = action.shape
        B, _ = obs_feat.shape

        s = obs_feat.reshape(B, 1, -1).expand(-1, N, -1)
        s_a = torch.cat([s, action.reshape(B, N, -1)], dim=-1).reshape(B * N, -1)
        out = self.energy_mlp(s_a)

        return out.reshape(B, N)


class PolarEnergyMLP(nn.Module):
    """Vanilla IBC with Polar harmonics head."""

    def __init__(
        self,
        obs_feat_dim: int,
        mlp_dim: int,
        num_layers: int,
        dropout: float,
        spec_norm: bool,
        radial_freq: int,
        angular_freq: int,
        min_radius: float,
        max_radius: float,
        num_radii: int = 100,
        num_phi: int = 360,
        boundary: str = "zero",
        initialize: bool = True,
    ):
        super().__init__()
        self.energy_mlp = MLP(
            [obs_feat_dim]
            + [mlp_dim] * num_layers
            + [radial_freq * (angular_freq * 2 + 1)],
            dropout=dropout,
            act_out=False,
            spec_norm=spec_norm,
        )

        self.ph = PolarHarmonics(
            radial_freq,
            angular_freq,
            min_radius,
            max_radius,
            num_radii,
            num_phi,
            boundary=boundary,
        )

    def forward(
        self,
        obs_feat: torch.Tensor,
        actions: torch.Tensor = None,
        return_coeffs: bool = False,
    ):
        """
        Compute the energy function for all actions using Polar Fourier transform. If actions are
        specified

        Args:
            obs_feat: Encoded observations.
            actions: Action coordinates to evaluate the polar harmoincs at.
        """
        B, _ = obs_feat.shape

        w = self.energy_mlp(obs_feat).view(B, 1, -1)
        if actions is not None:
            B, N, _ = actions.shape
            w = w.repeat(1, N, 1).reshape(B * N, -1)
            out = self.ph(w, actions.view(B * N, 2)).view(B, N)
        else:
            out = self.ph(w.reshape(B, -1))

        if return_coeffs:
            return out, w
        else:
            return out


class CyclicEnergyMLP(nn.Module):
    """Equivariant IBC energy head which uses the discrete cyclic group Cn."""

    def __init__(
        self,
        obs_feat_dim: int,
        mlp_dim: int,
        lmax: int,
        dropout: float,
        N: int = 16,
        initialize: bool = True,
    ):
        super().__init__()
        self.Lmax = lmax

        self.G = group.CyclicGroup(N)
        self.gspace = gspaces.no_base_space(self.G)
        rho = self.gspace.regular_repr
        self.in_type = enn.FieldType(
            self.gspace, obs_feat_dim * [rho] + [self.gspace.irrep(1)]
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
            initialize=initialize,
        )

    def forward(self, obs_feat: torch.Tensor, action: torch.Tensor):
        """Compute energy for observation and action pairs."""
        B, N, _, _ = action.shape
        B, _ = obs_feat.shape

        s = obs_feat.reshape(B, 1, -1).expand(-1, N, -1)
        s_a = self.in_type(
            torch.cat([s, action.reshape(B, N, -1)], dim=-1).reshape(B * N, -1)
        )
        out = self.energy_mlp(s_a)

        return out.tensor.reshape(B, N)


class SO2EnergyMLP(nn.Module):
    """Equivariant IBC energy head which uses the continuous SO2 group."""

    def __init__(
        self,
        obs_feat_dim: int,
        mlp_dim: int,
        lmax: int,
        dropout: float,
        N: int = 16,
        initialize: bool = True,
    ):
        super().__init__()
        self.Lmax = lmax

        self.G = group.so2_group()
        self.gspace = gspaces.no_base_space(self.G)
        rho = self.G.spectral_regular_representation(*self.G.bl_irreps(L=lmax))

        self.in_type = enn.FieldType(
            self.gspace, obs_feat_dim * [rho] + [self.gspace.irrep(1)]
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
            initialize=initialize,
        )

    def forward(self, obs_feat: torch.Tensor, action: torch.Tensor):
        """Compute energy for observation and action pairs."""
        B, N, _, _ = action.shape
        B, _ = obs_feat.shape

        s = obs_feat.reshape(B, 1, -1).expand(-1, N, -1)
        s_a = self.in_type(
            torch.cat([s, action.reshape(B, N, -1)], dim=-1).reshape(B * N, -1)
        )
        out = self.energy_mlp(s_a)

        return out.tensor.reshape(B, N)


class SO2CircularEnergyMLP(nn.Module):
    """
    Equivariant IBC fourier energy head which uses the continuous SO2 group and circular harmonics.
    """

    def __init__(
        self,
        obs_feat_dim: int,
        mlp_dim: int,
        lmax: int,
        dropout: float,
        N: int = 16,
        num_phi: int = 360,
        initialize: bool = True,
    ):
        super().__init__()
        self.Lmax = lmax
        self.num_phi = num_phi

        self.G = group.so2_group()
        self.gspace = gspaces.no_base_space(self.G)
        rho = self.G.spectral_regular_representation(*self.G.bl_irreps(L=lmax))

        self.in_type = enn.FieldType(
            self.gspace, obs_feat_dim * [rho] + [self.gspace.irrep(0)]
        )
        out_type = enn.FieldType(
            self.gspace, [self.gspace.irrep(l) for l in range(self.Lmax + 1)]
        )

        self.energy_mlp = SO2MLP(
            self.in_type,
            channels=[mlp_dim] * 4,
            lmaxs=[lmax] * 4,
            out_type=out_type,
            N=N,
            dropout=dropout,
            act_out=False,
            initialize=initialize,
        )
        self.circular_harmonics = CircularHarmonics(lmax, num_phi)

    def forward(
        self,
        obs_feat: torch.Tensor,
        action_magnitude: torch.Tensor,
        action_theta: torch.Tensor = None,
    ):
        """Compute the energy function for all actions using Fourier transform."""
        B, N, _, _ = action_magnitude.shape
        B, _ = obs_feat.shape

        s = obs_feat.reshape(B, 1, -1).expand(-1, N, -1)
        s_a = self.in_type(
            torch.cat([s, action_magnitude.reshape(B, N, -1)], dim=-1).reshape(
                B * N, -1
            )
        )

        w = self.energy_mlp(s_a).tensor.view(B * N, -1)
        if action_theta is not None:
            out = self.circular_harmonics(w, action_theta.view(B * N, 1)).view(B, N)
        else:
            out = self.circular_harmonics(w).view(B, N, -1)

        return out


class SO2PolarEnergyMLP(nn.Module):
    """
    Equivariant IBC fourier energy head which uses the continuous SO2 group and polar harmonics.

    Args:
        obs_feat_dim - Dimensionality of encoded observations.
        mlp_dim - Dimensionality of the MLP.
        lmax - Maxiumum frequency within the MLP.
        num_layers - Number of layers in the MLP.
        N - Number of discrete activation features in the MLP.
        dropout - Dropout of the MLP.
        radial_freq - Maximum radial frequency of Polar harmonics (K).
        angular_freq - Maximum angular frequneyc of the Polar harmoincs (L).
        min_radius - Minimum value of the radius.
        max_radius - Maximum value of the radius.
        num_phi - Number of angular components in the basis grid.
        num_rho - Number of radial components in the basis grid.
        boundary - Boundary type of Polar Harmonics: zero or deri.
        initialize - Initialize the model weights.
    """

    def __init__(
        self,
        obs_feat_dim: int,
        mlp_dim: int,
        lmax: int,
        num_layers: int,
        radial_freq: int,
        angular_freq: int,
        dropout: float,
        min_radius: float,
        max_radius: float,
        N: int = 16,
        num_radii: int = 100,
        num_phi: int = 360,
        boundary: str = "zero",
        initialize: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.lmax = lmax

        self.G = group.so2_group()
        self.gspace = gspaces.no_base_space(self.G)
        rho = self.G.spectral_regular_representation(*self.G.bl_irreps(L=self.lmax))

        self.in_type = enn.FieldType(self.gspace, obs_feat_dim * [rho])
        out_type = enn.FieldType(
            self.gspace,
            radial_freq * [self.gspace.irrep(l) for l in range(angular_freq + 1)],
        )

        self.energy_mlp = SO2MLP(
            self.in_type,
            channels=[mlp_dim] * num_layers,
            lmaxs=[self.lmax] * num_layers,
            out_type=out_type,
            N=N,
            dropout=dropout,
            act_out=False,
            initialize=initialize,
        )
        self.ph = PolarHarmonics(
            radial_freq,
            angular_freq,
            min_radius,
            max_radius,
            num_radii,
            num_phi,
            boundary=boundary,
        )

    def forward(
        self,
        obs_feat: torch.Tensor,
        actions: torch.Tensor = None,
        return_coeffs: bool = False,
    ):
        """
        Compute the energy function for all actions using Polar Fourier transform. If actions are
        specified

        Args:
            obs_feat: Encoded observations.
            actions: Action coordinates to evaluate the polar harmoincs at.
            return_coeffs: Return Fourier coefficients.
        """
        B, _ = obs_feat.shape

        s = self.in_type(obs_feat)
        w = self.energy_mlp(s).tensor.view(B, 1, -1)
        if actions is not None:
            B, N, _ = actions.shape
            w = w.repeat(1, N, 1).reshape(B * N, -1)
            out = self.ph(w, actions.view(B * N, 2)).view(B, N)
        else:
            out = self.ph(w.reshape(B, -1))

        if return_coeffs:
            return out, w
        else:
            return out


class CylindricalEnergyMLP(nn.Module):
    """
    Equivariant IBC fourier energy head which uses the continuous SO2 group and cylindrical harmonics.
    """

    def __init__(
        self,
        obs_feat_dim,
        mlp_dim,
        lmax,
        num_layers,
        radial_freq,
        angular_freq,
        axial_freq,
        dropout,
        N=16,
        min_radius=0.0,
        max_radius=1.0,
        max_height=1.0,
        num_radii=100,
        num_phi=360,
        num_height=100,
        initialize=True,
    ):
        super().__init__()
        self.Lmax = lmax
        self.num_layers = num_layers
        self.radial_freq = radial_freq
        self.angular_freq = angular_freq
        self.axial_freq = axial_freq
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.num_radii = num_radii
        self.num_phi = num_phi
        self.num_height = num_height

        self.G = group.so2_group()
        self.gspace = gspaces.no_base_space(self.G)
        rho = self.G.spectral_regular_representation(*self.G.bl_irreps(L=lmax))

        self.in_type = enn.FieldType(self.gspace, obs_feat_dim * [rho])
        out_type = enn.FieldType(
            self.gspace,
            radial_freq
            * axial_freq
            * [self.gspace.irrep(l) for l in range(angular_freq + 1)],
        )

        self.energy_mlp = SO2MLP(
            self.in_type,
            channels=[mlp_dim] * num_layers,
            lmaxs=[lmax] * num_layers,
            out_type=out_type,
            N=N,
            dropout=dropout,
            act_out=False,
            initialize=initialize,
        )
        gripper_out = enn.FieldType(self.gspace, [self.gspace.trivial_repr])
        self.gripper_mlp = SO2MLP(
            self.in_type,
            channels=[mlp_dim] * num_layers,
            lmaxs=[lmax] * num_layers,
            out_type=gripper_out,
            N=N,
            dropout=dropout,
            act_out=False,
            initialize=initialize,
        )
        self.ch = CylindricalHarmonics(
            radial_freq,
            angular_freq,
            axial_freq,
            min_radius,
            max_radius,
            max_height,
            num_radii,
            num_phi,
            num_height,
            boundary="deri",
        )

    def forward(self, obs_feat, actions=None, return_coeffs: bool = False):
        """Compute the energy function for all actions using Fourier transform."""
        B, _ = obs_feat.shape

        s = self.in_type(obs_feat)
        w = self.energy_mlp(s).tensor.view(B, 1, -1)
        if actions is not None:
            B, N, _, _ = actions.shape
            w = w.repeat(1, N, 1).reshape(B * N, -1)
            energy = self.ch(
                w,
                actions.view(B * N, -1),
            ).view(B, N)
        else:
            energy = self.ch(w.reshape(B, -1))

        gripper_pred = torch.sigmoid(self.gripper_mlp(s).tensor)
        if return_coeffs:
            return energy, gripper_pred, w
        else:
            return energy, gripper_pred


class SO3CylindricalEnergyMLP(nn.Module):
    """
    Equivariant IBC fourier energy head which uses the continuous SO2 group and
    cylindrical harmonics.
    """

    def __init__(
        self,
        obs_feat_dim,
        mlp_dim,
        lmax,
        num_layers,
        radial_freq,
        angular_freq,
        axial_freq,
        so3_freq,
        dropout,
        N=16,
        min_radius=0.0,
        max_radius=1.0,
        max_height=1.0,
        num_radii=100,
        num_phi=360,
        num_height=100,
        num_so3=1000,
        initialize=True,
    ):
        super().__init__()
        self.Lmax = lmax
        self.num_layers = num_layers
        self.radial_freq = radial_freq
        self.angular_freq = angular_freq
        self.axial_freq = axial_freq
        self.so3_freq = so3_freq
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.num_radii = num_radii
        self.num_phi = num_phi
        self.num_height = num_height
        self.num_so3 = num_so3

        self.so2_group = group.so2_group(lmax)
        self.so3_group = group.so3_group(lmax)
        self.so2_id = (False, -1)
        self.gspace = gspaces.no_base_space(self.so2_group)
        rho = self.so2_group.spectral_regular_representation(
            *self.so2_group.bl_irreps(L=lmax)
        )

        self.in_type = enn.FieldType(self.gspace, obs_feat_dim * [rho])

        cylinder_out_type = enn.FieldType(
            self.gspace,
            radial_freq
            * axial_freq
            * [self.gspace.irrep(l) for l in range(angular_freq + 1)],
        )
        self.cylinder_coeffs_mlp = SO2MLP(
            self.in_type,
            channels=[mlp_dim] * num_layers,
            lmaxs=[lmax] * num_layers,
            out_type=cylinder_out_type,
            N=N,
            dropout=dropout,
            act_out=False,
            initialize=initialize,
        )

        so3_out_type = enn.FieldType(
            self.gspace,
            [
                self.so3_group.bl_regular_representation(L=so3_freq).restrict(
                    self.so2_id
                )
            ],
        )
        self.so3_coeffs_mlp = SO2MLP(
            self.in_type,
            channels=[mlp_dim] * num_layers,
            lmaxs=[lmax] * num_layers,
            out_type=so3_out_type,
            N=N,
            dropout=dropout,
            act_out=False,
            initialize=initialize,
        )

        gripper_out = enn.FieldType(self.gspace, [self.gspace.trivial_repr])
        self.gripper_mlp = SO2MLP(
            self.in_type,
            channels=[mlp_dim] * num_layers,
            lmaxs=[lmax] * num_layers,
            out_type=gripper_out,
            N=N,
            dropout=dropout,
            act_out=False,
            initialize=initialize,
        )

        self.cylindrical_harmonics = CylindricalHarmonics(
            radial_freq,
            angular_freq,
            axial_freq,
            min_radius,
            max_radius,
            max_height,
            num_radii,
            num_phi,
            num_height,
            boundary="deri",
        )
        self.so3_harmonics = SO3Harmonics(so3_freq, num_so3)

    def forward(self, obs_feat, actions=None):
        """Compute the energy function for all actions using Fourier transform."""
        B, Dz = obs_feat.shape

        s = self.in_type(obs_feat)

        pos_w = self.cylinder_coeffs_mlp(s).tensor.view(B, 1, -1)
        rot_w = self.so3_coeffs_mlp(s).tensor.view(B, 1, -1)
        if actions is not None:
            B, N, _, _ = actions.shape
            pos_w = pos_w.repeat(1, N, 1).reshape(B * N, -1)
            rot_w = rot_w.repeat(1, N, 1).reshape(B * N, -1)
            pos_energy = self.cylindrical_harmonics(
                pos_w,
                actions[:, :, 0, :3].view(B * N, -1),
            ).view(B, N)
            rot_energy = self.so3_harmonics(
                rot_w,
                actions[:, :, 0, 3:].view(B * N, 3),
            ).view(B, N)
        else:
            pos_energy = self.cylindrical_harmonics(pos_w.reshape(B, -1))
            rot_energy = self.so3_harmonics(rot_w.reshape(B, -1))

        gripper_pred = torch.sigmoid(self.gripper_mlp(s).tensor)
        return pos_energy, rot_energy, gripper_pred
