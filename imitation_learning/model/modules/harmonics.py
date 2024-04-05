import numpy as np
from scipy.special import jv

import torch
from torch import nn

class HarmonicFunction(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def generate_baseis_fns(self, L: int) -> torch.Tensor:
        raise NotImplementedError

class CircularHarmonics(HarmonicFunction):
    def __init__(
        self,
        maximum_angular_frequency: int,
        num_phi: int=360
    ):
        super().__init__()

        self.num_phi = num_phi
        self.maximum_angular_frequency = maximum_angular_frequency
        self.basis_fns = nn.Parameter(self.generate_basis_fns())

    def generate_basis_fns(self, phi: torch.Tensor=None) -> torch.Tensor:
        if phi is None:
            phi = torch.linspace(0, 2*torch.pi, self.num_phi).view(-1,1)
        basis_fns = [
            torch.tensor([1 / np.sqrt(2 * torch.pi)] * phi.size(0)).view(-1, 1).to(phi.device)
        ]
        for l in range(1, self.maximum_angular_frequency+1):
            basis_fns.append(torch.cos(l * phi) / np.sqrt(torch.pi))
            basis_fns.append(torch.sin(l * phi) / np.sqrt(torch.pi))

        return torch.stack(basis_fns).permute(1, 0, 2).float().squeeze().permute(1,0)

    def evaluate(self, w: torch.Tensor, phi: torch.Tensor=None) -> torch.Tensor:
        if phi is not None:
            basis_fns = self.generate_basis_fns(phi).permute(1,0)
            L = self.maximum_angular_frequency * 2 + 1
            return torch.bmm(w.view(-1, 1, L), basis_fns.view(-1, L, 1))
        else:
            return torch.mm(w, self.basis_fns)

class DiskHarmonics(HarmonicFunction):
    def __init__(
        self,
        radial_frequency: int,
        angular_frequency: int,
        max_radius: float,
        num_radii: int=None,
        num_phi: int=None,
    ):
        super().__init__()

        self.radial_frequency = radial_frequency
        self.angular_frequency = angular_frequency
        self.max_radius = max_radius
        self.num_radii = num_radii
        self.num_phi = num_phi

        self.basis_fns = nn.Parameter(self.generate_basis_fns2())

    def Psi(self, n, m, r, phi):
        # Radial basis function
        k = n**2+m
        N = self.max_radius**2 / 2 * jv(m+1, k*self.max_radius)**2
        R = (1 / np.sqrt(N) * jv(m, k*r.cpu())).to(r.device)

        # Angular basis function
        if m == 0:
            Phi = torch.tensor([1 / np.sqrt(2*torch.pi)] * phi.size(0)).float().to(phi.device)
            return [R.view(-1) * Phi.view(-1)]
        else:
            Phi = [
                torch.cos(m * phi) / np.sqrt(torch.pi),
                torch.sin(m * phi) / np.sqrt(torch.pi),
            ]
            return [R.view(-1) * Phi[0].view(-1), R.view(-1) * Phi[1].view(-1)]

    def Psi2(self, n, m, r, phi):
        # Radial basis function
        k = n**2+m
        N = self.max_radius**2 / 2 * jv(m+1, k*self.max_radius)**2
        R = (1 / np.sqrt(N) * jv(m, k*r.cpu())).to(r.device)

        # Angular basis function
        if m == 0:
            Phi = torch.tensor([1 / np.sqrt(2*torch.pi)] * phi.size(0)).float().to(phi.device)
            return [R.view(-1, 1) @ Phi.view(1, -1)]
        else:
            Phi = [
                torch.cos(m * phi) / np.sqrt(torch.pi),
                torch.sin(m * phi) / np.sqrt(torch.pi),
            ]
            return [R.view(-1, 1) @ Phi[0].view(1, -1), R.view(-1, 1) @ Phi[1].view(1, -1)]


    def generate_basis_fns(self, radii: torch.Tensor=None, phis: torch.Tensor=None) -> torch.Tensor:
        if radii is None:
            radii = torch.linspace(0, self.max_radius, self.num_radii).view(-1 ,1)
            phis = torch.linspace(0, 2*np.pi, self.num_phi)
        basis_fns = []
        for n in range(1,self.radial_frequency+1):
            for m in range(self.angular_frequency+1):
                basis_fns.extend(self.Psi(n, m, radii, phis))
        return torch.stack(basis_fns).view(self.radial_frequency, 2*self.angular_frequency+1, -1)

    def generate_basis_fns2(self, radii: torch.Tensor=None, phis: torch.Tensor=None) -> torch.Tensor:
        if radii is None:
            radii = torch.linspace(0, self.max_radius, self.num_radii).view(-1 ,1)
            phis = torch.linspace(0, 2*np.pi, self.num_phi)
        basis_fns = []
        for n in range(1,self.radial_frequency+1):
            for m in range(self.angular_frequency+1):
                basis_fns.extend(self.Psi2(n, m, radii, phis))
        return torch.stack(basis_fns).view(self.radial_frequency, 2*self.angular_frequency+1, -1)

    def evaluate(
        self,
        coeffs: torch.Tensor,
        radii: torch.Tensor=None,
        phis: torch.Tensor=None
    ) -> torch.Tensor:
        if radii is not None:
            basis_fns = self.generate_basis_fns(radii, phis)
            return torch.einsum("inm,inm->i", coeffs, basis_fns.permute(2,0,1))
        else:
            return torch.einsum("inm,tnm->it", coeffs, self.basis_fns.permute(2,0,1))
