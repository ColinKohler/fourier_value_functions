import numpy as np
from scipy.special import jv

import torch
from torch import nn

class HarmonicFunction(nn.Module):
    def __init__(
        self,
        maximum_frequency: int
    ):
        super().__init__()

        self.maximum_frequency = maximum_frequency

    def generate_baseis_fns(self, L: int) -> torch.Tensor:
        raise NotImplementedError

class CircularHarmonics(HarmonicFunction):
    def __init__(
        self,
        maximum_frequency: int,
        N: int=360
    ):
        '''

        '''
        super().__init__(maximum_frequency)

        self.basis_fns = nn.Parameter(self.generate_basis_fn_ball(maximum_frequency, N))

    def generate_basis_fn_ball(self, L: int, N: int) -> torch.Tensor:
        ''' Generate circular basis funcations for the desired frequency.

        Args:
            L: int - Maxmium basis function frequnecy

        Returns:
            torch.Tensor: Basis functions
        '''
        theta = torch.linspace(0, 2*torch.pi, N).view(-1,1)
        basis_fns = [
            torch.tensor([1 / np.sqrt(2 * torch.pi)] * theta.size(0)).view(-1, 1)
        ]
        for l in range(1, L+1):
            basis_fns.append(torch.cos(l * theta) / np.sqrt(torch.pi))
            basis_fns.append(torch.sin(l * theta) / np.sqrt(torch.pi))

        return torch.stack(basis_fns).permute(1, 0, 2).float().squeeze().permute(1,0)

    def generate_basis_fns(self, L: int, theta: torch.Tensor) -> torch.Tensor:
        ''' Generate circular basis funcations for the desired frequency.

        Args:
            L: int - Maxmium basis function frequnecy

        Returns:
            torch.Tensor: Basis functions
        '''
        theta = theta.view(-1,1)
        basis_fns = [
            torch.tensor([1 / np.sqrt(2 * torch.pi)] * theta.size(0)).view(-1, 1).to(theta.device)
        ]
        for l in range(1, L+1):
            basis_fns.append(torch.cos(l * theta) / np.sqrt(torch.pi))
            basis_fns.append(torch.sin(l * theta) / np.sqrt(torch.pi))

        return torch.stack(basis_fns).permute(1, 0, 2).float().squeeze().permute(1,0)

    def evaluate(self, w: torch.Tensor, theta: torch.Tensor=None) -> torch.Tensor:
        ''' Evaluate the harmonic function using the given coefficents and the basis functions generated at init.

        Args:
            w:

        Returns:
        '''
        if theta is not None:
            basis_fns = self.generate_basis_fns(self.maximum_frequency, theta)
            L = self.maximum_frequency * 2 + 1
            return torch.bmm(w.view(-1, 1, L), basis_fns.view(-1, L, 1))
        else:
            return torch.mm(w, self.basis_fns)

    def convert_to_polar_coords(self, x: torch.Tensor) -> torch.Tensor:
        ''' Convert regular coordinates to polar coordinates.

        Args:
            x:
        '''
        r = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
        theta = torch.arctan2(x[:, 1], (x[:, 0]))
        theta[torch.where(theta < 0)] += 2 * torch.pi

        return torch.concatenate((r[:,None], theta[:,None]), axis=1)


class DiskHarmonics(HarmonicFunction):
    def __init__(
        self,
        N: int,
        M: int,
        a: float,
        num_radii: int=None,
        num_phi: int=None,
    ):
        super().__init__()

        self.N = N
        self.M = M
        self.a = a
        self.num_radii = num_radii
        self.num_phi = num_phi

    def Phi(self, phi):
        Phi = []
        for m in range(self.M):
            Phi.append((1/np.sqrt(2*np.pi)) * np.exp(1j * m * phi))
        return Phi

    def N_n(self, n, m, a):
        k = n**2+m
        return self.a**2 / 2 * jv(m+1, k*a)**2

    def R(self, r):
        R = []
        for n in range(1,self.N+1):
            ns = []
            for m in range(self.M):
                k = n**2+m
                ns.append(1 / np.sqrt(self.N_n(n,m,self.a)) * jv(m, k*r))
            R.append(ns)
        return R

    def Psi(self, n, m, a, r, phi):
        k = n**2+m
        N = a**2 / 2 * jv(m+1, k*a)**2
        R = 1 / np.sqrt(N) * jv(m, k*r)
        if m == 0:
            Phi = torch.tensor([1 / np.sqrt(2*np.pi)] * [phi.size(0)])
        else:
            Phi = torch.tensor([
                torch.cos(m*phi) / np.sqrt(np.pi),
                torch.sin(m*phi) / np.sqrt(np.pi),
            ])

        Phi = 1 / np.sqrt(2*np.pi) * np.exp(1j * m * phi)
        return R.reshape(r.size(0),1) @ Phi.reshape(1,phi.size(0))

    def evaluate(
        self,
        coeffs: torch.Tensor,
        radii: torch.Tensor,
        phi: torch.Tensor
    ) -> torch.Tensor:
        basis = []
        for n in range(1,self.N+1):
            b = []
            for m in range(self.M):
                b.append(Psi(n, m, self.a, radii, phi))
            basis.append(b)
        basis = torch.tensor(basis)
        return torch.einsum("inm,tnm->t", coeffs, basis)

    def convert_to_polar_coords(self, x: torch.Tensor) -> torch.Tensor:
        ''' Convert regular coordinates to polar coordinates.

        Args:
            x:
        '''
        r = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
        theta = torch.arctan2(x[:, 1], (x[:, 0]))
        theta[torch.where(theta < 0)] += 2 * torch.pi

        return torch.concatenate((r[:,None], theta[:,None]), axis=1)
