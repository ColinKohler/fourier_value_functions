import numpy as np
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

        self.basis_fns = nn.Parameter(self.generate_basis_fns(maximum_frequency, N))

    def generate_basis_fns(self, L: int, N: int) -> torch.Tensor:
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

    def evaluate(self, w: torch.Tensor, action_theta=None) -> torch.Tensor:
        ''' Evaluate the harmonic function using the given coefficents and the basis functions generated at init.

        Args:
            w:

        Returns:
        '''
        if action_theta is not None:
            return
        else:
            return torch.mm(w,  self.basis_fns)

    def convert_to_polar_coords(self, x: torch.Tensor) -> torch.Tensor:
        ''' Convert regular coordinates to polar coordinates.

        Args:
            x:
        '''
        r = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
        theta = torch.arctan2(x[:, 1], (x[:, 0]))
        theta[torch.where(theta < 0)] += 2 * torch.pi

        return torch.concatenate((r[:,None], theta[:,None]), axis=1)

