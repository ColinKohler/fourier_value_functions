import numpy as np
import torch
from torch import nn
from fvf.model.modules.harmonics.harmonics import HarmonicFunction

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

