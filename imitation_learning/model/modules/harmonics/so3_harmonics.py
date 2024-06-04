import numpy as np
import torch
from escnn import group
from escnn.group.groups.so3_utils import _wigner_d_matrix, _change_param

from imitation_learning.model.modules.harmonics.harmonics import HarmonicFunction

class SO3Harmonics(HarmonicFunction):
    def __init__(
        self,
        frequency: int,
        num_grid_points: int=100
    ):
        super().__init__()

        self.frequency = frequency

        self.so3_group = group.SO3(self.frequency)
        self.grid = self.so3_group.grid('thomson', N=num_grid_points)

        self.D = []
        #for l in range(self.frequency):
        #    self.D.append(torch.stack([torch.from_numpy(_wigner_d_matrix(np.array(e.value), l, param=e.param)).float()]))

    def evaluate(
        self,
        f: torch.Tensor,
        R: torch.Tensor=None
    ) -> torch.Tensor:
        B = f.size(0)
        if R is not None:
            F = torch.zeros((B,) + R.shape[1])
            for l in range(self.frequnecy):
                f_l = f[:,l].view(2*l+1, 2*l+1)
                d_l = torch.from_numpy(_wigner_d_matrix(R, l, param='ZYZ')).float()
                F += torch.sum((2*l + 1) / (8*torch.pi) * torch.matmul(f_l, d_l), axis=1)
        else:
            F = torch.zeros((B,) +  self.r2d.shape).to(Pnm.device)
            for l in range(f.size(1)):
                d_l = self.D[l].unsqueeze(0).repeat(B,1,1,1).to(f.device)
                f_l = f[:,l].view(2*l+1, 2*l+1)
                F += torch.sum((2*l + 1) / (8*torch.pi) * torch.matmul(f_l, d_l), axis=1)

        return F
