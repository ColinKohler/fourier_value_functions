import numpy as np
import torch
from escnn import group
from escnn.group.groups.so3_utils import _wigner_d_matrix, _change_param, _grid
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_euler_angles, euler_angles_to_matrix, matrix_to_axis_angle

from imitation_learning.model.modules.harmonics.harmonics import HarmonicFunction

class SO3Harmonics(HarmonicFunction):
    def __init__(
        self,
        frequency: int,
        num_grid_points: int=100
    ):
        super().__init__()

        self.frequency = frequency
        self.num_grid_points = num_grid_points

        self.so3_group = group.SO3(self.frequency)
        #self.grid = _grid('hopf', N=num_grid_points, parametrization='ZYZ')
        self.grid = torch.from_numpy(_grid('hopf', N=num_grid_points, parametrization='ZYZ')).float()
        transform = euler_angles_to_matrix(torch.tensor([0., 0., 1*np.pi/2.]), 'XYZ')
        self.grid = matrix_to_euler_angles(transform @ euler_angles_to_matrix(self.grid, 'ZYZ'), 'ZYZ').numpy()

        self.num_grid_points = self.grid.shape[0]

        self.D = []
        for l in range(self.frequency+1):
            d_l = []
            for e in self.grid:
                d_l.append(torch.from_numpy(_wigner_d_matrix(e, l, param='ZYZ')).float())
            self.D.append(torch.stack(d_l))

    def evaluate(
        self,
        f: torch.Tensor,
        R: torch.Tensor=None
    ) -> torch.Tensor:
        B = f.size(0)
        if R is not None:
            F = torch.zeros((B)).to(f.device)
            li = 0
            for l in range(self.frequency+1):
                ld = 2*l+1
                f_l = f[:,li:li+ld**2].view(B, ld, ld)
                d_l = []
                for e in R.cpu().numpy():
                    d_l.append(torch.from_numpy(_wigner_d_matrix(e, l, param='ZYZ')).float())
                d_l = torch.stack(d_l).to(f_l.device)
                F += (2*l + 1) / (8*torch.pi) * torch.vmap(torch.trace)(f_l * d_l.transpose(2,1))
                li += ld**2
        else:
            F = torch.zeros((B*self.num_grid_points)).to(f.device)
            li = 0
            for l in range(self.frequency+1):
                ld = 2*l+1
                d_l = self.D[l].unsqueeze(0).repeat(B,1,1,1).to(f.device)
                f_l = f[:,li:li+ld**2].view(B, 1, ld, ld)
                f_l = f_l.repeat(1,self.num_grid_points, 1, 1)
                F += (2*l + 1) / (8*torch.pi) * torch.vmap(torch.trace)(f_l.view(B*self.num_grid_points,ld,ld) * d_l.view(B*self.num_grid_points,ld,ld).transpose(2,1))
                li += ld**2
            F = F.view(B, self.num_grid_points)

        return F
