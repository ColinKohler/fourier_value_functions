from escnn.group import *
from escnn.gspaces import *
from escnn.nn import FieldType
from escnn.nn import GeometricTensor
from escnn.nn.modules.equivariant_module import EquivariantModule

import torch
import torch.nn.functional as F

from typing import List, Tuple, Any

import numpy as np

def _build_kernel(G: Group, irrep: List[tuple]):
    kernel = []

    for irr in irrep:
        irr = G.irrep(*irr)

        c = int(irr.size//irr.sum_of_squares_constituents)
        k = irr(G.identity)[:, :c] * np.sqrt(irr.size)
        kernel.append(k.T.reshape(-1))

    kernel = np.concatenate(kernel)
    return kernel


class Fourier(EquivariantModule):
    def __init__(
            self,
            gspace: GSpace,
            channels: int,
            irreps: List,
            *grid_args,
            **grid_kwargs
    ):
        assert isinstance(gspace, GSpace)

        super().__init__()

        self.space = gspace

        G: Group = gspace.fibergroup

        self.rho = G.spectral_regular_representation(*irreps, name=None)
        self.out_type = FieldType(self.space, [self.rho] * channels)

        kernel = _build_kernel(G, irreps)
        assert kernel.shape[0] == self.rho.size
        kernel = kernel / np.linalg.norm(kernel)
        kernel = kernel.reshape(-1, 1)

        grid = G.grid(*grid_args, **grid_kwargs)
        A = np.concatenate(
            [
                self.rho(g) @ kernel
                for g in grid
            ], axis=1
        ).T

        eps = 1e-8
        Ainv = np.linalg.inv(A.T @ A + eps * np.eye(self.rho.size)) @ A.T

        self.register_buffer('Ainv', torch.tensor(Ainv, dtype=torch.get_default_dtype()))

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        x_hat = torch.einsum('bcg...,fg->bcf...', x, self.Ainv)
        x_hat = x_hat.reshape(x.shape[0], self.out_type.size)

        return GeometricTensor(x_hat, self.out_type)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        assert len(input_shape) >= 2
        assert input_shape[1] == self.in_type.size

        b, c = input_shape[:2]
        spatial_shape = input_shape[2:]

        return (b, self.out_type.size, *spatial_shape)
