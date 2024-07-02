import torch
from torch import nn

class HarmonicFunction(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def generate_baseis_fns(self) -> torch.Tensor:
        raise NotImplementedError

    def evaluate(self, coeffs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
