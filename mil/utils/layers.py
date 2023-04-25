from torch import nn
import torch


class Aggregate(nn.Module):
    """Simple pooling layer for mean/max pooling."""

    def __init__(self, pool: str = "mean", dim: int = 0):
        super().__init__()
        self.pool = pool
        self.dim = dim

    def forward(self, x):
        pool = getattr(torch, self.pool)
        result = pool(x, dim=self.dim)
        if self.pool == "max":
            result = result.values
        return result


class Select(nn.Module):
    """Simple layer for selecting a single attribute, i.e. selecting the `x` attribute from a `Data` object."""

    def __init__(self, attr: str):
        super().__init__()
        self.attr = attr

    def forward(self, x):
        return getattr(x, self.attr)


class SqueezeUnsqueeze(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x.unsqueeze(0)).squeeze(0)
