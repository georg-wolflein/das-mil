from torch import nn
import torch


class Aggregate(nn.Module):
    """Simple pooling layer for mean/max pooling."""

    def __init__(self, agg: str = "mean", dim: int = 0):
        super().__init__()
        self.agg = agg
        self.dim = dim

    def forward(self, x):
        pool = getattr(torch, self.agg)
        result = pool(x, dim=self.dim)
        if self.agg == "max":
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


class CNNFeatureExtractor(nn.Module):
    def __init__(self, feature_size: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(.1),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(-3, -1),
            nn.Dropout(.5),
            nn.Linear(20 * 4 * 4, feature_size),
            nn.ReLU()
        )

    def forward(self, instances: torch.Tensor):
        return self.cnn(instances)
