from torch import nn
import torch

"""Simple implementations of MIL feature extractors and classifiers."""


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


class Classifier(nn.Sequential):
    def __init__(self, feature_size: int, output_size: int = 1):
        super().__init__(
            nn.Linear(feature_size, output_size),
            nn.Sigmoid()
        )
