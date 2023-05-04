from torch import nn
import typing
import torch

from mil.utils import identity


class MILModel(nn.Module):
    """Structure of a multiple instance learning model.

    The model consists of three parts:
    1. A feature extractor that takes a bag and returns a Z-dimensional feature vector for each instance. A bag with N instances will thus be represented by a N x Z matrix.
    2. A pooling function that takes the feature matrix and returns a single feature vector for the bag. (N x Z) -> (1 x Z)
    3. A classifier that takes a bag and returns a single scalar value. (1 x Z) -> (1 x 1)
    """

    def __init__(self,
                 feature_extractor: nn.Module,
                 pooler: nn.Module,
                 classifier: nn.Module,
                 logit_to_prob: typing.Callable[[torch.Tensor], torch.Tensor] = identity):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.pooler = pooler
        self.classifier = classifier
        self.logit_to_prob = logit_to_prob

    def forward(self, x, edge_index, edge_attr, pos):
        features = self.feature_extractor(x)
        pooled = self.pooler(features, edge_index, edge_attr, pos)
        logit = self.classifier(pooled).squeeze(-1)
        prob = self.logit_to_prob(logit)
        return prob, logit
