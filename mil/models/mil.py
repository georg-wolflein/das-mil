from torch import nn

from mil.data.mnist import Bag


class MILModel(nn.Module):
    """Structure of a multiple instance learning model.

    The model consists of three parts:
    1. A feature extractor that takes a bag and returns a Z-dimensional feature vector for each instance. A bag with N instances will thus be represented by a N x Z matrix.
    2. A pooling function that takes the feature matrix and returns a single feature vector for the bag. (N x Z) -> (1 x Z)
    3. A classifier that takes a bag and returns a single scalar value. (1 x Z) -> (1 x 1)
    """

    def __init__(self, feature_extractor, pooler, classifier):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.pooler = pooler
        self.classifier = classifier

    def forward(self, bag: Bag):
        features = self.feature_extractor(bag)
        pooled_features = self.pooler(bag, features)
        logits = self.classifier(pooled_features)
        return logits
