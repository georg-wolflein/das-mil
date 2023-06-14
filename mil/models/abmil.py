from torch import nn
import torch.nn.functional as F
import torch
from einops import rearrange


class WeightedAverageAttention(nn.Module):
    """
    Implementation of the attention layer from the paper: "Attention-Based Deep Multiple Instance Learning", https://arxiv.org/pdf/1802.04712.pdf.

    The attention layer is a weighted average of the features, where the weights are calculated by a neural network.
    """

    def __init__(self, feature_size: int, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_size, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=-2),
        )

    def forward(self, features):
        H = features  # BxNxL

        # Attention weights
        A = self.attention(H)  # BxNx1

        attended_features = A * H  # BxNxL
        self.A = A

        # return torch.sum(attended_features, dim=-2)  # BxL
        return attended_features
