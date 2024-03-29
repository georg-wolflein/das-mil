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
            nn.Softmax(dim=-2)
        )

    def forward(self, features):
        H = features  # BxNxL

        # Attention weights
        A = self.attention(H)  # BxNx1

        # Context vector (weighted average of the features)
        M = torch.sum(A * H, dim=-2)  # BxL

        self.A = A
        return M


class GatedAttention(nn.Module):
    """
    Implementation of the gated attention layer from the paper: "Attention-Based Deep Multiple Instance Learning", https://arxiv.org/pdf/1802.04712.pdf.
    Code is based on the implementation at https://github.com/AMLab-Amsterdam/AttentionDeepMIL.
    """

    def __init__(self, feature_size: int, hidden_dim: int, attention_heads: int = 1):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(feature_size, hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(feature_size, hidden_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(hidden_dim, attention_heads)

    def forward(self, features):
        H = features  # NxL

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U)  # NxK
        A = A.transpose(-2, -1)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N # KxN

        M = A @ H  # KxL
        M = rearrange(M, "... k l -> ... (k l)")  # KL
        self.A = A
        return M
