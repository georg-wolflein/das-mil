from torch import nn
import torch.nn.functional as F
import torch
from einops.layers.torch import Rearrange
from einops import rearrange, einsum


class WeightedAverageAttention(nn.Module):
    """
    Implementation of the attention layer from the paper: "Feed-Forward Networks with Attention Can Solve Some Long-Term Memory Problems", https://arxiv.org/pdf/1512.08756.pdf.

    The attention layer is a weighted average of the features, where the weights are calculated by a neural network.
    The attention_heads parameter was added by me (not in the paper).
    """

    def __init__(self, feature_size: int, hidden_dim: int, attention_heads: int = 1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_size, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, attention_heads),
            Rearrange("... n k -> ... k n"),
            nn.Softmax(dim=-1),
        )

    def forward(self, features):
        H = features  # BxNxL
        A = self.attention(features)  # BxKxN
        M = einsum(A, H, "... k n, ... n l -> ... k l")  # BxKxL
        M = rearrange(M, "... k l -> ... (k l)")  # Bx(KL)
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


class AttentionHead(nn.Module):
    def __init__(self, feature_size: int, hidden_dim: int):
        super().__init__()
        self.keys = nn.Linear(feature_size, hidden_dim, bias=False)
        self.queries = nn.Linear(feature_size, hidden_dim, bias=False)
        self.values = nn.Linear(feature_size, feature_size, bias=False)

    def forward(self, features):
        H = features  # NxL
        L = features.shape[-1]

        k = self.keys(H)  # NxD
        q = self.queries(H)  # NxD
        A = q @ k.transpose(-2, -1)  # NxN
        A = A / L**.5  # scale by sqrt(L)
        A = F.softmax(A, dim=-1)  # softmax over N # NxN

        v = self.values(H)  # NxL
        M = A @ v
        return M


class MultiHeadAttention(nn.Module):
    def __init__(self, feature_size: int, hidden_dim: int, attention_heads: int = 1):
        super().__init__()
        self.attention_heads = nn.ModuleList(
            [AttentionHead(feature_size, hidden_dim) for _ in range(attention_heads)])

    def forward(self, features):
        return torch.cat([head(features) for head in self.attention_heads], dim=-1)
