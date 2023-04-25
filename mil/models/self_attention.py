from torch import nn
import torch.nn.functional as F
import torch


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
        self.A = A
        return M


class MultiHeadAttention(nn.Module):
    def __init__(self, feature_size: int, hidden_dim: int, attention_heads: int = 1):
        super().__init__()
        self.attention_heads = nn.ModuleList(
            [AttentionHead(feature_size, hidden_dim) for _ in range(attention_heads)])

    def forward(self, features):
        return torch.cat([head(features) for head in self.attention_heads], dim=-1)

    @property
    def A(self):
        return torch.stack([head.A for head in self.attention_heads], dim=0)
