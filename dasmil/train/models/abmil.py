import torch
from torch import nn
from omegaconf import ListConfig
from typing import Optional
import torch.nn.functional as F


class AttentionMIL(nn.Module):
    """
    Implementation of the attention layer from the paper: "Attention-Based Deep Multiple Instance Learning", https://arxiv.org/pdf/1802.04712.pdf.

    The attention layer is a weighted average of the features, where the weights are calculated by a neural network.
    """

    def __init__(
        self, targets: ListConfig, d_features: int, hidden_dim: int, dropout: float = 0.5, batchnorm: bool = False
    ):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d_features, hidden_dim), nn.ReLU())
        self.attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))
        self.pre_head = (
            nn.Sequential(nn.BatchNorm1d(hidden_dim), nn.Dropout(dropout)) if batchnorm else nn.Dropout(dropout)
        )
        self.heads = nn.ModuleDict(
            {
                target.column: nn.Linear(
                    in_features=hidden_dim,
                    out_features=len(target.classes) if target.type == "categorical" else 1,
                )
                for target in targets
            }
        )
        self.targets = targets

    def forward(self, feats, coords, mask, *args, **kwargs):
        embeddings = self.encoder(feats)  # B, N, D
        attention = self.attention(embeddings).squeeze(-1)  # B, N
        attention = torch.masked_fill(attention, ~mask, -torch.inf)  # B, N
        attention = F.softmax(attention, dim=-1)  # B, N
        embeddings = embeddings * attention.unsqueeze(-1)  # B, N, D
        slide_tokens = embeddings.sum(dim=-2)  # B, D
        slide_tokens = self.pre_head(slide_tokens)  # B, D

        # Apply the corresponding head to each slide-level token
        logits = {target.column: self.heads[target.column](slide_tokens).squeeze(-1) for target in self.targets}
        return logits
