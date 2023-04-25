"""Implementation of absolute positional encoding schemes."""

import numpy as np
import torch
import torch.nn as nn
import math


class AxialPositionalEncodingLayer(nn.Module):

    def __init__(self, feature_size: int, dropout: float = 0.):
        """Layer that adds positional encoding to the input. Half of the feature size is used for the x-axis and the other half for the y-axis."""
        super().__init__()
        assert feature_size % 4 == 0
        self.div_term = torch.exp(torch.arange(
            0, feature_size, 4) * (-math.log(10000.0) / feature_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos):
        pos_x, pos_y = pos.moveaxis(-1, 0).unsqueeze(-1)
        div_term = self.div_term

        pe = torch.zeros_like(x)
        pe[..., 0::4] = torch.sin(pos_x * div_term)
        pe[..., 1::4] = torch.cos(pos_x * div_term)
        pe[..., 2::4] = torch.sin(pos_y * div_term)
        pe[..., 3::4] = torch.cos(pos_y * div_term)

        # pe = self.dropout(pe) # TODO: check if this is necessary

        x = x + pe

        return x


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, G: int, M: int, F_dim: int, H_dim: int, D: int, gamma: float):
        """
        Learnable Fourier Features from https://arxiv.org/pdf/2106.02795.pdf (Algorithm 1) (code adapted from https://github.com/willGuimont/learnable_fourier_positional_encoding)
        Implementation of Algorithm 1: Compute the Fourier feature positional encoding of a multi-dimensional position
        Computes the positional encoding of a tensor of shape [N, G, M]
        :param G: positional groups (positions in different groups are independent)
        :param M: each point has a M-dimensional positional values
        :param F_dim: depth of the Fourier feature dimension
        :param H_dim: hidden layer dimension
        :param D: positional encoding dimension
        :param gamma: parameter to initialize Wr
        """
        super().__init__()
        self.G = G
        self.M = M
        self.F_dim = F_dim
        self.H_dim = H_dim
        self.D = D
        self.gamma = gamma

        # Projection matrix on learned lines (used in eq. 2)
        self.Wr = nn.Linear(self.M, self.F_dim // 2, bias=False)
        # MLP (GeLU(F @ W1 + B1) @ W2 + B2 (eq. 6)
        self.mlp = nn.Sequential(
            nn.Linear(self.F_dim, self.H_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.H_dim, self.D // self.G)
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)

    def forward(self, x):
        """
        Produce positional encodings from x
        :param x: tensor of shape [N, G, M] that represents N positions where each position is in the shape of [G, M],
                  where G is the positional group and each group has M-dimensional positional values.
                  Positions in different positional groups are independent
        :return: positional encoding for X
        """
        N, G, M = x.shape
        # Step 1. Compute Fourier features (eq. 2)
        projected = self.Wr(x)
        cosines = torch.cos(projected)
        sines = torch.sin(projected)
        F = 1 / np.sqrt(self.F_dim) * torch.cat([cosines, sines], dim=-1)
        # Step 2. Compute projected Fourier features (eq. 6)
        Y = self.mlp(F)
        # Step 3. Reshape to x's shape
        PEx = Y.reshape((N, self.D))
        return PEx


class FourierPositionalEncodingLayer(nn.Module):
    """Layer that adds Fourier positional encoding to the input."""

    def __init__(self, feature_size: int, pos_dim: int = 2):
        super().__init__()
        self.enc = LearnableFourierPositionalEncoding(
            G=1,
            M=pos_dim,
            F_dim=16,
            H_dim=16,  # hidden layer size
            D=feature_size,
            gamma=10
        )

    def forward(self, x, pos):
        pos = pos.unsqueeze(-2)  # Nx1x2 (expand group dimension)
        enc = self.enc(pos)
        x = x + enc
        return x


if __name__ == '__main__':
    G = 1
    M = 2
    x = torch.randn((97, G, M))
    enc = LearnableFourierPositionalEncoding(G, M, 768, 32, 100, 10)
    pex = enc(x)
    print(pex.shape)
    print(sum(x.numel() for x in enc.parameters()))
