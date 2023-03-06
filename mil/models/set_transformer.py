"""Implementation of the Set Transformer from the paper: "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks", https://arxiv.org/abs/1810.00825.

The code is adapted from their official implementation at https://github.com/juho-lee/set_transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128, num_heads=4):
        super().__init__()
        self.enc = nn.Sequential(
            SAB(dim_input, dim_hidden, num_heads),
            SAB(dim_hidden, dim_hidden, num_heads))
        self.dec = nn.Sequential(
            nn.Dropout(),
            PMA(dim_hidden, num_heads, num_outputs),
            nn.Dropout(),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, X):
        return self.dec(self.enc(X.unsqueeze(0))).squeeze(0).squeeze(0)


class InducedSetTransformer(SetTransformer):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128, num_heads=4, num_inds=32):
        super().__init__(dim_input, num_outputs, dim_output, dim_hidden, num_heads)
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds))


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O + F.relu(self.fc_o(O))
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)
