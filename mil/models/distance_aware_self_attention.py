import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric as pyg


class EmbeddingTable(nn.Module):
    """Embedding table with trainable embeddings."""

    def __init__(self, dim, num_embeddings=2, trainable: bool = True):
        super().__init__()
        self.embeddings = torch.empty(num_embeddings, dim)
        if trainable:
            self.embeddings = nn.Parameter(self.embeddings)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.normal_(self.embeddings, mean=0, std=1)

    def forward(self, weights):
        weights  # Nxnum_embeddings
        embeddings = self.embeddings  # num_embeddings x dim
        # embeddings = embeddings * torch.tensor([0, 1]).unsqueeze(-1)
        X = weights @ embeddings  # Nxdim
        return X


class ContinuousEmbeddingIndex(nn.Module):
    """Provides an embedding index for continuous values using interpolation via a sigmoid."""

    def __init__(self, num_embeddings=2):
        super().__init__()
        assert num_embeddings == 2
        self.bias = nn.Parameter(torch.empty(1))
        self.multiplier = nn.Parameter(torch.empty(1))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.bias.data.fill_(.5)
        self.multiplier.data.fill_(10)  # TODO: test different initializations

    def forward(self, x):
        x  # num_edges x 1
        x = torch.sigmoid((x - self.bias) * self.multiplier)
        x = torch.cat([x, 1 - x], dim=-1)
        return x


class DiscreteEmbeddingIndex(nn.Module):
    """Provides an embedding index for discrete values using one-hot encoding."""

    def __init__(self, num_embeddings=2):
        super().__init__()
        self.num_embeddings = num_embeddings

    def forward(self, x):
        x  # num_edges x 1
        x = x * (self.num_embeddings - 1)  # assume 0 <= x <= 1
        x = torch.round(x)
        x = torch.clamp(x, 0, self.num_embeddings - 1)
        return F.one_hot(x.long().squeeze(-1), self.num_embeddings).float()


class DistanceAwareSelfAttentionHead(nn.Module):
    def __init__(self, feature_size: int, hidden_dim: int, num_embeddings: int = 2, continuous: bool = True):
        super().__init__()
        self.keys = nn.Linear(feature_size, hidden_dim, bias=False)
        self.queries = nn.Linear(feature_size, hidden_dim, bias=False)
        self.values = nn.Linear(feature_size, feature_size, bias=False)

        self.embed_k = EmbeddingTable(
            hidden_dim, num_embeddings=num_embeddings)
        self.embed_q = EmbeddingTable(
            hidden_dim, num_embeddings=num_embeddings)
        self.embed_v = EmbeddingTable(
            feature_size, num_embeddings=num_embeddings)

        EmbeddingIndex = ContinuousEmbeddingIndex if continuous else DiscreteEmbeddingIndex
        self.index_k = EmbeddingIndex(num_embeddings=num_embeddings)
        # self.index_q = EmbeddingIndex(num_embeddings=num_embeddings)
        # self.index_v = EmbeddingIndex(num_embeddings=num_embeddings)

        self.dropout = nn.Dropout(.1)

    def forward(self, data: pyg.data.Data):
        self.data = data  # retain for visualization
        features = data.x
        N = features.shape[0]
        H = features  # NxL
        L = features.shape[-1]

        rk = self.embed_k(self.index_k(data.edge_attr))  # num_edges x D
        rq = self.embed_q(self.index_k(data.edge_attr))  # num_edges x D
        rv = self.embed_v(self.index_k(data.edge_attr))  # num_edges x L

        Rk = pyg.utils.to_dense_adj(
            data.edge_index, edge_attr=rk, max_num_nodes=data.num_nodes).squeeze(0)  # NxNxD
        Rq = pyg.utils.to_dense_adj(
            data.edge_index, edge_attr=rq, max_num_nodes=data.num_nodes).squeeze(0)  # NxNxD
        Rv = pyg.utils.to_dense_adj(
            data.edge_index, edge_attr=rv, max_num_nodes=data.num_nodes).squeeze(0)  # NxNxL

        self.Rk = Rk
        self.Rq = Rq
        self.Rv = Rv

        # Compute key, query, value vectors
        k = self.keys(H)  # NxD
        q = self.queries(H)  # NxD
        v = self.values(H)  # NxL

        # Compute attention scores (dot product) from classic self-attention
        A = q @ k.transpose(-2, -1)  # NxN
        self.A0 = A

        # Compute additional distance-aware terms for keys/queries

        # Term 1
        # TODO: check if this is the correct dimension
        q_repeat = q.unsqueeze(1).repeat(1, N, 1)  # NxNxD
        A = A + (q_repeat * Rk).sum(axis=-1)  # NxN

        # Term 2
        # TODO: check if this is the correct dimension
        k_repeat = k.unsqueeze(0).repeat(N, 1, 1)  # NxNxD
        A = A + (k_repeat * Rq).sum(axis=-1)  # NxN

        # Term 3
        A = A + (q_repeat * k_repeat).sum(axis=-1)  # NxN

        # Scale by sqrt(L)
        A = A / L**.5

        # Softmax over N
        A = F.softmax(A, dim=-1)  # NxN

        # Retain attention weights for visualization
        self.A = A

        # Apply dropout
        A = self.dropout(A)

        # Apply attention weights to values
        M = A @ v  # NxL

        # Infuse distance-aware term in the value computation
        # TODO: check if this is the correct dimension
        M = M + (A.unsqueeze(-1) * Rv).sum(axis=-2)  # NxL

        data.update(dict(x=M))  # NOTE: modifies data in-place
        return data
