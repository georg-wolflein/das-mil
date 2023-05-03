"""Implementation of the GNN from the paper: "Multiple instance learning with graph neural networks", https://arxiv.org/abs/1906.04881.

The code is adapted from their official implementation at https://github.com/juho-lee/set_transformer.
"""



import torch
import numpy as np
from torch import nn
import torch_geometric as pyg

from torch_geometric.nn import dense_diff_pool, DenseSAGEConv, global_max_pool
import torch.nn.functional as F



class GNN(nn.Module):
    """Naive GNN model.

    The layer parameter can be used to specify the type of GNN layer to use (e.g. pyg.nn.GCNConv, pyg.nn.GATConv, pyg.nn.DenseGCNConv).
    """

    def __init__(self, feature_size: int, hidden_dim: int, layer=pyg.nn.GCNConv, pooling_layer=pyg.nn.global_max_pool, num_layers=2):
        super().__init__()
        self.layer = layer
        self.pooling_layer = pooling_layer

        if isinstance(self.layer, pyg.nn.GCNConv):
            input_str = 'x, edge_index'
        else: 
            input_str = 'x, edge_index, edge_attr'

        layer_list = []

        for i in range(num_layers):
            in_dim, out_dim = hidden_dim, hidden_dim
            if i == 0:
                in_dim = feature_size

            layer_list.append((layer(in_dim, out_dim), f'{input_str} -> x'))
            layer_list.append(nn.ReLU(inplace=True))
        
        self.gnn = pyg.nn.Sequential(input_str, layer_list)

    def forward(self, features, edge_index, edge_attr):
        if isinstance(self.layer, pyg.nn.GCNConv):
            x = self.gnn(features, edge_index)
        else:
            x = self.gnn(features, edge_index, edge_attr)

        batch = torch.tensor([0] * len(features))
        x = self.pooling_layer(x, batch)
        
        return x

class MIL_GNN(nn.Module):
    """Implementation of the GNN model from https://arxiv.org/pdf/1906.04881.pdf.
    """

    def __init__(self, feature_size: int, hidden_dim: int):
        super().__init__()
        self.num_clusters = 1
        self.num_classes = 2

        self.gnn_embd = DenseSAGEConv(feature_size, hidden_dim)
        self.gnn_embd2 = DenseSAGEConv(hidden_dim, hidden_dim)

        self.gnn_pool = DenseSAGEConv(hidden_dim, self.num_clusters)
        self.mlp = nn.Linear(self.num_clusters, self.num_clusters, bias=True)
        
        self.path1_lin1 = nn.Linear(hidden_dim, hidden_dim, bias=True) 
        self.path1_lin2 = nn.Linear(hidden_dim, self.num_classes, bias=True)

        self.gnn_embd3 = DenseSAGEConv(hidden_dim, hidden_dim)
        self.path2_lin1 = nn.Linear(hidden_dim, hidden_dim, bias=True) 
        self.path2_lin2 = nn.Linear(hidden_dim, self.num_classes, bias=True)
        

    def forward(self, features, edge_index, edge_attr):
        adj = pyg.utils.to_dense_adj(edge_index)
        self.adj = torch.clone(adj)

        # main part of GNN
        # GNN_embed1
        x = F.leaky_relu(self.gnn_embd(features, adj), negative_slope=0.01)
        self.path1_in = torch.clone(x)
        loss_emb1 = self.auxiliary_loss(x, adj)

        # GNN_cluster
        c = F.leaky_relu(self.gnn_pool(x, adj), negative_slope=0.01)
        c = F.leaky_relu(self.mlp(c), negative_slope=0.01)

        # Coarsened graph   
        x, adj, loss, _ = dense_diff_pool(x, adj, c)
        self.path2_in = torch.clone(x)

        # GNN_embed2
        x = F.leaky_relu(self.gnn_embd2(x, adj), negative_slope=0.01) # [C, 500]
        loss_emb2 = self.auxiliary_loss(x, adj)

        # Concat
        x = x.view(1, -1)

        self.additional_loss = loss_emb1 + loss + loss_emb2

        return x
    
    def run_deep_supervision(self):
        # path 1
        x1 = global_max_pool(self.path1_in, torch.zeros(len(x1)))
        # MLP
        x1 = F.leaky_relu(self.path1_lin1(x1), 0.01)
        x1 = F.leaky_relu(self.path1_lin2(x1), 0.01)
        pred1 = F.softmax(x1.squeeze(), dim=0)

        # path 2
        x1 = F.leaky_relu(self.gnn_embd3(self.path2_in, self.adj), negative_slope=0.01)
        # MLP
        x1 = F.leaky_relu(self.path2_lin1(x1), 0.01)
        x1 = F.leaky_relu(self.path2_lin2(x1), 0.01)
        pred2 = F.softmax(x1.squeeze(), dim=0)

        return pred1, pred2
    
    def auxiliary_loss(self, x, adj):
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        x = x.unsqueeze(0) if x.dim() == 2 else x
        x = torch.softmax(x, dim=-1)

        link_loss = torch.norm(adj - torch.matmul(x, x.transpose(1, 2)), p=2) / adj.numel()
        return link_loss

