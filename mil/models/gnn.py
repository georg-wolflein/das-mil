import torch
import numpy as np
from torch import nn
import torch_geometric as pyg


class GNN(nn.Module):
    """GNN model.

    The layer parameter can be used to specify the type of GNN layer to use (e.g. pyg.nn.GCNConv, pyg.nn.GATConv, pyg.nn.DenseGCNConv).
    """

    def __init__(self, feature_size: int, hidden_dim: int, layer=pyg.nn.GCNConv):
        super().__init__()
        self.layer = layer
        self.gnn = pyg.nn.Sequential('x, connectivity', [
            (layer(feature_size, hidden_dim), 'x, connectivity -> x'),
            nn.ReLU(inplace=True),
            (layer(hidden_dim, feature_size), 'x, connectivity -> x'),
            nn.ReLU(inplace=True),
        ])

    def forward(self, data: pyg.data.Data):
        number_of_nodes = data.num_nodes

        # Assume complete graph
        # complete (directed) graph
        edge_index = np.array(nx.complete_graph(number_of_nodes).edges).T
        edge_index = np.concatenate(
            [edge_index, edge_index[::-1]], axis=-1)  # undirected graph

        # Use distance-based edge weights to drop out edges
        pos = data.pos
        if pos is not None:
            edge_weights = data.edge_attr.squeeze(-1)

            if False:  # NOTE (Georg): This is not working yet
                # normalise
                # NOTE (Georg): data.edge_attr is already normalised
                print(edge_weights)
                scaler = preprocessing.MinMaxScaler()
                edge_weights = scaler.fit_transform(
                    edge_weights.reshape(-1, 1)).reshape(1, -1)[0]
                print("Normalized ", edge_weights)

                # drop-out edge based on edge weights
                print(edge_index)
                tau = 0.5
                ei_1 = edge_index[0][(edge_weights < tau)]
                ei_2 = edge_index[1][(edge_weights < tau)]
                edge_index = np.array([ei_1, ei_2])

            # Assume fully connected graph
            if self.layer == pyg.nn.DenseGCNConv:
                # DenseGCNConv requires a dense adjacency matrix
                connectivity = torch.ones((number_of_nodes, number_of_nodes))
                connectivity = connectivity - \
                    torch.diag(torch.ones(number_of_nodes))
            else:
                connectivity = data.edge_index

        batch = torch.tensor([0] * number_of_nodes)
        x = self.gnn(data.x, connectivity)
        x = pyg.nn.global_max_pool(x, batch)
        return x
