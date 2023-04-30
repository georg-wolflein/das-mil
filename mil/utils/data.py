import torch
from torch_geometric.data import Data
from torch_geometric import transforms


class FullyConnectedGraphTransform(transforms.BaseTransform):
    def __call__(self, data: Data):
        if data.edge_index is not None:
            return data
        # Make fully connected graph
        items = data.x if data.x is not None else data.instance_labels
        n = items.shape[0]
        x, y = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
        edge_index = torch.stack([x.flatten(), y.flatten()], dim=0)
        # Remove self-loops
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
        data.edge_index = edge_index
        return data
