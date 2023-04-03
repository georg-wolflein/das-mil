import torch
from torch import nn
import torch_geometric as pyg
from scipy.spatial.distance import pdist, squareform

from mil.data.mnist import Bag


def bag_to_torch_geometric(bag: Bag, features: torch.Tensor,
                           collage_size: float = 0.,  # if 0, then edge_attr isn't normalized
                           compute_edge_index: bool = True,
                           compute_edge_attr: bool = True) -> pyg.data.Data:
    kwargs = {}

    # Compute edge_index of complete graph
    if compute_edge_index:
        n = features.shape[0]
        x, y = torch.meshgrid(torch.arange(n), torch.arange(n))
        edge_index = torch.stack([x.flatten(), y.flatten()], dim=0)
        kwargs["edge_index"] = edge_index

    # Calculate distances between nodes
    if compute_edge_attr:
        dist = squareform(pdist(bag.pos, metric="euclidean"))
        dist = torch.from_numpy(dist).float()  # NxN
        if collage_size != 0.:
            dist /= (2**.5 * collage_size)  # normalize based on collage size
        kwargs["edge_attr"] = dist.flatten().reshape(-1, 1)

    # Copy other attributes from bag
    for attr in ("bag_label", "instances", "pos", "instance_labels", "key_instances"):
        if hasattr(bag, attr):
            kwargs[attr] = getattr(bag, attr)

    return pyg.data.Data(x=features,
                         y=bag.bag_label,
                         **kwargs,
                         fully_connected=True)


class BagToTorchGeometric(nn.Module):
    def __init__(self, collage_size: float = 0., compute_edge_index: bool = True, compute_edge_attr: bool = True):
        super().__init__()
        self.collage_size = collage_size
        self.compute_edge_index = compute_edge_index
        self.compute_edge_attr = compute_edge_attr

    def forward(self, bag: Bag, features: torch.Tensor) -> pyg.data.Data:
        return bag_to_torch_geometric(bag, features,
                                      collage_size=self.collage_size,
                                      compute_edge_index=self.compute_edge_index,
                                      compute_edge_attr=self.compute_edge_attr)
