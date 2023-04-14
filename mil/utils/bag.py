import torch
from torch import nn
import torch_geometric as pyg
from torch_geometric import transforms

from mil.data.mnist import Bag


class BagToTorchGeometric(nn.Module):
    def __init__(self, collage_size: float = 0., compute_edge_index: bool = True, compute_edge_attr: bool = True):
        super().__init__()
        self.compute_edge_index = compute_edge_index
        if compute_edge_attr:
            # Distance transform is used to compute distances between instances, which are saved as edge_attr
            self.T = transforms.Distance(norm=collage_size is not None,
                                         max_value=float(collage_size),
                                         cat=False)
        else:
            self.T = transforms.Compose([])  # noop

    def forward(self, bag: Bag, features: torch.Tensor) -> pyg.data.Data:
        kwargs = {}

        # Compute edge_index of complete graph
        if self.compute_edge_index:
            n = features.shape[0]
            x, y = torch.meshgrid(torch.arange(n), torch.arange(n))
            edge_index = torch.stack([x.flatten(), y.flatten()], dim=0)
            kwargs["edge_index"] = edge_index

        # Copy other attributes from bag
        for attr in ("y", "instances", "pos", "instance_labels", "key_instances"):
            if hasattr(bag, attr):
                kwargs[attr] = getattr(bag, attr)

        data = pyg.data.Data(x=features,
                             **kwargs)
        if data.pos is not None:
            data.pos = data.pos.to(torch.float32)

        return self.T(data)  # computes edge_attr if needed
