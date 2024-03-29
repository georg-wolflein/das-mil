import numpy as np
import torch
import torch.utils.data as data_utils
import math
from torch_geometric.data import Data
from torch_geometric import transforms
from pathlib import Path
import h5py
import pandas as pd

from mil.utils.data import FullyConnectedGraphTransform

# Exclude erroneous files from the dataset (https://arxiv.org/pdf/1703.02442v2.pdf)
EXCLUDED_FILES = {
    "normal_144",
    "normal_086",
    "test_049"
}


class _Dataset(data_utils.Dataset):
    def __init__(self, max_patches_per_bag: int = None):
        self.max_patches_per_bag = max_patches_per_bag
        self.bags = []
        self.T = transforms.Compose([
            FullyConnectedGraphTransform(),
            transforms.Distance(norm=True,
                                max_value=float(
                                    46816 * math.sqrt(2.)),  # TODO: set this value correctly
                                cat=False)
        ])

    def __len__(self) -> int:
        return len(self.bags)

    @torch.no_grad()
    def __getitem__(self, index: int) -> Data:
        bag_label, h5file = self.bags[index]
        with h5py.File(h5file, "r") as f:
            # https://github.com/pytorch/pytorch/issues/28761
            features = torch.tensor(np.array(f["feats"]))
            coords = torch.tensor(np.array(f["coords"]))
        n = features.shape[0]
        if self.max_patches_per_bag is not None and n > self.max_patches_per_bag:
            indices = torch.randperm(n)[:self.max_patches_per_bag]
            features = features[indices]
            coords = coords[indices] + (224 // 2)  # center coordinates
        data = Data(y=torch.tensor(bag_label).float(),
                    x=features.float(),
                    pos=coords.float())
        return self.T(data)

    @torch.no_grad()
    def fake_bag(self, n: int = None) -> Data:
        if n is None:
            n = self.max_patches_per_bag
        feature_size = self[0].x.shape[1]
        data = Data(y=torch.tensor(0.).float(),
                    x=torch.rand(n, feature_size).float(),
                    pos=torch.rand(n, 2).float())
        return self.T(data)

    @property
    def num_positives(self) -> int:
        return sum(1 for bag_label, _ in self.bags if bag_label)

    @property
    def num_negatives(self) -> int:
        return sum(1 for bag_label, _ in self.bags if not bag_label)


class Camelyon16Dataset(_Dataset):
    def __init__(self, cache_dir: str, train: bool = True, max_patches_per_bag: int = None, reference_csv_file: str = None):
        super().__init__(max_patches_per_bag)
        if train:
            self._load_train_bags(cache_dir)
        else:
            self._load_test_bags(cache_dir, reference_csv_file)

    def _load_train_bags(self, cache_dir: str):
        cache_dir = Path(cache_dir)
        self.bags = []
        for h5file in cache_dir.glob("*.h5"):
            if h5file.stem in EXCLUDED_FILES:
                continue
            if h5file.name.startswith("tumor") or h5file.name.startswith("normal"):
                bag_label = h5file.name.startswith("tumor")
                self.bags.append((bag_label, h5file))

    def _load_test_bags(self, cache_dir: str, reference_csv_file: str):
        cache_dir = Path(cache_dir)
        reference_csv_file = Path(reference_csv_file)
        self.bags = []
        df = pd.read_csv(reference_csv_file, header=None)
        bag_labels = {
            row[0]: row[1].lower() == "tumor" for _, row in df.iterrows()
        }
        for h5file in cache_dir.glob("*.h5"):
            if h5file.stem in EXCLUDED_FILES:
                continue
            if h5file.name.startswith("test") and h5file.stem in bag_labels:
                bag_label = bag_labels[h5file.stem]
                self.bags.append((bag_label, h5file))
