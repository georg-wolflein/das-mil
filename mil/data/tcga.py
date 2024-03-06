import numpy as np
import torch
import torch.utils.data as data_utils
import math
from torch_geometric.data import Data
from torch_geometric import transforms
from pathlib import Path
import h5py
import pandas as pd

from .camelyon16 import _Dataset


class TCGADataset(_Dataset):
    def __init__(self, cache_dir: str, train: bool = True, max_patches_per_bag: int = None, csv_file: str = None):
        super().__init__(max_patches_per_bag) # TODO:, max_dist=
        cache_dir = Path(cache_dir)

        df = pd.read_csv(csv_file)
        df = df[df["train"] == train]
        self.bags = []

        for _, row in df.iterrows():
            bag_label = bool(row["label"])
            h5file = cache_dir / row["filename"]
            self.bags.append((bag_label, h5file))
