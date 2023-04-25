import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import torch_geometric as pyg
import networkx as nx
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform
from torch_geometric.datasets import TUDataset
import torch_geometric as pyg
import math
from einops import einsum
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from mil.data.dataset import Datasets
from mil.data.mnist import Bag, MNISTBags, OneHotMNISTBags, MNISTCollage, OneHotMNISTCollage, BagLabelComputer, TargetNumbersBagLabelComputer, DistanceBagLabelComputer, DistanceBasedTargetNumbersBagLabelComputer
from mil.utils import device, detach, human_format
from mil.utils.bag import BagToTorchGeometric
from mil.utils.visualize import print_one_hot_bag_with_attention, print_one_hot_bag, plot_attention_head, plot_bag, plot_one_hot_collage
from mil.utils.stats import print_prediction_stats
from mil.utils.layers import Aggregate, Select, SqueezeUnsqueeze
from mil.models import MILModel
from mil.models.simple import CNNFeatureExtractor, Classifier
from mil.models.gnn import GNN
from mil.models.positional_encoding import FourierPositionalEncodingLayer, AxialPositionalEncodingLayer
from mil.models.abmil import WeightedAverageAttention
from mil.models.self_attention import MultiHeadAttention
from mil.models.set_transformer import SetTransformer, InducedSetTransformer, SAB, ISAB
from mil.models.distance_aware_self_attention import DistanceAwareSelfAttentionHead

RESULTS_FILE = "train.csv"

GlobalHydra().clear()
hydra.initialize(config_path="conf")
cfg = hydra.compose("config.yaml", overrides=[])
print(OmegaConf.to_yaml(cfg))
