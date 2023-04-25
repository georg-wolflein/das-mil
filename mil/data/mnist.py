import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms as T
import typing
import abc
import itertools
import math
from torch_geometric.data import Data
from torch_geometric import transforms


class FullyConnectedGraphTransform(transforms.BaseTransform):
    def __call__(self, data: Data):
        if data.edge_index is not None:
            return data
        # Make fully connected graph
        items = data.x if data.x is not None else data.instance_labels
        n = items.shape[0]
        x, y = torch.meshgrid(torch.arange(n), torch.arange(n))
        edge_index = torch.stack([x.flatten(), y.flatten()], dim=0)
        # Remove self-loops
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
        data.edge_index = edge_index
        return data


class BagLabelComputer(abc.ABC):
    @abc.abstractmethod
    def compute_bag_label(self, instance_labels: torch.Tensor, pos: torch.Tensor = None) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Computes bag label and key instances from instance labels.

        Args:
            instance_labels (torch.Tensor): instance labels of shape (num_instances,).
            pos (torch.Tensor, optional): instance positions of shape (num_instances, 2). Defaults to None.

        Returns:
            typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: bag label, key instances of shape (num_instances,), key cliques of shape (num_key_instances, clique_size), where clique_size is usually 2.
        """
        pass


class TargetNumbersBagLabelComputer(BagLabelComputer):
    def __init__(self, target_numbers: typing.Union[int, typing.Tuple[int]], min_instances_per_target: int = 1):
        self.target_numbers = torch.tensor(target_numbers, requires_grad=False)
        if len(self.target_numbers.shape) == 0:
            self.target_numbers = self.target_numbers.unsqueeze(0)
        self.min_instances_per_target = min_instances_per_target

    def compute_bag_label(self, instance_labels, *args, **kwargs):
        instance_labels = instance_labels == self.target_numbers.unsqueeze(-1)
        key_instances = instance_labels.any(axis=0)
        count_per_target = instance_labels.sum(axis=-1)
        bag_label = (count_per_target >=
                     self.min_instances_per_target).all()
        if not bag_label:
            key_instances = torch.zeros_like(key_instances, dtype=bool)
            key_cliques = set()
        else:
            # Compute key cliques
            target_instances = {
                target: tuple(torch.argwhere(
                    instance_labels == target).cpu().detach().numpy().tolist())
                for target in self.target_numbers
            }
            groups = [list(itertools.combinations(instances, r))
                      for target, instances in target_instances.items()
                      for r in range(self.min_instances_per_target, len(instances) + 1)]
            key_cliques = set(set())
            for group in groups:
                prev_key_cliques = key_cliques
                key_cliques = set()
                for val in group:
                    for clique in prev_key_cliques:
                        key_cliques.add(sorted((*clique, val)))

        return bag_label.float(), key_instances, key_cliques


class DistanceBagLabelComputer(BagLabelComputer):
    def __init__(self, predicate: typing.Callable[[int, int, int], bool]):
        self.predicate = predicate

    def _key_instances(self, instance_labels, pos):
        nodes = [(i, {"label": label, "loc": loc}) for i, (label, loc)
                 in enumerate(zip(instance_labels, pos))]

        key_cliques = set()
        for ia, a in nodes:
            for ib, b in nodes:
                if ia >= ib:
                    continue
                dist = np.linalg.norm(a["loc"] - b["loc"])
                if self.predicate(a["label"], b["label"], dist):
                    # if dist < 50 and a["label"] in self.target_numbers and b["label"] in self.target_numbers and a["label"] != b["label"]:
                    key_cliques.add(tuple(sorted((ia, ib))))
        return np.array(sorted(set(itertools.chain.from_iterable(key_cliques)))), key_cliques

    def compute_bag_label(self, instance_labels, pos, *args, **kwargs):
        ki, key_cliques = self._key_instances(instance_labels, pos)
        bag_label = torch.tensor(len(ki) > 0)
        # Convert key instances to mask
        key_instances = torch.zeros_like(instance_labels, dtype=bool)
        key_instances[list(ki)] = True
        return bag_label.float(), key_instances, key_cliques


class DistanceBasedTargetNumbersBagLabelComputer(DistanceBagLabelComputer):
    def __init__(self, target_numbers: typing.Tuple[int], dist_predicate: typing.Callable[[int], bool]):
        super().__init__(lambda a, b, dist:
                         dist_predicate(dist)
                         and a in target_numbers
                         and b in target_numbers
                         and a != b)


class Digits(data_utils.Dataset):

    @torch.no_grad()
    def __init__(self, bag_label_computer: BagLabelComputer, num_digits: int = 10, min_bag_size: int = 2, mean_bag_size: int = 10, var_bag_size: int = 2, num_bags: int = 250, seed: int = 1, train: bool = True):
        self.compute_bag_label = bag_label_computer.compute_bag_label
        self.num_digits = num_digits
        self.mean_bag_size = mean_bag_size
        self.var_bag_size = var_bag_size
        self.num_bags = num_bags
        self.train = train
        self.min_bag_size = min_bag_size
        self.r = np.random.RandomState(seed)
        self.T = FullyConnectedGraphTransform()

        positive_bags = []
        negative_bags = []
        for positive, bags in [(True, positive_bags), (False, negative_bags)]:
            for _ in range(self.num_bags // 2):
                bag_size = int(self.r.normal(
                    self.mean_bag_size, self.var_bag_size))
                bag_size = max(bag_size, self.min_bag_size)
                while True:
                    instance_labels = torch.from_numpy(
                        self.r.randint(0, self.num_digits, bag_size))
                    bag_label, _, _ = self.compute_bag_label(instance_labels)
                    if bag_label == (1 if positive else 0):
                        bags.append(instance_labels)
                        break
        bags = positive_bags + negative_bags
        self.r.shuffle(bags)
        self.bags = bags

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index) -> Data:
        instance_labels = self.bags[index]
        bag_label, key_instances, key_cliques = self.compute_bag_label(
            instance_labels)
        bag = Data(y=bag_label, instance_labels=instance_labels,
                   key_instances=key_instances, key_cliques=key_cliques)
        return self.T(bag)


class OneHotMNISTBags(Digits):
    def __getitem__(self, index):
        bag = super().__getitem__(index)
        bag.x = torch.nn.functional.one_hot(
            bag.instance_labels, self.num_digits).float()
        return bag


normalize = T.Normalize((0.1307,), (0.3081,))
unnormalize = T.Normalize((-0.1307 / 0.3081,), (1 / 0.3081,))


def load_mnist_instances(train: bool = True, r: np.random.RandomState = np.random, shuffle: bool = True, normalize: bool = True) -> typing.Dict[int, torch.Tensor]:
    transforms = [T.ToTensor()]
    if normalize:
        transforms.append(T.Normalize((0.1307,), (0.3081,)))
    ds = datasets.MNIST("../datasets",
                        train=train,
                        download=True,
                        transform=T.Compose(transforms))
    loader = data_utils.DataLoader(ds,
                                   batch_size=len(ds),
                                   shuffle=False)
    instances, labels = next(iter(loader))

    # Shuffle
    if shuffle:
        indices = r.permutation(len(labels))
        instances = instances[indices]
        labels = labels[indices]

    # Collect instances by digit
    instances = {i: instances[labels == i] for i in range(10)}
    return instances


class MNISTBags(Digits):

    @torch.no_grad()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        instances = load_mnist_instances(train=self.train, r=self.r)

        imgs = []
        instance_indices = {i: 0 for i in range(self.num_digits)}
        for bag in self.bags:
            bag_imgs = []
            for label in bag:
                label = label.item()
                bag_imgs.append(instances[label][instance_indices[label]])
                instance_indices[label] += 1
                instance_indices[label] %= len(instances[label])
            imgs.append(torch.stack(bag_imgs))
        self.imgs = imgs

    def __getitem__(self, index):
        bag = super().__getitem__(index)
        bag.x = self.imgs[index]
        return bag


class DigitCollage(data_utils.Dataset):

    @torch.no_grad()
    def __init__(self, bag_label_computer: BagLabelComputer, num_digits: int = 10, min_bag_size: int = 2, mean_bag_size: int = 10, var_bag_size: int = 2, collage_size: int = 256, min_dist: int = 20, num_bags: int = 250, seed: int = 1, train: bool = True):
        self.compute_bag_label = bag_label_computer.compute_bag_label
        self.num_digits = num_digits
        self.min_bag_size = min_bag_size
        self.mean_bag_size = mean_bag_size
        self.var_bag_size = var_bag_size
        self.collage_size = collage_size
        self.min_dist = min_dist  # minimum distance between any two digits
        self.num_bags = num_bags
        self.train = train
        self.r = np.random.RandomState(seed)
        self.T = transforms.Compose([
            FullyConnectedGraphTransform(),
            transforms.Distance(norm=collage_size is not None,
                                max_value=float(
                                    collage_size * math.sqrt(2.)),
                                cat=False)
        ])

        # Generate bags
        positive_bags = []
        negative_bags = []
        for positive, bags in [(True, positive_bags), (False, negative_bags)]:
            for _ in range(self.num_bags // 2):
                bag_size = int(self.r.normal(
                    self.mean_bag_size, self.var_bag_size))
                bag_size = max(bag_size, self.min_bag_size)
                while True:
                    instance_labels, pos = self._propose_locations_and_labels(
                        bag_size)
                    bag_label, _, _ = self.compute_bag_label(
                        instance_labels, pos)
                    if bag_label == (1 if positive else 0):
                        bags.append((instance_labels, pos))
                        break
        bags = positive_bags + negative_bags
        self.r.shuffle(bags)
        self.bags = bags

    def _valid_locations(self, pos, min_dist):
        dist = np.sqrt(np.sum(
            (pos[np.newaxis, :, :] - pos[:, np.newaxis, :])**2, axis=-1))
        np.fill_diagonal(dist, np.inf)
        return np.all(dist >= min_dist)

    def _propose_locations_and_labels(self, bag_size, padding=28 // 2):
        instance_labels = self.r.randint(0, self.num_digits, bag_size)
        while not self._valid_locations(pos := self.r.uniform(padding, self.collage_size - padding, size=(bag_size, 2)), min_dist=self.min_dist):
            pass
        return torch.from_numpy(instance_labels.astype(np.int64)), torch.from_numpy(pos.astype(np.float32))

    def __getitem__(self, index):
        instance_labels, pos = self.bags[index]
        bag_label, key_instances, key_cliques = self.compute_bag_label(
            instance_labels, pos)
        bag = Data(y=bag_label, instance_labels=instance_labels,
                   key_instances=key_instances, key_cliques=key_cliques, pos=pos)
        return self.T(bag)

    def __len__(self):
        return len(self.bags)


class OneHotMNISTCollage(DigitCollage):
    def __getitem__(self, index):
        bag = super().__getitem__(index)
        ohe = torch.nn.functional.one_hot(
            bag.instance_labels, self.num_digits).float()
        bag.x = ohe
        return bag


class MNISTCollage(DigitCollage):

    @torch.no_grad()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Retrieve MNIST images
        mnist_instances = load_mnist_instances(train=self.train, r=self.r)

        # Generate instance images
        imgs = []
        instance_indices = {i: 0 for i in range(self.num_digits)}
        for instance_labels, pos in self.bags:
            bag_imgs = []
            for label in instance_labels:
                label = label.item()
                bag_imgs.append(
                    mnist_instances[label][instance_indices[label]])
                instance_indices[label] += 1
                instance_indices[label] %= len(mnist_instances[label])
            imgs.append(torch.stack(bag_imgs))
        self.imgs = imgs

    def __getitem__(self, index):
        bag = super().__getitem__(index)
        bag.x = self.imgs[index]
        return bag


@torch.no_grad()
def make_collage(bag: Data, collage_size: int = None) -> torch.tensor:
    if collage_size is None:
        collage_size = bag.pos.max() + 28

    collage_img = torch.zeros((collage_size, collage_size))
    for img, (x, y) in zip(bag.instances, bag.pos):
        img = unnormalize(img).squeeze()
        collage_img[y-14:y+14, x-14:x+14] += img
    return normalize(collage_img.unsqueeze(0)).squeeze()


if __name__ == "__main__":
    for train in [True, False]:
        loader = data_utils.DataLoader(MNISTBags(target_number=9,
                                                 min_instances_per_target=2,
                                                 mean_bag_size=10,
                                                 var_bag_size=2,
                                                 num_bags=100,
                                                 seed=1,
                                                 train=train),
                                       batch_size=1,
                                       shuffle=train)
        len_bag_list = []
        positive_bags = 0
        for i, (bag, bag_label, instance_labels) in enumerate(loader):
            len_bag_list.append(bag.shape[1])
            positive_bags += bag_label.float().item()
        print(
            f"Number positive {'train' if train else 'test'} bags: {positive_bags:.0f}/{len(loader)}")
        print(
            f"Number of instances per bag, mean: {np.mean(len_bag_list)}, max: {np.max(len_bag_list)}, min {np.min(len_bag_list)}")
        print()
