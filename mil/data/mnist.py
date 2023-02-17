import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms as T
import typing


class Bag(typing.NamedTuple):
    bag_label: torch.Tensor
    instance_labels: torch.Tensor
    key_instances: torch.Tensor  # mask of key instances
    instances: typing.Optional[torch.Tensor] = None
    instance_locations: typing.Optional[torch.Tensor] = None


class Digits(data_utils.Dataset):

    @torch.no_grad()
    def __init__(self, target_number: typing.Union[int, typing.Tuple[int]] = 9, min_instances_per_target: int = 1, num_digits: int = 10, mean_bag_length: int = 10, var_bag_length: int = 2, num_bags: int = 250, seed: int = 1, train: bool = True):
        self.target_numbers = torch.tensor(target_number, requires_grad=False)
        if len(self.target_numbers.shape) == 0:
            self.target_numbers = self.target_numbers.unsqueeze(0)
        if self.target_numbers.max() >= num_digits:
            raise ValueError(
                f"Target number must be less than {num_digits}, got {self.target_numbers.max()}")
        self.num_digits = num_digits
        self.min_instances_per_target = min_instances_per_target
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bags = num_bags
        self.train = train
        self.min_bag_length = self.target_numbers.numel() * self.min_instances_per_target
        self.r = np.random.RandomState(seed)

        positive_bags = []
        negative_bags = []
        for positive, bags in [(True, positive_bags), (False, negative_bags)]:
            for _ in range(self.num_bags // 2):
                bag_length = int(self.r.normal(
                    self.mean_bag_length, self.var_bag_length))
                bag_length = max(bag_length, self.min_bag_length)
                while True:
                    instance_labels = torch.tensor(
                        self.r.randint(0, self.num_digits, bag_length))
                    bag_label, _ = self.compute_bag_label(instance_labels)
                    if bag_label == (1 if positive else 0):
                        bags.append(instance_labels)
                        break
        bags = positive_bags + negative_bags
        self.r.shuffle(bags)
        self.bags = bags

    def compute_bag_label(self, instance_labels):
        instance_labels = instance_labels == self.target_numbers.unsqueeze(-1)
        key_instances = instance_labels.any(axis=0)
        count_per_target = instance_labels.sum(axis=-1)
        bag_label = (count_per_target >=
                     self.min_instances_per_target).all()
        if not bag_label:
            key_instances = torch.zeros_like(key_instances, dtype=bool)
        return bag_label.float(), key_instances

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index) -> Bag:
        instance_labels = self.bags[index]
        bag_label, key_instances = self.compute_bag_label(instance_labels)
        return Bag(bag_label, instance_labels, key_instances)


class OneHotMNISTBags(Digits):
    def __getitem__(self, index):
        bag_label, instance_labels, key_instances, *_ = super().__getitem__(index)
        ohe = torch.nn.functional.one_hot(
            instance_labels, self.num_digits).float()
        return Bag(bag_label, instance_labels, key_instances, ohe)


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


def undo_normalize(img: torch.Tensor) -> torch.Tensor:
    return img * 0.3081 + 0.1307


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
        bag_label, instance_labels, key_instances, *_ = super().__getitem__(index)
        return Bag(bag_label, instance_labels, key_instances, instances=self.imgs[index])


if __name__ == "__main__":
    for train in [True, False]:
        loader = data_utils.DataLoader(MNISTBags(target_number=9,
                                                 min_instances_per_target=2,
                                                 mean_bag_length=10,
                                                 var_bag_length=2,
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
