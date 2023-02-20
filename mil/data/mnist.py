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
    def __init__(self, target_numbers: typing.Union[int, typing.Tuple[int]] = 9, min_instances_per_target: int = 1, num_digits: int = 10, mean_bag_size: int = 10, var_bag_size: int = 2, num_bags: int = 250, seed: int = 1, train: bool = True):
        self.target_numbers = torch.tensor(target_numbers, requires_grad=False)
        if len(self.target_numbers.shape) == 0:
            self.target_numbers = self.target_numbers.unsqueeze(0)
        if self.target_numbers.max() >= num_digits:
            raise ValueError(
                f"Target number must be less than {num_digits}, got {self.target_numbers.max()}")
        self.num_digits = num_digits
        self.min_instances_per_target = min_instances_per_target
        self.mean_bag_size = mean_bag_size
        self.var_bag_size = var_bag_size
        self.num_bags = num_bags
        self.train = train
        self.min_bag_size = self.target_numbers.numel() * self.min_instances_per_target
        self.r = np.random.RandomState(seed)

        positive_bags = []
        negative_bags = []
        for positive, bags in [(True, positive_bags), (False, negative_bags)]:
            for _ in range(self.num_bags // 2):
                bag_size = int(self.r.normal(
                    self.mean_bag_size, self.var_bag_size))
                bag_size = max(bag_size, self.min_bag_size)
                while True:
                    instance_labels = torch.tensor(
                        self.r.randint(0, self.num_digits, bag_size))
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
        bag_label, instance_labels, key_instances, *_ = super().__getitem__(index)
        return Bag(bag_label, instance_labels, key_instances, instances=self.imgs[index])


class MNISTCollage(data_utils.Dataset):

    @torch.no_grad()
    def __init__(self, target_numbers: typing.Tuple[int] = (9, 7), num_digits: int = 10, mean_bag_size: int = 10, var_bag_size: int = 2, collage_size: int = 256, min_dist: int = 20, num_bags: int = 250, seed: int = 1, train: bool = True):
        self.target_numbers = target_numbers
        self.num_digits = num_digits
        self.mean_bag_size = mean_bag_size
        self.var_bag_size = var_bag_size
        self.collage_size = collage_size
        self.min_dist = min_dist
        self.num_bags = num_bags
        self.train = train
        self.min_bag_size = len(self.target_numbers)
        self.r = np.random.RandomState(seed)

        # Retrieve MNIST images
        mnist_instances = load_mnist_instances(train=self.train, r=self.r)

        # Generate bags
        positive_bags = []
        negative_bags = []
        for positive, bags in [(True, positive_bags), (False, negative_bags)]:
            for _ in range(self.num_bags // 2):
                bag_size = int(self.r.normal(
                    self.mean_bag_size, self.var_bag_size))
                bag_size = max(bag_size, self.min_bag_size)
                while True:
                    instance_labels, instance_locations = self._propose_locations_and_labels(
                        bag_size)
                    bag_label, key_instances = self.compute_bag_label(
                        instance_labels, instance_locations)
                    if bag_label == (1 if positive else 0):
                        bags.append((instance_labels, instance_locations))
                        break
        self.bags = positive_bags + negative_bags

        # Generate instance images
        imgs = []
        instance_indices = {i: 0 for i in range(self.num_digits)}
        for instance_labels, instance_locations in self.bags:
            bag_imgs = []
            for label in instance_labels:
                label = label.item()
                bag_imgs.append(
                    mnist_instances[label][instance_indices[label]])
                instance_indices[label] += 1
                instance_indices[label] %= len(mnist_instances[label])
            imgs.append(torch.stack(bag_imgs))
        self.imgs = imgs

    def _valid_locations(self, instance_locations, min_dist):
        dist = np.sqrt(np.sum(
            (instance_locations[np.newaxis, :, :] - instance_locations[:, np.newaxis, :])**2, axis=-1))
        np.fill_diagonal(dist, np.inf)
        return np.all(dist >= min_dist)

    def _propose_locations_and_labels(self, bag_size, padding=28 // 2):
        instance_labels = self.r.randint(0, self.num_digits, bag_size)
        while not self._valid_locations(instance_locations := self.r.uniform(padding, self.collage_size - padding, size=(bag_size, 2)), min_dist=self.min_dist):
            pass
        return instance_labels.astype(np.int32), instance_locations.astype(np.int32)

    def _key_instances(self, instance_labels, instance_locations):
        nodes = [(i, {"label": label, "loc": loc}) for i, (label, loc)
                 in enumerate(zip(instance_labels, instance_locations))]
        k = set()

        for ia, a in nodes:
            for ib, b in nodes:
                if ia >= ib:
                    continue
                dist = np.linalg.norm(a["loc"] - b["loc"])
                if dist < 50 and a["label"] in self.target_numbers and b["label"] in self.target_numbers and a["label"] != b["label"]:
                    k.update({ia, ib})
        return np.array(sorted(k))

    def compute_bag_label(self, instance_labels, instance_locations):
        ki = self._key_instances(instance_labels, instance_locations)
        bag_label = np.array(len(ki) > 0)
        # Convert key instances to mask
        key_instances = np.zeros_like(instance_labels, dtype=bool)
        key_instances[list(ki)] = True
        return bag_label.astype(float), key_instances

    def __getitem__(self, index):
        instance_labels, instance_locations = self.bags[index]
        bag_label, key_instances = self.compute_bag_label(
            instance_labels, instance_locations)
        return Bag(bag_label, instance_labels, key_instances, instances=self.imgs[index], instance_locations=instance_locations)

    def __len__(self):
        return len(self.bags)


@torch.no_grad()
def make_collage(bag: Bag, collage_size: int = None) -> torch.tensor:
    if collage_size is None:
        collage_size = bag.instance_locations.max() + 28

    collage_img = torch.zeros((collage_size, collage_size))
    for img, (x, y) in zip(bag.instances, bag.instance_locations):
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
