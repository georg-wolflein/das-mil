import matplotlib.pyplot as plt
import numpy as np

from .mnist import Bag, undo_normalize, make_collage


def print_one_hot_bag(bag: Bag):
    RED = '\033[91m'
    ENDC = '\033[0m'
    bag_str = " ".join((f"{RED}{instance_label}{ENDC}" if key_instance else f"{instance_label}")
                       for instance_label, key_instance in zip(bag.instance_labels, bag.key_instances))
    bag_str = f"{{{bag_str}}}"
    print(
        f"{'pos' if bag.bag_label else 'neg'} bag: {bag_str}")


def plot_collage(bag: Bag, collage_size: int = None, highlight_key_instances: bool = True):
    collage_img = make_collage(bag, collage_size=collage_size)
    key_instance_mask = np.zeros_like(collage_img, dtype=bool)
    collage_img = np.repeat(collage_img[:, :, np.newaxis], 3, axis=-1)

    for ki, (x, y) in zip(bag.key_instances, bag.instance_locations):
        if ki:
            key_instance_mask[y-14:y+14, x-14:x+14] = 1
    collage_img[key_instance_mask] += [1, 0, 0]
    collage_img = np.clip(collage_img, 0, 1)
    plt.figure()
    plt.imshow(collage_img)
    plt.gcf().suptitle(f"Bag label: {bag.bag_label.item() == 1.}")
    plt.axis("equal")


def visualize_bag(bag: Bag, highlight_key_instances: bool = True, collage_size: int = None):
    bag_label, instance_labels, key_instances, instances, instance_locations = bag
    if len(instances.shape) < 2 or instances.shape[-1] != instances.shape[-2]:
        raise ValueError(
            "Instances must be square images. Did you supply one-hot encoded bags?")
    if instance_locations is None:
        fig, axs = plt.subplots(1, instances.shape[0], figsize=(8, 2))
        for instance, instance_label, key_instance, ax in zip(instances, instance_labels, key_instances, axs):
            instance = instance.squeeze().numpy()
            instance = undo_normalize(instance) * 255
            instance = instance.astype(np.uint8)
            instance = np.repeat(instance[:, :, np.newaxis], 3, axis=2)
            if highlight_key_instances and key_instance:
                instance[:, :, 0] = 255
            ax.imshow(instance)
            ax.set_axis_off()
            ax.set_title(f"{instance_label}")
        plt.gcf().suptitle(f"Bag label: {bag_label.item() == 1.}")
    else:
        plot_collage(bag,
                     highlight_key_instances=highlight_key_instances,
                     collage_size=collage_size)
