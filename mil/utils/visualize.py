import matplotlib.pyplot as plt
import numpy as np
import torch

from ..data.mnist import Bag, unnormalize, make_collage


def label2char(label: bool) -> str:
    if label is None:
        return " "
    return "+" if label else "-"


def red(s: str, show: bool = True) -> str:
    if not show:
        return f"{s}"
    return f"\033[91m{s}\033[0m"


def print_one_hot_bag(bag: Bag, y_pred=None):
    bag_str = " ".join(red(instance_label, show=key_instance)
                       for instance_label, key_instance in zip(bag.instance_labels, bag.key_instances))
    bag_str = f"{{{bag_str}}}"
    print(f"{label2char(bag.bag_label)}{label2char(y_pred)} bag: {bag_str}")


def print_one_hot_bag_with_attention(bag: Bag, attention: torch.Tensor, y_pred=None):
    bag_str = " | ".join(red(f"{instance_label:4d}", show=key_instance)
                         for instance_label, key_instance in zip(bag.instance_labels, bag.key_instances))
    att_str = " | ".join(red(f"{a:.2f}", show=a > .1)
                         for a in attention.squeeze(-1))
    bag_line = f"{label2char(bag.bag_label)} bag | {bag_str} |"
    att_line = f"{label2char(y_pred)} att | {att_str} |"
    print(bag_line)
    print(att_line)


def plot_collage(bag: Bag, collage_size: int = None, highlight_key_instances: bool = True):
    collage_img = make_collage(bag, collage_size=collage_size)
    collage_img = unnormalize(collage_img.unsqueeze(0)).squeeze(0).numpy()
    key_instance_mask = np.zeros_like(collage_img, dtype=bool)
    collage_img = np.repeat(collage_img[:, :, np.newaxis], 3, axis=-1)

    if highlight_key_instances:
        for ki, (x, y) in zip(bag.key_instances, bag.instance_locations):
            if ki:
                key_instance_mask[y-14:y+14, x-14:x+14] = True
        collage_img[key_instance_mask] += [1., 0., 0.]
    collage_img = np.clip(collage_img, 0., 1.)
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
            instance = unnormalize(instance) * 255
            instance = instance.squeeze(0).numpy().astype(np.uint8)
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


def plot_attention_head(bag: Bag, A: torch.Tensor):
    plt.imshow(A, vmin=0., vmax=1., cmap="gray")
    l = bag.instance_labels.detach().cpu().numpy().tolist()
    plt.xticks(range(len(l)), l)
    plt.yticks(range(len(l)), l)
    for xtick, ytick, key_instance in zip(plt.gca().get_xticklabels(), plt.gca().get_yticklabels(), bag.key_instances):
        if key_instance:
            for tick in (xtick, ytick):
                tick.set_color("red")
