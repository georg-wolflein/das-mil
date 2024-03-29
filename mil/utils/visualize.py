import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data as Bag

from ..data.mnist import unnormalize, make_collage

SIGNIFICANT_ATTENTION_THRESHOLD = .1


def label2char(label: bool) -> str:
    if label is None:
        return " "
    return "+" if label else "-"


def red(s: str, show: bool = True) -> str:
    if not show:
        return f"{s}"
    return f"\033[91m{s}\033[0m"


def print_one_hot_bag(bag: Bag, y_pred: torch.Tensor = None):
    bag_str = " ".join(red(instance_label, show=key_instance)
                       for instance_label, key_instance in zip(bag.instance_labels, bag.key_instances))
    bag_str = f"{{{bag_str}}}"
    print(f"{label2char(bag.y)}{label2char(y_pred)} bag: {bag_str}")


def print_one_hot_bag_with_attention(bag: Bag, attention: torch.Tensor, y_pred: torch.Tensor = None):
    bag_str = " | ".join(red(f"{instance_label:4d}", show=key_instance)
                         for instance_label, key_instance in zip(bag.instance_labels, bag.key_instances))
    att_str = " | ".join(red(f"{a:.2f}", show=a > SIGNIFICANT_ATTENTION_THRESHOLD)
                         for a in attention.squeeze(-1))
    bag_line = f"{label2char(bag.y)} bag | {bag_str} |"
    att_line = f"{label2char(y_pred)} att | {att_str} |"
    print(bag_line)
    print(att_line)


def _make_title(bag_label: torch.Tensor, y_pred: torch.Tensor = None):
    title = f"Bag label: {bag_label.item():.0f}"
    if y_pred is not None:
        title += f", pred: {y_pred.item():.2f}"
    return title


def plot_collage(bag: Bag, highlight_key_instances: bool = True, collage_size: int = None, y_pred: torch.Tensor = None, attention: torch.Tensor = None):
    collage_img = make_collage(bag, collage_size=collage_size)
    collage_img = unnormalize(collage_img.unsqueeze(0)).squeeze(0).numpy()
    key_instance_mask = np.zeros_like(collage_img, dtype=bool)
    collage_img = np.repeat(collage_img[:, :, np.newaxis], 3, axis=-1)

    if highlight_key_instances:
        for ki, (x, y) in zip(bag.key_instances, bag.pos.int()):
            if ki:
                key_instance_mask[y-14:y+14, x-14:x+14] = True
        collage_img[key_instance_mask] += [1., 0., 0.]
    collage_img = np.clip(collage_img, 0., 1.)
    plt.figure()
    plt.imshow(collage_img)
    plt.gcf().suptitle(_make_title(bag.y, y_pred=y_pred))
    plt.axis("equal")
    plt.gca().xaxis.set_ticks([])
    plt.gca().yaxis.set_ticks([])

    return collage_img


def plot_one_hot_collage(bag: Bag, highlight_key_instances: bool = True, collage_size: int = None, y_pred: torch.Tensor = None, attention: torch.Tensor = None):
    if highlight_key_instances:
        plt.scatter(*bag.pos[~bag.key_instances].T, c="b")
        plt.scatter(*bag.pos[bag.key_instances].T, c="r")
    else:
        plt.scatter(*bag.pos.T, c="b")
    plt.axis("equal")
    plt.xlim(0, collage_size)
    plt.ylim(0, collage_size)
    for label, loc in zip(bag.instance_labels, bag.pos):
        plt.annotate(label.item(), xy=loc, xytext=(0, 5),
                     textcoords="offset points", horizontalalignment="center")


def plot_bag(bag: Bag, highlight_key_instances: bool = True, collage_size: int = None, y_pred: torch.Tensor = None, attention: torch.Tensor = None):
    if len(bag.x.shape) < 2 or bag.x.shape[-1] != bag.x.shape[-2]:
        raise ValueError(
            "Instances must be square images. Did you supply one-hot encoded bags?")
    if bag.pos is None:
        fig, axs = plt.subplots(1, bag.x.shape[0], figsize=(8, 2))
        if attention is None:
            attention = [None] * bag.x.shape[0]
        for instance, instance_label, key_instance, att, ax in zip(bag.x, bag.instance_labels, bag.key_instances, attention, axs):
            instance = unnormalize(instance) * 255
            instance = instance.squeeze(0).numpy().astype(np.uint8)
            instance = np.repeat(instance[:, :, np.newaxis], 3, axis=2)
            if highlight_key_instances and key_instance:
                instance[:, :, 0] = 255
            ax.imshow(instance)
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.set_title(f"{instance_label}")
            if att is not None:
                ax.set_xlabel(f"{att:.2f}")
                if att > SIGNIFICANT_ATTENTION_THRESHOLD:
                    ax.xaxis.label.set_color("red")
        plt.gcf().suptitle(_make_title(bag.y, y_pred=y_pred))
    else:
        return plot_collage(bag,
                            highlight_key_instances=highlight_key_instances,
                            collage_size=collage_size,
                            y_pred=y_pred,
                            attention=attention)


def plot_attention_head(bag: Bag, A: torch.Tensor, limit_range: bool = True, latex: bool = False):
    plt.imshow(A, cmap="gray",
               **(dict(vmin=0., vmax=1.) if limit_range else {}))
    l = bag.instance_labels.detach().cpu().numpy().tolist()
    if latex:
        l = [(f"\\color{{red}}{i}\\normalcolor" if ki else f"{i}")
             for i, ki in zip(l, bag.key_instances)]
    plt.xticks(range(len(l)), l)
    plt.yticks(range(len(l)), l)
    for xtick, ytick, key_instance in zip(plt.gca().get_xticklabels(), plt.gca().get_yticklabels(), bag.key_instances):
        if key_instance:
            for tick in (xtick, ytick):
                tick.set_color("red")
