import torch
from torch import nn
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import torch_geometric as pyg
import numpy as np
import pandas as pd
import torch_geometric as pyg
import hydra
import os

from mil.data.mnist import MNISTBags, OneHotMNISTBags, MNISTCollage, OneHotMNISTCollage
from mil.utils import device, human_format, set_seed
from mil.utils.visualize import print_one_hot_bag_with_attention, print_one_hot_bag, plot_attention_head, plot_bag, plot_one_hot_collage
from mil.utils.stats import print_prediction_stats
from mil.models.abmil import WeightedAverageAttention
from mil.models.self_attention import MultiHeadSelfAttention
from mil.models.distance_aware_self_attention import DistanceAwareSelfAttentionHead

os.environ["HYDRA_FULL_ERROR"] = "1"
RESULTS_FILE = "train.csv"


def error_score(y_pred, y):
    return 1. - ((y_pred > .5).float() == y).cpu().detach().float()


loss_function = nn.BCELoss()


def test_loss_and_error(model, loader):
    model.eval()

    total_loss = 0.
    total_error = 0.
    predictions = []

    with torch.no_grad():
        for i, bag in enumerate(loader):
            bag = device(bag)
            y = bag.y.float()

            # Calculate loss and metrics
            y_pred = model(bag.x, bag.edge_index, bag.edge_attr).squeeze()
            loss = loss_function(y_pred, y)

            predictions.append((bag.cpu().detach(), y_pred.detach().cpu()))

            error = error_score(y_pred, y)
            total_error += error
            total_loss += loss.detach().cpu()
    return total_loss / len(loader), total_error / len(loader), predictions


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg):
    set_seed(cfg.seed)

    train_dataset = hydra.utils.instantiate(cfg.dataset.train)
    test_dataset = hydra.utils.instantiate(cfg.dataset.test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

    model = hydra.utils.instantiate(cfg.model, _convert_="partial")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-2)

    stats = []

    model.train()
    print(
        f"Training model with {human_format(sum(p.numel() for p in model.parameters() if p.requires_grad))} parameters")

    for epoch in range(50):
        model.train()

        total_loss = 0.
        total_error = 0.
        for bag in train_loader:
            bag = device(bag)
            y = bag.y

            optimizer.zero_grad()

            # Calculate loss and metrics
            y_pred = model(bag.x, bag.edge_index, bag.edge_attr).squeeze()
            loss = loss_function(y_pred, y)

            error = error_score(y_pred, y)
            total_error += error

            # Backward pass
            loss.backward()

            total_loss += loss.detach().cpu()
            # Step
            optimizer.step()

        test_loss, test_error, _ = test_loss_and_error(model, test_loader)

        stats.append({
            "epoch": epoch,
            "loss": total_loss / len(train_loader),
            "error": total_error / len(train_loader),
            "test_loss": test_loss,
            "test_error": test_error
        })
        print(
            f"Epoch: {epoch:3d}, loss: {total_loss/len(train_loader):.4f}, error: {total_error/len(train_loader):.4f}, test_loss: {test_loss:.4f}, test_error: {test_error:.4f}")

    # Plot training and test loss/error
    stats = pd.DataFrame(stats)
    float_cols = [col for col in stats.columns if col != "epoch"]
    stats[float_cols] = stats[float_cols].astype(float)
    stats.to_csv(RESULTS_FILE, index=False)

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.title("Loss")
    plt.plot(stats["epoch"], stats["loss"], label="train")
    plt.plot(stats["epoch"], stats["test_loss"], label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(122)
    plt.title("Error")
    plt.plot(stats["epoch"], stats["error"], label="train")
    plt.plot(stats["epoch"], stats["test_error"], label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()

    test_loss, test_error, predictions = test_loss_and_error(
        model, test_loader)
    print(f"Test loss: {test_loss:.4f}, test error: {test_error:.4f}")

    print_prediction_stats(
        predictions, target_numbers=cfg.settings.mnist.target_numbers)


if __name__ == "__main__":
    main()
