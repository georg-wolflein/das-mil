import torch
from torch import nn
import hydra
import os
from tqdm import tqdm
from pathlib import Path
import wandb
from omegaconf import OmegaConf
import sys
from sklearn import metrics as skmetrics
import numpy as np
import functools
from collections import defaultdict


from mil.utils import human_format, set_seed

os.environ["HYDRA_FULL_ERROR"] = "1"


loss_function = nn.BCELoss()


def binarized(fn):
    @functools.wraps(fn)
    def binarized_fn(y_true: np.ndarray, y_pred: np.ndarray):
        y_true = y_true.astype(int)
        y_pred = (y_pred > 0.5).astype(int)
        return fn(y_true, y_pred)
    return binarized_fn


METRICS = {
    "acc": binarized(skmetrics.accuracy_score),
    "balanced_acc": binarized(skmetrics.balanced_accuracy_score),
    "auc": skmetrics.roc_auc_score,
    "f1": binarized(skmetrics.f1_score),
    "precision": binarized(skmetrics.precision_score),
    "recall": binarized(skmetrics.recall_score)
}


class History:
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = []
        self.y_pred = []
        self.custom_metrics = defaultdict(list)

    def update(self, y_true, y_pred, **custom_metrics):
        self.y_true.append(y_true.detach().cpu().item())
        self.y_pred.append(y_pred.detach().cpu().item())
        for metric, value in custom_metrics.items():
            self.custom_metrics[metric].append(value.detach().cpu().item())

    def compute_metrics(self):
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        metrics = {metric: METRICS[metric](y_true, y_pred)
                   for metric in METRICS}
        custom_metrics = {metric: np.mean(values)
                          for metric, values in self.custom_metrics.items()
                          }
        return {**metrics, **custom_metrics}


@torch.no_grad()
def test(cfg, model, loader, history, save_predictions=False):
    model.eval()

    predictions = []

    for bag in tqdm(loader, desc="Testing"):
        bag = bag.to(cfg.device)

        # Calculate loss and metrics
        y_pred = model(bag.x, bag.edge_index, bag.edge_attr).squeeze()

        if save_predictions:
            predictions.append((bag.detach().cpu(), y_pred.detach().cpu()))

        # Update metrics
        history.update(bag.y.detach().cpu(), y_pred.detach().cpu(),
                       loss=loss_function(y_pred, bag.y).detach().cpu())
    return predictions


def train_step(cfg, i, bag, model, optimizer, history: History, update: bool = True):
    bag = bag.to(cfg.device)

    optimizer.zero_grad()

    # Calculate loss and metrics
    y_pred = model(bag.x, bag.edge_index,
                   bag.edge_attr).squeeze()
    loss = loss_function(y_pred, bag.y)

    # Backward pass
    loss.backward()

    # Update metrics
    if update:
        history.update(bag.y.detach().cpu(), y_pred.detach().cpu(),
                       loss=loss.detach().cpu())

    # Update weights
    if update:
        optimizer.step()


def save_model(cfg, model, epoch):
    output_folder = Path(hydra.utils.get_original_cwd()
                         ) / "checkpoints" / cfg.wandb_id
    output_folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_folder / f"model_{epoch:d}.pt")
    torch.save(model.state_dict(), output_folder / f"model_latest.pt")


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def train(cfg):
    wandb.init(entity=cfg.wandb.entity,
               project=cfg.wandb.project,
               name=f"{cfg.name}_seed{cfg.seed if cfg.seed is not None else 'none'}",
               group=cfg.name,
               job_type="train",
               config={**OmegaConf.to_container(
                   cfg, resolve=True, throw_on_missing=True
               ), "overrides": " ".join(sys.argv[1:])},
               settings=wandb.Settings(start_method="thread"),)
    cfg.wandb_id = wandb.run.id

    set_seed(cfg.seed)

    train_dataset = hydra.utils.instantiate(cfg.dataset.train)
    test_dataset = hydra.utils.instantiate(cfg.dataset.test)

    wandb.run.tags = wandb.run.tags + (train_dataset.__class__.__name__,)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0], num_workers=0, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0], num_workers=0, pin_memory=False)

    model = hydra.utils.instantiate(cfg.model, _convert_="partial")
    model.to(cfg.device)

    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())

    train_history = History()
    test_history = History()

    model.train()
    print(
        f"Training model with {human_format(sum(p.numel() for p in model.parameters() if p.requires_grad))} parameters")

    for epoch in range(cfg.num_epochs):
        model.train()

        train_history.reset()
        test_history.reset()

        # Biggest bag first to avoid OOM
        train_step(cfg, -1, train_dataset.fake_bag(),
                   model, optimizer, history=None, update=False)

        # Train
        for i, bag in enumerate(pbar := tqdm(train_loader, desc=f"Epoch {epoch}")):
            pbar.set_description(f"Epoch {epoch}, bag size {bag.x.shape[0]}")
            train_step(cfg, i, bag, model, optimizer, train_history)

        # Test
        test(cfg, model, test_loader, test_history)

        log = {
            "epoch": epoch,
            **{f"train/{k}": v for k, v in train_history.compute_metrics().items()},
            **{f"test/{k}": v for k, v in test_history.compute_metrics().items()}
        }
        print(
            f"Epoch: {epoch:3d},",
            ", ".join(f"{k}: {v:.4f}" for k, v in log.items() if k not in ("epoch", "step")))
        wandb.log(log, step=epoch)

        # Save model
        if epoch % cfg.save_epoch_freq == 0 or epoch == cfg.num_epochs - 1:
            save_model(cfg, model, epoch)


if __name__ == "__main__":
    train()
