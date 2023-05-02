from typing import Union
import torch
from torch import nn
import hydra
import os
from tqdm import tqdm
from pathlib import Path
import wandb
from omegaconf import OmegaConf
import sys
from torcheval.metrics import BinaryAccuracy, BinaryAUROC, BinaryF1Score, AUC, Mean


from mil.utils import device, human_format, set_seed

os.environ["HYDRA_FULL_ERROR"] = "1"


loss_function = nn.BCELoss()


class LossMetric(Mean):

    @torch.inference_mode()
    def update(self, y_pred, y_true):
        loss = loss_function(y_pred, y_true).detach()
        super().update(loss)


METRICS = {
    "acc": BinaryAccuracy,
    "auroc": BinaryAUROC,
    "f1": BinaryF1Score,
    "auc": AUC,
    "loss": LossMetric
}


class MetricCollection:
    def __init__(self, metric_factories: dict = {}, **kwargs):
        self.metrics = {name: factory(**kwargs)
                        for name, factory in metric_factories.items()}
        self.reset()

    def reset(self, *args, **kwargs):
        for metric in self.metrics.values():
            metric.reset(*args, **kwargs)

    def update(self, a, b, *args, **kwargs):
        a = a.unsqueeze(0) if a.ndim == 0 else a
        b = b.unsqueeze(0) if b.ndim == 0 else b
        for metric in self.metrics.values():
            metric.update(a, b, *args, **kwargs)

    def compute(self, *args, **kwargs):
        return {name: metric.compute(*args, **kwargs) for name, metric in self.metrics.items()}

    def __getitem__(self, key):
        return self.metrics[key]

    def __setitem__(self, key, value):
        self.metrics[key] = value

    def __iter__(self):
        return iter(self.metrics.values())

    def __len__(self):
        return len(self.metrics)

    def items(self):
        return self.metrics.items()


@torch.no_grad()
def test(cfg, model, loader, metrics, save_predictions=False):
    model.eval()

    predictions = []

    for bag in tqdm(loader, desc="Testing"):
        bag = bag.to(cfg.device)
        y = bag.y.float()

        # Calculate loss and metrics
        y_pred = model(bag.x, bag.edge_index, bag.edge_attr).squeeze()

        if save_predictions:
            predictions.append((bag.detach().cpu(), y_pred.detach().cpu()))

        # Update metrics
        metrics.update(y_pred.detach(), y.detach())
    return metrics, predictions


def train_step(cfg, i, bag, model, optimizer, metrics: MetricCollection, update: bool = True):
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
        metrics.update(y_pred.detach(), bag.y.detach())

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
        train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

    model = hydra.utils.instantiate(cfg.model, _convert_="partial")
    model.to(cfg.device)

    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())

    train_metrics = MetricCollection(METRICS, device=cfg.device)
    test_metrics = MetricCollection(METRICS, device=cfg.device)

    model.train()
    print(
        f"Training model with {human_format(sum(p.numel() for p in model.parameters() if p.requires_grad))} parameters")

    step = 0

    for epoch in range(cfg.num_epochs):
        model.train()

        train_metrics.reset()
        test_metrics.reset()

        # Biggest bag first to avoid OOM
        train_step(cfg, -1, train_dataset.fake_bag(),
                   model, optimizer, train_metrics, update=False)
        for i, bag in enumerate(pbar := tqdm(train_loader, desc=f"Epoch {epoch}")):
            pbar.set_description(f"Epoch {epoch}, bag size {bag.x.shape[0]}")
            train_step(cfg, i, bag, model, optimizer, train_metrics)
            step += 1
            if step % cfg.log_step_freq == 0:
                wandb.log({
                    "epoch": epoch,
                    "step": step,
                    **{f"train/intermediate/{k}": v.item() for k, v in train_metrics.compute().items()}
                }, step=step)

        test(cfg, model, test_loader, test_metrics)

        log = {
            "epoch": epoch,
            "step": step,
            **{f"train/{k}": v.item() for k, v in train_metrics.compute().items()},
            **{f"test/{k}": v.item() for k, v in test_metrics.compute().items()}
        }
        print(
            f"Epoch: {epoch:3d},",
            ", ".join(f"{k}: {v.item():.4f}" for k, v in log.items() if k not in ("epoch", "step")))
        wandb.log(log, step=step)

        # Save model
        if epoch % cfg.save_epoch_freq == 0 or epoch == cfg.num_epochs - 1:
            save_model(cfg, model, epoch)


if __name__ == "__main__":
    train()
