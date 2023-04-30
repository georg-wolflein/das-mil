import torch
from torch import nn
import hydra
import os
from tqdm import tqdm
from pathlib import Path


from mil.utils import device, human_format, set_seed
from mil.utils.stats import print_prediction_stats

os.environ["HYDRA_FULL_ERROR"] = "1"


def error_score(y_pred, y):
    return 1. - ((y_pred > .5) == (y > .5)).float()


loss_function = nn.BCELoss()


def test_loss_and_error(model, loader, save_predictions=False):
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

            if save_predictions:
                predictions.append((bag.detach().cpu(), y_pred.detach().cpu()))

            error = error_score(y_pred, y).detach().cpu().item()
            total_error += error
            total_loss += loss.detach().cpu().item()
    return total_loss / len(loader), total_error / len(loader), predictions


def train_step(cfg, i, bag, model, optimizer, update: bool = True):
    bag = bag.to(cfg.device)

    optimizer.zero_grad()

    # Calculate loss and metrics
    y_pred = model(bag.x, bag.edge_index,
                   bag.edge_attr).squeeze()
    loss = loss_function(y_pred, bag.y)

    error = error_score(y_pred, bag.y)
    error_value = error.detach().cpu().item()

    # Backward pass
    loss.backward()

    # Update weights
    if update:
        optimizer.step()

    loss_value = loss.detach().cpu().item()

    # del bag.x
    # del bag.edge_index
    # del bag.edge_attr
    # del bag.y
    # del bag.pos
    # del bag
    # del error
    # del loss

    return loss_value, error_value


def save_model(cfg, model, epoch):
    output_folder = Path(hydra.utils.get_original_cwd()
                         ) / "models" / cfg.name
    output_folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_folder / f"model_{epoch:d}.pt")
    torch.save(model.state_dict(), output_folder / f"model_latest.pt")


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def train(cfg):
    set_seed(cfg.seed)

    train_dataset = hydra.utils.instantiate(cfg.dataset.train)
    test_dataset = hydra.utils.instantiate(cfg.dataset.test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

    model = hydra.utils.instantiate(cfg.model, _convert_="partial")

    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())

    model.train()
    print(
        f"Training model with {human_format(sum(p.numel() for p in model.parameters() if p.requires_grad))} parameters")

    model.to(cfg.device)

    for epoch in range(cfg.num_epochs):
        model.train()

        total_loss = 0.
        total_error = 0.
        # train_step(-1, train_dataset.fake_bag(), model, optimizer)
        for i, bag in enumerate(pbar := tqdm(train_loader, desc=f"Epoch {epoch}")):
            pbar.set_description(f"Epoch {epoch}, bag size {bag.x.shape[0]}")
            loss, error = train_step(cfg, i, bag, model, optimizer)
            total_loss += loss
            total_error += error

        test_loss, test_error, _ = test_loss_and_error(model, test_loader)

        # stats.append({
        #     "epoch": epoch,
        #     "loss": total_loss / len(train_loader),
        #     "error": total_error / len(train_loader),
        #     "test_loss": test_loss,
        #     "test_error": test_error
        # })
        print(
            f"Epoch: {epoch:3d}, loss: {total_loss/len(train_loader):.4f}, error: {total_error/len(train_loader):.4f}, test_loss: {test_loss:.4f}, test_error: {test_error:.4f}")

        # Save model
        if epoch % cfg.save_epoch_freq == 0 or epoch == cfg.num_epochs - 1:
            save_model(cfg, model, epoch)

    test_loss, test_error, predictions = test_loss_and_error(
        model, test_loader)
    print(f"Test loss: {test_loss:.4f}, test error: {test_error:.4f}")

    if "mnist" in train_dataset.__class__.__name__.lower():
        print_prediction_stats(
            predictions, target_numbers=cfg.settings.mnist.target_numbers)


if __name__ == "__main__":
    train()
