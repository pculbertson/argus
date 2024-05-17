import os
from dataclasses import dataclass

import numpy as np
import pypose as pp
import torch
import torch.nn as nn
import tyro
import wandb
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from wandb.util import generate_id

from argus.data import CameraCubePoseDataset, CameraCubePoseDatasetConfig
from argus.models import NCameraCNN, NCameraCNNConfig

torch.set_float32_matmul_precision("high")


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for training.

    Fields:
        batch_size: The batch size.
        learning_rate: The learning rate.
        n_epochs: The number of epochs.
        device: The device to train on.
        max_grad_norm: The maximum gradient norm.
        val_epochs: The number of epochs between validation.
        print_epochs: The number of epochs between printing.
        save_epochs: The number of epochs between saving.
        model_config: The configuration for the model.
        dataset_config: The configuration for the dataset.
        wandb_project: The wandb project name.
    """

    # training parameters
    batch_size: int = 128
    learning_rate: float = 1e-3
    n_epochs: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_grad_norm: float = 100.0

    # validation, printing, and saving
    val_epochs: int = 1
    print_epochs: int = 1
    save_epochs: int = 5

    # model and dataset parameters
    model_config: NCameraCNNConfig = NCameraCNNConfig()
    dataset_config: CameraCubePoseDatasetConfig = CameraCubePoseDatasetConfig()

    # wandb
    wandb_project: str = "argus-estimator"


def geometric_loss_fn(pred: torch.Tensor, target: pp.LieTensor) -> torch.Tensor:
    """The geometric loss function.

    The model predictions are in se(3), so they are 6-vectors that must be cast to se3 objects then exponentiated in
    order to compare with the targets, which are cube poses in SE(3). Finally, the loss is an L2 loss taken in the
    tangent space.

    Args:
        pred: The predicted poses in se(3) of shape (B, 6).
        target: The target poses in SE(3) of shape (B, 7).
    """
    return torch.linalg.norm((pp.se3(pred).Exp() @ target.Inv()).Log())


def initialize_training(cfg: TrainConfig) -> tuple[DataLoader, DataLoader, NCameraCNN, Optimizer, ReduceLROnPlateau]:
    """Sets up the training."""
    # dataloaders
    train_dataset = CameraCubePoseDataset(cfg.dataset_config, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    val_dataset = CameraCubePoseDataset(cfg.dataset_config, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True)

    # model
    model = NCameraCNN(cfg.model_config).to(cfg.device)
    model = torch.compile(model, mode="reduce-overhead")  # compiled model
    model(
        torch.rand(
            (cfg.batch_size, cfg.model_config.n_cams * 3, cfg.model_config.W, cfg.model_config.H),
            device=cfg.device,
        )
    )  # warming up the optimized model

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5, verbose=True)

    # loss function
    loss_fn = geometric_loss_fn

    # wandb
    wandb_id = generate_id()
    wandb.init(project=cfg.wandb_project, config=cfg, id=wandb_id, resume="allow")

    return train_dataloader, val_dataloader, model, optimizer, scheduler, loss_fn, wandb_id


def train(cfg: TrainConfig) -> None:
    """Main training loop."""
    train_dataloader, val_dataloader, model, optimizer, scheduler, loss_fn, wandb_id = initialize_training(cfg)
    for epoch in range(cfg.n_epochs):
        # training loop
        model.train()
        for example in tqdm(train_dataloader, desc=f"Epoch {epoch}", total=len(train_dataloader)):
            # loading data
            images = example["images"].to(cfg.device)
            cube_pose_SE3 = example["cube_pose"].to(cfg.device)

            # forward pass
            cube_pose_pred_se3 = model(images)
            loss = loss_fn(cube_pose_pred_se3, cube_pose_SE3)
            optimizer.zero_grad()
            loss.backward()
            wandb.log({"loss": loss.item()})

            # backward pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

        if epoch % cfg.print_epochs == 0:
            print(f"Loss: {loss.item()}")

        # validation loop
        if epoch % cfg.val_epochs == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for example in tqdm(val_dataloader, desc="Validation", total=len(val_dataloader)):
                    images = example["images"].to(cfg.device)
                    cube_pose_SE3 = example["cube_pose"].to(cfg.device)
                    cube_pose_pred_se3 = model(images)
                    loss = loss_fn(cube_pose_pred_se3, cube_pose_SE3)
                    val_loss += loss.item()

                val_loss /= len(val_dataloader)
                wandb.log({"val_loss": val_loss.item()})
                print(f"Validation loss: {val_loss}")

                # update learning rate based on val loss
                scheduler.step(val_loss)

        if epoch % cfg.save_epochs == 0:
            os.makedirs("outputs/models", exist_ok=True)
            torch.save(model.state_dict(), f"outputs/models/{wandb_id}.pth")


if __name__ == "__main__":
    cfg = tyro.cli(TrainConfig)
    train(cfg)
