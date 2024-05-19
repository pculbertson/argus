import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pypose as pp
import torch
import torch.nn as nn
import tyro
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from wandb.util import generate_id

import wandb
from argus import ROOT
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
        save_dir: The directory to save the model.
        model_config: The configuration for the model.
        dataset_config: The configuration for the dataset.
        compile_model: Whether to compile the model.
        wandb_project: The wandb project name.
        wandb_log: Whether to log to wandb.
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
    save_dir: str = ROOT + "/outputs/models"

    # model and dataset parameters
    model_config: NCameraCNNConfig = NCameraCNNConfig()
    dataset_config: CameraCubePoseDatasetConfig = CameraCubePoseDatasetConfig("/")  # dummy path, must be overwritten!
    compile_model: bool = True

    # wandb
    wandb_project: str = "argus-estimator"
    wandb_log: bool = True

    def __post_init__(self) -> None:
        """Assert that save_dir is a string."""
        assert isinstance(self.save_dir, str)


def geometric_loss_fn(pred: torch.Tensor, target: pp.LieTensor) -> torch.Tensor:
    """The geometric loss function.

    The model predictions are in se(3), so they are 6-vectors that must be cast to se3 objects then exponentiated in
    order to compare with the targets, which are cube poses in SE(3). Finally, the loss is an L2 loss taken in the
    tangent space.

    Args:
        pred: The predicted poses in se(3) of shape (B, 6).
        target: The target poses in SE(3) of shape (B, 7).

    Returns:
        losses: The losses of shape (B,).
    """
    return torch.sum((pp.se3(pred).Exp() @ target.Inv()).Log() ** 2, axis=-1)


def initialize_training(cfg: TrainConfig) -> tuple[DataLoader, DataLoader, NCameraCNN, Optimizer, ReduceLROnPlateau]:
    """Sets up the training."""
    # dataloaders
    train_dataset = CameraCubePoseDataset(cfg.dataset_config, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    val_dataset = CameraCubePoseDataset(cfg.dataset_config, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    # model
    model = NCameraCNN(cfg.model_config).to(cfg.device)
    if cfg.compile_model:
        model = torch.compile(model, mode="reduce-overhead")  # compiled model
        print("Compiling the model...")
        model(
            torch.zeros(
                (cfg.batch_size, cfg.model_config.n_cams * 3, cfg.model_config.H, cfg.model_config.W),
                device=cfg.device,
            )
        )  # warming up the optimized model by running dummy inputs

        # doing the same with the leftover size of the train/val datasets
        train_leftover = len(train_dataset) % cfg.batch_size
        val_leftover = len(val_dataset) % cfg.batch_size
        if train_leftover != 0:
            model(
                torch.zeros(
                    (train_leftover, cfg.model_config.n_cams * 3, cfg.model_config.H, cfg.model_config.W),
                    device=cfg.device,
                )
            )
        if val_leftover != 0:
            model(
                torch.zeros(
                    (val_leftover, cfg.model_config.n_cams * 3, cfg.model_config.H, cfg.model_config.W),
                    device=cfg.device,
                )
            )
        print("Model compiled!")

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5)

    # loss function
    loss_fn = geometric_loss_fn

    # wandb
    wandb_id = generate_id()
    if cfg.wandb_log:
        wandb.init(project=cfg.wandb_project, config=cfg, id=wandb_id, resume="allow")

    return train_dataloader, val_dataloader, model, optimizer, scheduler, loss_fn, wandb_id


def train(cfg: TrainConfig) -> None:
    """Main training loop."""
    train_dataloader, val_dataloader, model, optimizer, scheduler, loss_fn, wandb_id = initialize_training(cfg)
    for epoch in range(cfg.n_epochs):
        # training loop
        model.train()
        for example in tqdm(train_dataloader, desc=f"Epoch {epoch}/{cfg.n_epochs}", total=len(train_dataloader)):
            # loading data
            images = example["images"].to(cfg.device).to(torch.float32)
            cube_pose_SE3 = example["cube_pose"].to(cfg.device).to(torch.float32)  # quats are (x, y, z, w)

            # forward pass
            cube_pose_pred_se3 = model(images)  # therefore, the predicted quats are (x, y, z, w)
            loss = torch.mean(loss_fn(cube_pose_pred_se3, cube_pose_SE3))
            optimizer.zero_grad()
            loss.backward()
            if cfg.wandb_log:
                wandb.log({"loss": loss.item()})

            # backward pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

        if epoch % cfg.print_epochs == 0:
            print(f"    Loss: {loss.item()}")

        # validation loop
        if epoch % cfg.val_epochs == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for example in val_dataloader:
                    images = example["images"].to(cfg.device).to(torch.float32)
                    cube_pose_SE3 = example["cube_pose"].to(cfg.device).to(torch.float32)
                    cube_pose_pred_se3 = model(images)
                    loss = torch.mean(loss_fn(cube_pose_pred_se3, cube_pose_SE3))
                    val_loss += loss.item()

                val_loss /= len(val_dataloader)
                if cfg.wandb_log:
                    wandb.log({"val_loss": val_loss})
                print(f"    Validation loss: {val_loss}")

                # update learning rate based on val loss
                scheduler.step(val_loss)

        if epoch % cfg.save_epochs == 0:
            if cfg.save_dir is not None:
                save_dir = Path(cfg.save_dir)
            else:
                save_dir = Path(ROOT + "/outputs/models")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), save_dir / f"{wandb_id}.pth")


if __name__ == "__main__":
    cfg = tyro.cli(TrainConfig)
    train(cfg)
