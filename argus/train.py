import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pypose as pp
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import tyro
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm
from wandb.util import generate_id

from argus import ROOT
from argus.data import Augmentation, AugmentationConfig, CameraCubePoseDataset, CameraCubePoseDatasetConfig
from argus.models import NCameraCNN, NCameraCNNConfig

torch.set_float32_matmul_precision("high")


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for training.

    For all path fields, you can either specify an absolute path, a relative path (with respect to where you are
    currently calling the data generation function), or a local path RELATIVE TO THE ROOT OF THE PACKAGE IN YOUR SYSTEM!
    For instance, if you pass "example_dir" to `save_dir`, the model will be saved under /path/to/argus/example_dir.

    Fields:
        dataset_config: The configuration for the dataset.
        model_config: The configuration for the model.
        compile_model: Whether to compile the model.
        batch_size: The batch size.
        learning_rate: The learning rate.
        n_epochs: The number of epochs.
        device: The device to train on.
        max_grad_norm: The maximum gradient norm.
        num_gpus: The number of GPUs to train with.
        random_seed: The random seed.
        multigpu: Whether to use multiple GPUs.
        amp: Whether to use automatic mixed precision.
        no_profiling: Whether to turn off profiling APIs.
        val_epochs: The number of epochs between validation.
        print_epochs: The number of epochs between printing.
        save_epochs: The number of epochs between saving.
        save_dir: The directory to save the model.
        wandb_project: The wandb project name.
        wandb_log: Whether to log to wandb.
    """

    # model and dataset parameters
    dataset_config: CameraCubePoseDatasetConfig
    model_config: NCameraCNNConfig = NCameraCNNConfig()
    compile_model: bool = False  # WARNING: compiling the model during training makes it hard to load later

    # training parameters
    batch_size: int = 32  # something maxes the GPU throughput far before the memory is saturated
    learning_rate: float = 1e-4
    n_epochs: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_grad_norm: float = 1.0
    num_gpus: int = torch.cuda.device_count()
    random_seed: int = 42

    # speed optimizations
    multigpu: bool = False
    amp: bool = False
    no_profiling: bool = False

    # validation, printing, and saving
    val_epochs: int = 1
    print_epochs: int = 1
    save_epochs: int = 5
    save_dir: str = ROOT + "/outputs/models"

    # data augmentation
    augmentation_config: AugmentationConfig = AugmentationConfig()
    use_augmentation: bool = True

    # wandb
    wandb_project: str = "argus-estimator"
    wandb_log: bool = True

    def __post_init__(self) -> None:
        """Assert that save_dir is a string."""
        assert isinstance(self.save_dir, str)
        if not os.path.exists(self.save_dir):  # absolute path
            if os.path.exists(ROOT + "/" + self.save_dir):
                self.save_dir = ROOT + "/" + self.save_dir
            else:
                os.makedirs(self.save_dir, exist_ok=True)

        assert self.num_gpus > 0, "The number of GPUs must be greater than 0!"
        assert (
            self.num_gpus <= torch.cuda.device_count()
        ), "The number of GPUs must be less than or equal to the number of GPUs on the system!"


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


def initialize_training(
    cfg: TrainConfig, rank: int = 0
) -> tuple[DataLoader, DataLoader, NCameraCNN, Optimizer, ReduceLROnPlateau]:
    """Sets up the training."""
    # set random seed
    torch.cuda.manual_seed_all(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    # local device
    if cfg.multigpu:
        device = torch.device("cuda", rank)
    else:
        device = torch.device(cfg.device)

    if cfg.multigpu:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=cfg.num_gpus)

    # dataloaders and augmentations
    if cfg.multigpu:
        print(f"Creating rank {rank} dataloaders...")
    else:
        print("Creating dataloaders...")
    num_workers = 8 if cfg.multigpu else 16  # completely empirically determined
    if cfg.amp:
        num_workers *= 2  # more workers for faster data loading with mixed precision
    try:
        train_dataset = CameraCubePoseDataset(cfg.dataset_config, cfg_aug=cfg.augmentation_config, train=True)
        val_dataset = CameraCubePoseDataset(cfg.dataset_config, cfg_aug=cfg.augmentation_config, train=False)

        if cfg.multigpu:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=cfg.num_gpus,
                rank=rank,
                shuffle=True,
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=cfg.num_gpus,
                rank=rank,
                shuffle=False,
            )
            train_shuffle = None
            val_shuffle = None
        else:
            train_sampler = None
            val_sampler = None
            train_shuffle = True
            val_shuffle = False

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=train_shuffle,
            num_workers=num_workers,
            pin_memory=True,
            sampler=train_sampler,
            multiprocessing_context="fork",  # this seems super important for speed when using multigpu!
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=val_shuffle,
            num_workers=num_workers,
            pin_memory=True,
            sampler=val_sampler,
            multiprocessing_context="fork",  # this seems super important for speed when using multigpu!
        )

    except RuntimeError:
        print("Data too large to load into memory. Please consider using a larger machine or a smaller dataset!")

    # model
    if cfg.multigpu:
        model = DDP(NCameraCNN(cfg.model_config).to(device), device_ids=[rank])
    else:
        model = NCameraCNN(cfg.model_config).to(device)

    # [DEBUG]
    # model.to(memory_format=torch.channels_last)  # this is a test to see if it speeds up the model

    if cfg.compile_model:
        model = torch.compile(model, mode="reduce-overhead")  # compiled model
        print("Using compiled model - first epoch will be slow!")

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    # loss function
    loss_fn = geometric_loss_fn

    # wandb
    wandb_id = generate_id()
    if cfg.wandb_log and rank == 0:
        wandb.init(project=cfg.wandb_project, config=cfg, id=wandb_id, resume="allow")

    # turn off profiling APIs
    if cfg.no_profiling:
        torch.autograd.profiler.record_function(enabled=False)
        torch.autograd.profiler.profile(enabled=False)
        torch.autograd.profiler.emit_nvtx(enabled=False)
        torch.autograd.set_detect_anomaly(mode=False)

    return (
        train_dataloader,
        val_dataloader,
        model,
        optimizer,
        scheduler,
        loss_fn,
        wandb_id,
        train_sampler,
        val_sampler,
        scaler,
    )


def rank_print(msg: str, rank: int = 0) -> None:
    """Prints only if rank is 0."""
    if rank == 0:
        print(msg)


def train(cfg: TrainConfig, rank: int = 0) -> None:
    """Main training loop."""
    # initializing the training
    (
        train_dataloader,
        val_dataloader,
        model,
        optimizer,
        scheduler,
        loss_fn,
        wandb_id,
        train_sampler,
        val_sampler,
        scaler,
    ) = initialize_training(cfg, rank=rank)

    # local device
    if cfg.multigpu:
        rank_print("Progress bar only shown for rank 0.", rank=rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device(cfg.device)

    for epoch in tqdm(range(cfg.n_epochs), desc="Epoch", disable=(rank != 0)):
        if cfg.multigpu:
            dist.barrier()
            train_sampler.set_epoch(epoch)

        # training loop
        model.train()
        avg_loss_in_epoch = []
        for example in tqdm(
            train_dataloader, desc="Iterations", total=len(train_dataloader), leave=False, disable=(rank != 0)
        ):
            with torch.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16, enabled=cfg.amp
            ):
                # loading data
                images = example["images"].to(device)  # (B, 6, H, W)
                # images = example["images"].to(device, memory_format=torch.channels_last).contiguous()  # (B, 6, H, W)
                cube_pose_SE3 = pp.SE3(example["cube_pose"].to(device))  # quats are (x, y, z, w)

                # forward pass
                cube_pose_pred_se3 = model(images)  # therefore, the predicted quats are (x, y, z, w)

            losses = loss_fn(cube_pose_pred_se3.to(torch.float32), cube_pose_SE3)
            loss = torch.mean(losses)

            if cfg.wandb_log and (not cfg.multigpu or rank == 0):
                wandb.log({"loss": loss.item()})

            # backward pass
            optimizer.zero_grad()
            # scaler.scale(loss).backward()
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            # scaler.step(optimizer)
            # scaler.update()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            avg_loss_in_epoch.append(losses)

        if epoch % cfg.print_epochs == 0:
            rank_print(f"    Avg. Loss in Epoch: {torch.mean(torch.cat(avg_loss_in_epoch)).item()}", rank=rank)

        # validation loop
        if epoch % cfg.val_epochs == 0:
            model.eval()
            with torch.no_grad():
                val_loss = []
                for example in val_dataloader:
                    images = example["images"].to(device)
                    # images = example["images"].to(device, memory_format=torch.channels_last).contiguous()
                    cube_pose_SE3 = pp.SE3(example["cube_pose"].to(device))
                    with torch.autocast(
                        device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16, enabled=cfg.amp
                    ):
                        cube_pose_pred_se3 = model(images)

                    losses = loss_fn(cube_pose_pred_se3.to(torch.float32), cube_pose_SE3)
                    val_loss.append(losses)

                val_loss = torch.mean(torch.cat(val_loss)).item()
                if cfg.wandb_log and (not cfg.multigpu or rank == 0):
                    wandb.log({"val_loss": val_loss})
                rank_print(f"    Validation loss: {val_loss}", rank=rank)

                # update learning rate based on val loss
                scheduler.step(val_loss)

        if epoch % cfg.save_epochs == 0:
            if cfg.save_dir is not None:
                save_dir = Path(cfg.save_dir)
            else:
                # Make outputs folder if not there.
                save_dir = Path(ROOT + "/outputs/models")
            os.makedirs(save_dir, exist_ok=True)
            if rank == 0:  # works for both single and multigpu
                torch.save(
                    getattr(model, "_orig_mod", model).state_dict(), save_dir / f"{wandb_id}.pth"
                )  # see: github.com/pytorch/pytorch/issues/101107#issuecomment-1869839379

    if cfg.multigpu:
        dist.destroy_process_group()


def _train_multigpu(rank: int, cfg: TrainConfig) -> None:
    """Trivial wrapper for train.

    Defined this way because rank must be the first argument in the function signature for mp.spawn.
    This function must also be defined at a module top level to allow pickling.
    """
    train(cfg, rank=rank)


if __name__ == "__main__":
    cfg = tyro.cli(TrainConfig)
    if cfg.multigpu:
        mp.spawn(_train_multigpu, args=(cfg,), nprocs=cfg.num_gpus, join=True)
    else:
        train(cfg, rank=0)
