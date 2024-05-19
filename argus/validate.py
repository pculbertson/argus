import os
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import pypose as pp
import torch
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader
from tqdm import tqdm

from argus import ROOT
from argus.data import CameraCubePoseDataset, CameraCubePoseDatasetConfig
from argus.models import NCameraCNN, NCameraCNNConfig
from argus.train import geometric_loss_fn


def plot_axes_from_pose(pose: pp.SE3, true: bool, ax: Optional[plt.Axes] = None) -> None:
    """Plots the axes of the pose.

    Args:
        pose: The pose to plot.
        true: Whether the pose is the true pose.
        ax: The axes to plot on. If None, a new figure is created.

    Returns:
        The axes object with the plotted pose axes.
    """
    pose = pose.matrix().detach().cpu().numpy()
    origin = pose[:3, -1]
    x_axis = pose[:3, 0]
    y_axis = pose[:3, 1]
    z_axis = pose[:3, 2]
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    ls = "-" if true else "--"
    ax.quiver(*origin, *x_axis, color="r", label="x", linestyle=ls, length=0.5)
    ax.quiver(*origin, *y_axis, color="g", label="y", linestyle=ls, length=0.5)
    ax.quiver(*origin, *z_axis, color="b", label="z", linestyle=ls, length=0.5)
    return ax


@dataclass(frozen=True)
class ValConfig:
    """The configuration dataclass for validation.

    Fields:
        model_path: The path to the model to validate.
        model_config: The configuration for the model.
        dataset_config: The configuration for the dataset.
        use_train: Whether to use the training set.
        device: The device to run on.
    """

    model_path: str
    dataset_config: CameraCubePoseDatasetConfig
    model_config: Optional[NCameraCNNConfig] = None
    use_train: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        """Sanity checks on inputs."""
        assert isinstance(self.model_path, str), "The model path must be a str!"
        assert self.model_path.endswith(".pth"), "The model path must end with '.pth'!"


def validate(cfg: ValConfig) -> None:
    """Validates the model on the dataset."""
    # unpacking config
    model_path = cfg.model_path
    model_config = cfg.model_config
    dataset_cfg = cfg.dataset_config
    use_train = cfg.use_train
    device = cfg.device

    # loading model
    ckpt_name = os.path.basename(model_path).split(".")[0]
    train_or_val = "train" if use_train else "validation"
    output_path = ROOT + f"/outputs/{train_or_val}_visuals/{ckpt_name}"

    model = NCameraCNN(model_config)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # dataloader
    dataset = CameraCubePoseDataset(dataset_cfg, train=use_train)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # for each data example, plot the true and predicted cube pose
    for i, example in tqdm(enumerate(dataloader), total=len(dataloader)):
        # forward pass
        images = example["images"].to(device).to(torch.float32)
        cube_pose_true = example["cube_pose"].to(device).to(torch.float32)
        cube_pose_pred = model(images)
        loss = torch.mean(geometric_loss_fn(cube_pose_pred, cube_pose_true))

        # plot the true and predicted cube poses
        fig = plt.figure(figsize=plt.figaspect(1.0 / 3.0))
        fig.suptitle(f"Cube Pose Prediction Validation | Checkpoint: {ckpt_name}")

        ax = fig.add_subplot(131, projection="3d")
        ax = plot_axes_from_pose(cube_pose_true[0], true=True, ax=ax)
        ax = plot_axes_from_pose(cube_pose_pred[0], true=False, ax=ax)
        ax.set_title(f"Example {i} | Loss: {loss.item():.3f}")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_aspect("equal")
        custom_lines = [
            Line2D([0], [0], color="black", linestyle="-", label="true"),
            Line2D([0], [0], color="black", linestyle="--", label="pred"),
        ]
        ax.legend(handles=custom_lines)

        ax = fig.add_subplot(132)
        image1 = images[0, :3].detach().cpu().numpy()
        ax.imshow(image1.transpose(1, 2, 0))
        ax.set_title("Camera 1")
        ax.axis("off")

        ax = fig.add_subplot(133)
        image2 = images[0, 3:6].detach().cpu().numpy()
        ax.imshow(image2.transpose(1, 2, 0))
        ax.set_title("Camera 2")
        ax.axis("off")

        # save the figure
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        fig.savefig(output_path + f"/example_{i}.png", bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    model_path = ROOT + "/outputs/models/c4vibi51.pth"
    dataset_cfg = CameraCubePoseDatasetConfig(dataset_path=ROOT + "/cube_unity_data.hdf5")
    cfg = ValConfig(model_path, dataset_cfg)
    validate(cfg)
