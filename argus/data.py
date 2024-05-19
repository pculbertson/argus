from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import kornia
import numpy as np
import pypose as pp
import torch
from torch.utils.data import Dataset

from argus.utils import xyzwxyz_to_xyzxyzw_SE3


@dataclass(frozen=True)
class CameraCubePoseDatasetConfig:
    """Configuration for the CameraCubePoseDataset.

    Args:
        dataset_path: The path to the dataset. Must lead to an hdf5 file.
        train: Whether to load the training or test set.
    """

    dataset_path: Optional[str] = None

    def __post_init__(self) -> None:
        """Checks that the dataset path is set and that it is a string for wandb serialization."""
        assert self.dataset_path is not None, "The dataset path must be set!"
        assert isinstance(self.dataset_path, str), "The dataset path must be a str!"


class CameraCubePoseDataset(Dataset):
    """The dataset for N cameras and a cube."""

    def __init__(self, cfg: CameraCubePoseDatasetConfig, train: bool = True) -> None:
        """Initializes the dataset.

        It is stored in an hdf5 file as follows.

        Attributes:
            - n_cams: the number of cameras.
            - W: the width of the images.
            - H: the height of the images.
        Structure:
            - train
                - images: The images of the cube of shape (n_data, n_cams, C, H, W). The pixel values should be
                    normalized between 0 and 1. There should be no alpha channel. When the images are retrieved
                    from the dataset, we flatten the shape to (n_cams * C, H, W)!
                - cube_poses: The poses of the cube of shape (n_data, 7), (x, y, z, qw, qx, qy, qz).
            - test
                - same fields as `train`.

        Args:
            cfg: The configuration for the dataset.
            train: Whether to load the training or test set. Default=True.
        """
        dataset_path = cfg.dataset_path

        assert Path(dataset_path).suffix == ".hdf5", "The dataset must be stored as an hdf5 file!"
        with h5py.File(dataset_path, "r") as f:
            if train:
                self.dataset = f["train"]
            else:
                self.dataset = f["test"]

            # extracting attributes
            self.n_cams = f.attrs["n_cams"]
            self.W = f.attrs["W"]
            self.H = f.attrs["H"]

            # grabbing the data
            _cube_poses = torch.from_numpy(self.dataset["cube_poses"][()])  # original quat order is (w, x, y, z)
            self.cube_poses = pp.SE3(xyzwxyz_to_xyzxyzw_SE3(_cube_poses))  # pp expects quat order to be (x, y, z, w)
            self.images = self.dataset["images"][()]  # (n_data, n_cams, 3, H, W)

    def __len__(self) -> int:
        """Number of datapoints, i.e., (N image, cube pose) tuples."""
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        """Returns the idx-th datapoint."""
        images = torch.tensor(self.images[idx]).reshape((-1, self.H, self.W))  # (n_cams * 3, H, W)
        return {
            "images": images.to(torch.float32),
            "cube_pose": self.cube_poses[idx],
        }
