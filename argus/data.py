import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import kornia
import numpy as np
import pypose as pp
import torch
from torch.utils.data import Dataset

from argus import ROOT
from argus.utils import get_tree_string, xyzwxyz_to_xyzxyzw_SE3


@dataclass(frozen=True)
class CameraCubePoseDatasetConfig:
    """Configuration for the CameraCubePoseDataset.

    For all path fields, you can either specify an absolute path, a relative path (with respect to where you are
    currently calling the data generation function), or a local path RELATIVE TO THE ROOT OF THE PACKAGE IN YOUR SYSTEM!
    For instance, if you pass "example_dir/data.hdf5" to `dataset_path`, the data will be loaded from
    /path/to/argus/example_dir/datda.hdf5.

    Args:
        dataset_path: The path to the dataset. Must lead to an hdf5 file or a directory of hdf5 files. Specifically, if
            you supply the path /path/to/dir, then the files must have the form /path/to/dir/dir_{i}.hdf5, where i is
            some integer.
    """

    dataset_path: Optional[str] = None

    def __post_init__(self) -> None:
        """Checks that the dataset path is set and that it is a string for wandb serialization."""
        assert self.dataset_path is not None, (
            "The dataset path must be provided (either an hdf5 file or a directory with them)!\n"
            "Here is a tree of the `outputs/data` directory to help:\n"
            f"{get_tree_string(ROOT + '/outputs/data', 'hdf5')}"
        )
        # check whether there's an extension or not
        if not Path(self.dataset_path).suffix:
            # iterate over all files in the directory
            for file in os.listdir(self.dataset_path):
                assert Path(file).suffix == ".hdf5", "The dataset must consist of hdf5 files!"
                assert (
                    str(Path(file).stem).split("_")[-1].isdigit()
                ), "The dataset must be named as `cube_unity_data_{n}.hdf5`!"
                assert "_".join(str(Path(file).stem).split("_")[:-1]) == str(Path(self.dataset_path).stem), (
                    "The dataset must be named as `{dir_name}_{n}.hdf5` where the directory name is the same as "
                    "the dataset path!"
                )
        else:
            assert Path(self.dataset_path).suffix == ".hdf5", "The dataset must be stored as an hdf5 file!"
        assert isinstance(self.dataset_path, str), "The dataset path must be a str!"
        if not os.path.exists(self.dataset_path):  # absolute path
            if os.path.exists(ROOT + "/" + self.dataset_path):
                self.dataset_path = ROOT + "/" + self.dataset_path
            else:
                raise FileNotFoundError(f"The specified path does not exist: {self.dataset_path}!")


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
        # case 1: single file
        if Path(dataset_path).suffix:
            with h5py.File(dataset_path, "r") as f:
                if train:
                    dataset = f["train"]
                else:
                    dataset = f["test"]

                # extracting attributes
                self.n_cams = f.attrs["n_cams"]
                self.W = f.attrs["W"]
                self.H = f.attrs["H"]

                # grabbing the data
                _cube_poses = torch.from_numpy(dataset["cube_poses"][()])  # original quat order is (w, x, y, z)
                self.cube_poses = pp.SE3(xyzwxyz_to_xyzxyzw_SE3(_cube_poses))  # pp quat order is (x, y, z, w)
                self.images = dataset["images"][()]  # (n_data, n_cams, 3, H, W)

        # case 2: multiple files
        else:
            # iterate over all files in the directory
            for i, file in enumerate(os.listdir(dataset_path)):
                with h5py.File(dataset_path + "/" + file, "r") as f:
                    if train:
                        dataset = f["train"]
                    else:
                        dataset = f["test"]

                    # extracting attributes
                    self.n_cams = f.attrs["n_cams"]
                    self.W = f.attrs["W"]
                    self.H = f.attrs["H"]

                    # grabbing the data
                    _cube_poses = torch.from_numpy(dataset["cube_poses"][()])
                    if i == 0:
                        self.cube_poses = pp.SE3(xyzwxyz_to_xyzxyzw_SE3(_cube_poses))
                        self.images = dataset["images"][()]
                    else:
                        self.cube_poses = torch.cat(
                            (self.cube_poses, pp.SE3(xyzwxyz_to_xyzxyzw_SE3(_cube_poses))), dim=0
                        )
                        self.images = np.concatenate((self.images, dataset["images"][()]), axis=0)

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


@dataclass(frozen=True)
class AugmentationConfig:
    """Configuration for data augmentation."""

    # color jiggle
    brightness: float = 0.2
    contrast: float = 0.4
    saturation: float = 0.4
    hue: float = 0.025

    # flags
    color_jiggle: bool = True
    planckian_jitter: bool = True
    random_erasing: bool = True
    blur: bool = True


class Augmentation(torch.nn.Module):
    """Data augmentation module for the images and pixel coordinates."""

    def __init__(self, cfg: AugmentationConfig, train: bool = True) -> None:
        """Initialize the augmentations."""
        super().__init__()
        self.cfg = cfg
        self.train = train
        self.transforms = []

        # constructing the augmentation sequence
        if cfg.random_erasing:
            self.transforms.append(
                kornia.augmentation.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(2.0, 3.0), same_on_batch=False)
            )
            self.transforms.append(
                kornia.augmentation.RandomErasing(
                    p=0.5,
                    scale=(0.02, 0.05),
                    ratio=(0.8, 1.2),
                    same_on_batch=False,
                    value=1,
                )
            )

        if cfg.planckian_jitter:
            self.transforms.append(kornia.augmentation.RandomPlanckianJitter(mode="blackbody"))

        if cfg.color_jiggle:
            self.transforms.append(
                kornia.augmentation.ColorJiggle(
                    brightness=cfg.brightness,
                    contrast=cfg.contrast,
                    saturation=cfg.saturation,
                    hue=cfg.hue,
                )
            )

        if cfg.blur:
            self.transforms.append(kornia.augmentation.RandomGaussianBlur((5, 5), (3.0, 8.0), p=0.5))

        self.transform_op = kornia.augmentation.AugmentationSequential(*self.transforms, data_keys=["image"])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Applies the augmentations to the images."""
        if len(self.transforms) > 0 and self.train:
            images = self.transform_op(images)
        return images


if __name__ == "__main__":
    # [DEBUG] do a dry run of the datagen + save images to file
    import cv2
    import tyro

    dataset_cfg = CameraCubePoseDatasetConfig(dataset_path=ROOT + "/outputs/data/cube_unity_data_medium.hdf5")
    augmentation_cfg = tyro.cli(AugmentationConfig)
    train_dataset = CameraCubePoseDataset(dataset_cfg, train=True)

    augmentation = Augmentation(augmentation_cfg, train=True)

    # Read and augment first image, and display with opencv.
    for ii in range(len(train_dataset)):
        imgs = train_dataset[ii]["images"]
        imgs = augmentation(imgs.reshape(-1, 3, train_dataset.H, train_dataset.W)).numpy()[0]
        cv2.imshow("image", imgs.transpose(1, 2, 0))

        cv2.waitKey(0)

    cv2.waitKey(0)
