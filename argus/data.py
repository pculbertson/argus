import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import h5py
import kornia
import numpy as np
import pypose as pp
import torch
from PIL import Image
from torch.utils.data import Dataset

from argus import ROOT
from argus.utils import draw_spaghetti, get_tree_string, xyzwxyz_to_xyzxyzw_SE3


@dataclass(frozen=True)
class AugmentationConfig:
    """Configuration for data augmentation."""

    # color jiggle
    brightness: Union[float, tuple[float, float]] = (0.8, 1.0)
    contrast: Union[float, tuple[float, float]] = (0.5, 1.2)
    saturation: Union[float, tuple[float, float]] = (0.25, 1.2)
    hue: Union[float, tuple[float, float]] = (-0.1, 0.1)

    # spaghetti
    num_spaghetti: int = 10

    # flags
    color_jiggle: bool = True
    planckian_jitter: bool = True
    random_erasing: bool = False
    blur: bool = True
    motion_blur: bool = True
    plasma_shadow: bool = True
    salt_and_pepper: bool = False


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
                    same_on_batch=True,
                    p=1.0,
                )
            )

        if cfg.blur:
            self.transforms.append(kornia.augmentation.RandomGaussianBlur((5, 5), (3.0, 8.0), p=0.5))

        if cfg.motion_blur:
            self.transforms.append(kornia.augmentation.RandomMotionBlur(3, 35.0, 0.5, p=0.7))

        if cfg.plasma_shadow:
            self.transforms.append(
                kornia.augmentation.RandomPlasmaShadow(
                    roughness=(0.1, 0.4), shade_intensity=(-0.6, 0.0), shade_quantity=(0.0, 0.5), p=1.0
                )
            )

        if cfg.salt_and_pepper:
            self.transforms.append(kornia.augmentation.RandomSaltAndPepperNoise(p=0.7))

        self.transform_op = kornia.augmentation.AugmentationSequential(*self.transforms, data_keys=["image"])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Applies the augmentations to the images."""
        if len(self.transforms) > 0 and self.train:
            images = self.transform_op(images)
        return images


@dataclass(frozen=True)
class CameraCubePoseDatasetConfig:
    """Configuration for the CameraCubePoseDataset.

    For all path fields, you can either specify an absolute path, a relative path (with respect to where you are
    currently calling the data generation function), or a local path RELATIVE TO THE ROOT OF THE PACKAGE IN YOUR SYSTEM!
    For instance, if you pass "example_dir/data" to `dataset_path`, the data will be loaded from
    /path/to/argus/example_dir/data/data.hdf5 and the image data will be from /path/to/argus/example_dir/data/img/.

    Args:
        dataset_path: The path to the dataset. Must lead to a directory with an hdf5 file and images.
        center_crop: The size of the center crop. Specified as the height and then the width of the crop.
    """

    dataset_path: Optional[str] = None
    center_crop: Optional[tuple[int, int]] = (256, 256)

    def __post_init__(self) -> None:
        """Checks that the dataset path is set and that it is a string for wandb serialization."""
        assert isinstance(self.dataset_path, str), "The dataset path must be a str!"
        if not os.path.exists(self.dataset_path):  # absolute path
            if os.path.exists(ROOT + "/" + self.dataset_path):
                self.dataset_path = ROOT + "/" + self.dataset_path
            else:
                raise FileNotFoundError(f"The specified path does not exist: {self.dataset_path}!")
        assert self.dataset_path is not None, (
            "The dataset path must be provided!\n"
            "Here is a tree of the `outputs/data` directory to help:\n"
            f"{get_tree_string(ROOT + '/outputs/data', 'hdf5')}"
        )
        # check whether there's an extension
        assert not Path(self.dataset_path).suffix, "The dataset path must point to a directory!"
        if Path(self.dataset_path).is_dir():
            assert os.path.exists(
                self.dataset_path + f"/{Path(self.dataset_path).stem}.hdf5"
            ), f"There must be an hdf5 file with the name {Path(self.dataset_path).stem}.hdf5!"
            assert os.path.exists(self.dataset_path + "/img"), "The dataset must have an `img` directory!"


class CameraCubePoseDataset(Dataset):
    """The dataset for N cameras and a cube."""

    def __init__(
        self, cfg_dataset: CameraCubePoseDatasetConfig, cfg_aug: Optional[AugmentationConfig] = None, train: bool = True
    ) -> None:
        """Initializes the dataset.

        It is stored in an hdf5 file as follows.

        Attributes:
            - n_cams: the number of cameras.
            - W: the width of the images.
            - H: the height of the images.
        Structure:
            - train
                - cube_poses: The poses of the cube of shape (n_data, 7), (x, y, z, qw, qx, qy, qz).
                - q_leap: The state of the LEAP hand of shape (n_data, 16). Mostly used for debugging or viz.
                - img_stems: The paths to the dataset images relative to the dataset root.
            - test
                - same fields as `train`.

        Args:
            cfg_dataset: The configuration for the dataset.
            cfg_aug: The configuration for the data augmentation.
            train: Whether to load the training or test set. Default=True.
        """
        # loading the data
        dataset_path = cfg_dataset.dataset_path
        with h5py.File(dataset_path + f"/{Path(dataset_path).stem}.hdf5", "r") as f:
            if train:
                dataset = f["train"]
            else:
                dataset = f["test"]

            # extracting attributes
            self.n_cams = f.attrs["n_cams"]

            # grabbing the data
            _cube_poses = torch.from_numpy(dataset["cube_poses"][()])  # original quat order is (w, x, y, z)
            self.cube_poses = pp.SE3(xyzwxyz_to_xyzxyzw_SE3(_cube_poses))  # pp quat order is (x, y, z, w)
            self.q_leap = torch.from_numpy(dataset["q_leap"][()])
            _img_stems = dataset["img_stems"][()]
            self.img_stems = [byte_string.decode("utf-8") for byte_string in _img_stems]

        # composing augmentation transform
        if cfg_aug is not None:
            self.augmentation = Augmentation(cfg_aug, train=train)
        else:
            self.augmentation = None

        self.cfg_aug = cfg_aug

        # assigning useful attributes
        self.dataset_path = dataset_path
        self.center_crop = cfg_dataset.center_crop

    def __len__(self) -> int:
        """Number of datapoints, i.e., (N image, cube pose) tuples."""
        return self.cube_poses.shape[0]

    def __getitem__(self, idx: int) -> dict:
        """Returns the idx-th datapoint."""
        img_stem = self.img_stems[idx]
        img_a = Image.open(f"{self.dataset_path}/{img_stem}_a.png")  # (H, W, 3)
        img_b = Image.open(f"{self.dataset_path}/{img_stem}_b.png")  # (H, W, 3)

        # Draw random arcs if self.augmentation.spaghetti is True
        if self.cfg_aug.num_spaghetti > 0:
            img_a = draw_spaghetti(img_a, self.cfg_aug.num_spaghetti)
            img_b = draw_spaghetti(img_b, self.cfg_aug.num_spaghetti)

        _images = np.concatenate([np.array(img_a), np.array(img_b)], axis=-1).transpose(2, 0, 1)  # (n_cams * 3, H, W)
        images = torch.tensor(_images) / 255.0  # (n_cams * 3, H, W)
        if self.center_crop and images.shape[-2:] != self.center_crop:  # crop if not already cropped and requested
            images = kornia.geometry.transform.center_crop(
                images.unsqueeze(0), (self.center_crop[0], self.center_crop[1])
            ).squeeze(0)
        if self.augmentation is not None:
            H, W = images.shape[-2:]
            images = self.augmentation(images.reshape((self.n_cams, 3, H, W))).reshape(-1, H, W)
        return {
            "images": images.to(torch.float32),
            "cube_pose": self.cube_poses[idx].to(torch.float32),
        }


if __name__ == "__main__":
    # [DEBUG] do a dry run of the datagen + save images to file
    import cv2
    import tyro

    dataset_cfg = CameraCubePoseDatasetConfig(dataset_path=ROOT + "/outputs/data/cube_unity_data_large")
    augmentation_cfg = tyro.cli(AugmentationConfig)
    train_dataset = CameraCubePoseDataset(dataset_cfg, cfg_aug=augmentation_cfg, train=True)

    # Read and augment first image, and display with opencv.
    for ii in range(len(train_dataset)):
        imgs = train_dataset[ii]["images"]
        H, W = imgs.shape[-2:]
        img = imgs.reshape((train_dataset.n_cams, 3, H, W))[0]
        cv2.imshow("image", img.permute(1, 2, 0).numpy())

        cv2.waitKey(0)

    cv2.waitKey(0)
