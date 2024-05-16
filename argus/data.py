from pathlib import Path

import h5py
import kornia
import numpy as np
import pypose as pp
import torch
from torch.utils.data import Dataset

from argus.utils import xyzwxyz_to_xyzxyzw_SE3


class CameraCubePoseDataset(Dataset):
    """The dataset for N cameras and a cube."""

    def __init__(self, dataset_path: Path, train: bool = True) -> None:
        """Initializes the dataset.

        It is stored in an hdf5 file as follows.

        Attributes:
            - n_cams: the number of cameras.
            - W: the width of the images.
            - H: the height of the images.
        Structure:
            - train
                - images: The images of the cube of shape (n_data, n_cams, H, W, C). The pixel values should be
                    normalized between 0 and 1. The alpha channel should be the last one. When the images are retrieved
                    from the dataset, we swap the shape order to become (n_cams * C, H, W)!
                - cube_poses: The poses of the cube of shape (n_data, 7), (x, y, z, qw, qx, qy, qz).
                - image_filenames: The filenames of the associated data given as tuples of length n_cams.
                    Primarily used for debugging.
            - test
                - same fields as `train`.

        Args:
            dataset_path: The path to the dataset. Must lead to an hdf5 file.
            train: Whether to load the training or test set.
        """
        assert dataset_path.suffix == ".hdf5", "The dataset must be stored as an hdf5 file!"
        with h5py.File(dataset_path, "r") as f:
            if self.train:
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
            self.images = self.dataset["images"][()][..., :3]  # (n_data, n_cams, H, W, 3)
            self.image_filenames = self.dataset["image_filenames"][()]  # list of tuples, (n_data, (n_cams,))

    def __len__(self) -> int:
        """Number of datapoints, i.e., (N image, cube pose) tuples."""
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        """Returns the idx-th datapoint."""
        # converts image shapes (n_cams, H, W, C) -> (n_cams * 3, H, W)
        # see: github.com/kornia/kornia/blob/3ce96a35bedf505bf416af21e5f01b5861c998df/kornia/utils/image.py#L10
        images = kornia.utils.image_to_tensor(self.images[idx]).reshape((-1, self.H, self.W))
        return {
            "images": images,
            "cube_pose": self.object_poses[idx],
            "image_filenames": self.image_filenames[idx],
        }
