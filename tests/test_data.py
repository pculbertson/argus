import h5py
import numpy as np
import pypose as pp
import pytest
import torch

from argus.data import CameraCubePoseDataset, CameraCubePoseDatasetConfig
from argus.utils import xyzxyzw_to_xyzwxyz_SE3

# ######### #
# UTILITIES #
# ######### #


@pytest.fixture(scope="session")
def dummy_data(tmp_path_factory):
    """A fixture for creating a dummy dataset used in all tests."""

    def create_dataset(group, n_data):
        images = np.random.rand(n_data, 2, 376, 672, 4)  # uniform on [0, 1]
        cube_poses = xyzxyzw_to_xyzwxyz_SE3(pp.randn_SE3(n_data)).numpy()  # (x, y, z, qw, qx, qy, qz)
        image_filenames = [(f"img_{i}a_test.png", f"img_{i}b_test.png") for i in range(n_data)]
        group.create_dataset("images", data=images)
        group.create_dataset("cube_poses", data=cube_poses)
        group.create_dataset("image_filenames", data=image_filenames)

    dummy_file = tmp_path_factory.mktemp("tmp") / "dummy_dataset.hdf5"
    with h5py.File(dummy_file, "w") as f:
        # attributes
        f.attrs["n_cams"] = 2
        f.attrs["W"] = 672
        f.attrs["H"] = 376

        # create the training set
        train = f.create_group("train")
        create_dataset(train, 10)

        # create the test set
        test = f.create_group("test")
        create_dataset(test, 5)

    return dummy_file


def run_assertions(dataset, expected_len):
    """A helper function to run assertions on the items in the dataset."""
    assert (
        len(dataset) == expected_len
    ), f"The length of the dataset is incorrect! Expected {expected_len}, got {len(dataset)}"
    example = dataset[0]
    assert set(example.keys()) == {"images", "cube_pose", "image_filenames"}, "The keys of the example are incorrect!"
    assert example["images"].shape == (2 * 3, 376, 672), "The shape of the images is incorrect!"
    assert example["cube_pose"].shape == (7,), "The shape of the cube poses is incorrect!"
    assert len(example["image_filenames"]) == 2, "The image filenames should be length 2!"
    assert isinstance(example["image_filenames"], tuple), "The image filenames should be a tuple!"


# ##### #
# TESTS #
# ##### #


def test_len(dummy_data):
    """Tests the __len__ method of the dataset."""
    # load the dataset
    cfg = CameraCubePoseDatasetConfig(dummy_data, train=True)
    dataset = CameraCubePoseDataset(cfg)
    run_assertions(dataset, 10)

    cfg = CameraCubePoseDatasetConfig(dummy_data, train=False)
    dataset = CameraCubePoseDataset(cfg)
    run_assertions(dataset, 5)


def test_get_item(dummy_data):
    """Tests the __getitem__ method of the dataset."""
    # load the dataset
    cfg = CameraCubePoseDatasetConfig(dummy_data, train=True)
    dataset = CameraCubePoseDataset(cfg)
    run_assertions(dataset, 10)

    cfg = CameraCubePoseDatasetConfig(dummy_data, train=False)
    dataset = CameraCubePoseDataset(cfg)
    run_assertions(dataset, 5)
