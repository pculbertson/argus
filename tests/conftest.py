import json

import h5py
import numpy as np
import pypose as pp
import pytest

from argus.models import NCameraCNN, NCameraCNNConfig
from argus.utils import xyzxyzw_to_xyzwxyz_SE3


@pytest.fixture(scope="session")
def dummy_data_path(tmp_path_factory) -> str:
    """A fixture for creating a dummy dataset used in all tests."""

    def create_dataset(group, n_data):
        images = np.random.rand(n_data, 2, 3, 376, 672)  # uniform on [0, 1]
        cube_poses = xyzxyzw_to_xyzwxyz_SE3(pp.randn_SE3(n_data)).numpy()  # (x, y, z, qw, qx, qy, qz)
        group.create_dataset("images", data=images)
        group.create_dataset("cube_poses", data=cube_poses)

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

    return str(dummy_file)


@pytest.fixture(scope="session")
def dummy_json_path(tmp_path_factory) -> str:
    """A fixture for creating a dummy JSON file used in all tests."""
    dummy_file = tmp_path_factory.mktemp("tmp") / "dummy_sim_residuals.json"
    data = []
    for i in range(10):
        data.append(
            {
                "dt": 0.01,
                "s": [i for _ in range(45)],
                "a": [i for _ in range(16)],
                "sp_pred": [i for _ in range(45)],
                "sp_actual": [i for _ in range(45)],
            }
        )
    with open(dummy_file, "w") as f:
        json.dump(data, f)
    return str(dummy_file)


@pytest.fixture(scope="session")
def dummy_save_dir(tmp_path_factory) -> str:
    """A fixture for the save directory."""
    dummy_dir = tmp_path_factory.mktemp("tmp") / "outputs/models"
    dummy_dir.mkdir(parents=True, exist_ok=True)
    return str(dummy_dir)


@pytest.fixture(scope="session")
def dummy_model() -> NCameraCNN:
    """A fixture for the model."""
    return NCameraCNN(NCameraCNNConfig(n_cams=2, W=672, H=376))
