import json
from pathlib import Path

import h5py
import numpy as np
import pypose as pp
import pytest
from PIL import Image

from argus.models import NCameraCNN, NCameraCNNConfig
from argus.utils import xyzxyzw_to_xyzwxyz_SE3


@pytest.fixture(scope="session")
def dummy_data_path(tmp_path_factory) -> str:
    """A fixture for creating a dummy dataset used in all tests."""

    def create_dataset(group, n_data, n_data_start=0):
        cube_poses = xyzxyzw_to_xyzwxyz_SE3(pp.randn_SE3(n_data)).numpy()  # (x, y, z, qw, qx, qy, qz)
        q_leap = np.random.randn(n_data, 23)
        group.create_dataset("cube_poses", data=cube_poses)
        group.create_dataset("q_leap", data=q_leap)
        img_stems = [f"img/img{i}" for i in range(n_data_start, n_data_start + n_data)]
        group.create_dataset("img_stems", data=np.array([s.encode("utf-8") for s in img_stems]))

    # creating dummy data
    n_data = 15
    dir_path = tmp_path_factory.mktemp("tmp")
    dummy_file = Path(dir_path) / f"{dir_path.stem}.hdf5"
    img_dir = dir_path / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_data):
        img_stem = f"img{i}"

        img_a = np.random.rand(256, 256, 3)
        dummy_img_a = Image.fromarray((img_a * 255).astype(np.uint8))
        dummy_img_a.save(Path(img_dir / f"{img_stem}_a.png"))

        img_b = np.random.rand(256, 256, 3)
        dummy_img_b = Image.fromarray((img_b * 255).astype(np.uint8))
        dummy_img_b.save(Path(img_dir / f"{img_stem}_b.png"))

    with h5py.File(dummy_file, "w") as f:
        # attributes
        f.attrs["n_cams"] = 2
        f.attrs["W"] = 256
        f.attrs["H"] = 256

        # create the training set
        train = f.create_group("train")
        create_dataset(train, 10)

        # create the test set
        test = f.create_group("test")
        create_dataset(test, 5, n_data_start=10)

    return str(dir_path)


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
    return NCameraCNN(NCameraCNNConfig(n_cams=2))
