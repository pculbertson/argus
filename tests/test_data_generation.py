from pathlib import Path
from sys import platform

import h5py
import numpy as np
import pytest

from argus import ROOT
from argus.data_generation import GenerateDataConfig, generate_data


def test_datagen(tmp_path_factory, dummy_json_path) -> None:
    """Test the datagen on making both 1 and multiple files."""
    # test whether mac or ubuntu
    if platform == "linux" or platform == "linux2":
        extension = ".x86_64"
    elif platform == "darwin":
        extension = ".app"
    elif platform == "win32":
        # skip the test with a warning
        pytest.skip("Skipping test on Windows platform.")
    else:
        # raise an error
        raise OSError("Unknown OS platform.")

    # try to find the unity executable
    env_exe_path = Path(ROOT) / f"outputs/unity/leap_env{extension}"
    if not env_exe_path.exists():
        # skip the test with a warning
        pytest.skip(f"Unity executable not found! This test assumes it is in {env_exe_path}.")

    # case 1: one file
    parent_dir = tmp_path_factory.mktemp("tmp")
    cfg1 = GenerateDataConfig(
        env_exe_path=str(env_exe_path),
        mjpc_data_path=dummy_json_path,
        output_data_path=str(parent_dir / "test1.hdf5"),
        n_agents=1,
        n_file_split=10,
    )
    generate_data(cfg1)
    assert Path(parent_dir / "test1.hdf5").exists()

    # case 2: multiple files
    cfg2 = GenerateDataConfig(
        env_exe_path=str(env_exe_path),
        mjpc_data_path=dummy_json_path,
        output_data_path=str(parent_dir / "test2.hdf5"),
        n_agents=1,
        n_file_split=5,
    )
    generate_data(cfg2)
    assert Path(parent_dir / "test2/test2_0.hdf5").exists()
    assert Path(parent_dir / "test2/test2_1.hdf5").exists()

    # check that in each case, the data is correctly saved
    # we know the true data generated in conftest.py
    cube_poses_true = np.repeat(np.linspace(0, 9, 10)[:, None], 7, axis=1)

    # case 1
    cube_poses = []
    with h5py.File(parent_dir / "test1.hdf5", "r") as f:
        for pose in f["train"]["cube_poses"]:
            cube_poses.append(pose)
        for pose in f["test"]["cube_poses"]:
            cube_poses.append(pose)

    # sort the data
    cube_poses = np.array(cube_poses)
    idxs = np.argsort(cube_poses[:, 0])
    cube_poses = cube_poses[idxs]
    assert np.allclose(cube_poses, cube_poses_true)

    # case 2
    cube_poses = []
    with h5py.File(parent_dir / "test2/test2_0.hdf5", "r") as f:
        for pose in f["train"]["cube_poses"]:
            cube_poses.append(pose)
        for pose in f["test"]["cube_poses"]:
            cube_poses.append(pose)
    with h5py.File(parent_dir / "test2/test2_1.hdf5", "r") as f:
        for pose in f["train"]["cube_poses"]:
            cube_poses.append(pose)
        for pose in f["test"]["cube_poses"]:
            cube_poses.append(pose)

    # sort the data
    cube_poses = np.array(cube_poses)
    idxs = np.argsort(cube_poses[:, 0])
    cube_poses = cube_poses[idxs]
    assert np.allclose(cube_poses, cube_poses_true)
