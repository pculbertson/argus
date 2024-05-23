import numpy as np
import pypose as pp
import pytest
import torch

from argus.utils import (
    convert_pose_mjpc_to_unity,
    convert_pose_unity_to_mjpc,
    convert_unity_quat_to_euler,
    get_pose,
    xyzwxyz_to_xyzxyzw_SE3,
    xyzxyzw_to_xyzwxyz_SE3,
)


def test_xyzwxyz_to_xyzxyzw_SE3():
    """Tests the conversion of (x, y, z, qw, qx, qy, qz) to (x, y, z, qx, qy, qz, qw)."""
    # single dimension
    xyzwxyz = torch.tensor([1, 2, 3, 0.5, 0.6, 0.7, 0.8])
    expected = torch.tensor([1, 2, 3, 0.6, 0.7, 0.8, 0.5])
    assert torch.allclose(xyzwxyz_to_xyzxyzw_SE3(xyzwxyz), expected)

    # batched
    xyzwxyz = torch.tensor([[1, 2, 3, 0.5, 0.6, 0.7, 0.8], [4, 5, 6, 0.1, 0.2, 0.3, 0.4]])
    expected = torch.tensor([[1, 2, 3, 0.6, 0.7, 0.8, 0.5], [4, 5, 6, 0.2, 0.3, 0.4, 0.1]])
    assert torch.allclose(xyzwxyz_to_xyzxyzw_SE3(xyzwxyz), expected)


def test_xyzxyzw_to_xyzwxyz_SE3():
    """Tests the conversion of (x, y, z, qx, qy, qz, qw) to (x, y, z, qw, qx, qy, qz)."""
    # single dimension
    xyzxyzw = torch.tensor([1, 2, 3, 0.6, 0.7, 0.8, 0.5])
    expected = torch.tensor([1, 2, 3, 0.5, 0.6, 0.7, 0.8])
    assert torch.allclose(xyzxyzw_to_xyzwxyz_SE3(xyzxyzw), expected)

    # batched
    xyzxyzw = torch.tensor([[1, 2, 3, 0.6, 0.7, 0.8, 0.5], [4, 5, 6, 0.2, 0.3, 0.4, 0.1]])
    expected = torch.tensor([[1, 2, 3, 0.5, 0.6, 0.7, 0.8], [4, 5, 6, 0.1, 0.2, 0.3, 0.4]])
    assert torch.allclose(xyzxyzw_to_xyzwxyz_SE3(xyzxyzw), expected)

    # with pypose
    xyzxyzw = pp.randn_SE3(2)
    try:
        assert torch.allclose(xyzxyzw_to_xyzwxyz_SE3(xyzwxyz_to_xyzxyzw_SE3(xyzxyzw)), xyzxyzw)
    except Exception:
        pytest.fail("The conversion failed on a pypose SE3 object!")


def test_convert_pose_mjpc_to_unity() -> None:
    """Tests the conversion from Mujoco's to Unity's coordinate system."""
    # test rotation about x
    pose_mjpc = np.array([[0.1, 0.2, 0.3, 0.92387953, 0.38268343, 0.0, 0.0]])  # rotate +45 about +x in mjpc
    pose_unity = convert_pose_mjpc_to_unity(pose_mjpc)
    euler = convert_unity_quat_to_euler(pose_unity[0, 3:])
    assert np.allclose(pose_unity, np.array([[-0.2, 0.3, 0.1, 0.0, 0.0, -0.38268343, 0.92387953]]))
    assert np.allclose(euler, np.array([0.0, 0.0, -45.0]))

    # test rotation about y
    pose_mjpc = np.array([[0.1, 0.2, 0.3, 0.92387953, 0.0, 0.38268343, 0.0]])  # rotate +45 about +y in mjpc
    pose_unity = convert_pose_mjpc_to_unity(pose_mjpc)
    euler = convert_unity_quat_to_euler(pose_unity[0, 3:])
    assert np.allclose(pose_unity, np.array([[-0.2, 0.3, 0.1, 0.38268343, 0.0, 0.0, 0.92387953]]))
    assert np.allclose(euler, np.array([45.0, 0.0, 0.0]))

    # test rotation about z
    pose_mjpc = np.array([[0.1, 0.2, 0.3, 0.92387953, 0.0, 0.0, 0.38268343]])  # rotate +45 about +z in mjpc
    pose_unity = convert_pose_mjpc_to_unity(pose_mjpc)
    euler = convert_unity_quat_to_euler(pose_unity[0, 3:])
    assert np.allclose(pose_unity, np.array([[-0.2, 0.3, 0.1, 0.0, -0.38268343, 0.0, 0.92387953]]))
    assert np.allclose(euler, np.array([0.0, -45.0, 0.0]))


def test_convert_pose_unity_to_mjpc() -> None:
    """Tests the conversion from Unity's to Mujoco's coordinate system."""
    # test by converting a pose from mjpc to unity and back to mjpc
    pose_mjpc = np.random.rand(2, 7)  # implicitly tests batching
    pose_mjpc[..., 3:] /= np.linalg.norm(pose_mjpc[..., 3:], axis=-1, keepdims=True)
    assert np.allclose(pose_mjpc, convert_pose_unity_to_mjpc(convert_pose_mjpc_to_unity(pose_mjpc)))


def test_get_pose(dummy_model) -> None:
    """Tests the get_pose function with a compiled model."""
    x = torch.randn(2, 6, 376, 672)
    model_compiled = torch.compile(dummy_model, mode="reduce-overhead")
    pose = get_pose(x, model_compiled)
    assert pose.shape == (2, 7)  # should be a pypose object
