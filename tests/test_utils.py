import torch

from argus.utils import xyzwxyz_to_xyzxyzw_SE3, xyzxyzw_to_xyzwxyz_SE3


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
