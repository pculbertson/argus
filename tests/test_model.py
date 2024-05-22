import pytest
import torch

from argus.models import NCameraCNN


def test_forward(dummy_model_se3: NCameraCNN, dummy_model_cts_6d: NCameraCNN) -> None:
    """Tests the forward pass of both model types."""
    # unbatched, assert throws error
    x = torch.randn(6, 376, 672)
    with pytest.raises(AssertionError):
        pose = dummy_model_se3(x)

    # batch, se3
    x = torch.randn(2, 6, 376, 672)
    pose = dummy_model_se3(x)
    assert pose.shape == (2, 6)

    # batch, cts_6d
    x = torch.randn(2, 6, 376, 672)
    pose = dummy_model_cts_6d(x)
    assert pose.shape == (2, 9)
