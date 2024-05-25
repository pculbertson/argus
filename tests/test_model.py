import pytest
import torch

from argus.models import NCameraCNN


def test_forward(dummy_model: NCameraCNN) -> None:
    """Tests the forward pass."""
    # unbatched, assert throws error
    x = torch.randn(6, 256, 256)
    with pytest.raises(AssertionError):
        pose = dummy_model(x)

    # batch
    x = torch.randn(2, 6, 256, 256)
    pose = dummy_model(x)
    assert pose.shape == (2, 6)
