import pytest
import torch

from argus.models import NCameraCNN, NCameraCNNConfig


@pytest.fixture(scope="session")
def model() -> NCameraCNN:
    """A fixture for the model."""
    return NCameraCNN(NCameraCNNConfig(n_cams=2, W=672, H=376))


def test_forward(model: NCameraCNN) -> None:
    """Tests the forward pass."""
    # unbatched, assert throws error
    x = torch.randn(6, 376, 672)
    with pytest.raises(AssertionError):
        pose = model(x)

    # batch
    x = torch.randn(2, 6, 376, 672)
    pose = model(x)
    assert pose.shape == (2, 6)
