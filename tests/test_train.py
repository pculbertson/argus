import pypose as pp
import torch
import wandb

from argus.train import TrainConfig, geometric_loss_fn


def test_wandb() -> None:
    """Tests whether wandb is set up correctly."""
    assert wandb.login()


def test_geometric_loss_fn() -> None:
    """Tests the geometric loss function."""
    # unbatched
    pred = torch.randn(6)
    target = pp.randn_SE3()
    loss = geometric_loss_fn(pred, target)
    assert loss.shape == torch.Size([])

    # batched
    pred = torch.randn(32, 6)
    target = pp.randn_SE3(32)
    loss = geometric_loss_fn(pred, target)
    assert loss.shape == torch.Size([32])
