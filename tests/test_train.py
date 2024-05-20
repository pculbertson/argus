from pathlib import Path

import pypose as pp
import pytest
import torch
import wandb

from argus.data import CameraCubePoseDatasetConfig
from argus.models import NCameraCNNConfig
from argus.train import TrainConfig, geometric_loss_fn, train


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


def test_train(dummy_save_dir, dummy_data_path, dummy_model) -> None:
    """Tests that the training loop runs all the way through properly for 1 iteration."""
    train_cfg = TrainConfig(
        batch_size=10,
        learning_rate=1e-3,
        n_epochs=1,
        device="cuda",
        max_grad_norm=100.0,
        random_seed=42,
        val_epochs=1,
        print_epochs=1,
        save_epochs=1,
        save_dir=dummy_save_dir,
        model_config=NCameraCNNConfig(
            n_cams=2,
            W=672,
            H=376,
        ),
        dataset_config=CameraCubePoseDatasetConfig(
            dataset_path=dummy_data_path,
        ),
        compile_model=False,  # don't compile to speed up the test
        wandb_project="argus-estimator",
        wandb_log=False,  # don't log during test
    )
    try:
        train(train_cfg)
    except Exception:
        pytest.fail("The training loop did not run all the way through!")
    assert Path(dummy_save_dir).exists()
    assert any(Path(dummy_save_dir).glob("*.pth"))

    # for efficiency, also tests the random seed by training the same model twice
    dummy_model.load_state_dict(torch.load(list(Path(dummy_save_dir).glob("*.pth"))[0]))
    output1 = dummy_model(torch.ones(1, 2 * 3, 376, 672))
    for p in Path(dummy_save_dir).glob("*.pth"):
        p.unlink()  # deletes the old model
    train(train_cfg)
    dummy_model.load_state_dict(torch.load(list(Path(dummy_save_dir).glob("*.pth"))[0]))
    output2 = dummy_model(torch.ones(1, 2 * 3, 376, 672))
    assert torch.allclose(output1, output2)
