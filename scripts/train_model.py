from argus.data import CameraCubePoseDatasetConfig
from argus.models import NCubeCNNConfig
from argus.train import TrainConfig, train


def main(dataset_path: str):
    """Sets up the training."""
    train_cfg = TrainConfig(
        batch_size=128,
        learning_rate=1e-3,
        n_epochs=100,
        device="cuda",
        max_grad_norm=100.0,
        val_epochs=1,
        print_epochs=1,
        save_epochs=5,
        model_config=NCubeCNNConfig(
            n_cams=2,
            W=672,
            H=376,
        ),
        dataset_config=CameraCubePoseDatasetConfig(
            dataset_path=dataset_path,
        ),
        wandb_project="argus-estimator",
    )
    train(train_cfg)


if __name__ == "__main__":
    dataset_path = "path/to/dataset.hdf5"  # TODO(ahl): update this when we have a dataset
    main(dataset_path)
