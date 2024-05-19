from argus import ROOT
from argus.data import CameraCubePoseDatasetConfig
from argus.models import NCameraCNNConfig
from argus.train import TrainConfig, train


def main(dataset_path: str):
    """Sets up the training."""
    train_cfg = TrainConfig(
        batch_size=64,
        learning_rate=1e-3,
        n_epochs=100,
        device="cuda",
        max_grad_norm=1.0,
        val_epochs=5,
        print_epochs=1,
        save_epochs=5,
        save_dir=ROOT + "/outputs/models",
        model_config=NCameraCNNConfig(
            n_cams=2,
            H=376,
            W=672,
        ),
        dataset_config=CameraCubePoseDatasetConfig(
            dataset_path=dataset_path,
        ),
        compile_model=False,  # compiled models are hard to serialize
        wandb_project="argus-estimator",
        wandb_log=True,
    )
    train(train_cfg)


if __name__ == "__main__":
    dataset_path = ROOT + "/cube_unity_data.hdf5"
    main(dataset_path)
