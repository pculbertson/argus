import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import imageio.v2 as imageio
import kornia
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
import tyro
from PIL import Image

from argus import ROOT
from argus.data import CameraCubePoseDatasetConfig
from argus.models import NCameraCNN, NCameraCNNConfig
from argus.utils import get_pose, xyzxyzw_to_xyzwxyz_SE3


@dataclass
class ValRealConfig:
    """Configuration for the validation test.

    Fields:
        real_data_path: The path to the real-world data.
    """

    model_path: str
    dataset_config: CameraCubePoseDatasetConfig
    n_cams: int = 4


def validate_real(cfg: ValRealConfig) -> None:
    """Validate model performance on real-world data visually."""
    # load pose estimator model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NCameraCNN(NCameraCNNConfig()).to(device)
    model.load_state_dict(torch.load(cfg.model_path))
    model.eval()

    # mujoco model for rendering
    m = mujoco.MjModel.from_xml_path(ROOT + "/mujoco/leap/task.xml")
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, *cfg.dataset_config.center_crop)
    geom_name_to_hide = "goal"
    geom_id = d.body(geom_name_to_hide).id
    mujoco.mj_forward(m, d)

    # convenience paths
    dataset_path = cfg.dataset_config.dataset_path
    filename = cfg.dataset_config.dataset_path + f"/{Path(cfg.dataset_config.dataset_path).stem}.hdf5"
    output_dir = Path(ROOT) / f"outputs/real_validation_visuals/{Path(cfg.model_path).stem}"
    os.makedirs(output_dir, exist_ok=True)

    # saving image analysis frames
    np.set_printoptions(suppress=True)  # suppresses scientific notation during this function
    frames = []
    print(f"file name is {filename}")
    with h5py.File(filename, "r") as f:

        _img_stems = f['test']["img_stems"][()]
        img_stems = [byte_string.decode("utf-8") for byte_string in _img_stems]
        for i, img_stem in enumerate(img_stems):
            img_a = Image.open(f"{dataset_path}/{img_stem}_a.png")  # (H, W, 3)
            img_b = Image.open(f"{dataset_path}/{img_stem}_b.png")  # (H, W, 3)
            img_depth1 = Image.open(f"{dataset_path}/{img_stem}_depth_a.png")  # (H, W, 3)
            img_depth2 = Image.open(f"{dataset_path}/{img_stem}_depth_b.png")  # (H, W, 3)            
            img = np.concatenate([np.array(img_a), np.array(img_b), np.array(img_depth1), np.array(img_depth2)], axis=-1).transpose(2, 0, 1)  # (n_cams * 3, H, W)
            float_img = torch.from_numpy(img).to(device).to(torch.float32) / 255.0  # (n_cams * 3, H, W)

            if cfg.dataset_config.center_crop:
                H_crop, W_crop = cfg.dataset_config.center_crop
                _float_img = float_img.reshape(cfg.n_cams, 3, *float_img.shape[-2:])  # (n_cams, 3, H, W)
                _float_img = kornia.geometry.transform.center_crop(_float_img, (H_crop, W_crop))
                float_img = _float_img.reshape(-1, H_crop, W_crop).unsqueeze(0)  # (1, n_cams * 3, H_crop, W_crop)

            pred_pose_xyzw = get_pose(float_img, model)[0]  # (num_images, 7)
            pred_pose_wxyz = xyzxyzw_to_xyzwxyz_SE3(pred_pose_xyzw)  # (num_images, 7)
            pred_pose = pred_pose_wxyz.detach().cpu().numpy()
            d.qpos[:7] = pred_pose
            mujoco.mj_forward(m, d)

            float_img_numpy = float_img.detach().cpu().numpy()
            float_img_numpy = float_img_numpy.reshape(cfg.n_cams, 3, *float_img_numpy.shape[-2:]).transpose(0, 2, 3, 1)

            # top row: side 1
            plt.subplot(2, 2, 1)
            plt.imshow(float_img_numpy[0])
            plt.axis("off")

            plt.subplot(2, 2, 2)
            m.geom_rgba[geom_id, 3] = 0
            renderer.update_scene(d, camera="cam1")
            m.geom_rgba[geom_id, 3] = 1
            pred_img1 = renderer.render()
            plt.imshow(pred_img1)
            plt.axis("off")

            # bottom row: side 2
            plt.subplot(2, 2, 3)
            plt.imshow(float_img_numpy[1])
            plt.axis("off")

            plt.subplot(2, 2, 4)
            m.geom_rgba[geom_id, 3] = 0
            renderer.update_scene(d, camera="cam2")
            m.geom_rgba[geom_id, 3] = 1
            pred_img2 = renderer.render()
            plt.imshow(pred_img2)
            plt.axis("off")

            plt.suptitle(f"Pred pose {i}:\n{np.array2string(pred_pose, precision=3, floatmode='fixed')}")

            # save fig and append frame to list for gif making
            plt.savefig(output_dir / f"example_{i}.png", bbox_inches="tight")
            frames.append(imageio.imread(output_dir / f"example_{i}.png"))

    # saving out a gif
    imageio.mimsave(output_dir / "real_validation.gif", frames)
    np.set_printoptions(suppress=False)  # revert suppression


if __name__ == "__main__":
    cfg = tyro.cli(ValRealConfig)
    validate_real(cfg)
