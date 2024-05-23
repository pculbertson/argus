from dataclasses import dataclass

import h5py
import matplotlib.pyplot as plt
import torch
import tyro

import mujoco
from argus import ROOT
from argus.models import NCameraCNN, NCameraCNNConfig
from argus.utils import get_pose, xyzxyzw_to_xyzwxyz_SE3


@dataclass
class ValRealConfig:
    """Configuration for the validation test.

    Fields:
        real_data_path: The path to the real-world data.
    """

    model_path: str
    real_data_path: str = ROOT + "/outputs/data/real_images.hdf5"


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
    renderer = mujoco.Renderer(m, model.H, model.W)
    geom_name_to_hide = "goal"
    geom_id = d.body(geom_name_to_hide).id
    mujoco.mj_forward(m, d)

    with h5py.File(cfg.real_data_path, "r") as f:
        images = f["images"]  # (num_images, 2, H, W, 3)
        H, W = images.shape[2], images.shape[3]
        for img_pair in images:
            long_img_np = img_pair.transpose(0, 3, 1, 2)
            long_img = torch.tensor(long_img_np, dtype=torch.float32).reshape((-1, H, W)).to(device)[None, ...]
            pred_pose_xyzw = get_pose(long_img, model)[0]  # (num_images, 7)
            pred_pose_wxyz = xyzxyzw_to_xyzwxyz_SE3(pred_pose_xyzw)  # (num_images, 7)
            pred_pose = pred_pose_wxyz.detach().cpu().numpy()
            d.qpos[:7] = pred_pose
            mujoco.mj_forward(m, d)

            # top row: side 1
            plt.subplot(2, 2, 1)
            plt.imshow(img_pair[0])

            plt.subplot(2, 2, 2)
            m.geom_rgba[geom_id, 3] = 0
            renderer.update_scene(d, camera="cam1")
            m.geom_rgba[geom_id, 3] = 1
            pred_img1 = renderer.render()
            plt.imshow(pred_img1)

            # bottom row: side 2
            plt.subplot(2, 2, 3)
            plt.imshow(img_pair[1])

            plt.subplot(2, 2, 4)
            m.geom_rgba[geom_id, 3] = 0
            renderer.update_scene(d, camera="cam2")
            m.geom_rgba[geom_id, 3] = 1
            pred_img2 = renderer.render()
            plt.imshow(pred_img2)

            plt.axis("off")
            plt.show()


if __name__ == "__main__":
    cfg = tyro.cli(ValRealConfig)
    validate_real(cfg)
