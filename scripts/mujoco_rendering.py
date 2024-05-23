import mujoco
import matplotlib.pyplot as plt
import numpy as np

from argus import ROOT
from pathlib import Path
import h5py
import json

USE_JSON = False


def compute_camera_matrix(renderer, data, cam_name):
    """Returns the 3x4 camera matrix."""
    # If the camera is a 'free' camera, we get its position and orientation
    # from the scene data structure. It is a stereo camera, so we average over
    # the left and right channels. Note: we call `self.update()` in order to
    # ensure that the contents of `scene.camera` are correct.
    renderer.update_scene(data, camera=cam_name)
    pos = np.mean([camera.pos for camera in renderer.scene.camera], axis=0)
    z = -np.mean([camera.forward for camera in renderer.scene.camera], axis=0)
    y = np.mean([camera.up for camera in renderer.scene.camera], axis=0)
    rot = np.vstack((np.cross(y, z), y, z))
    fov = model.vis.global_.fovy

    # Translation matrix (4x4).
    translation = np.eye(4)
    translation[0:3, 3] = -pos

    # Rotation matrix (4x4).
    rotation = np.eye(4)
    rotation[0:3, 0:3] = rot

    # Focal transformation matrix (3x4).
    focal_scaling = (1.0 / np.tan(np.deg2rad(fov) / 2)) * renderer.height / 2.0
    2 * np.arctan(266 * renderer.height / 2.0) * 180 / np.pi
    focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]

    # Image matrix (3x3).
    image = np.eye(3)
    image[0, 2] = (renderer.width - 1) / 2.0
    image[1, 2] = (renderer.height - 1) / 2.0
    return image @ focal @ rotation @ translation


if __name__ == "__main__":
    print("starting script")
    # Load the MuJoCo model
    model = mujoco.MjModel.from_xml_path("/home/preston/mujoco_mpc/build/mjpc/tasks/leap/task.xml")
    data = mujoco.MjData(model)

    if USE_JSON:
        json_path = "/home/preston/argus/sim_residuals_small.json"
        with open(json_path, "r") as f:
            sim_residuals = json.load(f)

        assert len(sim_residuals[0]["s"]) == 45
        cube_poses = np.stack([pose["s"][:7] for pose in sim_residuals])
    # Load in the data from the hdf5 file.
    else:
        unity_data_path = Path(ROOT) / "outputs/data/cube_unity_data_small.hdf5"
        with h5py.File(unity_data_path, "r") as f:
            cube_poses = f["train"]["cube_poses"][()]
            images = f["train"]["images"][()][:, :3]
            q_leaps = f["train"]["q_leap"][()]

    # create renderer
    width, height = 672, 376
    renderer = mujoco.Renderer(model, height, width)

    mujoco.mj_forward(model, data)

    # cam matrix
    camera_matrix = compute_camera_matrix(renderer, data, "cam1")

    # VGA camera calibration params from ZED SDK
    # fx=266.0475
    # fy=266.085
    # cx=338.7475
    # cy=189.42525
    # k1=-0.0543471
    # k2=0.0274212
    # p1=-6.98159e-05
    # p2=-0.000346827
    # k3=-0.0109536

    # printing
    print("Camera Matrix:")
    for row in camera_matrix:
        print(row)

    # hide the goal cube
    geom_name_to_hide = "goal"
    geom_id = data.body(geom_name_to_hide).id
    model.geom_rgba[geom_id, 3] = 0

    # Update the scene with the specified camera
    renderer.update_scene(data, camera="cam1")
    model.geom_rgba[geom_id, 3] = 1

    # Render the image (assuming the default resolution, you can change the resolution if needed)
    pixels = renderer.render()
    breakpoint()
    print(pixels.shape)

    # Read the first 10 poses, and render them in mujoco. Display the rendered image.
    for i in range(100):
        data.qpos[:7] = cube_poses[i]
        if not USE_JSON:
            # set hand states as well.
            data.qpos[7 : (16 + 7)] = q_leaps[i]
        mujoco.mj_forward(model, data)

        # hide the goal cube
        geom_name_to_hide = "goal"
        geom_id = data.body(geom_name_to_hide).id
        model.geom_rgba[geom_id, 3] = 0

        renderer.update_scene(data, camera="cam1")
        pixels = renderer.render()

        if USE_JSON:
            plt.imshow(pixels)
            plt.axis("off")
            plt.title(f"Rendered Image {i}")
            plt.show()
        else:
            # Plot both the rendered image and the one from the dataset side by side.
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(pixels)
            ax[0].axis("off")
            ax[0].set_title(f"Mujoco Image {i}")

            # Permute the dims from (C, H, W) to (H, W, C)
            unity_img_rgb = images[i].transpose(1, 2, 0)
            ax[1].imshow(unity_img_rgb)
            ax[1].axis("off")
            ax[1].set_title(f"Unity Image {i}")

            plt.show()

    # # Save and display the image using matplotlib
    # plt.imshow(pixels)
    # plt.axis("off")  # Hide the axes
    # plt.title("Rendered Image from cam1")
    # plt.savefig("rendered_image.png", bbox_inches="tight", pad_inches=0)
    # plt.show()

    # # hide the goal cube
    # geom_name_to_hide = "goal"
    # geom_id = data.body(geom_name_to_hide).id
    # model.geom_rgba[geom_id, 3] = 0

    # # Update the scene with the specified camera
    # renderer.update_scene(data, camera="cam2")
    # model.geom_rgba[geom_id, 3] = 1

    # # Render the image (assuming the default resolution, you can change the resolution if needed)
    # pixels = renderer.render()
    # print(pixels.shape)

    # # Convert the image from RGB to BGR (if necessary) or handle it directly as RGB
    # # Save and display the image using matplotlib
    # plt.imshow(pixels)
    # plt.axis("off")  # Hide the axes
    # plt.title("Rendered Image from cam2")
    # plt.savefig("rendered_image.png", bbox_inches="tight", pad_inches=0)
    # plt.show()
