import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from mlagents_envs.base_env import ActionTuple
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


# TODO(ahl): when merging the unity stuff with the rest of argus, integrate these utils and tests properly

# ##### #
# UTILS #
# ##### #

def convert_pose_mjpc_to_unity(pose_mjpc: np.ndarray) -> np.ndarray:
    """Converts a pose from Mujoco's coordinate system to Unity's coordinate system.

    The differences between the coordinate systems are the handedness of the frames and the direction of the axes. The
    quaternion convention is also different.

    Args:
        pose_mjpc: Pose in Mujoco's coordinate system. Shape=(..., 7), where the last 4 elements are the quaternion in
            wxyz convention. The mjpc coordinate system is right-handed, with +x "forward," +y "right," and +z "up."

    Returns:
        pose_unity: Pose in Unity's coordinate system. Shape=(..., 7), where the last 4 elements are the quaternion in
            xyzw convention. The Unity coordinate system is left-handed, with +z "forward," +x "left," and +y "up."
    """
    # translations - here, we use the improper rotation matrix to rotate the translation vector
    R_mjpc_to_unity_left_hand = np.array(
        [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    trans_mjpc = pose_mjpc[..., :3, None]  # (..., 3, 1)
    trans_unity = (R_mjpc_to_unity_left_hand @ trans_mjpc).squeeze(-1)  # (..., 3)

    # rotations
    q_wxyz = pose_mjpc[..., 3:]  # (..., 4)
    q_xyzw = np.concatenate([q_wxyz[..., 1:], q_wxyz[..., :1]], axis=-1)  # (..., 4)
    quat_unity = np.concatenate(
        [
            -q_xyzw[..., 1:2],  # -y, rotation about y in mjpc is rotation about -x in unity
            q_xyzw[..., 2:3],  # z, rotation about z in mjpc is rotation about y in unity
            q_xyzw[..., 0:1],  # x, rotation about x in mjpc is rotation about x in unity
            -q_xyzw[..., 3:4],  # -w, switch angle sign for right to left hand convention
        ],
        axis=-1,
    )
    quat_unity[quat_unity[..., 3] < 0] = -quat_unity[quat_unity[..., 3] < 0]  # return with positive w

    # concatenating and returning
    pose_unity = np.concatenate([trans_unity, quat_unity], axis=-1)  # (..., 7)
    return pose_unity

def convert_pose_unity_to_mjpc(pose_unity: np.ndarray) -> np.ndarray:
    """Converts a pose from Unity's coordinate system to Mujoco's coordinate system.

    Inverse operation of `convert_pose_mjpc_to_unity`. For more info, check its docstring.
    """
    # translations
    R_unity_to_mjpc_left_hand = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    trans_unity = pose_unity[..., :3, None]  # (..., 3, 1)
    trans_mjpc = (R_unity_to_mjpc_left_hand @ trans_unity).squeeze(-1)  # (..., 3)

    # rotations
    q_xyzw = pose_unity[..., 3:]  # (..., 4)
    q_wxyz = np.concatenate([q_xyzw[..., -1:], q_xyzw[..., :-1]], axis=-1)  # (..., 4)
    quat_mjpc = np.concatenate(
        [
            -q_wxyz[..., 0:1],  # -w, switch angle sign for left to right hand convention
            q_wxyz[..., 3:4],  # z, rotation about z in unity is rotation about x in mjpc
            -q_wxyz[..., 1:2],  # -x, rotation about x in unity is rotation about -y in mjpc
            q_wxyz[..., 2:3],  # y, rotation about y in unity is rotation about z in mjpc
        ],
        axis=-1,
    )
    quat_mjpc[quat_mjpc[..., 0] < 0] = -quat_mjpc[quat_mjpc[..., 0] < 0]  # return with positive w

    # concatenating and returning
    pose_mjpc = np.concatenate([trans_mjpc, quat_mjpc], axis=-1)  # (..., 7)
    return pose_mjpc

def convert_unity_quat_to_euler(quat: np.ndarray) -> np.ndarray:
    """Converts a quaternion in the Unity convention to Euler angles in degrees.

    Useful for debugging by manually inputing Euler angles into the Unity editor.

    Args:
        quat: Quaternion in xyzw convention with Unity's axes. Shape=(..., 4).

    Returns:
        euler: Euler angles in degrees. Shape=(..., 3), where the last dimension is the roll-pitch-yaw angles.
    """
    euler = R.from_quat(quat).as_euler("XYZ", degrees=True)
    return euler

def convert_mjpc_q_leap_to_unity(q_mjpc: np.ndarray) -> np.ndarray:
    """Converts the hand configuration from mjpc's to Unity's coordinate system.

    * The mjpc convention is depth-first with finger order index, middle, ring, thumb.
    * The Unity convention is breadth-first with finger order middle, thumb, ring, and index.

    Args:
        q_mjpc: The hand state in Mujoco's coordinate system. Shape=(..., 16).

    Returns:
        q_unity: The hand state in Unity's coordinate system. Shape=(..., 16).
    """
    new_idxs = np.array(
        [
            4, 12, 8, 0,  # mcp joint indices on the mjpc LEAP hand
            5, 13, 9, 1,  # pip joint indices on the mjpc LEAP hand
            6, 14, 10, 2,  # dip joint indices on the mjpc LEAP hand
            7, 15, 11, 3,  # fingertip joints
        ]
    )
    q_unity = q_mjpc[..., new_idxs]
    return q_unity

# ##### #
# TESTS #
# ##### #

def test_convert_pose_mjpc_to_unity() -> None:
    """Tests the conversion from Mujoco's to Unity's coordinate system."""
    # test rotation about x
    pose_mjpc = np.array([[0.1, 0.2, 0.3, 0.92387953, 0.38268343, 0.0, 0.0]])  # rotate +45 about +x in mjpc
    pose_unity = convert_pose_mjpc_to_unity(pose_mjpc)
    euler = convert_unity_quat_to_euler(pose_unity[0, 3:])
    assert np.allclose(pose_unity, np.array([[-0.2, 0.3, 0.1, 0.0, 0.0, -0.38268343, 0.92387953]]))
    assert np.allclose(euler, np.array([0.0, 0.0, -45.0]))

    # test rotation about y
    pose_mjpc = np.array([[0.1, 0.2, 0.3, 0.92387953, 0.0, 0.38268343, 0.0]])  # rotate +45 about +y in mjpc
    pose_unity = convert_pose_mjpc_to_unity(pose_mjpc)
    euler = convert_unity_quat_to_euler(pose_unity[0, 3:])
    assert np.allclose(pose_unity, np.array([[-0.2, 0.3, 0.1, 0.38268343, 0.0, 0.0, 0.92387953]]))
    assert np.allclose(euler, np.array([45.0, 0.0, 0.0]))

    # test rotation about z
    pose_mjpc = np.array([[0.1, 0.2, 0.3, 0.92387953, 0.0, 0.0, 0.38268343]])  # rotate +45 about +z in mjpc
    pose_unity = convert_pose_mjpc_to_unity(pose_mjpc)
    euler = convert_unity_quat_to_euler(pose_unity[0, 3:])
    assert np.allclose(pose_unity, np.array([[-0.2, 0.3, 0.1, 0.0, -0.38268343, 0.0, 0.92387953]]))
    assert np.allclose(euler, np.array([0.0, -45.0, 0.0]))

def test_convert_pose_unity_to_mjpc() -> None:
    """Tests the conversion from Unity's to Mujoco's coordinate system."""
    # test by converting a pose from mjpc to unity and back to mjpc
    pose_mjpc = np.random.rand(2, 7)  # implicitly tests batching
    pose_mjpc[..., 3:] /= np.linalg.norm(pose_mjpc[..., 3:], axis=-1, keepdims=True)
    assert np.allclose(pose_mjpc, convert_pose_unity_to_mjpc(convert_pose_mjpc_to_unity(pose_mjpc)))

# ############### #
# DATA GENERATION #
# ############### #

def unity_setup(env_exe_path: str, n_agents: int = 10) -> None:
    """Sets up the Unity environment for data gen.

    Args:
        env_exe_path: Path to the Unity environment executable.
        n_agents: Number of agents in the environment.
    """
    if not os.path.exists(env_exe_path):
        raise FileNotFoundError(f"The specified path does not exist: {env_exe_path}")

    # creating env
    engine_configuration_channel = EngineConfigurationChannel()
    engine_configuration_channel.set_configuration_parameters(time_scale=20.0)
    env = UnityEnvironment(file_name=env_exe_path, side_channels=[engine_configuration_channel], num_areas=n_agents)
    env.reset()  # starts the env

    # useful info
    behavior_name = list(env.behavior_specs.keys())[0]
    behavior_specs = env.behavior_specs[behavior_name]
    expected_action_size = behavior_specs.action_spec.continuous_size

    return env, behavior_name, expected_action_size

def generate_random_camera_poses(
    n_agents: int,
    mu_trans: np.ndarray,
    mu_quat: np.ndarray,
    bounds_trans: float = 0.01,
    quat_stdev: float = 0.05,
) -> np.ndarray:
    """Generates random camera poses for n_agents.

    The camera poses are centered around where they nominally are in the CAD model and then perturbed slightly.

    Args:
        n_agents: Number of agents in the environment.
        mu_trans: Nominal camera location in meters. Shape=(3,).
        mu_quat: Nominal camera orientation quaternion in xyzw convention (drawn in tangent space). Shape=(4,).
        bounds_trans: Bounds of the uniform translation perturbations in meters.
        quat_stdev: Standard deviation of the Gaussian noise added to the quaternion (drawn in tangent space).

    Returns:
        cam_poses: Random camera poses for n_agents. Shape=(n_agents, 7), where the last 4 elements are the quaternion
            expressed in xyzw convention.
    """
    translations = mu_trans + np.random.uniform(-bounds_trans, bounds_trans, size=(n_agents, 3))

    # small perturbation to a quaternion
    # see: math.stackexchange.com/a/477151/876331
    omega = np.random.normal(0, quat_stdev, size=(n_agents, 3))
    theta = np.linalg.norm(omega, axis=-1)  # (n_agents,)
    qxyz = np.sin(theta[:, None]) * omega / theta[:, None]  # (n_agents, 3)
    qw = np.cos(theta)  # (n_agents,)
    exp_omega = R.from_quat(np.concatenate([qxyz, qw[:, None]], axis=-1))  # (n_agents, 4), xyzw convention for scipy
    quat = (exp_omega * R.from_quat(mu_quat)).as_quat()  # (n_agents, 4), xyzw convention for scipy

    cam_poses = np.concatenate([translations, quat], axis=-1)  # (n_agents, 7)
    return cam_poses

@dataclass
class GenerateDataConfig:
    """The config for generating the data.

    Fields:
        env_exe_path: Path to the Unity environment executable.
        mjpc_data_path: Path to the bagged mjpc sim data.
        n_agents: Number of agents in the environment.
        output_data_path: Path to save the generated data.
        cam1_nominal: Nominal camera pose for camera 1. The last 4 quat coords are xyzw convention. Shape=(7,).
        cam2_nominal: Nominal camera pose for camera 2. The last 4 quat coords are xyzw convention. Shape=(7,).
        bounds_trans: Bounds of the uniform translation perturbations in meters.
        quat_stdev: Standard deviation of the Gaussian noise added to the quaternion (drawn in tangent space).
        train_frac: Fraction of the data to use for training.
    """
    env_exe_path: str
    mjpc_data_path: str
    n_agents: int
    output_data_path: str = "cube_unity_data.hdf5"
    cam1_nominal: np.ndarray = np.array(
        [
            -0.14786571, 0.125994, 0.00858148,
            0.35355339, -0.35355339,  0.85355339, 0.14644661,
        ]
    )  # WARNING: all coordinates are expressed in Unity's y-up left-handed frame convention!!!
    cam2_nominal: np.ndarray = np.array(
        [
            0.14786571, 0.125994, 0.00858148,
            -0.35355339, -0.35355339, 0.85355339, -0.14644661,
        ]
    )
    bounds_trans: float = 0.01
    quat_stdev: float = 0.05
    train_frac: float = 0.9

    def __post_init__(self):
        if not os.path.exists(self.env_exe_path):
            raise FileNotFoundError(f"The specified path does not exist: {self.env_exe_path}!")
        if not os.path.exists(self.mjpc_data_path):
            raise FileNotFoundError(f"The specified path does not exist: {self.mjpc_data_path}!")
        assert Path(self.output_data_path).suffix == ".hdf5", "The data path must have the .hdf5 extension!"
        assert Path(self.mjpc_data_path).suffix == ".json", "The mjpc data must be contained in a json file!"
        assert Path(self.env_exe_path).suffix in [".x86_64", ".app"], "The Unity environment must be an executable!"

def generate_data(cfg: GenerateDataConfig) -> None:
    """Main data generation loop."""
    # unpacking variables from config
    env_exe_path = cfg.env_exe_path
    mjpc_data_path = cfg.mjpc_data_path
    n_agents = cfg.n_agents
    output_data_path = cfg.output_data_path
    cam1_nominal = cfg.cam1_nominal
    cam2_nominal = cfg.cam2_nominal
    bounds_trans = cfg.bounds_trans
    quat_stdev = cfg.quat_stdev
    train_frac = cfg.train_frac

    # retrieving the mjpc sim data
    f = open(mjpc_data_path)
    all_data = json.load(f)
    q_all = np.array([all_data[i]["s"] for i in range(len(all_data))])[..., :23]  # (n_data, :23)
    cube_poses_mjpc = q_all[..., :7]
    cube_poses_all = convert_pose_mjpc_to_unity(cube_poses_mjpc)  # (n_data, 7), UNITY coords
    q_leap_all = q_all[..., 7:]  # (n_data, 16)
    n_episodes = cube_poses_all.shape[0] // n_agents
    _cube_poses_truncated = cube_poses_all[:n_agents * n_episodes, :]  # (n_agents * n_episodes, 7)
    cube_poses_truncated = convert_pose_unity_to_mjpc(_cube_poses_truncated)  # MJPC coords

    # generating data
    env, behavior_name, expected_action_size = unity_setup(env_exe_path, n_agents=n_agents)
    images = []
    image_filenames = []

    print("Rendering image data...")
    for episode in tqdm(range(n_episodes), desc="Episodes"):
        env.reset()

        # computing actions to send to Unity agent (these are the states we want to set)
        cube_poses_batch = cube_poses_all[episode * n_agents : (episode + 1) * n_agents]  # (n_agents, 7)
        q_leap = q_leap_all[episode * n_agents : (episode + 1) * n_agents]  # (n_agents, 16)
        cam1_poses = generate_random_camera_poses(
            n_agents,
            cam1_nominal[:3],
            cam1_nominal[3:],
            bounds_trans=bounds_trans,
            quat_stdev=quat_stdev,
        )  # (n_agents, 7)
        cam2_poses = generate_random_camera_poses(
            n_agents,
            cam2_nominal[:3],
            cam2_nominal[3:],
            bounds_trans=bounds_trans,
            quat_stdev=quat_stdev,
        )  # (n_agents, 7)

        action = np.zeros((n_agents, expected_action_size))
        action[:, :7] = cam1_poses
        action[:, 7:14] = cam2_poses
        action[:, 14:21] = cube_poses_batch
        action[:, 21:] = convert_mjpc_q_leap_to_unity(q_leap)

        # advancing the Unity sim and rendering out observations
        action_tuple = ActionTuple(continuous=action)
        env.set_actions(behavior_name, action_tuple)
        env.step()
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        # observing the system
        cam1_obs = decision_steps.obs[0]  # (n_agents, 3, H, W)
        cam2_obs = decision_steps.obs[1]  # (n_agents, 3, H, W)

        # bagging the data
        images.append(np.concatenate([cam1_obs, cam2_obs], axis=1).reshape(n_agents, 6, 376, 672))
        for i in range(n_agents):
            img_idx = episode * n_agents + i
            image_filenames.append((f"img_{img_idx}a.png", f"img_{img_idx}b.png"))

            # debug - this saves some images out so you can check what they look like
            #########################################################################
            # img = cam1_obs[i].transpose(1, 2, 0)
            # img = (img - np.min(img)) / (np.max(img) - np.min(img))
            # img = (img * 255).astype(np.uint8)
            # img = Image.fromarray(img)
            # img.save(f"img_{img_idx}a.png")

            # img = cam2_obs[i].transpose(1, 2, 0)
            # img = (img - np.min(img)) / (np.max(img) - np.min(img))
            # img = (img * 255).astype(np.uint8)
            # img = Image.fromarray(img)
            # img.save(f"img_{img_idx}b.png")
            #########################################################################

    # generating hdf5 file
    print("Saving all data to file...")
    train_test_idx = int(train_frac * len(images) * n_agents)
    with h5py.File(output_data_path, "w") as f:
        f.attrs["n_cams"] = 2
        f.attrs["H"] = 376
        f.attrs["W"] = 672

        idxs = np.random.permutation(len(images))  # shuffle the data before splitting into train/test

        train = f.create_group("train")
        train.create_dataset("images", data=np.concatenate(images, axis=0)[idxs][:train_test_idx, ...])
        train.create_dataset("cube_poses", data=cube_poses_truncated[idxs][:train_test_idx, ...])
        train.create_dataset("image_filenames", data=np.array(image_filenames)[idxs][:train_test_idx].tolist())

        test = f.create_group("test")
        test.create_dataset("images", data=np.concatenate(images, axis=0)[idxs][train_test_idx:, ...])
        test.create_dataset("cube_poses", data=cube_poses_truncated[idxs][train_test_idx:, ...])
        test.create_dataset("image_filenames", data=np.array(image_filenames)[idxs][train_test_idx:].tolist())

    env.close()

if __name__ == "__main__":
    # cfg = tyro.cli(GenerateDataConfig)  # TODO(ahl): once stable, switch to tyro
    cfg = GenerateDataConfig(
        env_exe_path="/home/albert/research/argus/LeapProject/leap_env.x86_64",
        mjpc_data_path="sim_residuals.json",
        n_agents=1,
    )
    start = time.time()
    generate_data(cfg)
    end = time.time()
    print(f"Data generation took {end - start:.2f} seconds.")
