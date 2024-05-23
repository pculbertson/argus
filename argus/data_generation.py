import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import tyro
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from argus import ROOT
from argus.utils import convert_pose_mjpc_to_unity, convert_pose_unity_to_mjpc


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


def generate_random_light_source_poses(n_agents: int) -> np.ndarray:
    """Generates random light source poses for n_agents different agents.

    Note that these poses are expressed in Unity's y-up and left-handed coordinate system.

    Args:
        n_agents: Number of agents in the environment.

    Returns:
        light_poses: Random light source poses for n_agents. Shape=(n_agents, 7), where the last 4 elements are the
            quaternion expressed in xyzw convention.
    """
    # translations
    x = np.random.uniform(-0.254, 0.254, size=n_agents)  # +/- 10 inches
    z = np.random.uniform(-0.254, 0.254, size=n_agents)  # +/- 10 inches
    y = np.random.uniform(2.0, 3.0, size=n_agents)  # 2-3m, this is height in Unity

    # rotations
    rot_x_deg = np.random.uniform(30.0, 150.0, size=n_agents)  # 30-150 degrees
    rot_y_deg = np.random.uniform(0.0, 360.0, size=n_agents)  # 0-360 degrees
    rot_z_deg = np.random.uniform(-60.0, -60.0, size=n_agents)  # +/- 60 degrees
    quat_xyzw = R.from_euler("XYZ", np.stack([rot_x_deg, rot_y_deg, rot_z_deg], axis=-1), degrees=True).as_quat()

    light_poses = np.stack([x, y, z, quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2], quat_xyzw[:, 3]], axis=-1)
    return light_poses


@dataclass
class GenerateDataConfig:
    """The config for generating the data.

    For all path fields, you can either specify an absolute path, a relative path (with respect to where you are
    currently calling the data generation function), or a local path RELATIVE TO THE ROOT OF THE PACKAGE IN YOUR SYSTEM!
    For instance, if you pass "example_dir/data.hdf5" to `output_data_path`, it will be saved under
    /path/to/argus/example_dir/data.hdf5.

    If the mjpc data has more than n_file_split datapoints, then there will instead be multiple hdf5 files generation.
    The naming scheme will be /path/desired/data/data_{i}.hdf5, where `i` is the index of the multiple files generated.

    Fields:
        env_exe_path: Path to the Unity environment executable.
        mjpc_data_path: Path to the bagged mjpc sim data.
        output_data_path: Path to save the generated data.
        n_agents: Number of agents in the environment.
        cam1_nominal: Nominal camera pose for camera 1. The last 4 quat coords are xyzw convention. Shape=(7,).
        cam2_nominal: Nominal camera pose for camera 2. The last 4 quat coords are xyzw convention. Shape=(7,).
        bounds_trans: Bounds of the uniform translation perturbations in meters.
        quat_stdev: Standard deviation of the Gaussian noise added to the quaternion (drawn in tangent space).
        cam_rgb_range: The RGB values the camera can randomize over. Must be subset of the [0, 1] interval.
        train_frac: Fraction of the data to use for training.
        n_file_split: Number of episodes after which the output data are split into multiple files.
    """

    env_exe_path: str = ROOT + "/outputs/unity/leap_env.x86_64"
    mjpc_data_path: str = ROOT + "/outputs/data/sim_residuals.json"
    output_data_path: str = ROOT + "/outputs/data/cube_unity_data.hdf5"
    n_agents: int = 1
    cam1_nominal: Optional[np.ndarray] = None
    cam2_nominal: Optional[np.ndarray] = None
    bounds_trans: float = 0.01
    quat_stdev: float = 0.05
    cam_rgb_range: tuple[float] = (0.5, 1.0)
    train_frac: float = 0.9
    n_file_split: int = 2000

    def __post_init__(self):
        """Assigning defaults and doing sanity checks."""
        # check both local and absolute paths
        if not os.path.exists(self.env_exe_path):  # absolute path
            if os.path.exists(ROOT + "/" + self.env_exe_path):
                self.env_exe_path = ROOT + "/" + self.env_exe_path
            else:
                raise FileNotFoundError(f"The specified path does not exist: {self.env_exe_path}!")
        if not os.path.exists(self.mjpc_data_path):
            if os.path.exists(ROOT + "/" + self.mjpc_data_path):
                self.mjpc_data_path = ROOT + "/" + self.mjpc_data_path
            else:
                raise FileNotFoundError(f"The specified path does not exist: {self.mjpc_data_path}!")
        assert Path(self.output_data_path).suffix == ".hdf5", "The data path must have the .hdf5 extension!"
        assert Path(self.mjpc_data_path).suffix == ".json", "The mjpc data must be contained in a json file!"
        assert Path(self.env_exe_path).suffix in [".x86_64", ".app"], "The Unity environment must be an executable!"

        # setting nominal cam positions
        if self.cam1_nominal is None:
            self.cam1_nominal = np.array(
                [
                    -0.14786571,
                    0.125994,
                    0.00858148,
                    0.35355339,
                    -0.35355339,
                    0.85355339,
                    0.14644661,
                ]
            )  # WARNING: all coordinates are expressed in Unity's y-up left-handed frame convention!!!
        if self.cam2_nominal is None:
            self.cam2_nominal = np.array(
                [
                    0.14786571,
                    0.125994,
                    0.00858148,
                    -0.35355339,
                    -0.35355339,
                    0.85355339,
                    -0.14644661,
                ]
            )

        # checking on the rgb range parameter
        assert isinstance(self.cam_rgb_range, tuple), "cam_rgb_range must be a 2-tuple!"
        assert len(self.cam_rgb_range) == 2, "cam_rgb_range must be a 2-tuple!"
        assert 0 <= self.cam_rgb_range[0] < self.cam_rgb_range[1] <= 1, "cam_rgb_range must be a subset of [0, 1]!"


def generate_data(cfg: GenerateDataConfig) -> None:
    """Main data generation loop."""
    # unpacking variables from config
    env_exe_path = cfg.env_exe_path
    mjpc_data_path = cfg.mjpc_data_path
    n_agents = cfg.n_agents
    output_data_path = cfg.output_data_path
    cam1_nominal = cfg.cam1_nominal
    cam2_nominal = cfg.cam2_nominal
    cam_rgb_range = cfg.cam_rgb_range
    bounds_trans = cfg.bounds_trans
    quat_stdev = cfg.quat_stdev
    train_frac = cfg.train_frac
    n_file_split = cfg.n_file_split

    # retrieving the mjpc sim data
    with open(mjpc_data_path) as f:
        all_data = json.load(f)

    q_all = np.array([all_data[i]["s"] for i in range(len(all_data))])[..., :23]  # (n_data, :23)
    cube_poses_mjpc = q_all[..., :7]
    cube_poses_all = convert_pose_mjpc_to_unity(cube_poses_mjpc)  # (n_data, 7), UNITY coords
    q_leap_all = q_all[..., 7:]  # (n_data, 16)
    n_episodes = cube_poses_all.shape[0] // n_agents
    _cube_poses_truncated = cube_poses_all[: n_agents * n_episodes, :]  # (n_agents * n_episodes, 7)
    cube_poses_truncated = convert_pose_unity_to_mjpc(_cube_poses_truncated)  # MJPC coords
    q_leap_truncated = q_leap_all[: n_agents * n_episodes, :]  # (n_agents * n_episodes, 16)

    # generating data
    env, behavior_name, expected_action_size = unity_setup(env_exe_path, n_agents=n_agents)
    images = []

    # if n_episodes is large, chunk the hdf5s into multiple ones
    if n_agents * n_episodes > n_file_split:
        n_files = int(np.ceil(n_agents * n_episodes / n_file_split))
    else:
        n_files = 1
        p = output_data_path
    idx_file = 0

    # create parent directory if it doesn't exist
    parent_dir = Path(output_data_path).parent
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

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
        light_poses = generate_random_light_source_poses(n_agents)  # (n_agents, 7)

        action = np.zeros((n_agents, expected_action_size))
        action[:, :7] = cam1_poses
        action[:, 7:10] = np.random.uniform(*cam_rgb_range, size=(n_agents, 3))
        action[:, 10:17] = cam2_poses
        action[:, 17:20] = np.random.uniform(*cam_rgb_range, size=(n_agents, 3))
        action[:, 20:27] = cube_poses_batch
        action[:, 27:34] = light_poses
        action[:, 34:50] = q_leap

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

        # chunk when needed
        if len(images) >= cfg.n_file_split or episode == n_episodes - 1:
            if n_files == 1:
                p = output_data_path
            else:
                filename = Path(output_data_path).stem  # name without extension
                if not os.path.exists(parent_dir / f"{filename}"):
                    os.makedirs(parent_dir / f"{filename}", exist_ok=True)
                p = parent_dir / f"{filename}" / f"{filename}_{idx_file}.hdf5"

            print(f"Saving data to file {idx_file + 1}/{n_files}...")
            idxs = np.random.permutation(len(images))  # shuffle each chunk individually
            train_test_idx = int(train_frac * len(images) * n_agents)  # index for train/test split
            images_train = np.concatenate(images, axis=0)[idxs][:train_test_idx, ...]
            images_test = np.concatenate(images, axis=0)[idxs][train_test_idx:, ...]

            idx_start = n_file_split * idx_file
            idx_end = n_file_split * (idx_file + 1) if idx_file < n_files - 1 else n_episodes * n_agents
            cube_poses_train = cube_poses_truncated[idx_start:idx_end][idxs][:train_test_idx]
            cube_poses_test = cube_poses_truncated[idx_start:idx_end][idxs][train_test_idx:]

            q_leap_train = q_leap_truncated[idx_start:idx_end][idxs][:train_test_idx]
            q_leap_test = q_leap_truncated[idx_start:idx_end][idxs][train_test_idx:]
            idx_file += 1

            with h5py.File(p, "w") as f:
                f.attrs["n_cams"] = 2
                f.attrs["H"] = 376
                f.attrs["W"] = 672

                train = f.create_group("train")
                train.create_dataset("images", data=images_train)
                train.create_dataset("cube_poses", data=cube_poses_train)
                train.create_dataset("q_leap", data=q_leap_train)

                test = f.create_group("test")
                test.create_dataset("images", data=images_test)
                test.create_dataset("cube_poses", data=cube_poses_test)
                test.create_dataset("q_leap", data=q_leap_test)

            images = []

    env.close()


if __name__ == "__main__":
    cfg = tyro.cli(GenerateDataConfig)
    start = time.time()
    generate_data(cfg)
    end = time.time()
    print(f"Data generation took {end - start:.2f} seconds.")
