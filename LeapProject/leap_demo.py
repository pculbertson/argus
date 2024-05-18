import os
from dataclasses import dataclass

import h5py
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from mlagents_envs.base_env import ActionTuple
from PIL import Image
from scipy.spatial.transform import Rotation as R


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
    bounds_trans: float = 0.1,
    quat_stdev: float = 0.1,
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
    translations = np.random.uniform(-bounds_trans, bounds_trans, size=(n_agents, 3))

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
        n_agents: Number of agents in the environment.
        n_episodes: Number of episodes to run.
        data_path: Path to save the generated data.
        cam1_nominal: Nominal camera pose for camera 1. The last 4 quat coords are xyzw convention. Shape=(7,).
        cam2_nominal: Nominal camera pose for camera 2. The last 4 quat coords are xyzw convention. Shape=(7,).
        bounds_trans: Bounds of the uniform translation perturbations in meters.
        quat_stdev: Standard deviation of the Gaussian noise added to the quaternion (drawn in tangent space).
        train_frac: Fraction of the data to use for training.
    """
    env_exe_path: str
    n_agents: int
    n_episodes: int  # TODO(ahl): remove this, depends on the size of the mujoco data we're reading
    data_path: str = "cube_unity_data.hdf5"
    cam1_nominal: np.ndarray = np.array(
        [
            -0.14786571, 0.125994, 0.00858148,
            0.35355339, -0.35355339,  0.85355339, 0.14644661,  # why isn't this rendering properly?
        ]
    )
    cam2_nominal: np.ndarray = np.array(
        [
            0.14786571, 0.125994, 0.00858148,
            -0.35355339, -0.35355339, 0.85355339, -0.14644661,  # why isn't this rendering properly?
        ]
    )
    # cam1_nominal: np.ndarray = np.array([0.125994, -0.00858148, -0.14786571, -0.45576804, -0.060003, 0.70455634, -0.5406251])
    # cam2_nominal: np.ndarray = np.array([0.125994, -0.00858148, 0.14786571, -0.70455634, -0.5406251, 0.45576804, -0.060003])
    bounds_trans: float = 1e-6
    quat_stdev: float = 1e-6
    train_frac: float = 0.9

    def __post_init__(self):
        if not os.path.exists(self.env_exe_path):
            raise FileNotFoundError(f"The specified path does not exist: {self.env_exe_path}")

def generate_data(cfg: GenerateDataConfig) -> None:
    """Main data generation loop."""
    # unpacking variables from config
    env_exe_path = cfg.env_exe_path
    n_agents = cfg.n_agents
    n_episodes = cfg.n_episodes
    data_path = cfg.data_path
    cam1_nominal = cfg.cam1_nominal
    cam2_nominal = cfg.cam2_nominal
    bounds_trans = cfg.bounds_trans
    quat_stdev = cfg.quat_stdev
    train_frac = cfg.train_frac

    # generating data
    env, behavior_name, expected_action_size = unity_setup(env_exe_path, n_agents=n_agents)
    images = []
    cube_poses = []
    image_filenames = []

    for episode in range(n_episodes):
        env.reset()

        # set the states for the agents
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
        # cam1_poses = np.tile(cam1_nominal, (n_agents, 1))  # debug, send just the nominal
        # cam2_poses = np.tile(cam2_nominal, (n_agents, 1))  # debug, send just the nominal
        action = np.zeros((n_agents, expected_action_size))
        action[:, :7] = cam1_poses
        action[:, 7:14] = cam2_poses
        # TODO(ahl): load the cube and hand states from data into actions

        action_tuple = ActionTuple(continuous=action)
        env.set_actions(behavior_name, action_tuple)
        env.step()
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        # observing the system
        cam1_obs = decision_steps.obs[0]  # (n_agents, 3, H, W)
        cam2_obs = decision_steps.obs[1]  # (n_agents, 3, H, W)
        vec_obs = decision_steps.obs[2]  # (x, y, z, qw, qy, qx, qz, q_hand[16])
        assert cam1_obs.shape == cam2_obs.shape == (n_agents, 3, 376, 672)
        assert vec_obs.shape == (n_agents, 7)

        # # debug - this saves some images out so you can check what they look like
        # for i in range(cam1_obs.shape[0]):
        #     img = cam1_obs[i].transpose(1, 2, 0)
        #     img = (img - np.min(img)) / (np.max(img) - np.min(img))
        #     img = (img * 255).astype(np.uint8)
        #     img = Image.fromarray(img)
        #     img.save(f"img_{i}a.png")

        #     img = cam2_obs[i].transpose(1, 2, 0)
        #     img = (img - np.min(img)) / (np.max(img) - np.min(img))
        #     img = (img * 255).astype(np.uint8)
        #     img = Image.fromarray(img)
        #     img.save(f"img_{i}b.png")
        # breakpoint()

        # bagging the data
        images.append(np.concatenate([cam1_obs, cam2_obs], axis=1).reshape(n_agents, 6, 376, 672))
        cube_poses.append(vec_obs)
        for i in range(n_agents):
            img_idx = episode * n_agents + i
            image_filenames.append((f"img_{img_idx}a.png", f"img_{img_idx}b.png"))

    env.close()

    # generating hdf5 file
    train_test_idx = int(train_frac * len(images) * n_agents)
    with h5py.File(data_path, "w") as f:
        f.attrs["n_cams"] = 2
        f.attrs["H"] = 376
        f.attrs["W"] = 672

        train = f.create_group("train")
        train.create_dataset("images", data=np.concatenate(images, axis=0)[:train_test_idx])
        train.create_dataset("cube_poses", data=np.concatenate(cube_poses, axis=0)[:train_test_idx])
        train.create_dataset("image_filenames", data=image_filenames[:train_test_idx])

        test = f.create_group("test")
        test.create_dataset("images", data=np.concatenate(images, axis=0)[train_test_idx:])
        test.create_dataset("cube_poses", data=np.concatenate(cube_poses, axis=0)[train_test_idx:])
        test.create_dataset("image_filenames", data=image_filenames[train_test_idx:])

if __name__ == "__main__":
    # cfg = tyro.cli(GenerateDataConfig)  # TODO(ahl): once stable, switch to tyro
    cfg = GenerateDataConfig(
        env_exe_path="/home/albert/research/argus/LeapProject/leap_env.x86_64",
        n_agents=1,
        n_episodes=1,
    )
    generate_data(cfg)
