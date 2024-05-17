import os

import h5py
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from mlagents_envs.base_env import ActionTuple

# Path to the Unity environment executable
env_path = "/home/albert/research/argus/LeapProject/leap_env.x86_64"
if not os.path.exists(env_path):
    raise FileNotFoundError(f"The specified path does not exist: {env_path}")

# Set up the engine configuration
engine_configuration_channel = EngineConfigurationChannel()
engine_configuration_channel.set_configuration_parameters(time_scale=20.0)

# Create the Unity environment
env = UnityEnvironment(file_name=env_path, side_channels=[engine_configuration_channel])

# Start the environment
env.reset()

# Get the behavior name
behavior_name = list(env.behavior_specs.keys())[0]
print(f"Behavior name: {behavior_name}")

# Get the behavior specs
behavior_specs = env.behavior_specs[behavior_name]

# Ensure the action size matches the expected size
expected_action_size = behavior_specs.action_spec.continuous_size
print(f"Expected action size: {expected_action_size}")

# variables for hdf5
images = []
cube_poses = []
image_filenames = []

# run some episodes
n_episodes = 10
for episode in range(n_episodes):
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    tracked_agent = -1  # -1 indicates not yet tracking
    done = False  # For the tracked_agent
    while not done:
        if tracked_agent == -1 and len(decision_steps) >= 1:
            tracked_agent = decision_steps.agent_id[0]

        # ####### #
        # ACTIONS #
        # ####### #
        action = np.random.uniform(-0.1, 0.1, size=(1, expected_action_size))

        # sample a uniformly random quaternion
        uvw = np.random.uniform(0, 1, size=(3,))
        u = uvw[0]
        v = uvw[1]
        w = uvw[2]
        quat = np.array(
            [
                np.sqrt(1 - u) * np.sin(2 * np.pi * v),
                np.sqrt(1 - u) * np.cos(2 * np.pi * v),
                np.sqrt(u) * np.sin(2 * np.pi * w),
                np.sqrt(u) * np.cos(2 * np.pi * w),
            ]
        )
        action[0][3:7] = quat

        # Ensure the actions have the correct shape
        action_tuple = ActionTuple(continuous=action)
        env.set_actions(behavior_name, action_tuple)

        # Perform a step in the environment
        env.step()

        decision_steps, terminal_steps = env.get_steps(behavior_name)

        # ############ #
        # OBSERVATIONS #
        # ############ #
        # camera obs
        cam1_obs = decision_steps.obs[0]  # (agent_id, 3, H, W)
        cam2_obs = decision_steps.obs[1]  # (agent_id, 3, H, W)
        print(f"Camera 1 observation shape: {cam1_obs.shape}")
        print(f"Camera 1 observation type: {type(cam1_obs)}")
        print(f"Max value in camera 1 observation: {np.max(cam1_obs)}")
        print(f"Min value in camera 1 observation: {np.min(cam1_obs)}")

        # vector obs, (agent_id, 23)
        vec_obs = decision_steps.obs[2]  # (x, y, z, qw, qy, qx, qz, q_hand[16])
        print(f"Vector observation: {vec_obs}")
        print(f"Vector observation shape: {vec_obs.shape}")

        # bagging
        images.append(np.concatenate([cam1_obs, cam2_obs], axis=0))
        cube_poses.append(vec_obs[0, :7])
        image_filenames.append((f"img_{episode}a_test.png", f"img_{episode}b_test.png"))
        
        if tracked_agent in terminal_steps:
            done = True

    print(f"Episode {episode} finished")

# Close the environment
env.close()

# creating an hdf5 dataset
data_path = "test.hdf5"
train_test_idx = int(0.9 * len(images))
with h5py.File(data_path, "w") as f:
    f.attrs["n_cams"] = 2
    f.attrs["H"] = 376
    f.attrs["W"] = 672

    train = f.create_group("train")
    train.create_dataset("images", data=np.array(images[:train_test_idx]))
    train.create_dataset("cube_poses", data=np.array(cube_poses[:train_test_idx]))
    train.create_dataset("image_filenames", data=image_filenames[:train_test_idx])

    test = f.create_group("test")
    test.create_dataset("images", data=np.array(images[train_test_idx:]))
    test.create_dataset("cube_poses", data=np.array(cube_poses[train_test_idx:]))
    test.create_dataset("image_filenames", data=image_filenames[train_test_idx:])
