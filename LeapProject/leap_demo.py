import h5py
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from mlagents_envs.base_env import ActionTuple

# Path to the Unity environment executable
env_path = "/home/albert/research/argus/LeapProject/leap_env.x86_64"

# Verify the environment path
import os

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

# run some episodes
n_episodes = 10
for episode in range(n_episodes):
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    tracked_agent = -1 # -1 indicates not yet tracking
    done = False # For the tracked_agent
    while not done:
        if tracked_agent == -1 and len(decision_steps) >= 1:
            tracked_agent = decision_steps.agent_id[0]

        # ############ #
        # OBSERVATIONS #
        # ############ #
        # vector obs
        vec_obs = decision_steps.obs[0]
        print(f"Vector observation shape: {vec_obs.shape}")

        # camera obs
        cam1_obs = decision_steps.obs[1]
        cam2_obs = decision_steps.obs[2]
        print(f"Camera 1 observation shape: {cam1_obs.shape}")
        print(f"Camera 1 observation type: {type(cam1_obs)}")

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
        if tracked_agent in terminal_steps:
            done = True

    print(f"Episode {episode} finished")

# Close the environment
env.close()
