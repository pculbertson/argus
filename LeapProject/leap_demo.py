import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from mlagents_envs.base_env import ActionTuple

# Path to the Unity environment executable
# NOTE: this is a Mac-specific path
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
    print(len(decision_steps), len(terminal_steps))
    while len(terminal_steps) == 0:
        # Get the current state of the agent
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        # If no agents are found, log an error
        if len(decision_steps) == 0:
            print("No agents found in the environment.")
            break

        # Extract observations
        obs = decision_steps.obs[0]  # Assuming single observation
        print(f"Observations: {obs}")

        # Define actions (e.g., random actions)
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

    print(f"Episode {episode} finished")

# Close the environment
env.close()
