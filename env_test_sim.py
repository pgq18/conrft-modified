# import debugpy
# debugpy.listen(10010)
# print('wait debugger')
# debugpy.wait_for_client()
# print("Debugger Attached")

import sys
sys.path.insert(0, '/data/disk0/Home/pengguanqi/Workspace/VLA/conrft/serl_robot_infra')
sys.path.insert(0, '/data/disk0/Home/pengguanqi/Workspace/VLA/conrft')

from franka_sim.envs.panda_pick_gym_env import PandaPickCubeGymEnv
from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv
)
# from pynput import keyboard  # Disabled for headless environment
from examples.experiments.pick_cube_sim.config import TrainConfig

# env = PandaPickCubeGymEnv(render_mode="human")
# env = PandaPickCubeGymEnv(render_mode="human", image_obs=True, config=EnvConfig())
env = TrainConfig().get_environment()
import numpy as np

obs, _ = env.reset()

while True:
    actions = env.action_space.sample()
    # actions = np.zeros(env.action_space.sample().shape)
    next_obs, reward, done, truncated, info = env.step(actions)

    if done:
        obs, info = env.reset()
