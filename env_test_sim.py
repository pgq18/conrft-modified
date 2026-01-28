# import debugpy
# debugpy.listen(10010)
# print('wait debugger')
# debugpy.wait_for_client()
# print("Debugger Attached")

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from franka_sim.envs.panda_pick_gym_env import PandaPickCubeGymEnv
from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv
)
# from pynput import keyboard  # Disabled for headless environment
from examples.experiments.pick_cube_sim.config import TrainConfig
import cv2

# env = PandaPickCubeGymEnv(render_mode="human")
# env = PandaPickCubeGymEnv(render_mode="human", image_obs=True, config=EnvConfig())
env = TrainConfig().get_environment()
import numpy as np

obs, _ = env.reset()

while True:
    actions = env.action_space.sample()
    print(actions)
    # actions = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1]) # relative action
    # actions = np.zeros(env.action_space.sample().shape)
    next_obs, reward, done, truncated, info = env.step(actions)
    # print(next_obs["state"])

    # Display camera images
    if "wrist_1" in next_obs and "wrist_2" in next_obs:
        img1 = next_obs["wrist_1"]
        img2 = next_obs["wrist_2"]

        # Remove batch dimension: (1, H, W, C) -> (H, W, C)
        if img1.ndim == 4:
            img1 = img1[0]
        if img2.ndim == 4:
            img2 = img2[0]

        # Ensure uint8 type
        img1 = img1.astype(np.uint8)
        img2 = img2.astype(np.uint8)

        # Convert RGB to BGR (OpenCV uses BGR format)
        img1_bgr = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        img2_bgr = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

        # Resize for better viewing
        img1_resized = cv2.resize(img1_bgr, (320, 320))
        img2_resized = cv2.resize(img2_bgr, (320, 320))

        # Concatenate horizontally
        combined = np.hstack((img1_resized, img2_resized))
        cv2.imshow('Camera Views - Wrist 1 | Wrist 2', combined)
        cv2.waitKey(1)

    if done:
        obs, info = env.reset()

cv2.destroyAllWindows()
