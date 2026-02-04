import os
import jax
import jax.numpy as jnp
import numpy as np
import glfw
import gymnasium as gym

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv,
    # KeyBoardIntervention2
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

# from experiments.config import DefaultTrainingConfig
# from experiments.ram_insertion.wrapper import RAMEnv
from examples.experiments.config import DefaultTrainingConfig
# from examples.experiments.ram_insertion.wrapper import RAMEnv  # Not used for pick_cube_sim

from franka_sim.envs.panda_pick_gym_env import PandaPickCubeGymEnv

class EnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://127.0.0.2:5000/"
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "127122270146",
            "dim": (1280, 720),
            "exposure": 40000,
        },
        "wrist_2": {
            "serial_number": "127122270350",
            "dim": (1280, 720),
            "exposure": 40000,
        },
    }
    IMAGE_CROP = {
        "wrist_1": lambda img: img[150:450, 350:1100],
        "wrist_2": lambda img: img[100:500, 400:900],
    }
    TARGET_POSE = np.array([0.5881241235410154,-0.03578590131997776,0.27843494179085326, np.pi, 0, 0])
    GRASP_POSE = np.array([0.5857508505445138,-0.22036261105675414,0.2731021902359492, np.pi, 0, 0])
    RESET_POSE = TARGET_POSE + np.array([0, 0, 0.05, 0, 0.05, 0])
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.03, 0.02, 0.01, 0.01, 0.1, 0.4])
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.03, 0.02, 0.05, 0.01, 0.1, 0.4])
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.05
    ACTION_SCALE = (0.01, 0.06, 1)
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 100
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.0075,
        "translational_clip_y": 0.0016,
        "translational_clip_z": 0.0055,
        "translational_clip_neg_x": 0.002,
        "translational_clip_neg_y": 0.0016,
        "translational_clip_neg_z": 0.005,
        "rotational_clip_x": 0.01,
        "rotational_clip_y": 0.025,
        "rotational_clip_z": 0.005,
        "rotational_clip_neg_x": 0.01,
        "rotational_clip_neg_y": 0.025,
        "rotational_clip_neg_z": 0.005,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 250,
        "rotational_damping": 9,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.1,
        "translational_clip_y": 0.1,
        "translational_clip_z": 0.1,
        "translational_clip_neg_x": 0.1,
        "translational_clip_neg_y": 0.1,
        "translational_clip_neg_z": 0.1,
        "rotational_clip_x": 0.5,
        "rotational_clip_y": 0.5,
        "rotational_clip_z": 0.5,
        "rotational_clip_neg_x": 0.5,
        "rotational_clip_neg_y": 0.5,
        "rotational_clip_neg_z": 0.5,
        "rotational_Ki": 0.0,
    }


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist_1", "wrist_2"]
    classifier_keys = ["wrist_1", "wrist_2"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    # setup_mode = "single-arm-fixed-gripper"
    setup_mode = "single-arm-learned-gripper"
    task_desc = "Pick up the cube"
    octo_path = "/home/pgq/Models/octo-small-1.5"
    wallx_path = "/home/pgq/Models/wall-oss-flow"
    wallx_config_path = "/data/pgq/Workspace/VLA/conrft-modified/examples/experiments/pick_cube_sim/config_qact.yml"
    reward_neg = -0.05
    discount = 0.98
    random_steps = 0
    cta_ratio = 2

    # Walloss-specific configuration
    action_dim = 7  # For xyz + rpy + gripper
    agent_pos_dim = 7  # Valid state dimensions: xyz(3) + rpy(3) + gripper(1) = 7
    pred_horizon = 1  # For demo recording (single action)

    def get_environment(self, fake_env=False, save_video=False, classifier=False, render_mode="human", stack_obs_num=1):
        # env = RAMEnv(
        #     fake_env=fake_env,
        #     save_video=save_video,
        #     config=EnvConfig(),
        # )
        env = PandaPickCubeGymEnv(render_mode=render_mode, image_obs=True, config=EnvConfig())
        classifier=False
        # fake_env=True
        # env = GripperCloseEnv(env)
        if not fake_env:
            # env = SpacemouseIntervention(env)
            env = KeyBoardIntervention2(env)
            pass
        # env = RelativeFrame(env)
        # env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=stack_obs_num, act_exec_horizon=None)
        if classifier:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                # added check for z position to further robustify classifier, but should work without as well
                return int(sigmoid(classifier(obs)) > 0.85 and obs['state'][0, 6] > 0.04)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env

class KeyBoardIntervention2(gym.ActionWrapper):
    """
    Enhanced keyboard intervention wrapper with two operation modes.

    Mode 1 (L-key toggle): Full intervention mode where keyboard completely
    replaces model actions. Actions exponentially decay when no keys are pressed,
    creating smooth coasting behavior between manual commands.

    Mode 2 (Temporary): When L-key mode is disabled, pressing movement keys
    temporarily overrides model actions. Release immediately switches back to
    model control with no decay.

    Key bindings:
    - W/S: X-axis movement (forward/backward)
    - A/D: Y-axis movement (left/right)
    - H/J: Z-axis movement (up/down)
    - K: Toggle gripper state
    - L: Toggle between Mode 1 and Mode 2

    Attributes:
        intervened (bool): True = Mode 1 (L-key), False = Mode 2 (temporary)
        last_keyboard_action (np.ndarray): Last action for decay in Mode 1
        decay_coefficient (float): Exponential decay factor (default: 0.9)
        decay_threshold (float): Threshold to zero out actions (default: 0.01)
    """
    def __init__(self, env, action_indices=None):
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        # Initialize last_keyboard_action with correct dimension
        self.last_keyboard_action = np.zeros(self.action_space.shape[0])

        self.left, self.right = False, False
        self.action_indices = action_indices

        self.gripper_state = 'close'
        self.intervened = False
        self.action_length = 0.75
        self.current_action = np.array([0, 0, 0, 0, 0, 0])  # 分别对应 W, A, S, D 的状态
        self.flag = False
        self.action_space_scale = [0.2, 0, 1.0]
        # New state variables for enhanced intervention
        self.decay_coefficient = 0.3  # Exponential decay factor
        self.decay_threshold = 0.01  # Threshold to zero out actions
        self.key_states = {
            'w': False,
            'a': False,
            's': False,
            'd': False,
            'h': False,
            'j': False,
            'k': False,
            'l': False,
        }

        # 设置 GLFW 键盘回调
        glfw.set_key_callback(self.env._viewer.viewer.window, self.glfw_on_key)

    def glfw_on_key(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_W:
                self.key_states['w'] = True
            elif key == glfw.KEY_A:
                self.key_states['a'] = True
            elif key == glfw.KEY_S:
                self.key_states['s'] = True
            elif key == glfw.KEY_D:
                self.key_states['d'] = True
            elif key == glfw.KEY_H:
                self.key_states['h'] = True
            elif key == glfw.KEY_J:
                self.key_states['j'] = True
            elif key == glfw.KEY_K:
                self.key_states['k'] = True
                self.flag = True
            elif key == glfw.KEY_L:
                self.intervened = not self.intervened
                self.env.intervened = self.intervened

                # Reset decay state when exiting intervention mode
                if not self.intervened:
                    self.last_keyboard_action = np.zeros_like(self.last_keyboard_action)

                mode_name = "L-key full intervention (with decay)" if self.intervened else "Temporary intervention (no decay)"
                print(f"Intervention mode: {mode_name}")

        elif action == glfw.RELEASE:
            if key == glfw.KEY_W:
                self.key_states['w'] = False
            elif key == glfw.KEY_A:
                self.key_states['a'] = False
            elif key == glfw.KEY_S:
                self.key_states['s'] = False
            elif key == glfw.KEY_D:
                self.key_states['d'] = False
            elif key == glfw.KEY_H:
                self.key_states['h'] = False
            elif key == glfw.KEY_J:
                self.key_states['j'] = False
            elif key == glfw.KEY_K:
                self.key_states['k'] = False

        self.current_action = [
            int(self.key_states['w']) - int(self.key_states['s']),
            int(self.key_states['a']) - int(self.key_states['d']),
            int(self.key_states['h']) - int(self.key_states['j']),
            0,
            0,
            0,
        ]
        self.current_action = np.array(self.current_action, dtype=np.float64)
        self.current_action *= self.action_length

    def _build_keyboard_action(self):
        """
        Build keyboard action from current key states.

        Constructs a 6DOF (or 7DOF if gripper enabled) action vector based on
        currently pressed movement keys. Each movement direction uses the
        difference between key pairs (W-S for x-axis, A-D for y-axis, H-J for z-axis).

        Returns:
            np.ndarray: Action vector with shape (6,) or (7,) depending on gripper
                        action. Movement components are scaled by action_length,
                        gripper action is either 0.9 (close) or -0.9 (open).
        """
        # Build base 6DOF action from WASD + HJK keys
        base_action = np.array([
            int(self.key_states['w']) - int(self.key_states['s']),  # x axis
            int(self.key_states['a']) - int(self.key_states['d']),  # y axis
            int(self.key_states['h']) - int(self.key_states['j']),  # z axis
            0,  # roll (no keyboard control)
            0,  # pitch (no keyboard control)
            0,  # yaw (no keyboard control)
        ], dtype=np.float64) * self.action_length

        # Add gripper action if enabled
        if self.gripper_enabled:
            gripper_action = np.array([0.9]) if self.gripper_state == 'close' else np.array([-0.9])
            return np.concatenate([base_action, gripper_action])
        else:
            return base_action

    def _any_movement_key_pressed(self):
        """
        Check if any movement key (WASD, HJK) is currently pressed.

        Determines if manual intervention should be active by checking if any
        of the movement keys (forward/backward/left/right/up/down) are pressed.
        Excludes gripper toggle (K) and mode toggle (L) keys.

        Returns:
            bool: True if any movement key is pressed, False otherwise.
        """
        movement_keys = ['w', 'a', 's', 'd', 'h', 'j']
        return any(self.key_states[key] for key in movement_keys)

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Transform action based on current intervention mode and key states.

        Implements two-mode intervention logic:
        - Mode 1 (L-key enabled): Full intervention with exponential decay.
          When keys are pressed, keyboard actions are used and stored.
          When no keys are pressed, stored actions decay exponentially.
        - Mode 2 (L-key disabled): Temporary intervention. Only when keys
          are pressed, keyboard actions override model actions. Release
          immediately returns to model control.

        Args:
            action (np.ndarray): Original action from the model policy.
                                Shape should match the environment's action space.

        Returns:
            tuple[np.ndarray, bool]: A tuple containing:
                - The final action to be executed (keyboard or model)
                - Boolean indicating whether intervention occurred (True)
                  or model action was used (False)
        """
        # Build keyboard action from current key states
        keyboard_action = self._build_keyboard_action()
        any_key_pressed = self._any_movement_key_pressed()

        # Handle gripper state toggle (K key)
        # When K is pressed, always return keyboard action with updated gripper state
        if self.flag:
            if self.gripper_state == 'open':
                self.gripper_state = 'close'
            elif self.gripper_state == 'close':
                self.gripper_state = 'open'

            # Rebuild keyboard action with updated gripper state
            keyboard_action = self._build_keyboard_action()
            self.flag = False

            # Store for decay in Mode 1
            if self.intervened:
                self.last_keyboard_action = keyboard_action.copy()

            # Apply action index filtering if needed
            if self.action_indices is not None:
                filtered_expert_a = np.zeros_like(keyboard_action)
                filtered_expert_a[self.action_indices] = keyboard_action[self.action_indices]
                keyboard_action = filtered_expert_a

            return keyboard_action, True

        if self.intervened:
            # Mode 1: L-key full intervention mode with decay
            # Always decay first, then update pressed key dimensions
            self.last_keyboard_action *= self.decay_coefficient

            if any_key_pressed:
                # Keys pressed: update only the dimensions with active keys
                # Mask indicates which dimensions have active key presses
                key_mask = keyboard_action != 0
                self.last_keyboard_action[key_mask] = keyboard_action[key_mask]

            # Zero out actions below threshold (except gripper)
            if self.gripper_enabled:
                # Apply threshold only to non-gripper dimensions
                mask = np.abs(self.last_keyboard_action[:-1]) < self.decay_threshold
                self.last_keyboard_action[:-1][mask] = 0.0
                # Restore gripper value from gripper_state (gripper doesn't decay)
                gripper_action = 0.9 if self.gripper_state == 'close' else -0.9
                self.last_keyboard_action[-1] = gripper_action
            else:
                mask = np.abs(self.last_keyboard_action) < self.decay_threshold
                self.last_keyboard_action[mask] = 0.0

            expert_a = self.last_keyboard_action

            # Apply action index filtering if needed
            if self.action_indices is not None:
                filtered_expert_a = np.zeros_like(expert_a)
                filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
                expert_a = filtered_expert_a

            return expert_a, True

        else:
            # Mode 2: Temporary intervention mode (no decay)
            if any_key_pressed:
                # Keys pressed: temporarily use keyboard action
                expert_a = keyboard_action

                if self.action_indices is not None:
                    filtered_expert_a = np.zeros_like(expert_a)
                    filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
                    expert_a = filtered_expert_a

                return expert_a, True
            else:
                # No keys pressed: use model action
                return action, False

    def step(self, action):
        new_action, replaced = self.action(action)
        print("-------------------------------------------")
        print("action: ", action)
        if replaced:
            print("intervene_action: ", new_action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.gripper_state = 'open'
        self.last_keyboard_action = np.zeros_like(self.last_keyboard_action)
        return obs, info