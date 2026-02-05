import numpy as np
import jax
import cv2
import torch

def calc_return_to_go(rewards, terminals, gamma, reward_scale, reward_bias, reward_neg, is_sparse_reward):
    """
    A config dict for getting the default high/low rewrd values for each envs
    """
    if len(rewards) == 0:
        return np.array([])

    if is_sparse_reward:
        reward_neg = reward_neg * reward_scale + reward_bias
    else:
        assert not is_sparse_reward, "If you want to try on a sparse reward env, please add the reward_neg value in the ENV_CONFIG dict."

    if is_sparse_reward and np.all(np.array(rewards) == reward_neg):
        """
        If the env has sparse reward and the trajectory is all negative rewards,
        we use r / (1-gamma) as return to go.
        For exapmle, if gamma = 0.99 and the rewards = [-1, -1, -1],
        then return_to_go = [-100, -100, -100]
        """
        return_to_go = [float(reward_neg / (1-gamma))] * len(rewards)
    else:
        return_to_go = [0] * len(rewards)
        prev_return = 0
        for i in range(len(rewards)):
            return_to_go[-i-1] = rewards[-i-1] + gamma * \
                prev_return * (1 - terminals[-i-1])
            prev_return = return_to_go[-i-1]

    return np.array(return_to_go, dtype=np.float32)


def add_mc_returns_to_trajectory(trajectory, gamma, reward_scale, reward_bias, reward_neg, is_sparse_reward):
    """
    undate every transition in the trajectory and add mc_returns
    return the updated trajectory
    """
    rewards = [t['rewards'] for t in trajectory]
    terminals = [t['dones'] for t in trajectory]

    mc_returns = calc_return_to_go(
        rewards=rewards,
        terminals=terminals,
        gamma=gamma,
        reward_scale=reward_scale,
        reward_bias=reward_bias,
        reward_neg=reward_neg,
        is_sparse_reward=is_sparse_reward,
    )

    for i, transition in enumerate(trajectory):
        transition['mc_returns'] = mc_returns[i]

    return trajectory

class ModelWrapper(torch.nn.Module):
    """Wrapper class for models that require keyword arguments."""

    def __init__(self, model, input_dict):
        super().__init__()
        self.model = model
        # Store the input structure, thop will replace tensor values with hooks
        self.input_dict = input_dict

    def forward(self):
        # thop doesn't pass any arguments when using this approach
        # We use the stored input_dict with tensors modified by thop's hooks
        return self.model(**self.input_dict)

def add_embeddings_to_trajectory(backbone, trajectory, model, tasks, image_keys, config=None):
    """
    undate every transition in the trajectory and add embeddings
    return the updated trajectory
    Handles both Octo model (with tasks) and ResNet encoder (without tasks)
    """

    if backbone == "octo":
        for i in range(len(trajectory)):
            observation = trajectory[i]['observations']

            # image_primary = observation["side_policy_256"]
            # image_wrist = observation["wrist_1"]
            image_primary = observation[image_keys[0]]
            image_wrist = observation[image_keys[1]]

            # Resize images to match Octo's expected sizes:
            # image_primary: 256x256 (keep as is)
            # image_wrist: 256x256 -> 128x128
            if image_wrist.ndim == 4:  # (window, H, W, C)
                image_wrist = np.stack([cv2.resize(frame, (128, 128), interpolation=cv2.INTER_LINEAR)
                                        for frame in image_wrist])
            elif image_wrist.ndim == 3:  # (H, W, C)
                image_wrist = cv2.resize(image_wrist, (128, 128), interpolation=cv2.INTER_LINEAR)

            # Add batch dimension
            image_primary = image_primary[np.newaxis, ...]
            image_wrist = image_wrist[np.newaxis, ...]

            timestep_pad_mask = np.array([[True, True]])

            observation = {
                "image_primary": image_primary,
                "image_wrist": image_wrist,
                "timestep_pad_mask": timestep_pad_mask
                }

            action_embeddings = model.sample_transformer(observation, tasks,)
            # Now, action_embeddings is (batch_size, window_size, embedding_size)

            # remove window_size dimension
            action_embeddings = action_embeddings[:, -1, :]

            trajectory[i]['embeddings'] = action_embeddings
    elif backbone == "walloss":
        from wall_x.serving.policy.utils import prepare_batch
        from scipy.spatial.transform import Rotation as R

        processor = model.processor
        device = next(model.parameters()).device

        # Get configuration parameters
        camera_key = image_keys  # ["wrist_1", "wrist_2"]
        action_dim = config.action_dim if config else 7  # Default for pick_cube
        agent_pos_dim = config.agent_pos_dim if config else 7
        pred_horizon = 1  # Single action for demo recording
        fixed_action_dim = 20  # Wall-X standard
        max_length = 2048
        image_factor = 28
        min_pixels = 4 * 28 * 28
        max_pixels = 16384 * 28 * 28

        # Normalization parameters for 7D state: [x, y, z, roll, pitch, yaw, gripper]
        # Ranges from _CARTESIAN_BOUNDS and conventions
        state_min = np.array([0.2, -0.3, 0.0, -np.pi, -np.pi/2, -np.pi, 0.0])
        state_max = np.array([0.6, 0.3, 0.5, np.pi, np.pi/2, np.pi, 1.0])
        state_delta = state_max - state_min  # [0.4, 0.6, 0.5, 2*pi, pi, 2*pi, 1.0]

        for i in range(len(trajectory)):
            observation = trajectory[i]['observations']

            # Process state: extract and convert from 20D to 7D
            # Original 20D state structure (alphabetically sorted by Dict):
            # - [0]: gripper_pose (1D)
            # - [1:4]: tcp_force (3D)
            # - [4:11]: tcp_pose (7D): xyz(3) + quaternion(4)
            # - [11:14]: tcp_torque (3D)
            # - [14:20]: tcp_vel (6D)
            original_state = observation["state"]  # 20D array

            # Ensure state is 1D array
            if original_state.ndim != 1:
                if original_state.shape[0] == 0:
                    # Empty state, skip this observation
                    print(f"Warning: Empty state at trajectory index {i}, skipping")
                    continue
                # If 2D with shape (1, 20), flatten to (20,)
                original_state = original_state.flatten()

            if original_state.shape[0] != 20:
                raise ValueError(f"Expected state to have 20 elements, got {original_state.shape[0]}")

            # Extract gripper (index 0)
            gripper = original_state[0]

            # Extract tcp_pose (indices 4:11, i.e., 7 dimensions)
            tcp_pose_quat = original_state[4:11]  # xyz(3) + quat(4)

            # Convert quaternion to euler angles
            xyz = tcp_pose_quat[:3]  # Position (indices 4:6)
            quat = tcp_pose_quat[3:]  # Quaternion (indices 7:10)
            euler = R.from_quat(quat).as_euler("xyz")  # Roll, pitch, yaw

            # Combine into 7D state: xyz(3) + euler(3) + gripper(1)
            processed_state = np.concatenate([xyz, euler, np.array([gripper])])

            # Normalize to [-1, 1] using min-max normalization
            # Formula: normalized = (value - min) / delta * 2 - 1
            normalized_state = (processed_state - state_min) / state_delta * 2 - 1
            normalized_state = np.clip(normalized_state, -1.0, 1.0)

            # Prepare obs dict for Wall-X
            wallx_obs = {
                **observation,  # Includes camera images
                "prompt": tasks,  # Task instruction text
                "state": normalized_state,  # 7D normalized to [-1, 1]
                "dataset_names": ["penggq/task_0"],
            }

            # Prepare batch using Wall-X utility
            input_batch = prepare_batch(
                obs=wallx_obs,
                processor=processor,
                camera_key=camera_key,
                agent_pos_dim=agent_pos_dim,
                action_dim=action_dim,
                pred_horizon=pred_horizon,
                fixed_action_dim=fixed_action_dim,
                max_length=max_length,
                image_factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                predict_mode="diffusion",  # Required to add <|action|> tokens
                device=device,
            )

            # Get embeddings from model
            with torch.no_grad():
                outputs = model(
                    **input_batch,
                    action_dim=action_dim,
                    pred_horizon=pred_horizon,
                    mode="get_embeddings",
                    # predict_mode="fast",
                )

                # Extract action-type embeddings from the last hidden state
                # Get last layer hidden states: [batch_size, seq_len, hidden_dim]
                hidden_states = outputs

                # Get input_ids to identify action tokens
                input_ids = input_batch["input_ids"]

                # Create action token mask to find action token positions
                action_token_id = model.action_token_id_set["action_token_id"]
                action_mask = (input_ids == action_token_id)  # [batch_size, seq_len]

                # Extract embeddings only from action token positions
                action_embeddings = hidden_states[action_mask]  # [num_action_tokens, hidden_dim]

                print(f"Found {action_embeddings.shape} action tokens")

                # For demo recording with pred_horizon=1, there should be only 1 action token
                # Take the first one to get shape [1, hidden_dim]
                action_embeddings = action_embeddings[0:1] if action_embeddings.ndim == 2 else action_embeddings.unsqueeze(0)

            # Convert to numpy and store (handle BFloat16)
            trajectory[i]['embeddings'] = action_embeddings.float().cpu().numpy().squeeze(0)

    return trajectory


def add_next_embeddings_to_trajectory(trajectory):
    """
    undate every transition in the trajectory and add next_embeddings
    return the updated trajectory
    """
    for i in range(len(trajectory)):
        if i == len(trajectory) - 1:
            trajectory[i]['next_embeddings'] = trajectory[i]['embeddings']
        else:
            trajectory[i]['next_embeddings'] = trajectory[i+1]['embeddings']

    return trajectory
