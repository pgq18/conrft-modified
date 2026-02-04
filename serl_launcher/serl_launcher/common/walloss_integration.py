"""Utility functions for integrating Walloss (PyTorch) with JAX agent."""

import numpy as np
import jax.numpy as jnp
import torch
from scipy.spatial.transform import Rotation as R


def generate_walloss_embeddings_batch(
    walloss_model,
    observations_list,
    task_desc,
    config,
    device="cuda"
):
    """
    Generate action embeddings for a batch of observations using walloss.

    Args:
        walloss_model: Qwen2_5_VLMoEForAction model
        observations_list: List of observation dicts
        task_desc: Task description string
        config: Experiment config with walloss parameters

    Returns:
        JAX array of embeddings: [batch_size, embedding_dim]
    """
    from wall_x.serving.policy.utils import prepare_batch

    walloss_model.eval()
    embeddings_list = []

    # Normalization parameters for 7D state: [x, y, z, roll, pitch, yaw, gripper]
    state_min = np.array([0.2, -0.3, 0.0, -np.pi, -np.pi/2, -np.pi, 0.0])
    state_max = np.array([0.6, 0.3, 0.5, np.pi, np.pi/2, np.pi, 1.0])
    state_delta = state_max - state_min

    # Process each observation
    for obs in observations_list:
        # Extract and process state (20D -> 7D normalized)
        # Original 20D state structure (alphabetically sorted by Dict):
        # - [0]: gripper_pose (1D)
        # - [1:4]: tcp_force (3D)
        # - [4:11]: tcp_pose (7D): xyz(3) + quaternion(4)
        # - [11:14]: tcp_torque (3D)
        # - [14:20]: tcp_vel (6D)
        original_state = obs["state"]  # 20D

        # Ensure state is 1D array
        if original_state.ndim != 1:
            if original_state.shape[0] == 0:
                raise ValueError(f"Empty state array in observation")
            # If 2D with shape (1, 20) or similar, flatten to (20,)
            original_state = original_state.flatten()

        if original_state.shape[0] != 20:
            raise ValueError(f"Expected state to have 20 elements, got {original_state.shape[0]}")

        # Extract gripper (index 0)
        gripper = original_state[0]

        # Extract tcp_pose (indices 4:11, i.e., 7 dimensions)
        tcp_pose_quat = original_state[4:11]  # xyz(3) + quat(4)

        # Convert quaternion to euler angles
        xyz = tcp_pose_quat[:3]  # Position
        quat = tcp_pose_quat[3:]  # Quaternion
        euler = R.from_quat(quat).as_euler("xyz")  # Roll, pitch, yaw

        # Combine into 7D state: xyz(3) + euler(3) + gripper(1)
        processed_state = np.concatenate([xyz, euler, np.array([gripper])])

        # Normalize to [-1, 1]
        normalized_state = (processed_state - state_min) / state_delta * 2 - 1
        normalized_state = np.clip(normalized_state, -1.0, 1.0)

        # Prepare batch
        wallx_obs = {
            **obs,
            "prompt": task_desc,
            "state": normalized_state,
            "dataset_names": ["penggq/task_0"],
        }

        input_batch = prepare_batch(
            obs=wallx_obs,
            processor=walloss_model.processor,
            camera_key=config.image_keys,
            agent_pos_dim=config.agent_pos_dim,
            action_dim=config.action_dim,
            pred_horizon=1,
            fixed_action_dim=20,
            max_length=2048,
            image_factor=28,
            min_pixels=4 * 28 * 28,
            max_pixels=16384 * 28 * 28,
            predict_mode="diffusion",  # Required to add <|action|> tokens
            device=device,
        )

        # Get embeddings
        with torch.no_grad():
            outputs = walloss_model(
                **input_batch,
                action_dim=config.action_dim,
                pred_horizon=1,
                mode="get_embeddings",
            )

            # Extract action-type embeddings from the last hidden state
            # Get last layer hidden states: [batch_size, seq_len, hidden_dim]
            # hidden_states = outputs["hidden_states"][-1]
            hidden_states = outputs

            # Get input_ids to identify action tokens
            input_ids = input_batch["input_ids"]

            # Create action token mask to find action token positions
            action_token_id = walloss_model.action_token_id_set["action_token_id"]
            action_mask = (input_ids == action_token_id)  # [batch_size, seq_len]

            # Extract embeddings only from action token positions
            action_embedding = hidden_states[action_mask]  # [num_action_tokens, hidden_dim]

            # For demo recording with pred_horizon=1, there should be only 1 action token
            # Take the first one to get shape [1, hidden_dim]
            action_embedding = action_embedding[0:1] if action_embedding.ndim == 2 else action_embedding.unsqueeze(0)

        # Convert to numpy (handle BFloat16 by converting to float first)
        embeddings_list.append(action_embedding.float().cpu().numpy())

    # Stack and convert to JAX
    embeddings_np = np.vstack(embeddings_list)  # [batch_size, hidden_dim]
    return jnp.array(embeddings_np)


def generate_walloss_embeddings_single(
    walloss_model,
    observation,
    task_desc,
    config,
    device="cuda"
):
    """
    Generate action embedding for a single observation.

    Returns:
        JAX array: [embedding_dim]
    """
    embeddings = generate_walloss_embeddings_batch(
        walloss_model, [observation], task_desc, config, device
    )
    return embeddings[0]  # [embedding_dim]
