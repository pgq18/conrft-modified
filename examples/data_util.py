import numpy as np
import jax


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


def add_embeddings_to_trajectory(trajectory, model, tasks, image_keys):
    """
    undate every transition in the trajectory and add embeddings
    return the updated trajectory
    Handles both Octo model (with tasks) and ResNet encoder (without tasks)
    """
    import cv2

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
        # batch_size, window_size = image_primary.shape[:2]
        # timestep_pad_mask = np.ones((batch_size, window_size), dtype=bool)
        # timestep = np.arange(window_size)[None, :].repeat(batch_size, axis=0)

        observation = {
            "image_primary": image_primary,
            "image_wrist": image_wrist,
            "timestep_pad_mask": timestep_pad_mask
            }
        # observation = {
        #     "image_primary": image_primary,
        #     "image_wrist": image_wrist,
        #     "timestep": timestep,
        #     "timestep_pad_mask": timestep_pad_mask,
        #     "pad_mask_dict": {
        #         "image_primary": timestep_pad_mask,
        #         "image_wrist": timestep_pad_mask,
        #         "timestep": timestep_pad_mask,
        #     },
        #     "task_completed": np.zeros((batch_size, window_size), dtype=np.float32),
        # }

        action_embeddings = model.sample_transformer(observation, tasks,)
        # Now, action_embeddings is (batch_size, window_size, embedding_size)

        # remove window_size dimension
        action_embeddings = action_embeddings[:, -1, :]

        trajectory[i]['embeddings'] = action_embeddings

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
