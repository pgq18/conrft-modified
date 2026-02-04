import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time
import cv2
import yaml

from experiments.mappings import CONFIG_MAPPING
from data_util import add_mc_returns_to_trajectory, add_embeddings_to_trajectory, add_next_embeddings_to_trajectory

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20,
                     "Number of successful demos to collect.")
flags.DEFINE_float("reward_scale", 1.0, "reward_scale ")
flags.DEFINE_float("reward_bias", 0.0, "reward_bias")
flags.DEFINE_string("backbone", "octo", "Backbone to use.")
flags.DEFINE_integer("stack_obs_num", 2, "Number of frames to stack")

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["data"]["model_type"] = config.get("model_type")

    return config

def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    image_keys = config.image_keys
    env = config.get_environment(
        fake_env=False, save_video=False, classifier=True, stack_obs_num=FLAGS.stack_obs_num)

    obs, info = env.reset()
    print("Observation keys: ", obs.keys())
    print("Image keys: ", image_keys)
    print("Reset done")

    # Initialize model after we have the first observation
    if FLAGS.backbone == "octo":
        print("Using Octo model for encoder")
        from octo.model.octo_model import OctoModel
        model = OctoModel.load_pretrained(config.octo_path)
        tasks = model.create_tasks(texts=[config.task_desc]) # instruction embedding
    elif FLAGS.backbone == "walloss":
        # todo: load pretrained walloss model
        print("Using pretrained walloss for encoder")
        from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
        train_config = load_config(config.wallx_config_path)
        model = Qwen2_5_VLMoEForAction.from_pretrained(
            config.wallx_path, train_config=train_config
        )
        model.eval()
        model = model.to("cuda")
        model = model.bfloat16()
        tasks = config.task_desc
    else:
        print("Backbone not supported")
        model = None
        tasks = None

    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0

    while success_count < success_needed:
        actions = np.zeros(env.action_space.sample().shape)
        next_obs, rew, done, truncated, info = env.step(actions)
        returns += rew
        if "intervene_action" in info:
            actions = info["intervene_action"]
        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
                infos=info,
            )
        )
        trajectory.append(transition)

        pbar.set_description(f"Return: {returns:.2f}")

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

        obs = next_obs
        if done:
            if info["succeed"]:
                trajectory = add_mc_returns_to_trajectory(
                    trajectory, config.discount, FLAGS.reward_scale, FLAGS.reward_bias, config.reward_neg, is_sparse_reward=True)
                trajectory = add_embeddings_to_trajectory(
                    FLAGS.backbone, trajectory, model, tasks=tasks, image_keys=image_keys, config=config)
                trajectory = add_next_embeddings_to_trajectory(trajectory)
                for transition in trajectory:
                    transitions.append(copy.deepcopy(transition))
                success_count += 1
                pbar.update(1)
            trajectory = []
            returns = 0
            obs, info = env.reset()
            time.sleep(2.0)

    if not os.path.exists("./demo_data"):
        os.makedirs("./demo_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_name}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run(main)
