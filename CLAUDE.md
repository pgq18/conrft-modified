# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ConRFT** (Consistency Reinforced Fine-Tuning) is a research project implementing reinforced fine-tuning for Vision-Language-Action (VLA) models, specifically targeting the Octo model for robotic manipulation tasks. The project uses a two-stage training approach: Cal-ConRFT (calibration with demonstrations) followed by HIL-ConRFT (human-in-the-loop online fine-tuning).

The codebase is built on top of HIL-SERL and implements an asynchronous actor-learner architecture using agentlace for networked training between real robot hardware and training nodes.

## Installation & Setup

### Environment Setup

```bash
# Create conda environment
conda create -n hilserl python=3.10
conda activate hilserl

# Install JAX (GPU example)
pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install Octo fork (required custom fork)
git clone git@github.com:cccedric/octo.git
cd octo
pip install -e .
pip install -r requirements.txt

# Install serl_launcher
cd serl_launcher
pip install -e .
pip install -r requirements.txt

# Install serl_robot_infra (for real robot only)
cd serl_robot_infra
pip install -e .
```

### External Dependencies

For real robot training, you must also:
- Install `libfranka` and `franka_ros` (see [Franka documentation](https://frankaemika.github.io/docs/requirements.html))
- Install `serl_franka_controllers` from https://github.com/rail-berkeley/serl_franka_controllers
- Set up ROS environment for robot control

## Common Development Commands

### Data Collection

```bash
# Collect success/failure data for reward classifier
cd examples
/home/pengguanqi/miniconda3/envs/hilserl/bin/python record_success_fail.py --exp_name task1_pick_banana --successes_needed 200

# Record human demonstrations (press space during execution)
/home/pengguanqi/miniconda3/envs/hilserl/bin/python record_demos_octo.py --exp_name task1_pick_banana --successes_needed 30

# Train reward classifier from collected data
cd examples/experiments/task1_pick_banana
/home/pengguanqi/miniconda3/envs/hilserl/bin/python ../../train_reward_classifier.py --exp_name task1_pick_banana
```

### Training

Training uses an actor-learner architecture with two threads:

```bash
# Stage I: Cal-ConRFT (pretraining on demonstrations)
cd examples/experiments/task1_pick_banana
bash run_learner_conrft_pretrain.sh

# Stage II: HIL-ConRFT (online fine-tuning with human interventions)
# Run both actor and learner concurrently:
bash run_learner_conrft.sh  # Terminal 1
bash run_actor_conrft.sh    # Terminal 2
```

### Evaluation

```bash
# Edit run_actor_conrft.sh to uncomment and set:
# --eval_checkpoint_step=CHECKPOINT_NUMBER
# --eval_n_trajs=20

bash run_actor_conrft.sh
```

### Robot Server (Real Robot)

```bash
# Start Franka robot server (requires ROS environment)
bash serl_robot_infra/robot_servers/launch_right_server.sh

# Useful server commands
curl -X POST http://<FRANKA_SERVER_URL>:5000/getpos_euler  # Get current pose
curl -X POST http://<FRANKA_SERVER_URL>:5000/jointreset    # Reset joints
```

### Simulation Testing

```bash
# Test simulation environment
/home/pengguanqi/miniconda3/envs/hilserl/bin/python env_test_sim.py
```

## Architecture & Code Structure

### Three-Package Structure

The codebase is organized as three installable Python packages:

1. **serl_launcher/** - Core RL training framework
2. **serl_robot_infra/** - Real robot infrastructure (Franka arm, cameras, grippers)
3. **franka_sim/** - MuJoCo simulation environment

### Actor-Learner Architecture

- **Actor Thread** (`examples/train_conrft_octo.py --actor`):
  - Runs policy in environment (real robot or simulation)
  - Collects experience transitions
  - Sends data to learner via agentlace network
  - Receives updated policy weights from learner

- **Learner Thread** (`examples/train_conrft_octo.py --learner`):
  - Trains policy using ConRFT algorithm
  - Receives data from actor via agentlace
  - Periodically syncs updated weights to actor
  - Saves checkpoints to disk

### Key Components

#### Core Agent Implementation

**Location:** `serl_launcher/serl_launcher/agents/continuous/conrft_single_octo_cp.py`

**Class:** `ConrftCPOctoAgentSingleArm`

The agent implements:
- Consistency policy learning with Octo VLA model
- Critic-regularized policy optimization
- Return-to-go computation and action embeddings
- Support for multi-camera observations

#### Configuration System

**Base Config:** `examples/experiments/config.py`

**Experiment Configs:** `examples/experiments/<task_name>/config.py`

Each experiment has its own configuration class defining:
- `EnvConfig`: Server URLs, camera settings, poses, workspace limits
- `TrainConfig`: Agent hyperparameters, image keys, classifier keys

**Experiment Mapping:** `examples/experiments/mappings.py` maps experiment names to configs

#### Vision & Reward System

- **Reward Classifier:** `serl_launcher/serl_launcher/networks/reward_classifier.py`
  - Trains on camera images to predict task success/failure
  - Supports multi-camera inputs
  - Used for both reward signaling and episode termination

- **Octo Integration:** `serl_launcher/serl_launcher/common/encoding.py`
  - `OctoEncodingWrapper` integrates Octo VLA model
  - Handles multi-modal observations (images + proprioception)

#### Data Flow

1. Actor collects transitions in environment
2. Transitions sent to learner via `agentlace` (networked data store)
3. Learner adds return-to-go and action embeddings
4. Learner trains policy using ConRFT loss (Q-weighted + BC-weighted)
5. Learner periodically syncs updated policy to actor
6. Human interventions via spacemouse captured as additional training data

### Wrappers and Preprocessing

**Location:** `serl_launcher/serl_launcher/wrappers/`

Key wrappers:
- `chunking.py` - Action chunking for temporal smoothing
- `norm.py` - Observation normalization
- `serl_obs_wrappers.py` - SERL-specific observation transformations
- `front_camera_wrapper.py` - Camera selection and handling
- `video_recorder.py` - Episode video recording

### Replay Buffer

**Standard:** `serl_launcher/serl_launcher/data/replay_buffer.py`
**Memory-Efficient:** `serl_launcher/serl_launcher/data/memory_efficient_replay_buffer.py`

The memory-efficient version is recommended for real robot training to handle large datasets.

## Experiment Workflow

### Setting Up a New Task

1. **Create experiment folder:**
   ```bash
   mkdir examples/experiments/my_new_task
   ```

2. **Create config.py** with `EnvConfig` and `TrainConfig` classes:
   - Set `SERVER_URL` for robot/simulation
   - Configure `REALSENSE_CAMERAS` with camera serial numbers
   - Set `IMAGE_CROP` for each camera
   - Define `TARGET_POSE`, `RESET_POSE`, and workspace limits
   - Specify `image_keys` for policy and `classifier_keys` for reward

3. **Add to mappings.py:**
   ```python
   from experiments.my_new_task.config import MyNewTaskConfig
   CONFIG_MAPPING = {
       "my_new_task": MyNewTaskConfig,
       # ... other tasks
   }
   ```

4. **Create launch scripts:** `run_actor_conrft.sh` and `run_learner_conrft.sh`

### Full Training Pipeline

1. **Collect classifier data** (success/failure examples from camera)
2. **Train reward classifier**
3. **Record demonstrations** (typically 30 episodes)
4. **Stage I: Pretrain** on demonstrations (Cal-ConRFT)
5. **Stage II: Online training** with human interventions (HIL-ConRFT)
6. **Evaluate** trained policy

## Important Implementation Details

### Octo Model Integration

- Uses custom Octo fork: `git@github.com:cccedric/octo.git`
- Fork adds custom functions while preserving core capabilities
- Octo provides pre-trained VLA model for fine-tuning

### Action Chunking

- Policy outputs chunks of actions (default: window size)
- Chunking provides temporal smoothness and stability
- Configured via `chunking.py` wrapper

### Multi-Camera Support

- Multiple cameras configured in `REALSENSE_CAMERAS`
- Image crops set per camera in `IMAGE_CROP`
- Different cameras can be used for policy vs reward classifier
- Camera keys specified in `image_keys` and `classifier_keys`

### Intervention Handling

- Spacemouse interventions captured during training
- Interventions provide corrective demonstrations
- Data automatically tagged as intervention vs autonomous
- Learner weights intervention data appropriately

### Return-to-Go Computation

- Computed per transition using discounted future rewards
- Added to replay buffer via `add_mc_returns_to_trajectory()`
- Used by consistency policy for temporal credit assignment

## Robot Infrastructure (serl_robot_infra)

### Flask Server Interface

**Location:** `serl_robot_infra/robot_servers/franka_server.py`

HTTP POST endpoints for robot control:
- `/pose` - Command end-effector pose
- `/getpos_euler` - Get current pose (xyz+rpy)
- `/jointreset` - Reset to joint positions
- `/startimp` / `/stopimp` - Start/stop impedance control
- `/close_gripper` / `/open_gripper` - Gripper control
- See full list in `serl_robot_infra/README.md`

### Camera System

- Uses RealSense cameras via `pyrealsense2`
- Camera configuration in experiment config files
- Exposure, gain, and cropping configured per camera
- Multi-camera support for different viewpoints

### Spacemouse Intervention

- 3DConnexion spacemouse for human intervention
- Allows real-time override during policy execution
- Intervention data saved for training
- Located in `serl_robot_infra/franka_env/spacemouse/`

## Simulation Environment (franka_sim)

### MuJoCo-Based Simulation

**Location:** `franka_sim/franka_sim/envs/panda_pick_gym_env.py`

Provides safe testing environment before real robot deployment:
- Pick and place tasks
- Operational space control
- Similar API to real robot environment

### Testing Scripts

- `test/test_gym_env_human.py` - Interactive testing with keyboard
- `test/test_gym_env_render.py` - Rendering and visualization

## Debugging Tips

### Check Robot Server Connection

```bash
# Test server is running
curl -X POST http://<SERVER_URL>:5000/getpos_euler
```

### Camera Issues

- Verify camera serial numbers in `REALSENSE_CAMERAS`
- Check image crops in `IMAGE_CROP`
- Use RealSense Viewer to test cameras independently
- Adjust exposure if images are over/under-exposed

### Training Not Converging

- Verify reward classifier accuracy (should be >90%)
- Collect more classifier data for failure modes
- Check demonstration quality (30 diverse demos recommended)
- Ensure workspace limits allow safe exploration
- Adjust `q_weight` and `bc_weight` hyperparameters

### Agentlace Connection Issues

- Ensure actor and learner use same IP/port
- Check firewall settings
- Verify `XLA_PYTHON_CLIENT_MEM_FRACTION` settings in launch scripts

## Key Files Reference

| File | Purpose |
|------|---------|
| `examples/train_conrft_octo.py` | Main training entry point (actor/learner) |
| `examples/record_demos_octo.py` | Record human demonstrations |
| `examples/train_reward_classifier.py` | Train reward classifier |
| `serl_launcher/agents/continuous/conrft_single_octo_cp.py` | Core ConRFT agent implementation |
| `serl_launcher/utils/launcher.py` | Agent factory functions |
| `examples/experiments/config.py` | Base configuration class |
| `examples/experiments/mappings.py` | Experiment name to config mapping |
| `docs/franka_walkthrough.md` | Detailed step-by-step robot training guide |

## Citation

If you use this codebase, please cite:

```bibtex
@inproceedings{chen2025conrft,
    title={ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy},
    author={Yuhui Chen and Shuai Tian and Shugao Liu and Yingting Zhou and Haoran Li and Dongbin Zhao},
    booktitle={Proceedings of Robotics: Science and Systems, {RSS} 2025},
    year={2025}
}
```
