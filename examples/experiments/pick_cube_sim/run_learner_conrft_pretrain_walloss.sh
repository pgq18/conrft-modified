#!/bin/bash
# Pretraining script with Walloss backbone
# Paths are configured in config.py (walloss_path and walloss_config_path)

export XLA_PYTHON_CLIENT_PREALLOCATE=true && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.90 && \
python ../../train_conrft_octo.py "$@" \
    --exp_name=pick_cube_sim \
    --q_weight=0.1 \
    --bc_weight=1.0 \
    --demo_path=./demo_data/pick_cube_sim_30_demos_2026-02-02_15-23-40.pkl \
    --pretrain_steps=20000 \
    --debug=False \
    --learner \
    --checkpoint_path=/home/pgq/Workspace/VLA/conrft-modified/examples/experiments/pick_cube_sim/checkpoint_walloss \
    --backbone "walloss" \
