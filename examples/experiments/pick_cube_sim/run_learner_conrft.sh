export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
python ../../train_conrft_octo.py "$@" \
    --exp_name=pick_cube_sim \
    --checkpoint_path=/root/online_rl/conrft/examples/experiments/pick_cube_sim/checkpoint \
    --q_weight=1.0 \
    --bc_weight=0.1 \
    --demo_path=./demo_data/pick_cube_sim_30_demos_2026-01-27_20-19-11.pkl \
    --pretrain_steps=20000 \
    --debug=False \
    --learner \