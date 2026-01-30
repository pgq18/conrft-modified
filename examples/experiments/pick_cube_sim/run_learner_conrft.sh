export XLA_PYTHON_CLIENT_PREALLOCATE=true && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.9 && \
python ../../train_conrft_octo.py "$@" \
    --exp_name=pick_cube_sim \
    --checkpoint_path=/home/pgq/Workspace/VLA/conrft-modified/examples/experiments/pick_cube_sim/checkpoint_1 \
    --q_weight=1.0 \
    --bc_weight=0.5 \
    --demo_path=./demo_data/pick_cube_sim_30_demos_2026-01-30_15-11-16.pkl \
    --pretrain_steps=20000 \
    --debug=False \
    --learner \