export XLA_PYTHON_CLIENT_PREALLOCATE=true && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95 && \
python ../../train_conrft_octo.py "$@" \
    --exp_name=pick_cube_sim \
    --q_weight=0.1 \
    --bc_weight=1.0 \
    --demo_path=./demo_data/pick_cube_sim_30_demos_2026-01-27_20-19-11.pkl \
    --pretrain_steps=20000 \
    --debug=False \
    --learner \
    --checkpoint_path=/home/pgq/Workspace/VLA/conrft-modified/examples/experiments/pick_cube_sim/checkpoint \
