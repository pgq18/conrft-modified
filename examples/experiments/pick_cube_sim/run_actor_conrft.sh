export XLA_PYTHON_CLIENT_PREALLOCATE=true && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.8 && \
python ../../train_conrft_octo.py "$@" \
    --exp_name=pick_cube_sim \
    --checkpoint_path=/home/pgq/Workspace/VLA/conrft-modified/examples/experiments/pick_cube_sim/checkpoint_1 \
    --actor \
    --ip "10.20.85.133" \
    # --eval_checkpoint_step=26000 \