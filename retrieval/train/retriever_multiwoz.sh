#!/bin/bash

args=(
    #Model

    # Data
	--total_batch_size 100000
	--total_val_batch_size 1000
	--total_test_batch_size 1000
    --batch_size 8
    --val_batch_size 1
    --test_batch_size 1
    --num_workers 8

    # Trainer
    --mode train
    --lr 1e-3
    --accumulate_grad_batches 8
    --max_epochs 10
	--check_val_every_n_epoch 1
    --val_check_interval 0.2
	--log_every_n_steps 1
	--enable_checkpointing
	--default_root_dir checkpoints
    --monitor val_loss
	--save_results
    #--fast_dev_run
)

echo "python main.py ${args[@]} $@"

python main.py "${args[@]}" "$@"
