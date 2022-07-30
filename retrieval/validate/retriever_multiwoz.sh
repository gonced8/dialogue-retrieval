#!/bin/bash

args=(
    #Model
	--original_model_name sentence-transformers/all-MiniLM-L6-v2

    # Data
	--total_batch_size 100000
	--total_val_batch_size 1000
	--total_test_batch_size 1000
    --batch_size 4
    --val_batch_size 4
    --test_batch_size 1
	--mrr_total 10
    --num_workers 8

    # Trainer
    --mode validate
    --lr 1e-3
    --accumulate_grad_batches 16
    --max_epochs 10
	--check_val_every_n_epoch 1
    --val_check_interval 0.05
	--log_every_n_steps 1
	--enable_checkpointing
	--default_root_dir checkpoints
    --monitor val_loss
	--save_val
	--save_test
    --fast_dev_run
)

echo "python main.py ${args[@]} $@"

python main.py "${args[@]}" "$@"
