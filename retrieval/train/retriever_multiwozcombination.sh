#!/bin/bash

args=(
    # Model
	--model_name retriever
	--original_model_name sentence-transformers/all-mpnet-base-v2

    # Data
	--data_name multiwoz_combination
	--total_batch_size 100000
	--total_val_batch_size 1000
	--total_test_batch_size 1000
    --batch_size 4
    --val_batch_size 4
    --test_batch_size 1
    --num_workers 16
	--transformation data/quantile_transformer.joblib

    # Trainer
    --mode train
    --lr 1e-6
    --accumulate_grad_batches 16
    --max_epochs 20
	--check_val_every_n_epoch 1
    --val_check_interval 0.2
	--log_every_n_steps 1
	--enable_checkpointing
	--default_root_dir checkpoints
    --monitor val_loss
	--monitor_mode min
	--save_examples
    #--fast_dev_run
)

echo "python main.py ${args[@]} $@"

python main.py "${args[@]}" "$@"
