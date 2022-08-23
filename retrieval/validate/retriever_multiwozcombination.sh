#!/bin/bash

args=(
    # Model
	--model_name retriever
	#--original_model_name sentence-transformers/all-MiniLM-L6-v2
	--original_model_name sentence-transformers/all-mpnet-base-v2

    # Data
	--data_name multiwoz_combination
	--total_val_batch_size 1000
    --val_batch_size 1
	--candidates 100
    --num_workers 8
	#--transformation data/quantile_transformer.joblib

    # Trainer
    --mode validate
	--enable_checkpointing
	--default_root_dir checkpoints
	--save_examples
	--ckpt_path checkpoints/retriever_multiwoz/version_2/checkpoints/best.ckpt
    #--fast_dev_run
)

echo "python main.py ${args[@]} $@"

python main.py "${args[@]}" "$@"
