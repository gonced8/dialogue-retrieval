#!/bin/bash

args=(
    # Trainer
	--data_name multiwoz_single
	--model_name next_turn
    --mode test
	--enable_checkpointing
	--default_root_dir checkpoints
	#--save_val
	--ckpt_path checkpoints/retriever_multiwoz/version_2/checkpoints/best.ckpt
    #--fast_dev_run

    # Model
	#--original_model_name sentence-transformers/all-MiniLM-L6-v2
	--original_model_name sentence-transformers/all-mpnet-base-v2
	--index_directory data/multiwoz/index_ours
	--k 10

    # Data
    --test_batch_size 8
    --num_workers 8
)

echo "python main.py ${args[@]} $@"

python main.py "${args[@]}" "$@"
