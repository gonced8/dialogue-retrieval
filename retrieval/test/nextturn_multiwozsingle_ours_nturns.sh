#!/bin/bash

args=(
    # Trainer
	--data_name multiwoz_single
	--model_name next_turn
    --mode test
	--enable_checkpointing
	--save_examples
	--ckpt_path checkpoints/retriever_multiwozcombination/version_13/checkpoints/last.ckpt
    #--fast_dev_run

    # Model
	--original_model_name sentence-transformers/all-mpnet-base-v2
	--index_directory data/multiwoz/index/ours_nturns
	--k 10

    # Data
    --test_batch_size 8
    --num_workers 8
)

echo "python main.py ${args[@]} $@"

python main.py "${args[@]}" "$@"
