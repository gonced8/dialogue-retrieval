#!/bin/bash

args=(
    # Trainer
	--data_name multiwoz_dialogue
	--model_name retriever_answerer
    --mode test
	#--enable_checkpointing
	#--save_examples
	#--ckpt_path checkpoints/retriever_multiwozcombination/version_2/checkpoints/best.ckpt
    --fast_dev_run

    # Model
	--retriever_encoder sentence-transformers/all-mpnet-base-v2
	--index_directory data/multiwoz/index/train_st
	--index_dataset ../data/multiwoz/processed2/train.json
	--n_candidates 5

    # Data
    --test_batch_size 64
    --num_workers 8
)

echo "python main.py ${args[@]} $@"

python main.py "${args[@]}" "$@"
