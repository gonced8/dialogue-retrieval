#!/bin/bash

args=(
    # Model
	--model_name retriever_answerer
	--retriever_encoder sentence-transformers/all-mpnet-base-v2
	--n_candidates 10

    # Data
	--data_name multiwoz_dialogue
    --train_data ../data/multiwoz/processed2/train.json
    --test_data ../data/multiwoz/processed2/test.json
    --heuristic lcs++
    --test_batch_size 64
    --index_batch_size 256
    --num_workers 8

    # Trainer
    --mode test
	--enable_checkpointing
	--save_examples
	--default_root_dir checkpoints
	#--ckpt_path
    #--fast_dev_run
)

echo "python main.py ${args[@]} $@"

time python main.py "${args[@]}" "$@"
