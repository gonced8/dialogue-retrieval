#!/bin/bash

args=(
    # Trainer
	#--data_name multiwoz_dialogue
	--model_name retriever_answerer
	#--enable_checkpointing
	#--save_examples
	#--ckpt_path checkpoints/retriever_multiwozcombination/version_2/checkpoints/best.ckpt
    #--fast_dev_run

    # Model
	--retriever_encoder sentence-transformers/all-mpnet-base-v2
	--index_directory data/multiwoz/index/train_st
	--index_dataset ../data/multiwoz/processed2/train.json
	--n_candidates 3

    # Data
    #--test_batch_size 64
    #--num_workers 8
)

echo "streamlit run demo.py -- ${args[@]} $@"

streamlit run demo.py -- "${args[@]}" "$@"
