#!/bin/bash

args=(
    # Model
	--model_name retriever_answerer
	--retriever_encoder sentence-transformers/all-mpnet-base-v2
    --n_candidates 10
    #--negative_examples

    # Data
	--data_name multiwoz_dialogue
    --train_data ../data/multiwoz/processed2/train.json
    --val_data ../data/multiwoz/processed2/val.json
    --train_dataset results/train_dataset_top10.json
    --heuristic answer_rougeL
    #--negative_st_best
    --train_batch_size 8
    --val_batch_size 64
    --index_batch_size 128
    --num_workers 8

    # Trainer
    --mode train
    --lr 1e-5
    --accumulate_grad_batches 8
    --max_epochs 20
	--check_val_every_n_epoch 1
    --val_check_interval 0.5
	--log_every_n_steps 10
	--enable_checkpointing
	--default_root_dir checkpoints
    --monitor rougeL
	--monitor_mode max
    --patience 5
    #--fast_dev_run
)

date '+%Y-%m-%d_%H%M%S'
echo "python main.py ${args[@]} $@"

time python main.py "${args[@]}" "$@"