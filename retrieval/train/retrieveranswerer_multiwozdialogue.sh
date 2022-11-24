#!/bin/bash

args=(
    # Model
	--model_name retriever_answerer
	--retriever_encoder sentence-transformers/all-mpnet-base-v2

    # Data
	--data_name multiwoz_dialogue
    --train_data ../data/multiwoz/processed2/train.json
    --results_st results/train_retrieval_st.json
    --train_batch_size 4
    --val_batch_size 64
    --test_batch_size 64
    --num_workers 8

    # Trainer
    --mode train
    --lr 1e-6
    --accumulate_grad_batches 16
    --max_epochs 20
	--check_val_every_n_epoch 1
    --val_check_interval 0.2
	--log_every_n_steps 1
	#--enable_checkpointing
	--default_root_dir checkpoints
    --monitor val_loss
	--monitor_mode min
    --fast_dev_run
)

echo "python main.py ${args[@]} $@"

python main.py "${args[@]}" "$@"