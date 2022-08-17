#!/bin/bash
args=(
	--index_directory data/multiwoz/index/ours1
	--ckpt_path checkpoints/retriever_multiwozcombination/version_1/checkpoints/best.ckpt
)

COMMAND="python -m data.multiwoz.index"

echo "$COMMAND ${args[@]} $@"

$COMMAND "${args[@]}" "$@"
