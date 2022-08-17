#!/bin/bash
args=(
	--index_directory data/multiwoz/index/ours0
	--ckpt_path checkpoints/retriever_multiwozcombination/version_0/checkpoints/best.ckpt
)

COMMAND="python -m data.multiwoz.index"

echo "$COMMAND ${args[@]} $@"

$COMMAND "${args[@]}" "$@"
