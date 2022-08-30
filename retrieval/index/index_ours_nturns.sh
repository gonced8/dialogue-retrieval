#!/bin/bash
args=(
	--index_directory data/multiwoz/index/ours_nturns
	--ckpt_path checkpoints/retriever_multiwozcombination/version_13/checkpoints/last.ckpt
)

COMMAND="python -m data.multiwoz.index"

echo "$COMMAND ${args[@]} $@"

$COMMAND "${args[@]}" "$@"
