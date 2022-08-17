#!/bin/bash
args=(
	--index_directory data/multiwoz/index/baseline
	--ckpt_path None
)

COMMAND="python -m data.multiwoz.index"

echo "$COMMAND ${args[@]} $@"

$COMMAND "${args[@]}" "$@"
