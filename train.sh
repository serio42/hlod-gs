#!/bin/sh
set -eu
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 DATASET_DIRECTORY OUTPUT_DIRECTORY [train.py options...]" >&2
    exit 2
fi
source_dir=$1
model_dir=$2
shift 2
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "${HLOD_PYTHON:-python}" "$repo_dir/train.py" -s "$source_dir" -m "$model_dir" --eval "$@"
