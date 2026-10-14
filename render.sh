#!/bin/sh
set -eu
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 MODEL_DIRECTORY [render.py options...]" >&2
    exit 2
fi
model_dir=$1
shift
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "${HLOD_PYTHON:-python}" "$repo_dir/render.py" -m "$model_dir" --skip_train "$@"
