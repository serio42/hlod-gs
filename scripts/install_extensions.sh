#!/bin/sh
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python_bin=${HLOD_PYTHON:-python}

fetch_source() {
    name=$1
    url=$2
    revision=$3
    source_dir="$repo_dir/submodules/$name"
    if [ ! -e "$source_dir/.git" ]; then
        git clone "$url" "$source_dir"
        git -C "$source_dir" checkout --detach "$revision"
    fi
    if [ "$(git -C "$source_dir" rev-parse HEAD)" != "$revision" ]; then
        echo "$name must be at $revision; refusing to change an existing checkout." >&2
        exit 1
    fi
}
fetch_source diff-gaussian-rasterization https://github.com/graphdeco-inria/diff-gaussian-rasterization.git 59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d
raster_dir="$repo_dir/submodules/diff-gaussian-rasterization"
patch_file="$repo_dir/patches/diff-gaussian-rasterization-low-pass.patch"
if git -C "$raster_dir" apply --check "$patch_file" 2>/dev/null; then
    git -C "$raster_dir" apply "$patch_file"
elif ! git -C "$raster_dir" apply --reverse --check "$patch_file" 2>/dev/null; then
    echo 'Rasterizer source differs from the pinned HLoD patch.' >&2
    exit 1
fi
git -C "$raster_dir" submodule update --init --recursive
fetch_source simple-knn https://gitlab.inria.fr/bkerbl/simple-knn.git 86710c2d4b46680c02301765dd79e465819c8f19
"$python_bin" -m pip install --no-build-isolation "$raster_dir" "$repo_dir/submodules/simple-knn"
