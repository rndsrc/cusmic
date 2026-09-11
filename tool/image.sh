#!/bin/sh
# Build one image or a complete CUDA profile through Docker Bake.
set -eu

version=$1
cuda=$2
platform=$3
role=$4
archs=${5:-}
revision=$(git rev-parse HEAD 2>/dev/null || echo dev)

case "$role" in
    all) target="cuda${cuda}" ;;
    full) target="cusmic-cuda${cuda}" ;;
    *-cuda12|*-cuda13) target="$role" ;;
    *) target="${role}-cuda${cuda}" ;;
esac

docker buildx bake --load --var VERSION="$version" --var REVISION="$revision" \
    --var CUDA_ARCHS="$archs" \
    --var PLATFORM="$platform" "$target"
