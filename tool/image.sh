#!/bin/sh
# Build one image or a complete CUDA profile through Docker Bake.
set -eu

version=$1
cuda=$2
platform=$3
role=$4
revision=$(git rev-parse HEAD 2>/dev/null || echo dev)

case "$role" in
    all) target="cuda${cuda}" ;;
    full) target="cuda${cuda}-cusmic" ;;
    *) target="cuda${cuda}-${role}" ;;
esac

docker buildx bake --load --var VERSION="$version" --var REVISION="$revision" \
    --var PLATFORM="$platform" "$target"
