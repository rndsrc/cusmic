#!/bin/sh
# Build the selected image role, or all four runnable roles.
set -eu

version=$1
cuda=$2
platform=$3
role=$4

case "$role:$cuda" in
	all:13|cupysmic:*|cupysmic-slim:*|cudasmic:13|cudasmic-slim:13) ;;
	*) echo "CUDA $cuda is not available for $role in this build" >&2; exit 2 ;;
esac

build_role()
{
	case "$1" in
		cupysmic)      stage=cli;      name=cupysmic; suffix= ;;
		cupysmic-slim) stage=api;      name=cupysmic; suffix=-slim ;;
		cudasmic)      stage=cuda-cli; name=cudasmic; suffix= ;;
		cudasmic-slim) stage=cuda-api; name=cudasmic; suffix=-slim ;;
		*) echo "Unknown image role: $1" >&2; exit 2 ;;
	esac

	tag="${version}${suffix}"
	if [ "$cuda" != 13 ]; then
		tag="${tag}-cuda${cuda}"
	fi

	docker buildx build --load --platform "$platform" --target "$stage" \
		--build-arg VERSION="$version" --build-arg CUDA="$cuda" \
		-t "rndsrc/$name:$tag" .
}

if [ "$role" = all ]; then
	for role in cupysmic cupysmic-slim cudasmic cudasmic-slim; do
		build_role "$role"
	done
else
	build_role "$role"
fi
