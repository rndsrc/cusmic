#!/bin/sh
# Run GPU-required exact checks, then write the matched benchmark report.
set -eu

cd /src
mkdir -p /data/results

if sh test/check.sh all python /usr/local/cuda/bin/nvcc build/cuda --require-gpu \
    > /data/results/check.log 2>&1; then
    cat /data/results/check.log
else
    cat /data/results/check.log
    exit 1
fi

if python -m bench.run --frames 1 4 16 --warmups 4 --repeats 16 \
    --output /data/results > /data/results/bench.log 2>&1; then
    cat /data/results/bench.log
else
    cat /data/results/bench.log
    exit 1
fi
