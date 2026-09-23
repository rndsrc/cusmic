#!/bin/bash
# Write check and benchmark reports even when one fails.
set -euo pipefail

cd /src
mkdir -p /data/results

echo "Running checks..."
check_status=0
if sh test/check.sh all python /usr/local/cuda/bin/nvcc build/cuda \
    2>&1 | tee /data/results/check.log; then
    :
else
    check_status=$?
fi

echo "Running benchmarks..."
bench_status=0
if python -m bench.run --frames 1 4 16 --warmups 4 --repeats 16 \
    --output /data/results "$@" 2>&1 | tee /data/results/bench.log; then
    :
else
    bench_status=$?
fi

printf 'checks=%s\nbenchmarks=%s\n' "$check_status" "$bench_status" \
    | tee /data/results/status.txt
if [ "$check_status" -ne 0 ] || [ "$bench_status" -ne 0 ]; then
    exit 1
fi
