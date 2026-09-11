#!/bin/sh
# Benchmark all backends even if the CUDA compiler or a measurement fails.
set -u

python=$1
nvcc=$2
build=$3
shift 3
export CUSMIC_CUDA_BENCH="$build/bench"

failed=0
compiler=${nvcc%% *}
if [ "${CHECK_PREBUILT:-0}" != 1 ]; then
	rm -f "$build/bench"
	if command -v "$compiler" >/dev/null 2>&1; then
		make "$build/bench" || failed=1
	else
		echo "CUDA compiler unavailable: $compiler" >&2
		failed=1
	fi
fi

"$python" -m bench.run "$@" || failed=1
exit "$failed"
