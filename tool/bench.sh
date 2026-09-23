#!/bin/sh
# Benchmark all backends even if the CUDA compiler or a measurement fails.
set -u

python=$1
nvcc=$2
build=$3
backends=$4
shift 4
export CUSMIC_CUDA_BENCH="$build/bench"

failed=0
compiler=${nvcc%% *}
case " $backends " in
*" cuda "*)
	if [ "${CHECK_PREBUILT:-0}" != 1 ]; then
		rm -f "$build/bench"
		if command -v "$compiler" >/dev/null 2>&1; then
			make "$build/bench" || failed=1
		else
			echo "CUDA compiler unavailable: $compiler" >&2
			failed=1
		fi
	fi
	;;
esac

"$python" -m bench.run --backends $backends "$@" || failed=1
exit "$failed"
