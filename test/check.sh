#!/bin/sh
# Run the matching Python and C/CUDA checks for one selection.
set -eu

mode=$1
python=$2
nvcc=$3
build=$4
shift 4

case "$mode" in
	all)
		"$python" -m pytest -q -rs "$@"
		checks="test_io test_api test_batch test_reference"
		;;
	unit)
		"$python" -m pytest -q -rs -m 'not e2e' "$@"
		checks="test_io test_api test_batch"
		;;
	e2e)
		"$python" -m pytest -q -rs -m e2e "$@"
		checks=test_reference
		;;
	*) echo "Unknown check selection: $mode" >&2; exit 2 ;;
esac

compiler=${nvcc%% *}
if command -v "$compiler" >/dev/null 2>&1; then
	build_checks=1
elif [ "${CHECK_PREBUILT:-0}" = 1 ]; then
	build_checks=0
else
	if [ "${GPU_REQUIRED:-0}" = 1 ]; then
		echo "CUDA compiler unavailable: $compiler" >&2
		exit 1
	fi
	echo "CUDA C/C++ checks skipped: compiler unavailable ($compiler)"
	exit 0
fi

for check in $checks; do
	if [ "$build_checks" = 1 ]; then
		make "$build/$check"
	elif [ ! -x "$build/$check" ]; then
		echo "Missing prebuilt CUDA check: $build/$check" >&2
		exit 1
	fi
	"$build/$check"
done
