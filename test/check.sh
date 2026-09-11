#!/bin/sh
# Run the matching Python and C/CUDA checks for one selection.
set -eu

mode=$1
python=$2
nvcc=$3
build=$4
shift 4

failed=0
compiler=${nvcc%% *}
if [ "$mode" != unit ] && command -v "$compiler" >/dev/null 2>&1; then
	if ! make bin/cudasmic; then
		failed=1
	fi
elif [ "$mode" != unit ] && [ ! -x bin/cudasmic ]; then
	echo "Missing CUDA FITS command: bin/cudasmic" >&2
	failed=1
fi

case "$mode" in
	all)
		if ! "$python" -m pytest -q -rs "$@"; then failed=1; fi
		checks="test_io test_api test_batch test_reference"
		;;
	unit)
		if ! "$python" -m pytest -q -rs -m 'not e2e' "$@"; then failed=1; fi
		checks="test_io test_api test_batch"
		;;
	e2e)
		if ! "$python" -m pytest -q -rs -m e2e "$@"; then failed=1; fi
		checks=test_reference
		;;
	*) echo "Unknown check selection: $mode" >&2; exit 2 ;;
esac

if command -v "$compiler" >/dev/null 2>&1; then
	build_checks=1
elif [ "${CHECK_PREBUILT:-0}" = 1 ]; then
	build_checks=0
else
	echo "CUDA C/C++ checks unavailable: compiler missing ($compiler)" >&2
	exit 1
fi

for check in $checks; do
	if [ "$build_checks" = 1 ]; then
		if ! make "$build/$check"; then
			failed=1
			continue
		fi
	elif [ ! -x "$build/$check" ]; then
		echo "Missing prebuilt CUDA check: $build/$check" >&2
		failed=1
		continue
	fi
	if ! "$build/$check"; then failed=1; fi
done

exit "$failed"
