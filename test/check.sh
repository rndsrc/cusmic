#!/bin/sh
# Run the matching Python and C/CUDA checks for one selection.
set -u

mode=$1
python=$2
nvcc=$3
build=$4
shift 4

case "${CUSMIC_REFERENCE:-exact}" in
	exact|close) ;;
	*) echo "CUSMIC_REFERENCE must be exact or close" >&2; exit 2 ;;
esac

case "$mode" in
	all)
		checks="test_io test_api test_batch test_reference"
		;;
	unit)
		set -- -m 'not e2e' "$@"
		checks="test_io test_api test_batch"
		;;
	e2e)
		set -- -m e2e "$@"
		checks=test_reference
		;;
	*) echo "Unknown check selection: $mode" >&2; exit 2 ;;
esac

passed=0
failed=0
compiler=${nvcc%% *}

run()
{
	printf '\n==> %s\n' "$*"
	if "$@"; then
		passed=$((passed + 1))
		echo "PASS: $*"
		return 0
	fi
	failed=$((failed + 1))
	echo "FAIL: $*" >&2
	return 1
}

prepare()
{
	if [ "${CHECK_PREBUILT:-0}" = 1 ]; then
		if [ -x "$1" ]; then return 0; fi
		echo "FAIL: missing prebuilt executable $1" >&2
	elif [ "$1" = "$build/test_io" ] || command -v "$compiler" >/dev/null 2>&1; then
		run make "$1"
		return $?
	else
		echo "FAIL: cannot build $1; CUDA compiler unavailable ($compiler)" >&2
	fi
	failed=$((failed + 1))
	return 1
}

printf 'Checking %s; reference policy: %s\n' "$mode" "${CUSMIC_REFERENCE:-exact}"
if [ "$mode" != unit ]; then prepare "${CUSMIC_CUDA_CLI:-bin/cudasmic}" || :; fi
run "$python" -m pytest -v -rs "$@" || :

for check in $checks; do
	if prepare "$build/$check"; then run "$build/$check" || :; fi
done

printf '\nCheck workflow: %s steps passed, %s failed.\n' "$passed" "$failed"
[ "$failed" -eq 0 ]
