# Tests and reference data

From the repository root on a GPU host, install the matching CuPy wheel and
test tools, then run both Python and C/CUDA checks:

```sh
python -m pip install '.[cuda13,test,cli]' ruff
make                # Show targets
make build
make check
```

Use `cuda12` instead of `cuda13` with a CUDA 12 runtime. `make unit-test` and
`make e2e-test` run the checks separately. A missing native compiler is an
error. GPU-dependent tests fail, rather than skip,
without a GPU; device switching also fails with only one GPU. Remaining checks
still run and are reported.

`data/` contains one input frame, its error map, and the saved L.A.Cosmic
reference with a `CRMASK` extension. Tests, benchmarks, and the demo use the
same files. One host test checks the reference against pinned
`lacosmic==1.4.0`; GPU tests allow 32 float64 epsilons of cleaned-pixel
rounding and require the saved mask exactly. The generated scaled FITS case
checks both commands: zero iterations remain bitwise equal, while cleaned
pixels may differ within the same tolerance.
Normal tests never regenerate the committed reference.

To create *candidate* reference files for review:

```sh
python -m pip install '.[test]'
make mkref REFDIR=dist/reference
```

`mkref` records generator and package versions in FITS headers and refuses
to overwrite existing files. Reference generation may round differently
across CPU/library builds, so use the committed files for routine checks.
