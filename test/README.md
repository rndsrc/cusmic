# Tests and references

Run from the repository root with CuPy, Astropy, Click and pytest installed:
```sh
make unit-test
make e2e-test PYTEST_ARGS=--require-gpu GPU_REQUIRED=1
make check
```
These targets run Python and C/CUDA checks. Without a local CUDA compiler,
the C/CUDA checks are skipped unless `GPU_REQUIRED=1` is set. Unavailable GPUs
are skipped by pytest unless `--require-gpu` is given.
A successful host run does not qualify GPU results.

`data/` holds one input scene, its error map, and the L.A.Cosmic cleaned
reference with a `CRMASK` extension. Tests, benchmarks and the demo share
these files. Tests compare float64 bits and masks exactly.

Generate candidates explicitly, then review them before replacing fixtures:
```sh
python -m pip install 'lacosmic==1.4.0'
make mkref REFDIR=dist/reference
```
`mkref.py` records generator settings and package versions in FITS headers
and refuses to overwrite existing files. CPU/library differences can affect
rounding during generation; committed files remain the test reference.
Normal testing never regenerates them.
