# Tests and references

Run from the repository root with CuPy, Astropy, Click and pytest installed:
```sh
make unit
make e2e PYTEST_ARGS=--require-gpu
make test
```
`unit` runs focused checks; `e2e` exercises the public API and FITS command.
Unavailable GPUs are skipped unless `--require-gpu` is given.
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

In the full container, use `-m pytest` or `test/mkref.py /data/reference`
in place of its default command. Reference generation needs no GPU.
