# cusmic

L.A.Cosmic cosmic-ray removal in CuPy, with exact float64 reference
checks.

## Python

With CuPy working on your GPU, install this checkout:
```sh
python -m pip install --no-deps .
```
This preserves your CUDA-specific CuPy installation.

```python
import cupy as cp
from cusmic import remove_cosmics

cleaned, mask = remove_cosmics(cp.asarray(data, dtype=cp.float64), error=error)
```

Provide a positive error map, or `effective_gain` and `readnoise`.
Errors override the noise model.
Images are nonempty float64 frames or stacks.
Calibration is converted to float64 and matches the whole stack or one frame;
gain, read noise and background also accept scalars.
Inputs are borrowed; outputs own their data.
Nonfinite input pixels are preserved.
A mask excludes detection seeds and replacement donors;
reference-compatible growth may enter masked pixels.

Reuse settings and select frames with ordinary slices:
```python
from cusmic import Cleaner, Image

cleaner = Cleaner()
cleaned, mask = cleaner(Image(movie[start:stop], error=error))
```
Calls select the input GPU and use its current CUDA stream.
Callers keep borrowed buffers valid and establish events when sharing
between streams.

## FITS command

```sh
python -m pip install click astropy
python -m cusmic input.fits cleaned.fits --error error.fits
```
The command reads one frame as float64 and saves cleaned pixels plus a
`CRMASK` extension.
It preserves the input header and refuses overwrite.
Use `--help` for gain, read noise and detection options.

## Containers

The host needs an NVIDIA driver, Docker and the NVIDIA Container Toolkit.
The images provide the user-space CUDA runtime; use `--gpus all` to expose
host-managed GPUs.

| Target | Image | Contents |
| --- | --- | --- |
| `api` | `rndsrc/cupysmic:<VERSION>-api` | Python API |
| `cli` | `rndsrc/cupysmic:<VERSION>-cli` | API and FITS command |
| `full` | `rndsrc/cusmic:<VERSION>` | CLI, tests, references, benchmarks and Jupyter demo |

Build locally (ARM64 for Spark, CUDA 13):
```sh
make image TARGET=api
make image TARGET=cli
make image TARGET=full
```
An exact version tag supplies `VERSION`; other checkouts use `0.0.0`.
Set `VERSION` to override it and `PLATFORM=linux/amd64` for an x86-64 host.
Builds record the commit ID for benchmark results.
`CUDA=12` selects the compatibility build and appends `-cuda12` to its tag.
An explicit `-cuda13` alias denotes the same stack as the default.
Defaults stay fixed within a release; CUDA compatibility builds require a
compatible host driver and separate GPU qualification.

```sh
docker run --rm --gpus all -v "$PWD:/data" rndsrc/cupysmic:<VERSION>-api script.py
docker run --rm --gpus all -v "$PWD:/data" rndsrc/cupysmic:<VERSION>-cli \
    input.fits cleaned.fits --error error.fits
docker run --rm --gpus all rndsrc/cusmic:<VERSION>
```
The full image defaults to GPU-required tests. For host checks or reference
generation, replace its arguments with `-m pytest -q` or
`test/mkref.py /data/reference` (mount an output directory).

The multi-stage [Dockerfile](Dockerfile) shares the core layer across targets.
The API image has no shell, pip, CLI packages, demo or test data.
CuPy retains NVRTC, its builtins and headers to compile kernels at runtime.
`rndsrc/cudasmic:<VERSION>-api` and `-cli` are reserved for the CUDA C/C++
implementation; this branch currently builds CuPy only.

## Checks and examples

With CuPy installed:
```sh
python -m pip install astropy click pytest ruff
make check
make test PYTEST_ARGS=--require-gpu
make bench BENCH_ARGS="--frames 4 --output bench/results/run.jsonl"
```

- [Tests and reference generation](test/README.md): `make unit`, `make e2e`,
  `make test`, `make mkref`.
- [Benchmarks](bench/README.md): warmed cleaning, transfers and complete calls;
  JSON records for comparing hardware and software.
- [Demo notebook](demo/demo.ipynb): API use, reference images and timings;
  runs locally or in Google Colab.

All three use the same saved scene in `test/data/`.
Normal tests compare float64 pixels bit for bit and masks exactly, without
regenerating data. CPU L.A.Cosmic provides the reference; cusmic requires a GPU.

[Checks](.github/workflows/check.yml) run on AMD64 and ARM64.
The [image workflow](.github/workflows/images.yml) builds both architectures
under each tag and checks them without a GPU. Publishing is a manual action
on a GPU-verified release tag; configure `DOCKERHUB_USERNAME` as a repository
variable and `DOCKERHUB_TOKEN` as a secret. Tag pushes build but do not publish.
