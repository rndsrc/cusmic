# cusmic

GPU L.A.Cosmic cosmic-ray removal with CuPy and CUDA C/C++ APIs and
FITS commands.
Both implementations support float64 images and independent frames in
a stack.
L.A.Cosmic 1.4.0 supplies the CPU reference and benchmarks;
cusmic has no CPU cleaner.

The v0.2.x line preserves the reference's float64 operation order.
The v0.3.x line allows rounding differences while keeping the same
algorithm.
Tests require identical masks for both;
see the [reference policy](test/README.md#reference-policy).

## Install

Install from this checkout with the CuPy wheel matching your CUDA
runtime:

```sh
python -m pip install '.[cuda13,cli]'
# For CUDA 12, use '.[cuda12,cli]'.
```

If CuPy is already installed, use `.[cli]`.
For the Python API alone, omit `cli`;
`pip install .` does not install or replace CuPy.
Cleaning requires a compatible NVIDIA GPU and driver.

## Clean a FITS image

```sh
cupysmic input.fits cleaned.fits --error error.fits
# Or estimate noise from gain (electrons/ADU) and read noise (electrons):
cupysmic input.fits cleaned.fits --gain 2 --readnoise 5
```

`python -m cusmic` runs the same command.
After building the CUDA executable, replace `cupysmic` with
`bin/cudasmic`.

Both commands read one 2D image, apply FITS scaling and integer
`BLANK` values, and write a float64 primary image plus a byte `CRMASK`
extension.
They preserve image metadata, update checksums, and refuse to
overwrite existing files.
Read noise defaults to zero; `--error` overrides the noise model.
Use `--help` for detection thresholds and iteration options.

## Python and C APIs

```python
from cusmic import remove_cosmics
from cusmic.io import read_fits

image, _ = read_fits("input.fits", dtype="float64")
error, _ = read_fits("error.fits", dtype="float64")
cleaned, mask = remove_cosmics(
    image, contrast=3, cr_threshold=5, neighbor_threshold=3, error=error,
)
```

The function follows L.A.Cosmic's parameter names and returns
independent CuPy arrays:
cleaned float64 pixels and a boolean detection mask.
NumPy inputs are uploaded; CuPy inputs stay on their device.
Input pixels are never modified.

* `data` must be a nonempty float64 frame `(height, width)` or stack
  `(frames, height, width)`.
  Noncontiguous CuPy slices are supported.
* Supply a positive, finite `error` map, or both `effective_gain`
  (positive) and `readnoise` (nonnegative).
  An error map takes precedence.
* Maps may match the entire input or one frame shared across a stack.
  Gain, read noise, and `background` also accept scalars.
  Background is added before cleaning and subtracted afterward;
  data, error, and background use image units.
* `mask` excludes pixels from detection seeds and replacement donors.
  Growth from neighboring detections can still flag masked pixels.
  Nonfinite input pixels are preserved and never flagged.
* `maxiter=0` disables detection and needs no noise model.
  Background arithmetic still applies and can round the result.

For repeated calls, reuse `Cleaner` settings with `Image` inputs.
Arrays are borrowed:
keep them unchanged until work on the input device's current stream
finishes.
Use events to establish readiness when passing data between streams.
See the API docstrings for parameters and [src/cusmic.h](src/cusmic.h)
for the host, device, and batch C interfaces.

## Build and check

The CUDA build requires `nvcc`, a C compiler, Make, and CFITSIO
headers and libraries.
Set `CUDA_ARCHS` for the target GPU, such as `121` for Spark with CUDA
13.
Local builds default to `CUDA_ARCH=75`;
container defaults are below.

```sh
python -m pip install -e '.[cuda13,cli,test,bench]' ruff
make build CUDA_ARCHS=121
make check REFERENCE=exact
make bench REFERENCE=exact
```

Use `REFERENCE=exact` for v0.2.x; `exact` is the tooling default.
The editable install keeps Python checks aligned with this checkout.
Run `make` for targets and options, including `unit`, `e2e`, `ref`,
`container`, and `clean`.
`make clean` removes build outputs, benchmark results, and caches.

The [test guide](test/README.md) covers host-only checks, GPU
requirements, and reference generation.
The [benchmark guide](bench/README.md) covers backend selection,
timing reports, shared-tool A/B comparisons, and the recorded GB10
measurements.
CPU-only benchmarks need `.[bench]` and no CUDA installation.

## Recorded performance

Historical GB10 results for a 512 × 512 float64 frame, with four
warmups and sixteen samples; median milliseconds per frame:

| Frames | CPU L.A.Cosmic | Exact CuPy | Exact CUDA |
| ---: | ---: | ---: | ---: |
|  1 | 680.481 | 10.190 | 6.629 |
|  4 | 683.897 |  9.614 | 6.252 |
| 16 | 680.504 |  9.800 | 6.658 |

GPU calls include allocation and transfers, with disk I/O excluded.
The measurements used v0.2.5; v0.2.6 preserves those exact algorithms.
See the [benchmark report](bench/README.md#recorded-gb10-measurements)
for the environment, timing spread, and validation limits.

## Docker images

The host supplies the NVIDIA driver and Container Toolkit.
Images provide the user-space runtime;
pass `--gpus all` to run GPU code.
Build all five roles or select the combined test/benchmark image:

```sh
make container VERSION=0.2.6-dev
make container VERSION=0.2.6-dev TARGET=full
```

| Role | Tag | Contents |
| --- | --- | --- |
| CuPy CLI | `rndsrc/cupysmic:<version>` | CuPy API and FITS command |
| CuPy API | `rndsrc/cupysmic:<version>-slim` | CuPy API |
| CUDA CLI | `rndsrc/cudasmic:<version>` | C library and FITS command |
| CUDA API | `rndsrc/cudasmic:<version>-slim` | C library and header |
| Full | `rndsrc/cusmic:<version>` | Both implementations, tests, benchmarks, demo |

Builds default to ARM64 and CUDA 13, targeting architectures 87 and
121.
`CUDA=12` selects CUDA 12, architecture 87, and a `-cuda12` tag
suffix.
For x86-64, set `PLATFORM=linux/amd64` and the appropriate
`CUDA_ARCHS`, such as `86`.
`VERSION` defaults to an exact Git version tag, or `0.0.0.dev0`
when none matches; overrides must be valid Python versions.

```sh
mkdir -p results
docker run --rm --gpus all -e CUSMIC_REFERENCE=exact \
    -v "$PWD/results:/data/results" rndsrc/cusmic:0.2.6-dev
```

The full image runs checks followed by benchmarks, continuing after
failures.
It saves logs, measurements, comparison tables, failures, and
`status.txt` to the mounted directory, and exits nonzero if either
phase fails.
Use `exact` for v0.2.x.  Append benchmark options such as `--frames 1
4 --warmups 4 --repeats 16` after the image name.
The [benchmark guide](bench/README.md#docker) includes benchmark-only
runs.
ARM64 images can show CLI help on a Mac; cleaning needs an NVIDIA
host.

## Demo

The [notebook](demo/demo.ipynb) displays the saved scene and compares
CPU and CuPy results and timings.
Install `.[demo,bench]` alongside CuPy for local Jupyter, use its
Colab setup, or open the copy included in the full image.
