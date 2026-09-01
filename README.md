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
Calibration matches the whole stack or one frame;
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

The host needs an NVIDIA driver, Docker and the NVIDIA Container
Toolkit.
The host manages the driver;
the image carries the user-space CUDA runtime.

| Target | Contents |
| --- | --- |
| `slim` | CuPy core |
| `cli`  | Core and FITS command |
| `test` | Tests, reference data and benchmark |

Build for an x86-64 GPU host:
```sh
docker build --platform linux/amd64 --build-arg VERSION=${VERSION} \
    --target test -t cusmic:${VERSION}-test .
```
For NVIDIA Spark, select ARM64 and the CUDA 13 runtime:
```sh
docker build --platform linux/arm64/v8 --build-arg VERSION={VERSION} \
    --build-arg CUPY_PACKAGE=cupy-cuda13x --build-arg CUDA_VERSION=13.0.2 \
    --target test -t cusmic:${VERSION}-test .
```
Choose `slim` or `cli` for a smaller runtime.
Builders retain compilers, download caches and dependency tests.
CuPy still needs NVRTC and headers to compile its kernels.
No image contains the NVIDIA driver.

```sh
docker run --rm --gpus all cusmic:${VERSION}-test
docker run --rm --gpus all -v "$PWD:/data" cusmic:${VERSION}-slim <SCRIPT.py>
```
Use `--target slim -t cusmic:${VERSION}-slim` to build the second image.

## Checks and examples

```sh
python -m pip install astropy click pytest
python -m pytest -q --require-gpu
python -m demo.benchmark --warmups 5 --repeats 15
```
Tests read committed references without installing lacosmic or
regenerating data.
GPU checks fail explicitly when required hardware is unavailable.
The [notebook](demo/demo.ipynb) generates the example and reference
images.
