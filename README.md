# cusmic

Implementing L.A.Cosmic using CuPy.


## Install

With CUDA and CuPy already working, install from this checkout:
```sh
python -m pip install --no-deps .
```
`--no-deps` keeps your existing CuPy installation, including a
CUDA-specific wheel such as `cupy-cuda12x`.
The core needs CuPy, which also installs NumPy.

The optional `cli` extra adds Click and Astropy for FITS I/O.
With an existing CuPy installation, add these separately:
```sh
python -m pip install click astropy
```


## Docker

The images use Python 3.12 on Debian Trixie, CuPy 14.2.0, and CUDA
12.9.1 components.

From this checkout, build for x86-64 Linux:
```sh
docker build --platform linux/arm64 --build-arg VERSION=<VERSION> -t rndsrc/cusmic:<VERSION> .
```
Building and displaying help do not require a GPU.

The Linux host needs an NVIDIA GPU, a driver compatible with CUDA
12.9, Docker, and the configured
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The host manages the driver; it does not need Python, CuPy, or the
CUDA toolkit installed.
`--gpus all` exposes its GPUs and driver to the container.

Check GPU access and runtime kernel compilation:
```sh
docker run --rm --gpus all rndsrc/cusmic:<VERSION> \
    -c "import cupy as cp; print(cp.arange(5).sum().item())"
```
This should print `10`; it requires an actual NVIDIA GPU.

Mount your working directory at `/data`, then run a Python script or
the FITS CLI:
```sh
docker run --rm --gpus all -v "$PWD:/data" rndsrc/cusmic:<VERSION> \
    input.fits cleaned.fits --error error.fits
```
Add `-v <CACHE>:/tmp/cupy` to reuse compiled kernels between runs.


## Clean a FITS image

```sh
python -m cusmic input.fits cleaned.fits --error error.fits
```
The error map contains positive 1-sigma errors in the same units and
shape as the input.

Alternatively, supply gain in electrons/ADU and read noise in
electrons:
```sh
cusmic input.fits cleaned.fits --gain 2 --readnoise 5
```
`--error` takes precedence over the noise model.
`--contrast`, `--cr-threshold`, `--neighbor-threshold`, and
`--maxiter` control detection;
see `cusmic --help`.

The CLI accepts finite, nonempty 2D FITS images, including `.fits.gz`,
and cleans in float64.
It reads the primary image, or the first extension if the primary is
empty.
The output keeps the image header and contains the cleaned pixels in
the primary HDU plus a `CRMASK` extension (1 = detected cosmic ray, 0
= unflagged).
Existing output files are never overwritten; choose a new filename for
each run.

[`demo/demo.ipynb`](demo/demo.ipynb) installs cusmic from GitHub and
lacosmic from PyPI, recreates the reference image, and compares their
cleaned images and cosmic-ray masks.
