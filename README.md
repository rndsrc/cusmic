# cusmic

L.A.Cosmic cosmic-ray removal in CuPy and CUDA C/C++. The v0.2.x releases
preserve float64 operation order and match the saved L.A.Cosmic reference bit
for bit. v0.3.x uses a fused Laplacian with the same mathematical stencil but
different rounding; well-separated reference detections must still match.
For finite reference pixels, the check is
`abs(cleaned-reference) <= 32*eps*(1+abs(reference))`, with `eps` the float64
machine epsilon. Threshold ties may change after rounding. L.A.Cosmic supplies
the CPU benchmark and reference; cusmic has no CPU cleaner.

## Install and use

Install the CuPy wheel that matches your CUDA runtime:

```sh
python -m pip install '.[cuda13]'
# Use '.[cuda12]' on a CUDA 12 host.
```

CuPy already installed? `python -m pip install .` leaves it alone. Add `cli`
for the FITS command, and `test` or `bench` only when you need them:

```sh
python -m pip install '.[cuda13,cli,test,bench]' ruff
```

```python
import cupy as cp
from cusmic import remove_cosmics

cleaned, mask = remove_cosmics(cp.asarray(data, dtype=cp.float64),
                              1, 5, 5, error=error)
```

The three thresholds and their names follow L.A.Cosmic. Cleaning requires a
GPU and returns independent CuPy arrays. NumPy inputs are uploaded to the GPU.
Provide an error map, or `effective_gain` and `readnoise`; an error map takes
precedence. Images are nonempty float64 frames or stacks. Use ordinary CuPy
slices to select frames, and keep borrowed inputs valid through the call.

With `cli` installed, the CuPy command is `cupysmic` (or `python -m cusmic`).
The CUDA C/C++ command is `bin/cudasmic` after `make build`:

```sh
cupysmic input.fits cleaned.fits --error error.fits
bin/cudasmic input.fits cleaned-cuda.fits --error error.fits
```

Both commands read one FITS frame, write cleaned pixels and a `CRMASK`
extension, and refuse to overwrite files. They interpret FITS scaling and
integer `BLANK` consistently. Use `--help` for the small set of options.
The public C header is [src/cusmic.h](src/cusmic.h).

## Build and measure locally

The CUDA C/C++ build needs `nvcc`, a C compiler, Make, and CFITSIO headers.
Run `make` to see the available targets. On a GPU host with the Python extras
above:

```sh
make build
make check
make bench
make clean            # Remove generated build and benchmark files
```

`make check` runs Python and C/CUDA checks. GPU-dependent tests fail explicitly
when no GPU is available, or when a two-GPU test has only one GPU; the other
checks still run. `make bench` measures CPU L.A.Cosmic, CuPy, and CUDA for 1,
4, and 16 frames with four warmups and sixteen samples. It writes a report of
completed measurements and backend failures. On a CPU-only host, install
`.[bench]` and run `python -m bench.cpu --frames 1` to measure L.A.Cosmic
alone. Compare v0.2.5 and a v0.3 candidate on the same GPU with
`python -m bench.ab BASELINE_RESULTS CANDIDATE_RESULTS`. See
[test/README.md](test/README.md) and [bench/README.md](bench/README.md)
for the fixtures and timing boundaries.

## Docker images

The host supplies the NVIDIA driver and NVIDIA Container Toolkit. The images
supply the matching user-space CUDA runtime and CuPy wheel; pass `--gpus all`
when running GPU code. The default build is ARM64/CUDA 13:

```sh
VERSION=local make image                  # All five roles
VERSION=local make image TARGET=full      # Combined check/benchmark image
```

| Role | Tag | Contents |
| --- | --- | --- |
| CuPy CLI | `rndsrc/cupysmic:<version>` | CuPy API and FITS command |
| CuPy API | `rndsrc/cupysmic:<version>-slim` | CuPy API only |
| CUDA CLI | `rndsrc/cudasmic:<version>` | C API and FITS command |
| CUDA API | `rndsrc/cudasmic:<version>-slim` | C library and header |
| Full | `rndsrc/cusmic:<version>` | Both implementations, tests, benchmarks, demo |

For a CUDA 12 build, set `CUDA=12`; each tag then ends in `-cuda12`. Set
`PLATFORM=linux/amd64` for an x86-64 build, and set `CUDA_ARCHS` to that GPU's
compute capability (for example, `CUDA_ARCHS=86`). The defaults target GPU
architectures 87 and 121. An exact Git version tag supplies `VERSION`
automatically; otherwise set it as above.

```sh
mkdir -p results
docker run --rm --gpus all -v "$PWD/results:/data/results" rndsrc/cusmic:local
```

The full image runs checks and then benchmarks, even if a check fails. Logs,
completed measurements, failures, a comparison table, and a status file go to
`results/`. Unavailable timings are marked in the table. Its exit status
reports failures.
On a Mac, the ARM64 images can start and show CLI help, but GPU cleaning
requires an NVIDIA host.

The [demo notebook](demo/demo.ipynb) shows the saved scene and CuPy interface.
