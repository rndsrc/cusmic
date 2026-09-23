# Tests and reference data

Run commands from the repository root.
Tests use the installed Python package;
an editable install follows changes in this checkout:

```sh
python -m pip install -e '.[cuda13,test,cli]' ruff
make check REFERENCE=close
```

Use `cuda12` for a CUDA 12 runtime, and `REFERENCE=exact` for v0.2.x.
CUDA C/C++ checks also need `nvcc`, a C compiler, and CFITSIO headers
and libraries.
Set `CUDA_ARCHS` for your GPU when compiling; see the [build
instructions](../README.md#build-and-check).

## Select checks

| Command | Checks |
| --- | --- |
| `make check` | Ruff and all Python and C/CUDA checks                |
| `make lint`  | Python style                                         |
| `make unit`  | Validation, FITS I/O, cleaning, batches, and streams |
| `make e2e`   | Saved-reference API checks and FITS command checks   |

Pass pytest options through `PYTEST_ARGS`, for example `make unit
PYTEST_ARGS='-x'`.
Each workflow reports live progress and counts successful and failed
steps; pytest reports individual test results.
Independent checks continue after build or test failures.

GPU checks fail explicitly if CuPy or a CUDA device is unavailable.
The device switching test requires **two GPUs**, so a single-GPU run
cannot pass the full suite.
To run only Python checks that need no CUDA:

```sh
python -m pip install -e '.[test]'
python -m pytest -m host -q
```

This covers settings validation, FITS I/O and invalid CLI inputs,
benchmark validation/reporting, and the saved CPU reference.
It does not validate GPU arithmetic.
The C FITS test also needs no GPU or CUDA compiler:

```sh
make build/cuda/test_io
build/cuda/test_io
```

## Reference policy

`test/data/` contains the shared 512 × 512 input, error map, and saved
L.A.Cosmic reference with its `CRMASK` extension.
The scene uses seed 0 and 200 injected cosmic-ray trails;
the saved mask contains 6,518 detected pixels.
A host test checks it against pinned `lacosmic==1.4.0`.

| Policy | Pixel comparison | Intended release |
| --- | --- | --- |
| `exact` (default) | Identical float64 bits                           | v0.2.x |
| `close`           | Finite error ≤ `32 * eps * (1 + abs(reference))` | v0.3.x |

Here `eps` is float64 machine epsilon.  Masks always match exactly;
special nonfinite values retain their bits.
Make commands use `REFERENCE=exact|close`; direct Python and Docker
commands use `CUSMIC_REFERENCE=exact|close`.
Use the same policy for tests and benchmarks.

Generated FITS cases check scaling and command parity without adding
saved fixtures.
Routine tests never regenerate the committed reference.
The `ref` target defaults to `test/data/`;
use `REFDIR` for candidates to review:

```sh
make ref                        # Writes test/data/; refuses existing files
make ref REFDIR=dist/reference  # Generate separate candidates
```

The generator records the seed, detection settings, and
L.A.Cosmic/NumPy versions in FITS headers and refuses overwrites.
CPU and library builds may round differently;
use the committed files for routine checks.

## Docker

The full image runs these checks with prebuilt C/CUDA executables,
followed by benchmarks.
See the [container instructions](../README.md#docker-images) for
builds, reference policy, and mounted reports.
Failures remain visible in logs and the exit status;
they do not prevent the benchmark phase from running.
