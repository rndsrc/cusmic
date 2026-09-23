# Benchmarks

The tools in this checkout measure CPU L.A.Cosmic, the installed CuPy
package, and the compiled CUDA library using the same saved FITS
scene.
For installation and CUDA build requirements, see the [main
README](../README.md#build-and-check).
Run commands from the repository root.

## Run locally

```sh
make bench REFERENCE=close
make bench BACKENDS='cpu cupy' REFERENCE=close  # No CUDA C/C++ build
make bench BACKENDS=cpu BENCH_ARGS='--frames 1 --warmups 1 --repeats 2'
```

Defaults are 1, 4, and 16 frames, four warmups, and sixteen measured
samples per case.
CPU-only runs need just `pip install '.[bench]'`.
You can also run the Python orchestrator directly;
it expects the CUDA executable to be built already if that backend is
selected:

```sh
CUSMIC_REFERENCE=close python -m bench.run --backends cpu cupy --frames 1 4 16
```

`REFERENCE=exact` is the Make default for v0.2.x;
select `close` for v0.3.x.
Direct Python and Docker runs use `CUSMIC_REFERENCE`.
The [test guide](../test/README.md#reference-policy) defines both
policies.
GPU records retain actual bit equality and maximum error even in close
mode.
CPU L.A.Cosmic always uses the exact saved reference.
Every measured cleaning result is checked outside the timing interval,
and repeated GPU outputs must be identical.

Cases run independently:
a missing GPU, build error, or failed comparison returns a nonzero
status while preserving completed measurements.

## Read reports

Results default to `bench/results/`.
Set `--output DIRECTORY` in `BENCH_ARGS` or the Python runner to keep
separate runs.
Reusing a directory replaces the selected cases;
use a fresh directory for each comparison.

| File | Contents |
| --- | --- |
| `BACKEND-FRAMES.json` | Successful case, environment, reference checks, timing samples |
| `BACKEND-FRAMES.log`  | Backend progress and errors                                    |
| `summary.txt`         | Per-frame medians, means, sample SD, range, and speedups       |
| `comparison.csv`      | Median timings and speedups                                    |
| `failures.json`       | Failed cases; absent after a fully successful run              |

First use is reported separately in milliseconds per whole call.
GPU stages measure upload, resident cleaning, download, and complete
calls, including synchronization.
Complete calls include allocation and transfers;
disk I/O, progress output, and reference checks are excluded.
Separately timed stages need not sum to the complete call.
CPU clean/total fields describe the same ordinary calls.

Individual backends (`python -m bench.cpu`, `python -m bench.bench`,
and `build/cuda/bench`) emit JSON on stdout and progress on stderr.
Their `--output` option appends JSONL records;
the orchestrator instead writes one JSON file per case.

CuPy records identify the installed package path.
Editable and Git URL installations supply a revision;
ordinary wheel/local-copy installations report `unknown` unless the
build supplies `CUSMIC_REVISION`.
Editable installs also report tracked source changes.
CUDA records embed the revision at compilation;
rebuild after changing sources or flags.
A revision alone does not prove that a compiled library came from an
unmodified tree.

## A/B with one shared toolset

Keep tests, benchmark scripts, and saved data in this checkout.
Install each implementation from a separate worktree into the same
Python environment so dependencies and input paths stay fixed:

```sh
git worktree add --detach ../cusmic-a v0.2.6
git worktree add --detach ../cusmic-b v0.3.1
for side in a b; do
    python -m pip install --no-deps -e "../cusmic-$side"
    policy=exact
    if [ "$side" = b ]; then policy=close; fi
    CUSMIC_REFERENCE=$policy python -m bench.run --backends cupy \
        --output "bench/results/$side"
done
python -m bench.ab bench/results/a bench/results/b --backends cupy
```

The report requires distinct recorded Git revisions, matching
hardware, runtimes, dependencies, settings, frame counts, and timing
parameters, plus passing reference checks.
Known dirty source trees are rejected.
Build from clean worktrees and leave the saved data unchanged;
records do not capture all source and input modifications.
Use `--frames` to compare a subset.

To include CUDA, replace the runner command inside the loop with:

```sh
make -B bench SRC="../cusmic-$side/src" BACKENDS='cupy cuda' \
    REFERENCE="$policy" BENCH_ARGS="--output bench/results/$side"
```

`SRC` selects the CUDA implementation;
`-B` prevents stale library reuse when switching sources.
Then omit `--backends cupy` from the A/B report.
Use the same `SRC` and `REFERENCE` with `make -B e2e` for shared
reference checks.
Run full `make check` from each version's own checkout because unit
regressions can require fixes absent from the other release.
Restore your development installation afterward:

```sh
python -m pip install --no-deps -e .
```

A/B output includes median and mean speedups and sample spread.
Use an idle GPU and repeat in reverse order when investigating small
differences or pauses.

## Docker

Only the full image includes tests and benchmark tools.
Build it with `make container TARGET=full VERSION=0.3.1-dev`.
The default command runs checks and then benchmarks, saving both logs
and `status.txt`:

```sh
mkdir -p results
docker run --rm --gpus all -e CUSMIC_REFERENCE=close \
    -v "$PWD/results:/data/results" rndsrc/cusmic:0.3.1-dev
```

Use `exact` for v0.2.x.  Append benchmark options after the image name, such as
`--frames 1 --backends cpu cupy`.  To run only benchmarks:

```sh
docker run --rm --gpus all -e CUSMIC_REFERENCE=close --entrypoint python \
    -v "$PWD/results:/data/results" rndsrc/cusmic:0.3.1-dev \
    -m bench.run --output /data/results --frames 1 4 16
```

For A/B testing, build one full image per implementation revision with
the same toolset, run each into a separate result directory, then pass
those directories to `python -m bench.ab`.
Existing release images retain their original scripts until rebuilt.
See the [container guide](../README.md#docker-images) for CUDA
profiles, platforms, and host driver requirements.

## Recorded GB10 measurements

NVIDIA GB10, a 512 × 512 float64 reference image, four warmups, and
sixteen measured calls.
Values below are median milliseconds per frame.
GPU complete calls include allocation, upload, cleaning, and download;
disk I/O is excluded.
The CPU ran L.A.Cosmic 1.4, and CuPy was 14.2 with CUDA 13.0.2.

| Frames | CPU L.A.Cosmic | v0.2.5 CuPy | v0.3.0 CuPy | v0.2.5 CUDA | v0.3.0 CUDA |
| ---: | ---: | ---: | ---: | ---: | ---: |
|  1 | 680.481 | 10.190 | 9.732 | 6.629 | 6.547 |
|  4 | 683.897 |  9.614 | 8.592 | 6.252 | 6.041 |
| 16 | 680.504 |  9.800 | 8.907 | 6.658 | 6.553 |

These historical runs measured v0.2.5 (exact) and v0.3.0 (close).
v0.2.6 and v0.3.1 retain their respective cleaning algorithms.
Although the optimized line permits rounding, these outputs matched
the saved pixels and masks bit for bit.

The initial 16-frame CuPy runs had unexplained pauses:
means were 12.313 and 14.095 ms/frame despite the lower optimized
median.
An ABBA ABBA repeat used four warmups and sixteen samples in each run.
Across 64 samples per implementation, mean complete-call time fell
from 9.257 ± 0.099 to 8.518 ± 0.093 ms/frame (mean ± sample SD), or
**8.0% less time**.
First-use and separate transfer timings remain in the original
benchmark reports.

These measurements predate the shared-tool rebuild; fresh GPU runs
are required to qualify the rebuilt releases.
The original single-GPU runs could not pass the two-GPU check.
