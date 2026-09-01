# Benchmark

Run from the repository root on a GPU:
```sh
make bench
make bench BENCH_ARGS="--frames 4 --warmups 5 --repeats 15 --output bench/results/run.jsonl"
```
The command uses `test/data/` and verifies exact agreement with the saved
reference. Each output line records hardware, software, settings and all
timing samples. Local results belong in the ignored `bench/results/`;
keep published measurements as CI or release artifacts.

- First result: complete call including CUDA initialization and any compilation.
  An existing CuPy disk cache can make this faster; its presence is recorded.
- Upload: pageable host arrays to GPU, including `Image` construction.
- Cleaning: complete `Cleaner` call with data and calibration already on the GPU.
- Download: cleaned pixels and mask to host.
- Total: independent complete upload, cleaning and download call.

Warmed samples synchronize the current stream before and after each interval.
Disk I/O and reference comparisons are outside these intervals. Allocation
and normal pool reuse remain part of the measured calls. The separately
measured stages need not add up to the end-to-end total.

To record results from a container:
```sh
mkdir -p bench/results
docker run --rm --gpus all -v "$PWD/bench/results:/results" \
    rndsrc/cusmic:0.0.0 -m bench.bench --frames 4 --output /results/run.jsonl
```
For an uncached first result, start a fresh container without a mounted CuPy
cache. Notebook timings use its already initialized CUDA context.
