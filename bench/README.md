# Benchmark

On an NVIDIA GPU, run `make bench` from the repository root. It benchmarks
CuPy and CUDA separately for 1, 4, and 16 frames, using four warmups and
sixteen samples. Use `python -m bench.run --frames 4 --output bench/results`
for a shorter run after building `build/cuda/bench`.

Both benchmarks use the saved image and error map in `test/data/`, repeat that
scene for stack timings, and check the cleaned pixels and mask against the
saved reference exactly. FITS reading and reference comparisons are outside
the warmed intervals.

The JSON records contain every sample and hardware details; `comparison.csv`
contains per-frame medians and CUDA/CuPy speedups. The intervals are:

- First result: an ordinary complete call, including first CUDA use.
- Upload: pageable host pixels and error map transferred to the GPU.
- Clean: an ordinary call with inputs resident on the GPU.
- Download: cleaned pixels and mask transferred to pageable host arrays.
- Total: a fresh ordinary upload, clean, and download call.

Each interval waits for GPU completion. Upload excludes CuPy `Image`
construction; total includes each backend's preparation. Separately timed
stages need not add up to total. Results belong in the ignored
`bench/results/` directory.
