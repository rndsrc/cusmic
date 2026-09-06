# Benchmark

On an NVIDIA GPU, run `make bench` from the repository root. It benchmarks
CPU L.A.Cosmic, CuPy and CUDA separately for 1, 4, and 16 frames, using
four warmups and sixteen samples. For a shorter run after building
`build/cuda/bench`, use `python -m bench.run --frames 4`.

On a CPU-only machine, use `python -m bench.cpu --frames 1` with NumPy,
Astropy and `lacosmic==1.4.0` installed.

All three benchmarks use the saved image and error map in `test/data/`,
repeat that scene for stack timings, and check the cleaned pixels and mask
against the saved reference exactly. FITS reading and reference comparisons are outside
the warmed intervals.

The JSON records contain every sample and hardware details; `comparison.csv`
contains per-frame medians and speedups against CPU and CuPy. The intervals are:

- First result: an ordinary complete call; GPU first use may initialize CUDA.
- Upload (GPU only): pageable host pixels and error map transferred to the GPU.
- Clean: an ordinary call with inputs resident on the host or GPU.
- Download (GPU only): cleaned pixels and mask transferred to host arrays.
- Total: a complete host call or a fresh GPU upload, clean, and download call.

GPU intervals wait for completion. CPU clean and total use the same
ordinary-call samples because there is no host/device transfer. Upload
excludes CuPy `Image` construction; total includes each backend's
preparation. Separately timed stages need not add up to total. Results belong in the ignored
`bench/results/` directory.
