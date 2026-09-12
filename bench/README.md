# Benchmark

On a GPU host, install the matching CuPy wheel and pinned CPU comparator,
build CUDA C/C++, then measure all three implementations:

```sh
python -m pip install '.[cuda13,bench]'
make build
make bench
```

Use `cuda12` with a CUDA 12 runtime. If CuPy is already installed, `.[bench]`
adds L.A.Cosmic without replacing CuPy. On a CPU-only host, run
`python -m pip install '.[bench]'` and `python -m bench.cpu --frames 1` to
measure L.A.Cosmic alone.

`make bench` uses the saved frame and error map in `test/data/` for 1, 4,
and 16 frames, with four warmups and sixteen repeated samples. A shorter
run is `python -m bench.run --frames 4`. The runner records completed
backends and explicit failures when a GPU is unavailable. It reports
completed measurements even when a backend fails; unavailable timings
are marked in the comparison table.

Each backend checks its pixels and mask against the saved reference outside
the timed interval. GPU records retain both bitwise agreement and the
32-epsilon pixel comparison used after arithmetic reordering; saved-scene masks
must still match exactly. JSON files hold all samples and hardware details;
`comparison.csv` contains available per-frame medians and speedups;
`failures.json` identifies missing backends. Output is written
to the ignored `bench/results/` directory by default, or to `/data/results`
in the full Docker image.

GPU timings wait for completion and separate first use, upload, resident
cleaning, download, and ordinary total. CPU cleaning and total use the same
ordinary calls. Separately timed stages need not add up to total because
each timing has its own setup.
