"""Time CPU L.A.Cosmic on the saved scene, processing stack frames in order."""

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import lacosmic
import numpy as np
from astropy import log as astropy_log
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "test/data"
SETTINGS = dict(contrast=1, cr_threshold=5, neighbor_threshold=5, maxiter=4)


def positive_int(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def clean_batch(data, error):
    if data.ndim == 2:
        return lacosmic.remove_cosmics(data, error=error, **SETTINGS)

    results = [lacosmic.remove_cosmics(frame, error=error, **SETTINGS)
               for frame in data]
    return tuple(np.stack(parts) for parts in zip(*results))


def exact_result(result, reference):
    cleaned, mask = result
    expected, expected_mask = reference
    np.testing.assert_array_equal(cleaned.view("uint64"),
                                  np.broadcast_to(expected, cleaned.shape).view("uint64"))
    np.testing.assert_array_equal(mask, np.broadcast_to(expected_mask, mask.shape))


def timed(call):
    start = perf_counter()
    result = call()
    return 1000 * (perf_counter() - start), result


def source_info():
    revision = os.environ.get("CUSMIC_REVISION", "unknown")
    dirty = None
    if (ROOT / ".git").exists():
        try:
            revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            dirty = bool(subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=ROOT, text=True))
        except (OSError, subprocess.CalledProcessError):
            pass
    return dict(source_revision=revision, source_dirty=dirty)


def benchmark(data, error, reference, frames, warmups, repeats):
    def call():
        return clean_batch(data, error)
    print(f"CPU: first result ({warmups} warmups, {repeats} samples, "
          f"{frames} frames)", file=sys.stderr, flush=True)
    first_ms, first = timed(call)
    exact_result(first, reference)

    for i in range(warmups):
        print(f"CPU: warmup {i + 1}/{warmups}", file=sys.stderr, flush=True)
        call()

    samples = []
    for i in range(repeats):
        print(f"CPU: ordinary total {i + 1}/{repeats}", file=sys.stderr, flush=True)
        elapsed, last = timed(call)
        samples.append(elapsed)
        exact_result(last, reference)

    stats = dict(median=float(np.median(samples)), minimum=min(samples),
                 maximum=max(samples), samples=samples)
    return dict(
        backend="cpu", reference_exact=True, frames=frames,
        shape=list(data.shape), dtype=str(data.dtype), settings=SETTINGS,
        warmups=warmups, repeats=repeats, first_result_ms=first_ms,
        milliseconds=dict(clean_ms=stats, total_ms=stats),
        frames_per_second=1000 * frames / stats["median"],
        detected_pixels=int(first[1].sum()),
        stopping="convergence or maxiter", allocation="ordinary calls",
        **source_info(), python=platform.python_version(), numpy=np.__version__,
        lacosmic=lacosmic.__version__, system=platform.platform(),
        cpu=platform.processor() or platform.machine(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA / "input.fits.gz")
    parser.add_argument("--error", type=Path, default=DATA / "error.fits.gz")
    parser.add_argument("--reference", type=Path, default=DATA / "reference.fits.gz")
    parser.add_argument("--frames", type=positive_int, default=1)
    parser.add_argument("--warmups", type=positive_int, default=4)
    parser.add_argument("--repeats", type=positive_int, default=16)
    parser.add_argument("--output", type=Path, help="Append a JSON record to this file")
    args = parser.parse_args()

    astropy_log.setLevel("ERROR")
    data = np.asarray(fits.getdata(args.input), dtype=np.float64)
    error = np.asarray(fits.getdata(args.error), dtype=np.float64)
    expected = np.asarray(fits.getdata(args.reference), dtype=np.float64)
    mask = np.asarray(fits.getdata(args.reference, extname="CRMASK"), dtype=bool)
    if args.frames > 1:
        data = np.repeat(data[None], args.frames, axis=0)

    record = benchmark(data, error, (expected, mask), args.frames,
                       args.warmups, args.repeats)
    line = json.dumps(dict(record, input=str(args.input)))
    print("CPU: reference passed; done", file=sys.stderr, flush=True)
    print(line)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a") as stream:
            stream.write(line + "\n")


if __name__ == "__main__":
    main()
