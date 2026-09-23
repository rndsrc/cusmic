"""Compare matched GPU benchmarks from two installed revisions."""

import argparse
import json
import math
import re
import statistics as stats
from pathlib import Path

FRAMES = (1, 4, 16)
BACKENDS = ("cupy", "cuda")
STAGES = ("clean_ms", "total_ms")


def load(directory, backend, frames):
    record = json.loads((directory / f"{backend}-{frames}.json").read_text())
    shape = record["shape"]
    n = shape[0] if len(shape) == 3 else 1
    if record["backend"] != backend or n != frames:
        raise ValueError(f"wrong case in {directory / f'{backend}-{frames}.json'}")
    return record


def settings(record):
    values = dict(record["settings"])
    values.setdefault("laplacian_dtype", "float64")
    return values


def samples(record, stage):
    values = record["milliseconds"][stage]["samples"]
    if len(values) != record["repeats"] or not values or any(
            not math.isfinite(t) or t <= 0 for t in values):
        raise ValueError("stage samples must match repeats and be positive and finite")
    return values


def compare(baseline, candidate):
    same = ("backend", "dtype", "gpu", "cuda_runtime", "cuda_driver",
            "shape", "warmups", "repeats", "host_memory", "allocation")
    if baseline["backend"] == "cupy":
        same += ("cupy", "numpy", "python")
    if any(baseline[key] != candidate[key] for key in same):
        raise ValueError("GPU, software, shape or timing settings differ")
    before = baseline.get("source_revision")
    after = candidate.get("source_revision")
    if before == after or any(
            not isinstance(revision, str) or
            not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", revision)
            for revision in (before, after)):
        raise ValueError("A/B comparison needs distinct Git source revisions")
    counts = baseline.get("detected_pixels"), candidate.get("detected_pixels")
    if counts[0] != counts[1] or any(
            not isinstance(n, int) or isinstance(n, bool) or n < 0
            for n in counts):
        raise ValueError("A/B comparison needs the same detected-pixel count")
    if baseline["backend"] == "cupy":
        path = baseline.get("input")
        if not isinstance(path, str) or not path or path != candidate.get("input"):
            raise ValueError("CuPy A/B comparison needs the same input path")
    if settings(baseline) != settings(candidate):
        raise ValueError("algorithm settings differ")
    for record in (baseline, candidate):
        if record.get("source_dirty") is True:
            raise ValueError("A/B comparison requires unmodified source trees")
        if record.get("mask_disagreements", 0) != 0:
            raise ValueError("reference masks differ")
        if record.get("reference_exact") is True:
            continue
        if (record.get("reference_mode") == "exact" or
                record.get("reference_close") is not True or
                record.get("mask_disagreements") != 0):
            raise ValueError("reference pixels must pass their policy and masks must match")
        error = record.get("max_abs_error")
        if error is None or not math.isfinite(error) or error < 0:
            raise ValueError("rounded results must report a finite reference error")
    return [(stage, samples(baseline, stage), samples(candidate, stage))
            for stage in STAGES]


def spread(values, frames):
    return (f"{min(values) / frames:.3f}..{max(values) / frames:.3f}"
            f" (SD {(stats.stdev(values) if len(values) > 1 else 0) / frames:.3f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="baseline results directory")
    parser.add_argument("candidate", type=Path, help="candidate results directory")
    parser.add_argument("--backends", choices=BACKENDS, nargs="+", default=BACKENDS,
                        help="Implementations to compare (default: cupy cuda)")
    parser.add_argument("--frames", type=int, nargs="+", default=FRAMES)
    args = parser.parse_args()
    print("Median warmed milliseconds per frame; speedup >1 means candidate is faster")
    print("Frames Backend Stage     Baseline Candidate  Speedup")
    try:
        for frames in args.frames:
            for backend in args.backends:
                old = load(args.baseline, backend, frames)
                new = load(args.candidate, backend, frames)
                for stage, before, after in compare(old, new):
                    a, b = stats.median(before) / frames, stats.median(after) / frames
                    name = "resident" if stage == "clean_ms" else "ordinary"
                    print(f"{frames:>6} {backend:>7} {name:>8} {a:>12.3f} {b:>8.3f} "
                          f"{a / b:>7.2f}x")
                    mean_a = stats.mean(before) / frames
                    mean_b = stats.mean(after) / frames
                    print(f"       mean: baseline {mean_a:.3f}; candidate {mean_b:.3f}; "
                          f"speedup {mean_a / mean_b:.3f}x "
                          f"({100 * (mean_b / mean_a - 1):+.1f}% time)")
                    print(f"       sample range: baseline {spread(before, frames)}; "
                          f"candidate {spread(after, frames)}")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
