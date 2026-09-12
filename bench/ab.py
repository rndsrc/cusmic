"""Compare warmed v0.2.5 and v0.3 GPU results from the same host.

Run ``python -m bench.ab BASELINE_RESULTS CANDIDATE_RESULTS`` after both full
images finish their benchmarks. A speedup above one means v0.3 is faster.
"""

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
    if values.pop("laplacian_dtype", "float64") != "float64":
        raise ValueError("A/B comparison requires a float64 Laplacian")
    return values


def samples(record, stage):
    values = record["milliseconds"][stage]["samples"]
    if len(values) != 16 or any(not math.isfinite(t) or t <= 0 for t in values):
        raise ValueError("each stage needs sixteen positive finite samples")
    return values


def compare(baseline, candidate):
    same = ("backend", "dtype", "gpu", "cuda_runtime", "cuda_driver",
            "shape", "warmups", "repeats", "host_memory", "allocation")
    if baseline["backend"] == "cupy":
        same += ("cupy", "numpy", "python")
    if any(baseline[key] != candidate[key] for key in same):
        raise ValueError("GPU, software, shape or timing settings differ")
    old_version = baseline.get("cusmic")
    new_version = candidate.get("cusmic")
    if (old_version != "0.2.5" or not isinstance(new_version, str) or
            not new_version.startswith("0.3.")):
        raise ValueError("A/B comparison needs v0.2.5 and a v0.3 candidate")
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
    if (baseline["dtype"] != "float64" or baseline["warmups"] != 4 or
            baseline["repeats"] != 16 or settings(baseline) != settings(candidate)):
        raise ValueError("A/B comparison requires matching float64 settings and 4/16 timing")
    if (baseline["reference_exact"] is not True or
            candidate.get("reference_close") is not True or
            candidate.get("mask_disagreements") != 0):
        raise ValueError("baseline must be bit exact; candidate pixels close and masks exact")
    error = candidate.get("max_abs_error")
    if error is None or not math.isfinite(error) or error < 0:
        raise ValueError("candidate must report a finite reference error")
    return [(stage, samples(baseline, stage), samples(candidate, stage))
            for stage in STAGES]


def spread(values, frames):
    return (f"{min(values) / frames:.3f}..{max(values) / frames:.3f}"
            f" (SD {stats.stdev(values) / frames:.3f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="v0.2.5 bench/results directory")
    parser.add_argument("candidate", type=Path, help="v0.3 bench/results directory")
    args = parser.parse_args()
    print("Median warmed milliseconds per frame; speedup >1 means v0.3 is faster")
    print("Frames Backend Stage       v0.2.5     v0.3  Speedup")
    try:
        for frames in FRAMES:
            for backend in BACKENDS:
                old = load(args.baseline, backend, frames)
                new = load(args.candidate, backend, frames)
                for stage, before, after in compare(old, new):
                    a, b = stats.median(before) / frames, stats.median(after) / frames
                    name = "resident" if stage == "clean_ms" else "ordinary"
                    print(f"{frames:>6} {backend:>7} {name:>8} {a:>12.3f} {b:>8.3f} "
                          f"{a / b:>7.2f}x")
                    print(f"       sample range: exact {spread(before, frames)}; "
                          f"optimized {spread(after, frames)}")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
