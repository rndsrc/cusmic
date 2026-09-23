"""Benchmark CPU L.A.Cosmic, CuPy and CUDA in separate processes."""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("first", "upload", "clean", "download", "total")


def positive_int(value):
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return count


def measure(backend, frames, warmups, repeats):
    command = {
        "cpu": [sys.executable, "-m", "bench.cpu"],
        "cupy": [sys.executable, "-m", "bench.bench"],
        "cuda": [os.environ.get("CUSMIC_CUDA_BENCH", str(ROOT / "build/cuda/bench"))],
    }[backend]
    command += ["--frames", str(frames), "--warmups", str(warmups),
                "--repeats", str(repeats)]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"{backend}, {frames} frame(s): {result.stderr.strip() or result.stdout.strip()}")
    record = json.loads(result.stdout)
    shape = record["shape"]
    measured_frames = shape[0] if len(shape) == 3 else 1
    if (record["backend"] != backend or measured_frames != frames or
            record["reference_exact"] is not True or
            record["warmups"] != warmups or record["repeats"] != repeats):
        raise ValueError(f"unexpected {backend} result for {frames} frame(s)")
    return record


def per_frame(record, stage, frames):
    ms = (record["first_result_ms"] if stage == "first" else
          record["milliseconds"][stage + "_ms"]["median"])
    return ms / frames


def comparison(frames, records):
    cpu = records.get("cpu")
    cupy = records.get("cupy")
    cuda = records.get("cuda")
    row = {"frames": frames}
    if cpu:
        row["cpu_first_ms_per_frame"] = per_frame(cpu, "first", frames)

    for stage in STAGES:
        if cupy:
            a = per_frame(cupy, stage, frames)
            row[f"cupy_{stage}_ms_per_frame"] = a
        if cuda:
            b = per_frame(cuda, stage, frames)
            row[f"cuda_{stage}_ms_per_frame"] = b
        if cupy and cuda:
            row[f"cuda_vs_cupy_{stage}_speedup"] = a / b
        if stage in ("clean", "total"):
            if cpu:
                c = per_frame(cpu, stage, frames)
                row[f"cpu_{stage}_ms_per_frame"] = c
            if cpu and cupy:
                row[f"cupy_vs_cpu_{stage}_speedup"] = c / a
            if cpu and cuda:
                row[f"cuda_vs_cpu_{stage}_speedup"] = c / b
    return row


def format_cell(row, key, width, decimals, suffix=""):
    if key not in row:
        return f"{'--':>{width}}"
    return f"{row[key]:>{width}.{decimals}f}{suffix}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", choices=("cpu", "cupy", "cuda"),
                        default=["cpu", "cupy", "cuda"])
    parser.add_argument("--frames", type=positive_int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--warmups", type=positive_int, default=4)
    parser.add_argument("--repeats", type=positive_int, default=16)
    parser.add_argument("--output", type=Path, default=ROOT / "bench/results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    failures = []
    for frames in args.frames:
        records = {}
        for backend in args.backends:
            print(f"Measuring {backend}, {frames} frame(s)...", flush=True)
            path = args.output / f"{backend}-{frames}.json"
            path.unlink(missing_ok=True)
            try:
                record = measure(backend, frames, args.warmups, args.repeats)
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                failure = dict(backend=backend, frames=frames, error=str(exc))
                failures.append(failure)
                print(f"Unavailable: {failure['error']}", file=sys.stderr, flush=True)
                continue
            path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            records[backend] = record
        rows.append(comparison(frames, records))

    with (args.output / "comparison.csv").open("w", newline="") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    failure_path = args.output / "failures.json"
    if failures:
        failure_path.write_text(json.dumps(failures, indent=2) + "\n")
    else:
        failure_path.unlink(missing_ok=True)

    print("\nMedian milliseconds per frame (warmed):")
    for stage in ("clean", "total"):
        print(f"{stage.title()} calls:")
        print("Frames    CPU ms   CuPy ms   CUDA ms  CuPy/CPU  CUDA/CPU  CUDA/CuPy")
        for row in rows:
            print(f"{row['frames']:>6}  "
                  f"{format_cell(row, f'cpu_{stage}_ms_per_frame', 8, 3)}  "
                  f"{format_cell(row, f'cupy_{stage}_ms_per_frame', 8, 3)}  "
                  f"{format_cell(row, f'cuda_{stage}_ms_per_frame', 8, 3)}  "
                  f"{format_cell(row, f'cupy_vs_cpu_{stage}_speedup', 11, 2, 'x')}  "
                  f"{format_cell(row, f'cuda_vs_cpu_{stage}_speedup', 11, 2, 'x')}  "
                  f"{format_cell(row, f'cuda_vs_cupy_{stage}_speedup', 9, 2, 'x')}")
    print(f"Samples and comparison: {args.output}")
    if failures:
        print(f"Incomplete: {len(failures)} backend measurement(s) failed; "
              f"details: {failure_path}", file=sys.stderr)
    return bool(failures)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
