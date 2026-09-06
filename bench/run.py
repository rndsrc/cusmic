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
        "cuda": [str(ROOT / "build/cuda/bench")],
    }[backend]
    command += ["--frames", str(frames), "--warmups", str(warmups),
                "--repeats", str(repeats)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "mod") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"{backend}, {frames} frame(s): {result.stderr.strip() or result.stdout.strip()}")
    record = json.loads(result.stdout)
    if (record["backend"] != backend or record["reference_exact"] is not True or
            record["warmups"] != warmups or record["repeats"] != repeats):
        raise ValueError(f"unexpected {backend} result for {frames} frame(s)")
    return record


def per_frame(record, stage, frames):
    ms = (record["first_result_ms"] if stage == "first" else
          record["milliseconds"][stage + "_ms"]["median"])
    return ms / frames


def comparison(frames, cpu, cupy, cuda):
    row = {"frames": frames, "cpu_first_ms_per_frame": per_frame(cpu, "first", frames)}
    for stage in STAGES:
        a = per_frame(cupy, stage, frames)
        b = per_frame(cuda, stage, frames)
        row[f"cupy_{stage}_ms_per_frame"] = a
        row[f"cuda_{stage}_ms_per_frame"] = b
        row[f"cuda_vs_cupy_{stage}_speedup"] = a / b
        if stage in ("clean", "total"):
            c = per_frame(cpu, stage, frames)
            row[f"cpu_{stage}_ms_per_frame"] = c
            row[f"cupy_vs_cpu_{stage}_speedup"] = c / a
            row[f"cuda_vs_cpu_{stage}_speedup"] = c / b
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=positive_int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--warmups", type=positive_int, default=4)
    parser.add_argument("--repeats", type=positive_int, default=16)
    parser.add_argument("--output", type=Path, default=ROOT / "bench/results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for frames in args.frames:
        records = {}
        for backend in ("cupy", "cuda", "cpu"):
            print(f"Measuring {backend}, {frames} frame(s)...", flush=True)
            record = measure(backend, frames, args.warmups, args.repeats)
            (args.output / f"{backend}-{frames}.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n")
            records[backend] = record
        rows.append(comparison(frames, records["cpu"], records["cupy"], records["cuda"]))

    with (args.output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nMedian milliseconds per frame (warmed):")
    for stage in ("clean", "total"):
        print(f"{stage.title()} calls:")
        print("Frames    CPU ms   CuPy ms   CUDA ms  CuPy/CPU  CUDA/CPU  CUDA/CuPy")
        for row in rows:
            print(f"{row['frames']:>6}  {row[f'cpu_{stage}_ms_per_frame']:>8.3f}"
                  f"  {row[f'cupy_{stage}_ms_per_frame']:>8.3f}"
                  f"  {row[f'cuda_{stage}_ms_per_frame']:>8.3f}"
                  f"  {row[f'cupy_vs_cpu_{stage}_speedup']:>11.2f}x"
                  f"  {row[f'cuda_vs_cpu_{stage}_speedup']:>11.2f}x"
                  f"  {row[f'cuda_vs_cupy_{stage}_speedup']:>9.2f}x")
    print(f"Samples and comparison: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
