"""Benchmark CPU L.A.Cosmic, CuPy and CUDA in separate processes."""

import argparse
import csv
import json
import math
import os
import statistics as stats
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryFile
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("first", "upload", "clean", "download", "total")


def positive_int(value):
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return count


def measure(backend, frames, warmups, repeats, log_path):
    command = {
        "cpu": [sys.executable, "-m", "bench.cpu"],
        "cupy": [sys.executable, "-m", "bench.bench"],
        "cuda": [os.environ.get("CUSMIC_CUDA_BENCH", str(ROOT / "build/cuda/bench"))],
    }[backend]
    command += ["--frames", str(frames), "--warmups", str(warmups),
                "--repeats", str(repeats)]
    with TemporaryFile(mode="w+") as output, log_path.open("w") as log:
        with subprocess.Popen(command, cwd=ROOT, text=True, stdout=output,
                              stderr=subprocess.PIPE) as process:
            for line in process.stderr:
                log.write(line)
                log.flush()
                print(line, end="", file=sys.stderr, flush=True)
        if process.returncode:
            raise RuntimeError(f"exit {process.returncode}; details: {log_path}")
        output.seek(0)
        record = json.load(output)
    shape = record["shape"]
    measured_frames = shape[0] if len(shape) == 3 else 1
    if (record["backend"] != backend or measured_frames != frames or
            record.get("reference_close", record.get("reference_exact")) is not True or
            record.get("mask_disagreements", 0) != 0 or
            (os.environ.get("CUSMIC_REFERENCE", "exact") == "exact" and
             record.get("reference_exact") is not True) or
            record["warmups"] != warmups or record["repeats"] != repeats):
        raise ValueError(f"unexpected {backend} result for {frames} frame(s)")
    first = record["first_result_ms"]
    if not math.isfinite(first) or first <= 0:
        raise ValueError("first-result time must be positive and finite")
    stages = ("clean", "total") if backend == "cpu" else STAGES[1:]
    for stage in stages:
        samples = record["milliseconds"][stage + "_ms"]["samples"]
        if len(samples) != repeats or any(
                not math.isfinite(value) or value <= 0 for value in samples):
            raise ValueError(f"{stage} samples must match repeats and be positive and finite")
    return record


def per_frame(record, stage, frames, stat="median"):
    ms = (record["first_result_ms"] if stage == "first" else
          getattr(stats, stat)(record["milliseconds"][stage + "_ms"]["samples"]))
    return ms / frames


def comparison(frames, records, stat="median"):
    cpu = records.get("cpu")
    cupy = records.get("cupy")
    cuda = records.get("cuda")
    row = {"frames": frames}
    if cpu:
        row["cpu_first_ms_per_frame"] = per_frame(cpu, "first", frames, stat)

    for stage in STAGES:
        if cupy:
            a = per_frame(cupy, stage, frames, stat)
            row[f"cupy_{stage}_ms_per_frame"] = a
        if cuda:
            b = per_frame(cuda, stage, frames, stat)
            row[f"cuda_{stage}_ms_per_frame"] = b
        if cupy and cuda:
            row[f"cuda_vs_cupy_{stage}_speedup"] = a / b
        if stage in ("clean", "total"):
            if cpu:
                c = per_frame(cpu, stage, frames, stat)
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

    mode = os.environ.get("CUSMIC_REFERENCE", "exact")
    if mode not in ("exact", "close"):
        parser.error("CUSMIC_REFERENCE must be exact or close")
    args.backends = list(dict.fromkeys(args.backends))
    args.frames = list(dict.fromkeys(args.frames))
    title = (f"Benchmarks: {', '.join(args.backends)}; frames {args.frames}; "
             f"{args.warmups} warmups, {args.repeats} samples; reference {mode}")
    print(title, file=sys.stderr, flush=True)
    rows = []
    means = []
    records_by_frame = []
    failures = []
    total = len(args.frames) * len(args.backends)
    done = 0
    for frames in args.frames:
        records = {}
        for backend in args.backends:
            done += 1
            start = perf_counter()
            print(f"[{done}/{total}] {backend}, {frames} frame(s)...",
                  file=sys.stderr, flush=True)
            path = args.output / f"{backend}-{frames}.json"
            path.unlink(missing_ok=True)
            try:
                record = measure(backend, frames, args.warmups, args.repeats,
                                 args.output / f"{backend}-{frames}.log")
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                failure = dict(backend=backend, frames=frames, error=str(exc))
                failures.append(failure)
                print(f"FAIL {backend}, {frames} frame(s): {failure['error']}",
                      file=sys.stderr, flush=True)
                continue
            path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            records[backend] = record
            quality = "exact" if record["reference_exact"] else "close"
            print(f"PASS {backend}: reference {quality}; "
                  f"total median {per_frame(record, 'total', frames):.3f} ms/frame "
                  f"({perf_counter() - start:.1f}s elapsed)",
                  file=sys.stderr, flush=True)
        rows.append(comparison(frames, records))
        means.append(comparison(frames, records, "mean"))
        records_by_frame.append((frames, records))

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

    lines = [title, "", "Warmed milliseconds per frame:",
             "Backend Frames Stage       Median      Mean        SD       Min       Max"]
    for frames, records in records_by_frame:
        for backend, record in records.items():
            lines.append(f"{backend}: first result {record['first_result_ms']:.3f} ms/call; "
                         f"source {record.get('source_revision', 'unknown')}")
            for stage, values in record["milliseconds"].items():
                samples = [x / frames for x in values["samples"]]
                sd = stats.stdev(samples) if len(samples) > 1 else 0.0
                lines.append(f"{backend:>7} {frames:>6} {stage[:-3]:>8} "
                             f"{stats.median(samples):>10.3f} {stats.mean(samples):>9.3f} "
                             f"{sd:>9.3f} {min(samples):>9.3f} {max(samples):>9.3f}")
    for label, table in (("Median", rows), ("Mean", means)):
        lines += ["", f"{label} speedups (>1 is faster):",
                  "Frames    Stage  CuPy/CPU  CUDA/CPU CUDA/CuPy"]
        for row in table:
            for stage in ("clean", "total"):
                lines.append(f"{row['frames']:>6} {stage:>8}  "
                             f"{format_cell(row, f'cupy_vs_cpu_{stage}_speedup', 8, 2)}  "
                             f"{format_cell(row, f'cuda_vs_cpu_{stage}_speedup', 8, 2)}  "
                             f"{format_cell(row, f'cuda_vs_cupy_{stage}_speedup', 8, 2)}")
    lines += ["", f"Completed {total - len(failures)}/{total} cases; "
              f"{len(failures)} failed. Results: {args.output}"]
    if failures:
        lines.append(f"Failures: {failure_path}")
    report = "\n".join(lines) + "\n"
    (args.output / "summary.txt").write_text(report)
    print(report, end="", flush=True)
    return bool(failures)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
