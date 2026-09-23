"""Host checks for benchmark correctness and failure reporting."""

import json
import sys

import numpy as np
import pytest

from bench import cpu, run

pytestmark = pytest.mark.host


def test_cpu_checks_every_measured_result(monkeypatch):
    reference = (np.ones((1, 1)), np.zeros((1, 1), dtype=bool))
    calls = 0

    def clean(data, error):
        nonlocal calls
        calls += 1
        # First result and warmup pass; only the first measured call is wrong.
        return np.full((1, 1), 2.0 if calls == 3 else 1.0), reference[1].copy()

    monkeypatch.setattr(cpu, "clean_batch", clean)
    with pytest.raises(AssertionError):
        cpu.benchmark(reference[0], reference[0], reference,
                      frames=1, warmups=1, repeats=2)


@pytest.mark.parametrize("samples", [[0], [float("nan")], [], [1, 2], None])
def test_runner_keeps_results_after_invalid_timings(tmp_path, monkeypatch, samples):
    def record(backend, times):
        stages = ("clean", "total") if backend == "cpu" else run.STAGES[1:]
        return dict(
            backend=backend, shape=[2, 2], reference_exact=True,
            warmups=1, repeats=1, first_result_ms=1,
            milliseconds={stage + "_ms": {"samples": times} for stage in stages},
        )

    executable = tmp_path / "backend"
    executable.write_text(
        f"#!{sys.executable}\nprint({json.dumps(record('cuda', samples))!r})\n")
    executable.chmod(0o755)
    monkeypatch.setenv("CUSMIC_CUDA_BENCH", str(executable))
    monkeypatch.setenv("CUSMIC_REFERENCE", "exact")
    output = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", [
        "bench.run", "--backends", "cuda", "cpu", "--frames", "1",
        "--warmups", "1", "--repeats", "1", "--output", str(output),
    ])
    measure = run.measure

    def measure_case(backend, *args):
        return record("cpu", [1]) if backend == "cpu" else measure(backend, *args)

    monkeypatch.setattr(run, "measure", measure_case)
    assert run.main() == 1
    assert not (output / "cuda-1.json").exists()
    assert (output / "cpu-1.json").exists()
    failures = json.loads((output / "failures.json").read_text())
    assert len(failures) == 1 and failures[0]["backend"] == "cuda"
    assert "Completed 1/2 cases; 1 failed" in (output / "summary.txt").read_text()
