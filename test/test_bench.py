"""Host checks for benchmark correctness and failure reporting."""

import numpy as np
import pytest

from bench import cpu

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
