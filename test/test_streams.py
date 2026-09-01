"""Caller-owned CUDA streams and borrowed storage."""

import numpy as np
import pytest


def test_stream_handoff(cp):
    from cusmic import Cleaner, Image

    producer = cp.cuda.Stream(non_blocking=True)
    consumer = cp.cuda.Stream(non_blocking=True)
    ready, done = cp.cuda.Event(), cp.cuda.Event()
    data, error = cp.empty((9, 9)), cp.empty((9, 9))
    delay = cp.RawKernel('''extern "C" __global__ void delay() {
        unsigned long long start = clock64();
        while (clock64() - start < 10000000) {}
    }''', "delay")
    with producer:
        delay((1,), (1,), ())
        data.fill(10)
        data[4, 4] = 1000
        error.fill(1)
        ready.record()
    with consumer:
        consumer.wait_event(ready)
        clean, mask = Cleaner()(Image(data, error=error))
        done.record()
    with producer:
        producer.wait_event(done)
        data.fill(0)
    producer.synchronize()
    np.testing.assert_array_equal(cp.asnumpy(clean), 10)
    assert int(mask.sum()) == 1


def test_input_device(cp):
    from cusmic import Cleaner, Image

    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("device switching needs two GPUs")
    device = cp.cuda.runtime.getDeviceCount() - 1
    with cp.cuda.Device(device):
        data, error = cp.full((9, 9), 10.0), cp.ones((9, 9))
        data[4, 4] = 1000
    with cp.cuda.Device(0):
        cleaned, mask = Cleaner()(Image(data, error=error))
        assert cp.cuda.Device().id == 0
        assert cleaned.device.id == mask.device.id == device
    with cp.cuda.Device(device):
        np.testing.assert_array_equal(cp.asnumpy(cleaned), 10)
        assert int(mask.sum()) == 1
