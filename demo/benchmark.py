"""Time complete cleaning calls, with and without host/device transfers."""

import json
import os
from pathlib import Path
from time import perf_counter

import click
import cupy as cp
import numpy as np
from cusmic import Cleaner, Image, __version__
from cusmic.io import read_fits


def timed(call):
    cp.cuda.get_current_stream().synchronize()
    start = perf_counter()
    result = call()
    cp.cuda.get_current_stream().synchronize()
    return 1000 * (perf_counter() - start), result


def benchmark(data, error, settings, repeats=15, warmups=5):
    """First result includes CUDA initialization; warmed samples wait for all work."""
    cache = Path(os.environ.get("CUPY_CACHE_DIR", "~/.cupy/kernel_cache")).expanduser()
    cache_existed = cache.exists()
    start = perf_counter()
    cleaner = Cleaner(**settings)

    def upload():
        return Image(cp.asarray(data), error=cp.asarray(error))

    def end_to_end():
        return tuple(cp.asnumpy(a) for a in cleaner(upload()))

    first = end_to_end()
    first_ms = 1000 * (perf_counter() - start)
    for _ in range(warmups):
        end_to_end()
    image = upload()
    for _ in range(warmups):
        timed(lambda: cleaner(image))
    samples = {name: [] for name in ("upload_ms", "clean_ms", "download_ms", "total_ms")}
    for _ in range(repeats):
        elapsed, uploaded = timed(upload)
        samples["upload_ms"].append(elapsed)
        del uploaded
        elapsed, result = timed(lambda: cleaner(image))
        samples["clean_ms"].append(elapsed)
        elapsed, downloaded = timed(lambda result=result: tuple(cp.asnumpy(a) for a in result))
        samples["download_ms"].append(elapsed)
        elapsed, complete = timed(end_to_end)
        samples["total_ms"].append(elapsed)
        for output in (downloaded, complete):
            np.testing.assert_array_equal(output[0].view("uint64"), first[0].view("uint64"))
            np.testing.assert_array_equal(output[1], first[1])
        del result, downloaded, complete
    device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    record = dict(
        host_memory="pageable", allocation="ordinary calls",
        source_revision=os.environ.get("CUSMIC_REVISION", "unknown"),
        kernel_cache_existed=cache_existed,
        stopping="convergence or maxiter", detected_pixels=int(first[1].sum()),
        backend="cupy", shape=list(data.shape), dtype=str(data.dtype), settings=settings,
        warmups=warmups, repeats=repeats, first_result_ms=first_ms,
        milliseconds={name: dict(median=float(np.median(times)), minimum=min(times), maximum=max(times))
                      for name, times in samples.items()},
        cusmic=__version__, cupy=cp.__version__, gpu=device["name"].decode(),
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
        cuda_driver=cp.cuda.runtime.driverGetVersion(),
    )
    return record, first


@click.command()
@click.option("--input", "path", default="test/input.fits.gz", type=click.Path(exists=True))
@click.option("--error", default="test/error.fits.gz", type=click.Path(exists=True))
@click.option("--reference", default="test/reference.fits.gz", type=click.Path(exists=True))
@click.option("--frames", default=1, type=click.IntRange(min=1))
@click.option("--repeats", default=15, type=click.IntRange(min=1), show_default=True)
@click.option("--warmups", default=5, type=click.IntRange(min=1), show_default=True)
def main(path, error, reference, frames, repeats, warmups):
    """Benchmark the saved L.A.Cosmic example; disk I/O is outside warmed timings."""
    data, _ = read_fits(path, dtype="float64")
    noise, _ = read_fits(error, dtype="float64")
    if frames > 1:
        data = np.repeat(data[None], frames, axis=0)
    settings = dict(contrast=1, cr_threshold=5, neighbor_threshold=5, maxiter=4)
    record, (cleaned, mask) = benchmark(data, noise, settings, repeats, warmups)
    expected, _ = read_fits(reference, dtype="float64")
    expected_mask, _ = read_fits(reference, ext="CRMASK")
    expected = np.broadcast_to(expected, cleaned.shape)
    expected_mask = np.broadcast_to(expected_mask, mask.shape)
    np.testing.assert_array_equal(cleaned.view("uint64"), expected.view("uint64"))
    np.testing.assert_array_equal(mask, expected_mask)
    print(json.dumps(dict(record, input=str(Path(path)), reference_exact=True)))


if __name__ == "__main__":
    main()
