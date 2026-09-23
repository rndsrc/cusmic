"""Time complete cleaning calls, with and without host/device transfers."""

import json
import os
import platform
import subprocess
from importlib.metadata import distribution
from pathlib import Path
from time import perf_counter
from urllib.parse import unquote, urlparse

import click
import cupy as cp
import cusmic
import numpy as np
from cusmic import Cleaner, Image, __version__
from cusmic.io import read_fits

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "test/data"


def source_info():
    """Identify the installed package, never infer it from the benchmark checkout."""
    direct = json.loads(distribution("cusmic").read_text("direct_url.json") or "{}")
    revision = direct.get("vcs_info", {}).get("commit_id", "unknown")
    dirty = None
    if direct.get("dir_info", {}).get("editable"):
        root = unquote(urlparse(direct["url"]).path)
        try:
            revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            dirty = bool(subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=root, text=True))
        except (OSError, subprocess.CalledProcessError):
            pass
    elif revision == "unknown":
        revision = os.environ.get("CUSMIC_REVISION", "unknown")
    return dict(source_revision=revision, source_dirty=dirty,
                package_path=str(Path(cusmic.__file__).resolve()))


def timed(call):
    cp.cuda.get_current_stream().synchronize()
    start = perf_counter()
    result = call()
    cp.cuda.get_current_stream().synchronize()
    return 1000 * (perf_counter() - start), result


def benchmark(data, error, settings, repeats=16, warmups=4):
    """First result includes CUDA initialization; warmed samples wait for all work."""
    cache = Path(os.environ.get("CUPY_CACHE_DIR", "~/.cupy/kernel_cache")).expanduser()
    cache_existed = cache.exists()
    click.echo(f"CuPy: first result ({warmups} warmups, {repeats} samples, "
               f"shape {data.shape})", err=True)
    start = perf_counter()
    cleaner = Cleaner(**settings)

    def transfer():
        return cp.asarray(data), cp.asarray(error)

    def upload():
        pixels, noise = transfer()
        return Image(pixels, error=noise)

    def end_to_end():
        return tuple(cp.asnumpy(a) for a in cleaner(upload()))

    first = end_to_end()
    first_ms = 1000 * (perf_counter() - start)
    for i in range(warmups):
        click.echo(f"CuPy: ordinary warmup {i + 1}/{warmups}", err=True)
        end_to_end()
    samples = {name: [] for name in ("upload_ms", "clean_ms", "download_ms", "total_ms")}
    for i in range(repeats):
        click.echo(f"CuPy: upload {i + 1}/{repeats}", err=True)
        elapsed, uploaded = timed(transfer)
        samples["upload_ms"].append(elapsed)
        del uploaded

    image = upload()
    for i in range(warmups):
        click.echo(f"CuPy: resident warmup {i + 1}/{warmups}", err=True)
        timed(lambda: cleaner(image))
    for i in range(repeats):
        click.echo(f"CuPy: clean/download {i + 1}/{repeats}", err=True)
        elapsed, result = timed(lambda: cleaner(image))
        samples["clean_ms"].append(elapsed)
        elapsed, downloaded = timed(lambda result=result: tuple(cp.asnumpy(a) for a in result))
        samples["download_ms"].append(elapsed)
        np.testing.assert_array_equal(downloaded[0].view("uint64"), first[0].view("uint64"))
        np.testing.assert_array_equal(downloaded[1], first[1])
        del result, downloaded

    image = None
    for i in range(repeats):
        click.echo(f"CuPy: ordinary total {i + 1}/{repeats}", err=True)
        elapsed, complete = timed(end_to_end)
        samples["total_ms"].append(elapsed)
        np.testing.assert_array_equal(complete[0].view("uint64"), first[0].view("uint64"))
        np.testing.assert_array_equal(complete[1], first[1])
        del complete
    device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    record = dict(
        host_memory="pageable", allocation="ordinary calls",
        **source_info(),
        python=platform.python_version(), numpy=np.__version__,
        system=platform.platform(), cpu=platform.machine(),
        kernel_cache_existed=cache_existed,
        stopping="convergence or maxiter", detected_pixels=int(first[1].sum()),
        backend="cupy", shape=list(data.shape), dtype=str(data.dtype), settings=settings,
        warmups=warmups, repeats=repeats, first_result_ms=first_ms,
        milliseconds={name: dict(median=float(np.median(times)), minimum=min(times), maximum=max(times), samples=times)
                      for name, times in samples.items()},
        frames_per_second=1000 * (len(data) if data.ndim == 3 else 1) / np.median(samples["total_ms"]),
        cusmic=__version__, cupy=cp.__version__, gpu=device["name"].decode(),
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
        cuda_driver=cp.cuda.runtime.driverGetVersion(),
    )
    return record, first


@click.command()
@click.option("--input", "path", default=str(DATA / "input.fits.gz"), type=click.Path(exists=True))
@click.option("--error", default=str(DATA / "error.fits.gz"), type=click.Path(exists=True))
@click.option("--reference", default=str(DATA / "reference.fits.gz"), type=click.Path(exists=True))
@click.option("--frames", default=1, type=click.IntRange(min=1))
@click.option("--repeats", default=16, type=click.IntRange(min=1), show_default=True)
@click.option("--warmups", default=4, type=click.IntRange(min=1), show_default=True)
@click.option("--output", type=click.Path(path_type=Path), help="Append a JSON record to this file.")
def main(path, error, reference, frames, repeats, warmups, output):
    """Benchmark the saved L.A.Cosmic example; disk I/O is outside warmed timings."""
    mode = os.environ.get("CUSMIC_REFERENCE", "exact")
    if mode not in ("exact", "close"):
        raise click.ClickException("CUSMIC_REFERENCE must be exact or close")
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
    eps = 32 * np.finfo(np.float64).eps
    if mode == "exact":
        np.testing.assert_array_equal(cleaned.view("uint64"), expected.view("uint64"))
    else:
        special = ~np.isfinite(expected)
        np.testing.assert_array_equal(
            cleaned[special].view("uint64"), expected[special].view("uint64"))
        np.testing.assert_allclose(cleaned, expected, rtol=eps, atol=eps, equal_nan=True)
    np.testing.assert_array_equal(mask, expected_mask)
    finite = np.isfinite(expected)
    difference = np.abs(cleaned[finite] - expected[finite])
    line = json.dumps(dict(
        record, input=str(Path(path)), reference_mode=mode, reference_close=True,
        reference_exact=bool(np.array_equal(cleaned.view("uint64"), expected.view("uint64"))),
        max_abs_error=float(difference.max()) if difference.size else 0.0,
        mask_disagreements=0,
    ))
    click.echo("CuPy: reference passed; done", err=True)
    print(line)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a") as stream:
            stream.write(line + "\n")


if __name__ == "__main__":
    main()
