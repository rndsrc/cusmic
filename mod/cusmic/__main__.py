# Copyright 2026 Chi-kwan Chan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from pathlib import Path

import click
import cupy as cp
import numpy as np
from astropy.io import fits

from . import remove_cosmics


INPUT  = click.Path(exists=True, dir_okay=False, path_type=Path)
OUTPUT = click.Path(dir_okay=False, path_type=Path)


def read_image(path):
    """Read a finite, two-dimensional FITS image in native float64 format."""
    try:
        data, header = fits.getdata(path, header=True)
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"{path}: {exc}") from exc

    if data.ndim != 2 or not data.size or data.dtype.kind not in "iuf":
        raise click.ClickException(f"{path}: expected a nonempty 2D real image")

    data = data.astype(np.float64)
    if not np.isfinite(data).all():
        raise click.ClickException(f"{path}: image contains nonfinite pixels")

    return data, header


@click.command()
@click.argument("source", type=INPUT)
@click.argument("output", type=OUTPUT)
@click.option("--error", type=INPUT, help="FITS error (1-sigma) image; overrides the noise model.")
@click.option("--gain", type=click.FloatRange(min=0, min_open=True),
              help="Gain in electrons/ADU; required without --error.")
@click.option("--readnoise", default=0.0, type=click.FloatRange(min=0), show_default=True,
              help="Read noise in electrons, used with --gain.")
@click.option("--contrast", default=3.0, type=click.FloatRange(min=0), show_default=True,
              help="Minimum contrast against fine structure.")
@click.option("--cr-threshold", default=5.0, type=click.FloatRange(min=0), show_default=True,
              help="Cosmic-ray detection threshold in sigma.")
@click.option("--neighbor-threshold", default=3.0, type=click.FloatRange(min=0), show_default=True,
              help="Neighbor detection threshold in sigma.")
@click.option("--maxiter", default=4, type=click.IntRange(min=0), show_default=True,
              help="Maximum cleaning iterations; zero copies the image.")
def main(source, output, error, gain, readnoise, contrast, cr_threshold, neighbor_threshold, maxiter):
    """Remove cosmic rays from SOURCE and write a new FITS OUTPUT."""

    if error is None and gain is None:
        raise click.UsageError("Provide --error or --gain (optionally --readnoise)")
    if not np.isfinite(readnoise) or (gain is not None and not np.isfinite(gain)):
        raise click.UsageError("Gain and read noise must be finite")
    if not np.isfinite([contrast, cr_threshold, neighbor_threshold]).all():
        raise click.UsageError("Detection thresholds must be finite")

    data, header = read_image(source)
    data = cp.asarray(data)

    noise = None
    if error is not None:
        noise, _ = read_image(error)
        if noise.shape != data.shape or (noise <= 0).any():
            raise click.ClickException("--error must match the image shape and be positive")
        noise = cp.asarray(noise)

    cleaned, mask = remove_cosmics(
        data,
        error=noise, effective_gain=gain, readnoise=readnoise,
        contrast=contrast, cr_threshold=cr_threshold, neighbor_threshold=neighbor_threshold, maxiter=maxiter,
    )

    cleaned = cp.asnumpy(cleaned)
    mask    = cp.asnumpy(mask).astype(np.uint8)

    header.add_history("Cosmic rays removed with cusmic")
    result = fits.HDUList([
        fits.PrimaryHDU(cleaned, header),
        fits.ImageHDU(mask, name="CRMASK"),
    ])
    result.writeto(output, checksum=True)

    click.echo(f"Saved {output} ({mask.sum():,} cosmic-ray pixels)")


if __name__ == "__main__":
    main()
