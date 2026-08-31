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

    data, header = fits.getdata(source, header=True)
    noise        = fits.getdata(error)

    cleaned, mask = remove_cosmics(
        cp.asarray(data),
        error=cp.asarray(noise), effective_gain=gain, readnoise=readnoise,
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
