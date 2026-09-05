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

from . import remove_cosmics
from .io import read_fits, write_fits

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

    if str(output).startswith("!"):
        raise click.ClickException("Choose a new output filename")
    if output.exists():
        raise click.ClickException(f"{output} already exists; choose a new output")

    if maxiter and error is None and gain is None:
        raise click.UsageError("Provide --error or --gain (optionally --readnoise)")

    try:
        data, header = read_fits(source, dtype="float64")
        if data.ndim != 2:
            raise ValueError("The FITS command expects one 2D image")
        noise = read_fits(error, dtype="float64")[0] if error else None
        cleaned, mask = remove_cosmics(
            data, error=noise, effective_gain=gain, readnoise=readnoise,
            contrast=contrast, cr_threshold=cr_threshold,
            neighbor_threshold=neighbor_threshold, maxiter=maxiter,
        )
        header.add_history("Cosmic rays removed with cusmic")
        write_fits(output, cleaned, mask, header=header)
    except (OSError, ValueError, TypeError, IndexError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Saved {output} ({int(mask.sum())} cosmic-ray pixels)")


if __name__ == "__main__":
    main()
