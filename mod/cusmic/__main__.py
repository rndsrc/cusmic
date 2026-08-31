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
@click.option("--error", required=True, type=INPUT, help="FITS error (1-sigma) image.")
def main(source, output, error):
    """Remove cosmic rays from SOURCE and write a new FITS OUTPUT."""

    data, header = fits.getdata(source, header=True)
    noise        = fits.getdata(error)

    cleaned, _   = remove_cosmics(cp.asarray(data), error=cp.asarray(noise))

    fits.writeto(output, cp.asnumpy(cleaned), header)
    click.echo(f"Saved {output}")


if __name__ == "__main__":
    main()
