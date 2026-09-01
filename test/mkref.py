"""Write a fresh L.A.Cosmic scene to a separate output directory."""

import argparse
from pathlib import Path

import lacosmic
import numpy as np
from astropy.io import fits
from lacosmic.utils import make_cosmic_rays, make_gaussian_sources


def mkref(directory):
    baseline, error = make_gaussian_sources((512, 512), seed=0)
    data = baseline + make_cosmic_rays(baseline.shape, n_cosmics=200, seed=0)
    settings = dict(contrast=1, cr_threshold=5, neighbor_threshold=5, maxiter=4)
    cleaned, mask = lacosmic.remove_cosmics(data, error=error, **settings)
    header = fits.Header({"LACOSMIC": lacosmic.__version__, "NUMPY": np.__version__,
                          "SEED": 0, "NCRAYS": 200, "CONTRAST": 1,
                          "CRTHRESH": 5, "NBTHRESH": 5, "MAXITER": 4})
    directory.mkdir(parents=True, exist_ok=True)
    for name, pixels in (("input", data), ("error", error), ("reference", cleaned)):
        hdus = [fits.PrimaryHDU(pixels, header)]
        if name == "reference":
            hdus.append(fits.ImageHDU(mask.astype("uint8"), name="CRMASK"))
        fits.HDUList(hdus).writeto(directory / f"{name}.fits.gz", checksum=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Output directory; existing files are not overwritten")
    mkref(parser.parse_args().directory)
