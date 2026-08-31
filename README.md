# cusmic

Implementing L.A.Cosmic using CuPy.

From a checkout, with CUDA and CuPy already working:

```sh
python -m pip install numpy astropy click
python -m pip install --no-deps .
```

This keeps your installed CuPy distribution, including a CUDA-specific
wheel.


## Clean a FITS image

```sh
python -m cusmic input.fits cleaned.fits --error error.fits
```

The error map contains positive 1-sigma errors in the same units and
shape as the input.
Alternatively, supply gain in electrons/ADU and read noise in
electrons:

```sh
cusmic input.fits cleaned.fits --gain 2 --readnoise 5
```

`--error` takes precedence over the noise model.
`--contrast`, `--cr-threshold`, `--neighbor-threshold`, and
`--maxiter` control detection;
see `cusmic --help`.

The CLI accepts finite, nonempty 2D FITS images, including `.fits.gz`,
and cleans in float64.
It reads the primary image, or the first extension if the primary is
empty.
The output keeps the image header and contains the cleaned pixels in
the primary HDU plus a `CRMASK` extension (1 = detected cosmic ray, 0
= unflagged).
Existing output files are never overwritten; choose a new filename for
each run.

[`demo/demo.ipynb`](demo/demo.ipynb) installs cusmic from GitHub and
lacosmic from PyPI, recreates the reference image, and compares their
cleaned images and cosmic-ray masks.
