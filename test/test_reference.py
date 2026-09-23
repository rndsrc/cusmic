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


import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.e2e


def reference_images():
    from cusmic.io import read_fits

    root = Path(__file__).parent / "data"
    return (
        read_fits(root / "input.fits.gz")[0],
        read_fits(root / "error.fits.gz")[0],
        read_fits(root / "reference.fits.gz")[0],
        read_fits(root / "reference.fits.gz", ext="CRMASK")[0],
    )


@pytest.mark.host
def test_saved_reference_lacosmic():
    from inspect import signature

    from cusmic import remove_cosmics as gpu_remove
    from lacosmic import remove_cosmics as cpu_remove

    image, error, expected, expected_mask = reference_images()
    assert signature(gpu_remove) == signature(cpu_remove)
    cleaned, mask = cpu_remove(image, 1, 5, 5, error=error, maxiter=4)
    np.testing.assert_array_equal(cleaned.view("uint64"), expected.view("uint64"))
    np.testing.assert_array_equal(mask, expected_mask)


def test_reference(cp, assert_pixels):
    from cusmic import remove_cosmics

    image, error, expected, expected_mask = reference_images()

    cleaned, mask = remove_cosmics(
        cp.asarray(image), error=cp.asarray(error),
        contrast=1, cr_threshold=5, neighbor_threshold=5, maxiter=4,
    )
    assert_pixels(cp.asnumpy(cleaned), expected)
    np.testing.assert_array_equal(cp.asnumpy(mask), expected_mask)


def test_cupy_cli_validation(cp, tmp_path):
    from click.testing import CliRunner
    from cusmic.__main__ import main
    from cusmic.io import read_fits

    source = Path(__file__).parent / "data/input.fits.gz"
    output = tmp_path / "copy.fits"
    args = [str(source), str(output), "--maxiter", "0"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    np.testing.assert_array_equal(read_fits(output)[0].view("uint64"),
                                  read_fits(source)[0].view("uint64"))
    assert not read_fits(output, ext="CRMASK")[0].any()
    assert CliRunner().invoke(main, args).exit_code != 0
    output = tmp_path / "invalid.fits"
    for options in (["--gain", "1", "--contrast", "nan"], ["--readnoise", "1"]):
        result = CliRunner().invoke(main, [str(source), str(output), *options])
        assert result.exit_code != 0 and not output.exists()


def test_scaled_fits_cli_parity(cp, tmp_path, assert_pixels):
    from astropy.io import fits
    from click.testing import CliRunner
    from cusmic.__main__ import main
    from cusmic.io import read_fits

    native = Path(__file__).resolve().parents[1] / os.environ.get(
        "CUSMIC_CUDA_CLI", "bin/cudasmic")
    if not native.exists():
        pytest.fail("Build bin/cudasmic for cross-implementation FITS checks", pytrace=False)

    data = np.ones((9, 9), dtype="int16")
    data[4, 4] = 10000
    hdu = fits.PrimaryHDU(data)
    hdu.header["BSCALE"] = 0.1
    hdu.header["BZERO"] = 10.2
    source, error = tmp_path / "scaled.fits", tmp_path / "error.fits"
    hdu.writeto(source)
    fits.PrimaryHDU(np.ones(data.shape)).writeto(error)

    for iterations in (0, 4):
        python_out = tmp_path / f"cupy-{iterations}.fits"
        cuda_out = tmp_path / f"cuda-{iterations}.fits"
        options = ["--error", str(error), "--maxiter", str(iterations)]
        result = CliRunner().invoke(main, [str(source), str(python_out), *options])
        assert result.exit_code == 0, result.output
        result = subprocess.run(
            [str(native), str(source), str(cuda_out), *options],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

        python_pixels = read_fits(python_out)[0]
        cuda_pixels = read_fits(cuda_out)[0]
        assert_pixels(python_pixels, cuda_pixels, exact=iterations == 0)
        np.testing.assert_array_equal(
            read_fits(python_out, ext="CRMASK")[0],
            read_fits(cuda_out, ext="CRMASK")[0],
        )
