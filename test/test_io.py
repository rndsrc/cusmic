"""FITS edge cases and CLI input checks that need no GPU."""

import numpy as np
import pytest
from astropy.io import fits
from cusmic.io import read_fits, write_fits

pytestmark = pytest.mark.host


def test_integer_blank(tmp_path):
    data = np.arange(9, dtype="int16").reshape(3, 3)
    data[1, 1] = -32768
    hdu = fits.PrimaryHDU(data)
    hdu.header["BLANK"] = -32768
    source, output = tmp_path / "input.fits", tmp_path / "output.fits"
    hdu.writeto(source)
    pixels, header = read_fits(source, dtype="float64")
    write_fits(output, pixels, np.zeros(data.shape, dtype="bool"), header=header)
    np.testing.assert_array_equal(read_fits(output)[0], pixels)
    assert not read_fits(output, ext="CRMASK")[0].any()
    assert np.isnan(pixels[1, 1])
    assert "BLANK" not in fits.getheader(output)
    assert header["BLANK"] == -32768


def test_float64_fits_pixels(tmp_path):
    source = tmp_path / "pixels.fits"
    bits = np.array([
        0x8000000000000000, 0x7FF0000000000000,
        0x7FF8000000001234, 0x401C000000000000,
    ], dtype="uint64")
    raw = np.array([[1, -32768], [2, 3]], dtype="int16")
    scaled = fits.ImageHDU(raw, name="SCALED")
    scaled.header["BSCALE"] = 0.1
    scaled.header["BZERO"] = 10.2
    scaled.header["BLANK"] = -32768
    unsigned = fits.ImageHDU(raw, name="UNSIGNED")
    unsigned.header["BZERO"] = 32768
    unsigned.header["BLANK"] = -32768
    fits.HDUList([
        fits.PrimaryHDU(bits.view("float64").reshape(2, 2)), scaled, unsigned,
    ]).writeto(source)

    pixels, _ = read_fits(source, dtype="float64")
    np.testing.assert_array_equal(pixels.view("uint64").ravel(), bits)

    pixels, _ = read_fits(source, dtype="float64", ext="SCALED")
    expected = np.array([
        0x4024999999999999, 0x7FF8000000000000,
        0x4024CCCCCCCCCCCC, 0x4025000000000000,
    ], dtype="uint64")
    np.testing.assert_array_equal(pixels.view("uint64").ravel(), expected)

    pixels, _ = read_fits(source, dtype="float64", ext="UNSIGNED")
    assert np.isnan(pixels[0, 1])
    np.testing.assert_array_equal(pixels[[0, 1, 1], [0, 0, 1]], [32769, 32770, 32771])


@pytest.mark.parametrize("data", [None, np.zeros((2, 3, 3))])
def test_cli_rejects_invalid_images(tmp_path, data):
    from click.testing import CliRunner
    from cusmic.__main__ import main

    source, output = tmp_path / "invalid.fits", tmp_path / "output.fits"
    fits.PrimaryHDU(data).writeto(source)
    result = CliRunner().invoke(main, [str(source), str(output), "--maxiter", "0"])
    assert result.exit_code == 1 and "Error:" in result.output
    assert not output.exists()
