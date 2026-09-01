"""FITS metadata and CLI shape checks do not need a GPU."""

import numpy as np
import pytest
from astropy.io import fits
from click.testing import CliRunner

pytest.importorskip("cupy")
from cusmic.__main__ import main
from cusmic.io import read_fits, write_fits


def test_integer_blank(tmp_path):
    data = np.arange(9, dtype="int16").reshape(3, 3)
    data[1, 1] = -32768
    hdu = fits.PrimaryHDU(data)
    hdu.header["BLANK"] = -32768
    source, output = tmp_path / "input.fits", tmp_path / "output.fits"
    hdu.writeto(source)
    pixels, header = read_fits(source, dtype="float64")
    write_fits(output, pixels, header=header)
    np.testing.assert_array_equal(read_fits(output)[0], pixels)
    assert np.isnan(pixels[1, 1])
    assert "BLANK" not in fits.getheader(output)
    assert header["BLANK"] == -32768


@pytest.mark.parametrize("data", [None, np.zeros((2, 3, 3))])
def test_cli_rejects_invalid_images(tmp_path, data):
    source, output = tmp_path / "invalid.fits", tmp_path / "output.fits"
    fits.PrimaryHDU(data).writeto(source)
    result = CliRunner().invoke(main, [str(source), str(output), "--maxiter", "0"])
    assert result.exit_code == 1 and "Error:" in result.output
    assert not output.exists()
