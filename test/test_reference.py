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

import numpy as np


def test_reference(cp):
    from cusmic import remove_cosmics
    from cusmic.io import read_fits

    root = Path(__file__).parent
    image, _ = read_fits(root / "input.fits.gz")
    error, _ = read_fits(root / "error.fits.gz")
    expected, _ = read_fits(root / "reference.fits.gz")
    expected_mask, _ = read_fits(root / "reference.fits.gz", ext="CRMASK")

    cleaned, mask = remove_cosmics(
        cp.asarray(image), error=cp.asarray(error),
        contrast=1, cr_threshold=5, neighbor_threshold=5, maxiter=4,
    )
    np.testing.assert_array_equal(cp.asnumpy(cleaned).view("uint64"), expected.view("uint64"))
    np.testing.assert_array_equal(cp.asnumpy(mask), expected_mask)


def test_cupy_cli_validation(cp, tmp_path):
    from click.testing import CliRunner
    from cusmic.__main__ import main
    from cusmic.io import read_fits

    source = Path(__file__).with_name("input.fits.gz")
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
