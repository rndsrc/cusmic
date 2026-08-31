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
    np.testing.assert_array_equal(cp.asnumpy(cleaned), expected)
    np.testing.assert_array_equal(cp.asnumpy(mask), expected_mask)
