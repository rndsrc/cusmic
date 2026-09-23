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

import numpy as np
import pytest


@pytest.fixture(scope="session")
def cp():
    try:
        import cupy as cp

        if not cp.cuda.runtime.getDeviceCount():
            raise RuntimeError("No CUDA GPU")
    except (ImportError, RuntimeError) as exc:
        pytest.fail(f"CUDA unavailable: {exc}", pytrace=False)
    return cp


@pytest.fixture(scope="session")
def assert_pixels():
    mode = os.environ.get("CUSMIC_REFERENCE", "exact")
    if mode not in ("exact", "close"):
        pytest.fail("CUSMIC_REFERENCE must be exact or close", pytrace=False)

    def compare(actual, expected, exact=False):
        if exact or mode == "exact":
            np.testing.assert_array_equal(actual.view("uint64"), expected.view("uint64"))
        else:
            special = ~np.isfinite(expected)
            np.testing.assert_array_equal(
                actual[special].view("uint64"), expected[special].view("uint64"))
            eps = 32 * np.finfo("float64").eps
            np.testing.assert_allclose(actual, expected, rtol=eps, atol=eps)
    return compare
