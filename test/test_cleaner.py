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


import numpy as np
import pytest


def test_dense_patch(cp):
    from cusmic import remove_cosmics

    image = cp.full((15, 15), 10.0)
    image[5:10, 5:10] = 1000
    cleaned, mask = remove_cosmics(image, error=cp.ones_like(image))

    np.testing.assert_array_equal(cp.asnumpy(cleaned), 10)
    np.testing.assert_array_equal(cp.asnumpy(mask), cp.asnumpy(image == 1000))


def test_input_validation(cp):
    from cusmic import Image

    for data in (cp.ones(3), cp.ones((0, 3)), cp.ones((1, 2, 3))):
        with pytest.raises(ValueError, match="nonempty 2D"):
            Image(data)
    for dtype in ("bool", "complex128", "float32"):
        with pytest.raises(TypeError, match="float64"):
            Image(cp.ones((3, 3), dtype=dtype))
    data = cp.ones((3, 3))
    for name in ("error", "mask", "background", "effective_gain", "readnoise"):
        with pytest.raises(ValueError, match=name):
            Image(data, **{name: cp.ones((1, 3))})
    for name, value in (("error", 0), ("effective_gain", -1),
                        ("readnoise", -1), ("background", np.inf)):
        with pytest.raises(ValueError, match=name):
            Image(data, **{name: cp.full_like(data, value)})
