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

SETTINGS = (3, 5, 3)


def test_dense_patch(cp):
    from cusmic import remove_cosmics

    image = cp.full((15, 15), 10.0)
    image[5:10, 5:10] = 1000
    cleaned, mask = remove_cosmics(image, *SETTINGS, error=cp.ones_like(image))

    np.testing.assert_array_equal(cp.asnumpy(cleaned), 10)
    np.testing.assert_array_equal(cp.asnumpy(mask), cp.asnumpy(image == 1000))


@pytest.mark.parametrize("mode", ("mirror", "reflect", "nearest", "wrap", "constant"))
def test_laplacian_borders(cp, mode):
    from cupyx.scipy.ndimage import convolve
    from cusmic.filters import mklaplacian

    kernel = cp.asarray([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype="float64")
    for pixels in ([[12, -4, 20]], [[12], [-4], [20]], [[12, -4], [20, 0]]):
        image = cp.asarray(pixels, dtype="float64")
        sampled = cp.repeat(cp.repeat(image / 4, 2, axis=0), 2, axis=1)
        lap2 = cp.maximum(convolve(sampled, kernel, mode=mode), 0)
        expected = (lap2[0::2, 0::2] + lap2[0::2, 1::2]) + (
            lap2[1::2, 0::2] + lap2[1::2, 1::2]
        )
        actual = mklaplacian(image.shape, image.dtype, mode)(image)
        np.testing.assert_array_equal(cp.asnumpy(actual), cp.asnumpy(expected))


def test_input_validation(cp):
    from cusmic import Cleaner, Image

    for data in (cp.ones(3), cp.ones((0, 3)), cp.ones((1, 2, 3, 4))):
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
    with pytest.raises(ValueError, match="Provide error"):
        Cleaner()(Image(data))


def test_cleaner_settings():
    from cusmic import Cleaner

    for settings in ({"contrast": -1}, {"cr_threshold": np.nan},
                     {"neighbor_threshold": np.inf}, {"maxiter": 1.5},
                     {"maxiter": True}, {"maxiter": -1}, {"border_mode": "invalid"}):
        with pytest.raises(ValueError):
            Cleaner(**settings)


def test_zero_iterations_background(cp):
    from cusmic import remove_cosmics

    image = cp.full((3, 3), 0.1)
    cleaned, mask = remove_cosmics(image, *SETTINGS, background=1e12, maxiter=0)
    expected = (np.full((3, 3), 0.1) + 1e12) - 1e12
    np.testing.assert_array_equal(cp.asnumpy(cleaned).view("uint64"), expected.view("uint64"))
    assert not mask.any()


def test_nonfinite_pixels(cp):
    from cusmic import remove_cosmics

    data = np.full((9, 9), 10.0)
    data[2:7, 2:7] = np.nan
    data[3, 3], data[3, 5], data[4, 4] = np.inf, -np.inf, 1000
    data.view("uint64")[2, 2] = 0x7ff8000000000001  # Preserve NaN payloads too.
    original = data.copy()
    for maxiter in (0, 4):
        clean, mask = remove_cosmics(data, *SETTINGS, error=np.ones_like(data),
                                      background=100, maxiter=maxiter)
        expected = data.copy()
        if maxiter:
            expected[4, 4] = 10
        np.testing.assert_array_equal(cp.asnumpy(clean).view("uint64"), expected.view("uint64"))
        assert cp.asnumpy(mask).sum() == bool(maxiter)
        np.testing.assert_array_equal(data.view("uint64"), original.view("uint64"))
    data[:] = np.nan
    clean, mask = remove_cosmics(data, *SETTINGS, error=np.ones_like(data))
    np.testing.assert_array_equal(cp.asnumpy(clean).view("uint64"), data.view("uint64"))
    assert not cp.asnumpy(mask).any()


def test_independent_calls(cp):
    from cusmic import Cleaner, Image

    cleaner = Cleaner()
    def clean_image(data, **kwargs):
        return cleaner(Image(data, **kwargs))
    xp = cp
    results = []
    for shape in ((9, 9), (7, 11)):
        data, error = xp.full(shape, 10.0), xp.ones(shape)
        data[3, 3] = 1000
        original = cp.asnumpy(data).copy()
        results.append(clean_image(data, error=error))
        np.testing.assert_array_equal(cp.asnumpy(data), original)
        np.testing.assert_array_equal(cp.asnumpy(error), 1)
    for cleaned, mask in results:
        np.testing.assert_array_equal(cp.asnumpy(cleaned), 10)
        assert cp.asnumpy(mask).sum() == 1


def test_replacement_batches(cp, monkeypatch):
    from cusmic.replace import replace

    monkeypatch.setattr("cusmic.replace.BUDGET", 1)
    data = cp.asarray([[0.0, 1000, 1000, 1000, 1000, 1000, 60]])
    targets = data == 1000
    donors = ~targets
    cleaned = replace(data, targets, donors)
    assert cleaned is data
    np.testing.assert_array_equal(cp.asnumpy(cleaned), [[0, 0, 0, 30, 60, 60, 60]])
    assert replace(data, cp.zeros(data.shape, dtype=bool), donors) is data
