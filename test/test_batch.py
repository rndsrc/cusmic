"""Sliced stacks agree with independent calls and preserve their inputs."""

import numpy as np


def test_sliced_stack(cp):
    from cusmic import Cleaner, Image

    movie = cp.full((4, 9, 18), 10.0)
    view = movie[::2, :, ::2]
    view[0, 4, 4] = 1000
    view[1] = cp.nan
    original = cp.asnumpy(movie).view("uint64").copy()
    image = Image(view, error=cp.ones((9, 9)))
    assert image.data is view
    cleaner = Cleaner()
    clean, mask = cleaner(image)
    for i in range(2):
        want, flags = cleaner(Image(view[i], error=image.error))
        np.testing.assert_array_equal(cp.asnumpy(clean[i]).view("uint64"),
                                      cp.asnumpy(want).view("uint64"))
        np.testing.assert_array_equal(cp.asnumpy(mask[i]), cp.asnumpy(flags))
    np.testing.assert_array_equal(cp.asnumpy(movie).view("uint64"), original)
    movie.fill(0)
    np.testing.assert_array_equal(cp.asnumpy(clean[0]), 10)
    assert cp.isnan(clean[1]).all() and not mask[1].any()
