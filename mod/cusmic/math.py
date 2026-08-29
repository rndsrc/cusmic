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


import cupy as cp
from cupyx.scipy.ndimage import convolve, median_filter, binary_dilation


RADIUS = 2     # replacement window is 5x5 (paper sec 3.1)
BATCH  = 8192  # replacement gathers at most BATCH x 25 values at a time


def mklaplacian(dtype, mode):

    # Discrete Laplacian for the 2x2-replicated image (paper eq 4);
    # the paper's factor 1/4 is supplied by the flux-conserving
    # replication.
    kernel = cp.asarray([
        [ 0,-1, 0],
        [-1, 4,-1],
        [ 0,-1, 0],
    ], dtype=dtype)

    def laplacian(image):  # closure on kernel and mode
        sampled = (image/4).repeat(2, axis=0).repeat(2, axis=1)
        lap2    = convolve(sampled, kernel, mode=mode)
        cp.maximum(lap2, 0, out=lap2)

        # Sum each 2x2 block as (a + b) + (c + d), order matters
        a, b = lap2[0::2, 0::2], lap2[0::2, 1::2]
        c, d = lap2[1::2, 0::2], lap2[1::2, 1::2]
        return (a + b) + (c + d)

    return laplacian


def mkgrow():
    structure = cp.ones((3, 3), dtype=bool)

    def grow(candidates, sig, allowed, cr_threshold, neighbor_threshold):
        """Grow at the cosmic threshold, then at the neighbor threshold."""
        for threshold in (cr_threshold, neighbor_threshold):
            candidates = binary_dilation(candidates, structure) & (sig > threshold) & allowed
        return candidates

    return grow


def significance(lap, noise, mode, order=5):
    """Remove smooth structure from Laplacian significance (eqs 11, 13)"""
    S = lap / (2 * noise)
    return S - median_filter(S, size=order, mode=mode)


def fine_structure(clean, noise, mode, floor=0.01):
    """Noise-normalized fine structure for star rejection (eq 14)"""
    m = median_filter(clean, size=3, mode=mode)
    F = m - median_filter(m, size=7, mode=mode)
    return cp.maximum(F / noise, floor)


def detect(sig, fine, allowed, contrast, cr_threshold):
    """Select significant pixels with sufficient Laplacian contrast."""
    return (sig > cr_threshold) & (sig / fine > contrast) & allowed


def local_median(clean, donors, targets, offsets):
    """Median of each clipped window; also report which windows have donors."""
    ny, nx = clean.shape
    y,  x  = targets.T
    dy, dx = offsets
    yy, xx = y[:, None] + dy, x[:, None] + dx
    inside = (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx)
    yy, xx = yy.clip(0, ny-1), xx.clip(0, nx-1)
    valid  = inside & donors[yy, xx]
    values = cp.where(valid, clean[yy, xx], clean.dtype.type(cp.inf))
    values.sort(axis=1)  # Donors first, ascending; the +inf padding sorts last.
    count = cp.count_nonzero(valid, axis=1)
    rows  = cp.arange(len(count))
    low, high = values[rows, (count-1)//2], values[rows, count//2]
    return cp.where(count%2, low, (low+high)/2), count > 0
