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
from cupyx.scipy.ndimage import binary_dilation, convolve, median_filter

ORDER  = 5     # median filter
FLOOR  = 0.01  # fine structure floor
NOISE  = 1e-5  # median floor in the noise model (eq 10)


def mklaplacian(shape, dtype, mode):
    ny, nx = shape[-2:]
    ndim = len(shape)

    # Discrete Laplacian for the 2x2-replicated image (paper eq 4);
    # the paper's factor 1/4 is supplied by the flux-conserving
    # replication.
    kernel = cp.asarray([
        [ 0,-1, 0],
        [-1, 4,-1],
        [ 0,-1, 0],
    ], dtype=dtype).reshape((1,) * (ndim - 2) + (3, 3))
    sampled = cp.empty((*shape[:-2], 2 * ny, 2 * nx), dtype=dtype)
    lap2 = cp.empty_like(sampled)

    def laplacian(image):  # closure on kernel and mode
        blocks = sampled.reshape(*shape[:-2], ny, 2, nx, 2)
        cp.divide(image[..., :, None, :, None], 4, out=blocks)
        convolve(sampled, kernel, mode=mode, output=lap2)
        cp.maximum(lap2, 0, out=lap2)

        # Sum each 2x2 block as (a + b) + (c + d), order matters
        a, b = lap2[..., 0::2, 0::2], lap2[..., 0::2, 1::2]
        c, d = lap2[..., 1::2, 0::2], lap2[..., 1::2, 1::2]
        return ((a + b) + c) + d if nx == 1 else (a + b) + (c + d)

    return laplacian


def mkgrow(ndim=2):
    structure = cp.ones((1,) * (ndim - 2) + (3, 3), dtype=bool)

    def grow(candidates, sig, cr_threshold, neighbor_threshold):
        """Grow at cosmic then neighbor thresholds; input exclusions apply to seeds."""
        for threshold in (cr_threshold, neighbor_threshold):
            candidates = binary_dilation(candidates, structure) & (sig > threshold)
        return candidates

    return grow


def median(image, size, mode):
    """Filter spatial axes only; frames never share pixels."""
    return median_filter(image, size=(1,) * (image.ndim - 2) + (size, size), mode=mode)


def noise_model(clean, gain, readnoise, mode):
    """Poisson and read noise from the 5x5 median (eq 10)"""
    m = cp.maximum(median(clean, ORDER, mode), NOISE)
    return cp.sqrt(readnoise*readnoise + gain*m) / gain


def significance(lap, noise, mode):
    """Remove smooth structure from Laplacian significance (eqs 11, 13)"""
    S = lap / (2 * noise)
    return S - median(S, ORDER, mode)


def fine_structure(clean, noise, mode):
    """Noise-normalized fine structure for star rejection (eq 14)"""
    m = median(clean, ORDER-2, mode)
    F = m - median(m, ORDER+2, mode)
    return cp.maximum(F / noise, FLOOR)


def detect(sig, fine, excluded, contrast, cr_threshold):
    """Select significant pixels with sufficient Laplacian contrast."""
    return (sig > cr_threshold) & (sig / fine > contrast) & cp.logical_not(excluded)
