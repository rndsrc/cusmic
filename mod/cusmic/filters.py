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
from cupyx.scipy.ndimage import binary_dilation, median_filter

ORDER = 5     # median filter
FLOOR = 0.01  # fine structure floor
NOISE = 1e-5  # median floor in the noise model (eq 10)


# Mirror and reflect both select the edge pixel of a 2x-replicated image.
PAD = {
    "mirror": "edge",
    "reflect": "edge",
    "nearest": "edge",
    "wrap": "wrap",
    "constant": "constant",
}


@cp.fuse()
def lap4(c, up, down, left, right):
    """Sum the four positive Laplacians of each replicated pixel (eq 4)."""
    c, up, down, left, right = (c / 4, up / 4, down / 4, left / 4, right / 4)
    ul = cp.maximum((c - up) + (c - left), 0)
    ur = cp.maximum((c - up) + (c - right), 0)
    dl = cp.maximum((c - down) + (c - left), 0)
    dr = cp.maximum((c - down) + (c - right), 0)
    return (ul + ur) + (dl + dr)


def mklaplacian(shape, dtype, mode):
    pad = [(0, 0)] * (len(shape) - 2) + [(1, 1), (1, 1)]
    border = PAD[mode]

    def laplacian(image):
        image = image.astype(dtype, copy=False)
        p = cp.pad(image, pad, mode=border)
        return lap4(
            image,
            p[..., :-2, 1:-1],
            p[..., 2:, 1:-1],
            p[..., 1:-1, :-2],
            p[..., 1:-1, 2:],
        )

    return laplacian


def mkgrow(ndim=2):
    structure = cp.ones((1,) * (ndim - 2) + (3, 3), dtype=bool)

    def grow(candidates, sig, cr_threshold, neighbor_threshold):
        """Grow at cosmic then neighbor thresholds; input exclusions apply to seeds."""
        for threshold in (cr_threshold, neighbor_threshold):
            candidates = binary_dilation(candidates, structure)
            cp.logical_and(candidates, sig > threshold, out=candidates)
        return candidates

    return grow


def median(image, size, mode):
    """Filter spatial axes only; frames never share pixels."""
    return median_filter(image, size=(1,) * (image.ndim - 2) + (size, size), mode=mode)


def noise_model(clean, gain, readnoise, mode):
    """Poisson and read noise from the 5x5 median (eq 10)"""
    noise = median(clean, ORDER, mode)
    cp.maximum(noise, NOISE, out=noise)

    rn2 = readnoise * readnoise
    cp.multiply(gain, noise, out=noise)
    cp.add(rn2, noise, out=noise)
    cp.sqrt(noise, out=noise)
    cp.divide(noise, gain, out=noise)
    return noise


def significance(lap, noise, mode):
    """Remove smooth structure from Laplacian significance (eqs 11, 13)"""
    sig = lap / (2 * noise)
    smooth = median(sig, ORDER, mode)
    cp.subtract(sig, smooth, out=sig)
    return sig


def fine_structure(clean, noise, mode):
    """Noise-normalized fine structure for star rejection (eq 14)"""
    med3 = median(clean, ORDER - 2, mode)
    med7 = median(med3, ORDER + 2, mode)
    fine = med3
    cp.subtract(fine, med7, out=fine)
    cp.divide(fine, noise, out=fine)
    cp.maximum(fine, FLOOR, out=fine)
    return fine


def detect(sig, fine, excluded, contrast, cr_threshold):
    """Select significant pixels with sufficient Laplacian contrast."""
    ratio = fine  # Fine structure is not needed after detection.
    cp.divide(sig, ratio, out=ratio)
    return (sig > cr_threshold) & (ratio > contrast) & cp.logical_not(excluded)
