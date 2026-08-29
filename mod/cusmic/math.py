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
from cupyx.scipy.ndimage import convolve


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
