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


from dataclasses import dataclass

import cupy as cp
from cupy import ndarray as Array
from cupyx.scipy.ndimage import median_filter


@dataclass
class Image:
    """Pixels and calibration data"""

    data:           Array
    error:          Array | None = None
    mask:           Array | None = None
    background:     Array | None = None
    effective_gain: Array | None = None
    readnoise:      float | None = None

    def noise(self, image=None, mode=None, order=5, floor=1e-5):
        if self.error is not None:
            return self.error
        else:
            """Poisson and read noise from the order*order median (eq 10)"""
            n = self.readnoise
            g = self.effective_gain
            m = median_filter(image, size=order, mode=mode)
            cp.maximum(m, floor, out=m)
            return cp.sqrt(n * n + g * m) / g
