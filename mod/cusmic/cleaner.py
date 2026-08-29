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

from .image import Image, Array
from .math  import *


@dataclass
class Cleaner:
    """Reusable detection settings"""

    contrast:           float = 3
    cr_threshold:       float = 5
    neighbor_threshold: float = 3
    maxiter:            int   = 4
    border_mode:        str   = "mirror"

    def __call__(self, image: Image) -> tuple[Array, Array]:
        laplacian = mklaplacian(image.data.dtype, self.border_mode)

        clean = cp.where(cp.isfinite(image.data), image.data, 0)
        if image.background is not None:
            clean += image.background

        allowed     = cp.logical_not(image.mask)
        cosmic_mask = cp.zeros(image.data.shape, dtype=bool)
        new         = 0

        for i in range(self.maxiter):
            lap   = laplacian(clean)
            noise = image.noise(clean, mode=self.border_mode)
            sig   = significance(lap, noise, mode=self.border_mode)
            fine  = fine_structure(clean, noise, mode=self.border_mode)

            candidates = detect(sig, fine, allowed, self.contrast, self.cr_threshold)

            print("Iteration {i+1}: {new} new cosmic-ray pixels")
            if not new:
                break

        if image.background is not None:
            clean -= image.background
        return cp.where(cosmic_mask, clean, image.data), cosmic_mask
