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


import logging
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real

import cupy as cp

from .filters import detect, fine_structure, mkgrow, mklaplacian, significance
from .image import Array, Image
from .replace import replace

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Cleaner:
    """Reusable detection settings"""

    contrast:           float = 3
    cr_threshold:       float = 5
    neighbor_threshold: float = 3
    maxiter:            int   = 4
    border_mode:        str   = "mirror"

    def __post_init__(self):
        for name in ("contrast", "cr_threshold", "neighbor_threshold"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")

        if (
            isinstance(self.maxiter, bool)
            or not isinstance(self.maxiter, Integral)
            or self.maxiter < 0
        ):
            raise ValueError("maxiter must be a nonnegative integer")
        modes = ("mirror", "reflect", "nearest", "wrap", "constant")
        if not isinstance(self.border_mode, str) or self.border_mode not in modes:
            raise ValueError(f"border_mode must be one of {', '.join(modes)}")

    def __call__(self, image: Image) -> tuple[Array, Array]:
        """Return the cleaned image and the cosmic-ray mask"""

        laplacian = mklaplacian(image.data.dtype, self.border_mode)
        grow      = mkgrow()

        clean = cp.where(cp.isfinite(image.data), image.data, 0)
        if image.background is not None:
            clean += image.background

        excluded = False if image.mask is None else image.mask
        donors = cp.logical_not(excluded)
        crmask  = cp.zeros(image.data.shape, dtype=bool)

        for i in range(self.maxiter):
            lap   = laplacian(clean)
            noise = image.noise(clean, mode=self.border_mode)
            sig   = significance(lap, noise, mode=self.border_mode)
            fine  = fine_structure(clean, noise, mode=self.border_mode)

            candidates = detect(sig, fine, excluded, self.contrast, self.cr_threshold)
            candidates = grow(candidates, sig, self.cr_threshold, self.neighbor_threshold)
            n_new = int(cp.count_nonzero(candidates & ~crmask))

            crmask |= candidates
            n_donors = int(cp.count_nonzero(donors & ~crmask))

            log.info("Iteration %d: %d new cosmic-ray pixels", i+1, n_new)
            if not n_new or not n_donors:
                break

            clean = replace(clean, crmask, excluded)

        if image.background is not None:
            clean -= image.background
        return clean, crmask
