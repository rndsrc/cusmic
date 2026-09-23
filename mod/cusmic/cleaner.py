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

from .image import Array, Image

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Cleaner:
    """Reusable detection settings; call with an Image to get pixels and a mask."""

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
        """Return independent arrays on the input device's current stream."""
        import cupy as cp

        from .filters import (
            detect,
            fine_structure,
            mkgrow,
            mklaplacian,
            significance,
        )
        from .replace import replace

        with image.data.device:
            if self.maxiter and image.error is None and (
                image.effective_gain is None or image.readnoise is None
            ):
                raise ValueError("Provide error, or both effective_gain and readnoise")

            clean = image.data.copy()
            crmask = cp.zeros(clean.shape, dtype=bool)
            invalid = ~cp.isfinite(clean)
            excluded = invalid if image.mask is None else invalid | image.mask
            cp.copyto(clean, 0, where=invalid)
            if image.background is not None:
                clean += image.background

            if self.maxiter:
                initial_donors = ~excluded
                donors = initial_donors & cp.isfinite(clean)
                if invalid.any() and initial_donors.any():
                    targets = invalid & initial_donors.any(axis=(-2, -1), keepdims=True)
                    replace(clean, targets, donors)

                laplacian = mklaplacian(clean.shape, clean.dtype, self.border_mode)
                grow = mkgrow(clean.ndim)

                for i in range(self.maxiter):
                    lap = laplacian(clean)
                    noise = image.noise(clean, mode=self.border_mode)
                    sig = significance(lap, noise, mode=self.border_mode)
                    fine = fine_structure(clean, noise, mode=self.border_mode)
                    cp.copyto(sig, 0, where=invalid)

                    candidates = detect(sig, fine, excluded, self.contrast, self.cr_threshold)
                    candidates = grow(candidates, sig, self.cr_threshold, self.neighbor_threshold)
                    cp.logical_and(candidates, ~crmask, out=candidates)
                    n_new = int(cp.count_nonzero(candidates))

                    crmask |= candidates

                    log.info("Iteration %d: %d new cosmic-ray pixels", i + 1, n_new)
                    if not n_new:
                        break

                    cp.logical_and(donors, ~candidates, out=donors)
                    replace(clean, crmask, donors)

            if image.background is not None:
                clean -= image.background
            cp.copyto(clean, image.data, where=invalid)
            return clean, crmask
