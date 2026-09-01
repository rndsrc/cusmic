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

from .filters import noise_model


@dataclass
class Image:
    """Borrowed float64 pixels and calibration validated on their device."""

    data:           Array
    error:          Array | None = None
    mask:           Array | None = None
    background:     Array | float | None = None
    effective_gain: Array | float | None = None
    readnoise:      Array | float | None = None

    def __post_init__(self):
        if not isinstance(self.data, Array):
            self.data = cp.asarray(self.data)
        if self.data.dtype != cp.float64:
            raise TypeError("data must contain float64 pixels")
        if self.data.ndim != 2 or not self.data.size:
            raise ValueError("data must be a nonempty 2D image")

        shape = self.data.shape
        fields = ("error",) if self.error is not None else ("effective_gain", "readnoise")
        with self.data.device:
            if not cp.isfinite(self.data).all():
                raise ValueError("data must be finite")
            for name in (*fields, "background"):
                value = getattr(self, name)
                if value is None:
                    continue
                value = cp.asarray(value)
                if value.dtype.kind not in "iuf":
                    raise TypeError(f"{name} must be real numeric data")
                if value.shape != shape and (name == "error" or value.ndim != 0):
                    raise ValueError(f"{name} has an incompatible shape")
                if not cp.isfinite(value).all():
                    raise ValueError(f"{name} must be finite")
                if name in ("error", "effective_gain") and (value <= 0).any():
                    raise ValueError(f"{name} must be positive")
                if name == "readnoise" and (value < 0).any():
                    raise ValueError("readnoise must be nonnegative")
                setattr(self, name, value.astype(cp.float64, copy=False))

            if self.mask is not None:
                self.mask = cp.asarray(self.mask, dtype=bool)
                if self.mask.shape != shape:
                    raise ValueError("mask must match the data")

    def noise(self, image=None, mode=None):
        """Given errors, else the noise model evaluated on the working image."""
        if self.error is not None:
            return self.error
        return noise_model(image, self.effective_gain, self.readnoise, mode)
