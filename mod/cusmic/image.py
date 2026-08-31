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

from cupy import ndarray as Array

from .filters import noise_model


@dataclass
class Image:
    """Pixels and calibration data"""

    data:           Array
    error:          Array | None = None
    mask:           Array | None = None
    background:     Array | None = None
    effective_gain: Array | None = None
    readnoise:      float | None = None

    def noise(self, image=None, mode=None):
        """Given errors, else the noise model evaluated on `clean`"""
        return noise_model(image, self.effective_gain, self.readnoise, mode) if self.error is None else self.error
