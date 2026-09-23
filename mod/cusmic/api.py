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


from .cleaner import Cleaner
from .image import Image


def remove_cosmics(
    data,
    contrast,
    cr_threshold,
    neighbor_threshold,
    error=None,
    mask=None,
    background=None,
    effective_gain=None,
    readnoise=None,
    maxiter=4,
    border_mode="mirror",
):
    """Return cleaned float64 pixels and a boolean cosmic-ray mask on the GPU.

    ``data`` is a nonempty float64 frame (height, width) or stack
    (frames, height, width). NumPy inputs are uploaded; CuPy views are borrowed.
    Frames are cleaned independently and outputs do not alias inputs.

    Supply a finite, positive 1-sigma ``error`` map in image units, or both
    ``effective_gain`` (electrons per image unit, positive) and ``readnoise``
    (electrons, nonnegative). Error overrides gain and read noise. Maps may
    match data or one shared frame; gain and read noise also accept scalars.

    ``contrast`` rejects fine structure; ``cr_threshold`` and
    ``neighbor_threshold`` set detection and growth thresholds in sigma.
    All three must be finite and nonnegative. ``mask`` excludes seed and
    donor pixels, but neighboring detections may grow into masked pixels.
    Nonfinite input pixels are preserved and never flagged.

    ``background`` is a finite scalar or map in image units, added before
    cleaning and subtracted afterward. ``maxiter`` is nonnegative; zero
    disables detection and needs no noise model, but still applies background
    arithmetic. Border modes are mirror, reflect, nearest, wrap, and constant
    (zero padding).

    Work uses the input device's current stream. Keep borrowed arrays unchanged
    until that stream finishes, and establish readiness across streams yourself.
    A replacement with no finite, unmasked donors raises ValueError.
    """

    image = Image(data, error, mask, background, effective_gain, readnoise)
    clean = Cleaner(contrast, cr_threshold, neighbor_threshold, maxiter, border_mode)
    return clean(image)
