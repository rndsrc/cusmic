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
    contrast=3,
    cr_threshold=5,
    neighbor_threshold=3,
    error=None,
    mask=None,
    background=None,
    effective_gain=None,
    readnoise=None,
    maxiter=4,
    border_mode="mirror",
):
    """Remove cosmic rays with L.A.Cosmic algorithm; same signature as lacosmic.remove_cosmics()"""

    image = Image(data, error, mask, background, effective_gain, readnoise)
    clean = Cleaner(contrast, cr_threshold, neighbor_threshold, maxiter, border_mode)
    return clean(image)
