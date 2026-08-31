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


"""FITS images on the host; requires the optional Astropy dependency."""

import cupy as cp
from astropy.io import fits


def read_fits(path, dtype=None, *, ext=None):
    """Return native-endian pixels and a header from a FITS image extension."""
    if isinstance(ext, str):
        ext = (ext, 1)
    data, header = fits.getdata(path, ext=ext, header=True, memmap=False)
    if data.dtype.kind not in "iuf":
        raise TypeError(f"{path}: expected real image pixels")
    return data.astype(dtype or data.dtype.newbyteorder("="), copy=False), header


def write_fits(path, data, mask=None, *, header=None, overwrite=False):
    """Write pixels and an optional CRMASK; refuse overwrites by default."""
    hdus = [fits.PrimaryHDU(cp.asnumpy(data), header)]
    if mask is not None:
        hdus.append(fits.ImageHDU(cp.asnumpy(mask).astype("uint8"), name="CRMASK"))
    fits.HDUList(hdus).writeto(path, checksum=True, overwrite=overwrite)
