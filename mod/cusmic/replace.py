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

RADIUS = 2  # replacement window is 5x5 (paper sec 3.1)
BUDGET = 1 << 18  # Gathered values per batch; at least one window is needed.


def local_median(clean, donors, targets, offsets):
    """Median of each clipped window; also report which windows have donors."""
    ny, nx = clean.shape[-2:]
    *f, y, x = targets.T[:, :, None]
    dy, dx = offsets
    yy, xx = y + dy, x + dx
    inside = (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx)
    yy.clip(0, ny - 1, out=yy)
    xx.clip(0, nx - 1, out=xx)
    at = (*f, yy, xx)
    valid = inside & donors[at]
    values = clean[at]
    cp.copyto(values, clean.dtype.type(cp.inf), where=~valid)
    values.sort(axis=1)  # Donors first, ascending; the +inf padding sorts last.
    count = cp.count_nonzero(valid, axis=1)
    rows  = cp.arange(len(count))
    low, high = values[rows, (count - 1) // 2], values[rows, count // 2]
    return cp.where(count % 2, 0.0 + low, ((0.0 + low) + high) / 2), count > 0


def replace(clean, crmask, donors):
    """Fill targets in place from fixed donors in expanding 5x5 windows."""
    targets = cp.argwhere(crmask)
    if not len(targets):
        return clean
    available = donors.any(axis=(-2, -1))
    if not (available[targets[:, 0]].all() if clean.ndim == 3 else available):
        raise ValueError("no finite replacement donors")
    ny, nx = clean.shape[-2:]
    for r in range(RADIUS, max(RADIUS, ny - 1, nx - 1) + 1):
        if not len(targets):
            break
        ry, rx = min(r, ny - 1), min(r, nx - 1)
        offsets = cp.mgrid[-ry:ry + 1, -rx:rx + 1].reshape(2, -1)
        batch = max(1, BUDGET // offsets.shape[1])
        pending = []
        for i in range(0, len(targets), batch):
            at = targets[i:i + batch]
            median, found = local_median(clean, donors, at, offsets)
            clean[tuple(at[found].T)] = median[found]
            pending.append(at[~found])
        targets = pending[0] if len(pending) == 1 else cp.concatenate(pending)
    return clean
