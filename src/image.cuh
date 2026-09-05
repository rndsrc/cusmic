/*
 * Copyright 2026 Chi-kwan Chan
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "cusmic.h"
#include "device.cuh"

struct counts {
	int todo, donors;
};

static __global__ void
background(double *clean, const cusmic_image *ims, double scalar, int sign, int n)
{
	clean = frame(clean, n);
	const double *map = ims[blockIdx.y].background;
	if (!map && isnan(scalar))
		return;
	int i = pixel_index();
	if (i < n)
		clean[i] += sign * (map ? map[i] : scalar);
}

static __global__ void
prepare_image(const cusmic_image *ims, double *clean, uint8_t *invalid, uint8_t *excluded,
	counts *count, int n)
{
	clean = frame(clean, n);
	invalid = frame(invalid, n);
	excluded = frame(excluded, n);
	count += blockIdx.y;
	const auto &im = ims[blockIdx.y];
	const double *data = im.data;
	const uint8_t *mask = im.mask;
	int i = pixel_index();
	bool hole = i < n && !isfinite(data[i]);
	bool allowed = i < n && !hole && !(mask && mask[i]);
	if (i < n) {
		invalid[i] = hole;
		excluded[i] = !allowed;
		clean[i] = hole ? 0 : data[i];
	}
	count_block(hole, &count->todo);
	count_block(allowed, &count->donors);
}

static __global__ void
restore_image(const double *clean, const uint8_t *crmask, const cusmic_image *ims, double *out,
	uint8_t *mask, int n)
{
	clean = frame(clean, n);
	crmask = frame(crmask, n);
	out = frame(out, n);
	mask = frame(mask, n);
	const double *data = ims[blockIdx.y].data;
	int i = pixel_index();

	if (i < n) {
		out[i] = isfinite(data[i]) ? clean[i] : data[i];
		mask[i] = crmask[i];
	}
}
