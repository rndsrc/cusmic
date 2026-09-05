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
#include "image.cuh"

static __device__ bool
is_donor(const double *clean, const uint8_t *crmask, const uint8_t *excluded, int i)
{
	return !crmask[i] && !excluded[i] && isfinite(clean[i]);
}

struct window {
	int x0, x1, y0, y1;
};

static __device__ window
region(int x, int y, int r, int w, int h)
{
	return {max(0, x - r), min(w - 1, x + r), max(0, y - r), min(h - 1, y + r)};
}

/* Ordered IEEE keys let large windows select a median without dynamic storage.
 */
static __device__ unsigned long long
key(double v)
{
	auto bits = (unsigned long long)__double_as_longlong(v);
	return bits >> 63 ? ~bits : bits ^ (1ull << 63);
}

static __device__ double
value(unsigned long long k)
{
	return __longlong_as_double((long long)(k >> 63 ? k ^ (1ull << 63) : ~k));
}

/* Each replacement target owns a complete warp, including inactive lanes. */
static __device__ int
warp_sum(int count)
{
#pragma unroll
	for (int step = 16; step; step /= 2)
		count += __shfl_down_sync(0xffffffff, count, step);
	return __shfl_sync(0xffffffff, count, 0);
}

template <class T>
static __device__ T
sort_warp(T v)
{
	int lane = threadIdx.x & 31;
#pragma unroll
	for (int span = 2; span <= 32; span *= 2) {
#pragma unroll
		for (int step = span / 2; step; step /= 2) {
			T peer = __shfl_xor_sync(0xffffffff, v, step);
			bool lower = ((lane & span) == 0) == ((lane & step) == 0);
			v = lower ? (peer < v ? peer : v) : (v < peer ? peer : v);
		}
	}
	return v;
}

static __device__ double
select_donor(const double *clean, const uint8_t *crmask, const uint8_t *excluded, window win, int w,
	int rank)
{
	int ww = win.x1 - win.x0 + 1, area = ww * (win.y1 - win.y0 + 1);
	unsigned long long prefix = 0, used = 0;
	for (int bit = 63; bit >= 0; --bit) {
		unsigned long long flag = 1ull << bit;
		int below = 0;
		for (int pos = threadIdx.x & 31; pos < area; pos += 32) {
			int i = (win.y0 + pos / ww) * w + win.x0 + pos % ww;
			if (!is_donor(clean, crmask, excluded, i))
				continue;
			auto k = key(clean[i]);
			below += (k & used) == prefix && !(k & flag);
		}
		below = warp_sum(below);
		used |= flag;
		if (rank >= below) {
			prefix |= flag;
			rank -= below;
		}
	}
	return value(prefix);
}
