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
#include "sort.cuh"

constexpr double FINE_FLOOR = 0.01;
constexpr double SIGNAL_FLOOR = 1e-5;

static __device__ int
border_index(int x, int n, int mode)
{
	if (x >= 0 && x < n)
		return x;
	if (mode == CUSMIC_CONSTANT)
		return -1;
	if (mode == CUSMIC_NEAREST)
		return x < 0 ? 0 : n - 1;
	if (n == 1)
		return 0;
	int period = mode == CUSMIC_REFLECT ? 2 * n : mode == CUSMIC_MIRROR ? 2 * (n - 1) : n;
	x %= period;
	if (x < 0)
		x += period;
	if (mode == CUSMIC_REFLECT && x >= n)
		return period - x - 1;
	if (mode == CUSMIC_MIRROR && x >= n)
		return period - x;
	return x;
}

template <int size>
static __global__ void
median(const double *a, double *out, int w, int h, int mode)
{
	a = frame(a, w * h);
	out = frame(out, w * h);
	int i = pixel_index();
	if (i >= w * h)
		return;
	double values[size * size];
	constexpr int r = size / 2;
	int xs[size], ys[size];
#pragma unroll
	for (int k = 0; k < size; ++k) {
		xs[k] = border_index(i % w + k - r, w, mode);
		ys[k] = border_index(i / w + k - r, h, mode);
	}
#pragma unroll
	for (int y = 0; y < size; ++y)
#pragma unroll
		for (int x = 0; x < size; ++x)
			values[y * size + x] = xs[x] < 0 || ys[y] < 0 ? 0 : a[ys[y] * w + xs[x]];
	sort_window(values);
	out[i] = values[size * size / 2];
}

static __device__ double
supersample(const double *clean, int x, int y, int w, int h, int mode)
{
	x = border_index(x, 2 * w, mode);
	y = border_index(y, 2 * h, mode);
	return x < 0 || y < 0 ? 0 : clean[(y / 2) * w + x / 2] / 4;
}

/* Sample the replicated image without storing it; retain each rounded
 * operation. */
static __device__ double
laplacian_at(const double *clean, int x, int y, int w, int h, int mode)
{
	double v = 0;
	v -= supersample(clean, x, y - 1, w, h, mode);
	v -= supersample(clean, x - 1, y, w, h, mode);
	v += 4 * supersample(clean, x, y, w, h, mode);
	v -= supersample(clean, x + 1, y, w, h, mode);
	v -= supersample(clean, x, y + 1, w, h, mode);
	return v < 0 ? 0 : v;
}

static __global__ void
laplacian(const double *clean, double *lap, int w, int h, int mode)
{
	clean = frame(clean, w * h);
	lap = frame(lap, w * h);
	int i = pixel_index();
	if (i >= w * h)
		return;
	int x = 2 * (i % w), y = 2 * (i / w);
	double a = laplacian_at(clean, x, y, w, h, mode);
	double b = laplacian_at(clean, x + 1, y, w, h, mode);
	double c = laplacian_at(clean, x, y + 1, w, h, mode);
	double d = laplacian_at(clean, x + 1, y + 1, w, h, mode);
	/* The reference sums singleton-width columns from left to right. */
	lap[i] = w == 1 ? ((a + b) + c) + d : (a + b) + (c + d);
}
