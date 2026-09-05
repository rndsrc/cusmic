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
#include "cusmic.h"
#include <cuda_runtime.h>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstring>
#include "cleaner.cuh"

#ifndef CUSMIC_VERSION
#define CUSMIC_VERSION "local-dev"
#endif

extern "C" const char *
cusmic_version(void)
{
	return CUSMIC_VERSION;
}

extern "C" int
cusmic_cuda_version(void)
{
	return CUDART_VERSION;
}

extern "C" void
cusmic_default_options(cusmic_options *options)
{
	if (options)
		*options = {3, 5, 3, NAN, 0, NAN, 4, CUSMIC_MIRROR};
}

static size_t
check_arguments(
	const cusmic_image *ims, size_t nf, const cusmic_options *o, double *clean, uint8_t *mask)
{
	if (!ims || !o || !clean || !mask || !nf || nf > 65535)
		throw std::invalid_argument("invalid images, batch size or output buffers");
	size_t w = ims[0].width, h = ims[0].height;
	if (!w || !h || w > size_t(INT_MAX) / 4 / h)
		throw std::invalid_argument("invalid image dimensions");
	size_t n = w * h;
	if (nf > SIZE_MAX / sizeof(double) / n)
		throw std::invalid_argument("batch is too large");
	if (o->maxiter < 0 || o->border < 0 || o->border > CUSMIC_WRAP ||
		!std::isfinite(o->contrast) || o->contrast < 0 || !std::isfinite(o->cr_threshold) ||
		o->cr_threshold < 0 || !std::isfinite(o->neighbor_threshold) ||
		o->neighbor_threshold < 0)
		throw std::invalid_argument("invalid detection settings");

	for (size_t f = 0; f < nf; ++f) {
		const auto &im = ims[f];
		if (!im.data || im.width != w || im.height != h)
			throw std::invalid_argument("frames must have data and the same shape");
		if (!im.error) {
			bool needs_gain = o->maxiter || !std::isnan(o->gain);
			if (!im.gain && needs_gain && (!std::isfinite(o->gain) || o->gain <= 0))
				throw std::invalid_argument("positive gain or error is required");
			if (!im.readnoise && (!std::isfinite(o->readnoise) || o->readnoise < 0))
				throw std::invalid_argument(
					"read noise must be finite and nonnegative");
		}
		if (!im.background && !std::isnan(o->background) && !std::isfinite(o->background))
			throw std::invalid_argument("background must be finite");
	}
	return n;
}

template <class Func>
static int
guarded(Func run, char *msg, size_t len)
{
	if (msg && len)
		msg[0] = 0;
	try {
		run();
		return CUSMIC_OK;
	} catch (const std::invalid_argument &e) {
		if (msg && len)
			std::snprintf(msg, len, "%s", e.what());
		return CUSMIC_INVALID;
	} catch (const std::bad_alloc &e) {
		if (msg && len)
			std::snprintf(msg, len, "%s", e.what());
		return CUSMIC_NO_MEMORY;
	} catch (const std::exception &e) {
		if (msg && len)
			std::snprintf(msg, len, "%s", e.what());
		return CUSMIC_CUDA_ERROR;
	}
}

static void
check_values(const double *data, size_t n, const char *name, int minimum)
{
	for (size_t i = 0; data && i < n; ++i)
		if (!std::isfinite(data[i]) || (minimum == 1 && data[i] <= 0) ||
			(minimum == 0 && data[i] < 0))
			throw std::invalid_argument(name);
}

static host_result
clean_host(const cusmic_image &im, const cusmic_options &o, size_t n)
{
	host_result out(n);
	if (!o.maxiter) {
		std::memcpy(out.clean.get(), im.data, n * sizeof(double));
		std::memset(out.mask.get(), 0, n);
		if (im.background || !std::isnan(o.background))
			for (size_t i = 0; i < n; ++i) {
				if (!std::isfinite(im.data[i]))
					continue;
				double bg = im.background ? im.background[i] : o.background;
				out.clean[i] += bg;
				out.clean[i] -= bg;
			}
	} else {
		const double *gmap = im.error ? nullptr : im.gain;
		const double *rnmap = im.error ? nullptr : im.readnoise;
		buffer<double> data(n), error(im.error ? n : 0), gain(gmap ? n : 0);
		buffer<double> readnoise(rnmap ? n : 0), bg(im.background ? n : 0), clean(n);
		buffer<uint8_t> excluded(im.mask ? n : 0), crmask(n);

		data.upload(im.data, n);
		error.upload(im.error, n);
		gain.upload(gmap, n);
		readnoise.upload(rnmap, n);
		bg.upload(im.background, n);
		excluded.upload(im.mask, n);

		cusmic_image gpu = {data.data, error.data, gain.data, readnoise.data, bg.data,
			excluded.data, im.width, im.height};
		clean_images(&gpu, 1, o, clean.data, crmask.data, nullptr);
		cuda_check(cudaMemcpy(
			out.clean.get(), clean.data, n * sizeof(double), cudaMemcpyDeviceToHost));
		cuda_check(cudaMemcpy(out.mask.get(), crmask.data, n, cudaMemcpyDeviceToHost));
	}
	return out;
}

extern "C" int
remove_cosmics(const cusmic_image *im, const cusmic_options *o, double *clean, uint8_t *mask,
	char *msg, size_t len)
{
	return guarded([&] {
		size_t n = check_arguments(im, 1, o, clean, mask);
		if (im->error)
			check_values(im->error, n, "error must be finite and positive", 1);
		else {
			check_values(im->gain, n, "gain must be finite and positive", 1);
			check_values(
				im->readnoise, n, "read noise must be finite and nonnegative", 0);
		}
		check_values(im->background, n, "background must be finite", -1);

		auto out = clean_host(*im, *o, n);

		std::memcpy(clean, out.clean.get(), n * sizeof(double));
		std::memcpy(mask, out.mask.get(), n);
	}, msg, len);
}

extern "C" int
remove_cosmics_device(const cusmic_image *im, const cusmic_options *o, double *clean, uint8_t *mask,
	void *stream, char *msg, size_t len)
{
	return remove_cosmics_batch_device(im, 1, o, clean, mask, stream, msg, len);
}

extern "C" int
remove_cosmics_batch_device(const cusmic_image *ims, size_t nf, const cusmic_options *o,
	double *clean, uint8_t *mask, void *stream, char *msg, size_t len)
{
	return guarded([&] {
		check_arguments(ims, nf, o, clean, mask);
		clean_images(ims, nf, *o, clean, mask, (cudaStream_t)stream);
	}, msg, len);
}
