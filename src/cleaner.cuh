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
#include "filters.cuh"
#include "replace.cuh"
#include <vector>

static __global__ void
accumulate(const double *clean, const uint8_t *cand, uint8_t *crmask, const uint8_t *excluded,
	counts *count, int n)
{
	clean = frame(clean, n);
	cand = frame(cand, n);
	crmask = frame(crmask, n);
	excluded = frame(excluded, n);
	count += blockIdx.y;
	int i = pixel_index();
	bool fresh = i < n && cand[i] && !crmask[i];
	if (i < n)
		crmask[i] |= cand[i];
	count_block(fresh, &count->todo);
	count_block(i < n && is_donor(clean, crmask, excluded, i), &count->donors);
}

struct scratch {
	int w, h, n;
	size_t total;
	dim3 block, grid, replace_grid;
	buffer<cusmic_image> input;
	buffer<double> pixels;
	buffer<uint8_t> masks;
	buffer<counts> dcount;
	std::vector<counts> hcount;
	/* lap ends at snr; fine replaces med3 per pixel; med7 replaces tmp. */
	double *clean, *med3, *med7, *fine, *lap, *tmp, *sig, *noise;
	uint8_t *crmask, *cand, *grown, *excluded;

	scratch(int w, int h, size_t nf, bool model, double *output, uint8_t *mask)
	    : w(w), h(h), n(w * h), total(nf * n), block(256), grid((n - 1) / 256 + 1, nf),
	      replace_grid((n - 1) / 8 + 1, nf), input(nf), pixels((3 + model) * total),
	      masks(3 * total), dcount(nf), hcount(nf), clean(output), med3(pixels.data),
	      med7(med3 + total), fine(med3), lap(med3), tmp(med7),
	      sig(tmp + total), noise(model ? sig + total : nullptr), crmask(mask),
	      cand(masks.data), grown(cand + total), excluded(grown + total)
	{
	}
};

inline void
clear_counts(scratch &s, cudaStream_t stream)
{
	cuda_check(cudaMemsetAsync(s.dcount.data, 0, s.hcount.size() * sizeof(counts), stream));
}

inline void
read_counts(scratch &s, cudaStream_t stream)
{
	cuda_check(cudaGetLastError());
	cuda_check(cudaMemcpyAsync(s.hcount.data(), s.dcount.data, s.hcount.size() * sizeof(counts),
		cudaMemcpyDeviceToHost, stream));
	cuda_check(cudaStreamSynchronize(stream));
}

inline void
fill_holes(scratch &s, const cusmic_options &o, bool has_bg, cudaStream_t stream)
{
	bool fill = false;
	for (const auto &c : s.hcount)
		fill |= c.todo && c.donors;
	if (!o.maxiter || !fill)
		return;

	/* Background addition can overflow otherwise valid donors. */
	if (has_bg) {
		auto before = s.hcount;
		clear_counts(s, stream);
		accumulate<<<s.grid, s.block, 0, stream>>>(
			s.clean, s.crmask, s.crmask, s.excluded, s.dcount.data, s.n);
		read_counts(s, stream);
		for (size_t f = 0; f < s.hcount.size(); ++f)
			if (before[f].todo && before[f].donors && !s.hcount[f].donors)
				throw std::invalid_argument("no finite replacement donors");
	}
	replace<<<s.replace_grid, s.block, 0, stream>>>(
		s.clean, s.crmask, s.excluded, s.dcount.data, s.w, s.h);
}

inline void
find_cosmics(scratch &s, const cusmic_options &o, cudaStream_t stream)
{
	laplacian<<<s.grid, s.block, 0, stream>>>(s.clean, s.lap, s.w, s.h, o.border);
	if (s.noise) {
		median<5><<<s.grid, s.block, 0, stream>>>(s.clean, s.tmp, s.w, s.h, o.border);
		noise_model<<<s.grid, s.block, 0, stream>>>(
			s.tmp, s.input.data, o.gain, o.readnoise, s.noise, s.n);
	}

	snr<<<s.grid, s.block, 0, stream>>>(s.lap, s.input.data, s.noise, s.sig, s.n);
	median<5><<<s.grid, s.block, 0, stream>>>(s.sig, s.tmp, s.w, s.h, o.border);
	significance<<<s.grid, s.block, 0, stream>>>(s.tmp, s.input.data, s.sig, s.n);

	median<3><<<s.grid, s.block, 0, stream>>>(s.clean, s.med3, s.w, s.h, o.border);
	median<7><<<s.grid, s.block, 0, stream>>>(s.med3, s.med7, s.w, s.h, o.border);
	fine_structure<<<s.grid, s.block, 0, stream>>>(
		s.med3, s.med7, s.input.data, s.noise, s.fine, s.n);

	detect<<<s.grid, s.block, 0, stream>>>(
		s.sig, s.fine, s.excluded, s.cand, o.contrast, o.cr_threshold, s.n);
	grow<<<s.grid, s.block, 0, stream>>>(s.cand, s.sig, s.grown, o.cr_threshold, s.w, s.h);
	grow<<<s.grid, s.block, 0, stream>>>(
		s.grown, s.sig, s.cand, o.neighbor_threshold, s.w, s.h);
}

inline bool
update_mask(scratch &s, cudaStream_t stream)
{
	clear_counts(s, stream);
	accumulate<<<s.grid, s.block, 0, stream>>>(
		s.clean, s.cand, s.crmask, s.excluded, s.dcount.data, s.n);
	read_counts(s, stream);
	bool changed = false;
	for (const auto &c : s.hcount) {
		if (c.todo && !c.donors)
			throw std::invalid_argument("no finite replacement donors");
		changed |= c.todo != 0;
	}
	return changed;
}

inline void
clean_images(const cusmic_image *ims, size_t nf, const cusmic_options &o, double *output,
	uint8_t *mask, cudaStream_t stream)
{
	bool model = false, has_bg = !std::isnan(o.background);
	for (size_t f = 0; f < nf; ++f) {
		model |= !ims[f].error;
		has_bg |= ims[f].background != nullptr;
	}
	scratch s(ims[0].width, ims[0].height, nf, model, output, mask);
	cuda_check(cudaMemcpyAsync(
		s.input.data, ims, nf * sizeof(*ims), cudaMemcpyHostToDevice, stream));
	clear_counts(s, stream);
	prepare_image<<<s.grid, s.block, 0, stream>>>(
		s.input.data, s.clean, s.crmask, s.excluded, s.dcount.data, o.background, s.n);
	read_counts(s, stream);
	fill_holes(s, o, has_bg, stream);
	cuda_check(cudaMemsetAsync(s.crmask, 0, s.total, stream));

	for (int i = 0; i < o.maxiter; ++i) {
		find_cosmics(s, o, stream);
		if (!update_mask(s, stream))
			break;
		replace<<<s.replace_grid, s.block, 0, stream>>>(
			s.clean, s.crmask, s.excluded, s.dcount.data, s.w, s.h);
	}

	restore_image<<<s.grid, s.block, 0, stream>>>(
		s.clean, s.crmask, s.input.data, output, mask, o.background, s.n);
	cuda_check(cudaGetLastError());
	cuda_check(cudaStreamSynchronize(stream));
}
