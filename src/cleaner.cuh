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
	double *clean, *med3, *med7, *fine, *lap, *tmp, *sig, *noise;
	uint8_t *crmask, *cand, *grown, *excluded;

	scratch(int w, int h, size_t nf, bool model)
	    : w(w), h(h), n(w * h), total(nf * n), block(256), grid((n - 1) / 256 + 1, nf),
	      replace_grid((n - 1) / 8 + 1, nf), input(nf), pixels((7 + model) * total),
	      masks(4 * total), dcount(nf), hcount(nf), clean(pixels.data), med3(clean + total),
	      med7(med3 + total), fine(med7 + total), lap(fine + total), tmp(lap + total),
	      sig(tmp + total), noise(model ? sig + total : nullptr), crmask(masks.data),
	      cand(crmask + total), grown(cand + total), excluded(grown + total)
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
