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
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <utility>
#include <cstdint>

inline void
cuda_check(cudaError_t status)
{
	if (status == cudaErrorMemoryAllocation)
		throw std::bad_alloc();
	if (status != cudaSuccess)
		throw std::runtime_error(cudaGetErrorString(status));
}

template <class T> struct buffer {
	T *data = nullptr;
	explicit buffer(size_t n)
	{
		if (n > SIZE_MAX / sizeof(T))
			throw std::bad_alloc();
		if (n)
			cuda_check(cudaMalloc(&data, n * sizeof(T)));
	}
	~buffer()
	{
		cudaFree(data);
	}
	buffer(const buffer &) = delete;
	buffer &operator=(const buffer &) = delete;
	buffer(buffer &&other) noexcept : data(std::exchange(other.data, nullptr)) {}
	void
	upload(const T *source, size_t n)
	{
		if (source)
			cuda_check(cudaMemcpy(data, source, n * sizeof(T), cudaMemcpyHostToDevice));
	}
};

struct host_result {
	std::unique_ptr<double[]> clean;
	std::unique_ptr<uint8_t[]> mask;
	/* Every element is written before use. Future: reuse or pin these buffers. */
	explicit host_result(size_t n) : clean(new double[n]), mask(new uint8_t[n]) {}
};

static __device__ int
pixel_index()
{
	return blockIdx.x * blockDim.x + threadIdx.x;
}

/* One grid row per frame; spatial kernels never sample another frame. */
template <class T>
static __device__ T *
frame(T *data, int n)
{
	return data + size_t(blockIdx.y) * n;
}

/* Every thread participates, including inactive lanes in the last block. */
static __device__ void
count_block(bool selected, int *count)
{
	int total = __syncthreads_count(selected);
	if (!threadIdx.x && total)
		atomicAdd(count, total);
}
