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
#include <cstdio>
#include <cstring>

#define GPU(call)                                                                                \
	do {                                                                                       \
		cudaError_t err = (call);                                                            \
		if (err != cudaSuccess) {                                                            \
			std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(err));             \
			goto out;                                                                       \
		}                                                                                  \
	} while (0)

__global__ void
delay_upload(void)
{
	unsigned long long start = clock64();
	while (clock64() - start < 10000000) {
	}
}

__global__ void
mark_rays(double *data, int w, int n, int nf)
{
	int f = threadIdx.x;
	if (f < nf)
		data[f * n + (4 + f % 2) * w + 4 + f / 2] = 1000 + 100 * f;
}

int
main(void)
{
	enum { w = 9, h = 9, n = w * h, nf = 3 };
	double data[nf * n], error[n], batch[nf * n], single[n];
	uint8_t flags[nf * n], one_mask[n];
	double *dev_im = nullptr, *dev_error = nullptr, *dev_batch = nullptr, *dev_single = nullptr;
	uint8_t *dev_mask = nullptr, *dev_one_mask = nullptr;
	cudaStream_t producer = nullptr, consumer = nullptr;
	cudaEvent_t ready = nullptr;
	cusmic_image ims[nf] = {};
	cusmic_options o;
	char msg[256];
	int ret = 1;

	for (int i = 0; i < n; ++i)
		error[i] = 1;
	for (int f = 0; f < nf; ++f) {
		for (int i = 0; i < n; ++i)
			data[f * n + i] = 10 + f;
	}
	cusmic_default_options(&o);
	GPU(cudaMalloc((void **)&dev_im, sizeof(data)));
	GPU(cudaMalloc((void **)&dev_error, sizeof(error)));
	GPU(cudaMalloc((void **)&dev_batch, sizeof(batch)));
	GPU(cudaMalloc((void **)&dev_single, sizeof(single)));
	GPU(cudaMalloc((void **)&dev_mask, sizeof(flags)));
	GPU(cudaMalloc((void **)&dev_one_mask, sizeof(one_mask)));
	GPU(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking));
	GPU(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking));
	GPU(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));

	GPU(cudaMemcpy(dev_im, data, sizeof(data), cudaMemcpyHostToDevice));
	GPU(cudaMemcpy(dev_error, error, sizeof(error), cudaMemcpyHostToDevice));
	delay_upload<<<1, 1, 0, producer>>>();
	mark_rays<<<1, nf, 0, producer>>>(dev_im, w, n, nf);
	GPU(cudaGetLastError());
	GPU(cudaEventRecord(ready, producer));
	GPU(cudaStreamWaitEvent(consumer, ready, 0));
	for (int f = 0; f < nf; ++f) {
		ims[f].data = dev_im + f * n;
		ims[f].error = dev_error;
		ims[f].width = w;
		ims[f].height = h;
	}
	if (remove_cosmics_batch_device(ims, nf, &o, dev_batch, dev_mask, consumer, msg,
		sizeof(msg))) {
		std::fprintf(stderr, "batch: %s\n", msg);
		goto out;
	}
	GPU(cudaMemcpy(batch, dev_batch, sizeof(batch), cudaMemcpyDeviceToHost));
	GPU(cudaMemcpy(flags, dev_mask, sizeof(flags), cudaMemcpyDeviceToHost));
	for (int f = 0; f < nf; ++f) {
		if (remove_cosmics_device(&ims[f], &o, dev_single, dev_one_mask, consumer, msg,
			sizeof(msg))) {
			std::fprintf(stderr, "frame %d: %s\n", f, msg);
			goto out;
		}
		GPU(cudaMemcpy(single, dev_single, sizeof(single), cudaMemcpyDeviceToHost));
		GPU(cudaMemcpy(one_mask, dev_one_mask, sizeof(one_mask), cudaMemcpyDeviceToHost));
		int ray = (4 + f % 2) * w + 4 + f / 2;
		if (std::memcmp(batch + f * n, single, sizeof(single)) ||
			std::memcmp(flags + f * n, one_mask, sizeof(one_mask)) || !one_mask[ray]) {
			std::fprintf(stderr, "batch frame %d differs from single call\n", f);
			goto out;
		}
	}
	puts("distinct device frames match independent calls on caller stream");
	ret = 0;
out:
	if (ready)
		cudaEventDestroy(ready);
	if (producer)
		cudaStreamDestroy(producer);
	if (consumer)
		cudaStreamDestroy(consumer);
	cudaFree(dev_im);
	cudaFree(dev_error);
	cudaFree(dev_batch);
	cudaFree(dev_single);
	cudaFree(dev_mask);
	cudaFree(dev_one_mask);
	return ret;
}
