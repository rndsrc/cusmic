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
#ifndef CUSMIC_H
#define CUSMIC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum cusmic_border { CUSMIC_REFLECT, CUSMIC_CONSTANT, CUSMIC_NEAREST, CUSMIC_MIRROR, CUSMIC_WRAP };
enum cusmic_status { CUSMIC_OK, CUSMIC_INVALID, CUSMIC_CUDA_ERROR, CUSMIC_NO_MEMORY };

/* Contiguous float64 pixels and optional per-pixel maps. Error overrides
 * gain and read noise. Mask bytes are zero/nonzero; nonfinite pixels survive.
 */
struct cusmic_image {
	const double *data, *error;
	const double *gain; /* L.A.Cosmic: effective_gain. */
	const double *readnoise, *background;
	const uint8_t *mask;
	size_t width, height;
};

struct cusmic_options {
	double contrast, cr_threshold, neighbor_threshold;
	double gain; /* L.A.Cosmic: effective_gain. */
	double readnoise, background;
	int maxiter;
	int border; /* L.A.Cosmic: border_mode, selected by cusmic_border. */
};

/* Initialize before overriding fields; NaN means no background or unset gain.
 */
const char *
cusmic_version(void);

int
cusmic_cuda_version(void);

void
cusmic_default_options(struct cusmic_options *options);

/* Host arrays are borrowed. Outputs must not overlap inputs or each other.
 * Failures leave host outputs unchanged. Messages are NUL-terminated when
 * message_size > 0; no C++ exception crosses this interface. */
int
remove_cosmics(const struct cusmic_image *image, const struct cusmic_options *options,
	double *clean, uint8_t *mask, char *msg, size_t message_size);

/* Device arrays and stream belong to the caller's current GPU. The caller
 * validates calibration contents and establishes input readiness. stream is
 * cudaStream_t cast to void *, or NULL. Calls wait for completion; device
 * outputs are unspecified on failure. Ownership rules match the host API. */
int
remove_cosmics_device(const struct cusmic_image *image, const struct cusmic_options *options,
	double *clean, uint8_t *mask, void *stream, char *msg, size_t message_size);

/* Host descriptors select 1..65535 same-sized contiguous device frames.
 * Calibration may be shared; outputs are contiguous in descriptor order. */
int
remove_cosmics_batch_device(const struct cusmic_image *ims, size_t nf,
	const struct cusmic_options *options, double *clean, uint8_t *mask, void *stream, char *msg,
	size_t message_size);

#ifdef __cplusplus
}
#endif
#endif
