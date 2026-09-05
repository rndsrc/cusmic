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
#include <fitsio.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct fits_image {
	fitsfile *file;
	double *data;
	long axes[2];
};

/* Zero-initialize images; close_fits also releases partially read images. */
int
read_fits(const char *path, const char *extension, struct fits_image *im);

void
close_fits(struct fits_image *im);

int
write_fits(const char *path, const struct fits_image *im, const double *clean, const uint8_t *mask);

#ifdef __cplusplus
}
#endif
