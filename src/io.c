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
#include "io.h"
#include <limits.h>
#include <math.h>
#include <stdlib.h>

int
read_fits(const char *path, const char *extension, struct fits_image *im)
{
	int status = 0, ndim = 0, type = 0, anynull;
	double null = NAN;

	fits_open_diskfile(&im->file, path, READONLY, &status);
	if (extension)
		fits_movnam_hdu(im->file, IMAGE_HDU, (char *)extension, 0, &status);
	fits_get_img_dim(im->file, &ndim, &status);
	if (!status && !ndim && !extension) {
		fits_movabs_hdu(im->file, 2, NULL, &status);
		fits_get_img_dim(im->file, &ndim, &status);
	}
	if (status)
		return status;
	if (ndim != 2)
		return BAD_DIMEN;
	fits_get_img_size(im->file, 2, im->axes, &status);
	fits_get_img_type(im->file, &type, &status);
	if (status)
		return status;
	long w = im->axes[0], h = im->axes[1];
	if (w <= 0 || h <= 0 || w > INT_MAX / 4 / h)
		return BAD_DIMEN;
	im->data = malloc((size_t)(w * h) * sizeof(double));
	if (!im->data)
		return MEMORY_ALLOCATION;
	/* Null substitution is for integer BLANK values, not IEEE pixels. */
	fits_read_img(
		im->file, TDOUBLE, 1, w * h, type > 0 ? &null : NULL, im->data, &anynull, &status);
	return status;
}

void
close_fits(struct fits_image *im)
{
	int status = 0;
	if (im->file)
		fits_close_file(im->file, &status);
	free(im->data);
	*im = (struct fits_image){0};
}
