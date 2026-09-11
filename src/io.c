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
	double scale = 1, zero = 0, null = NAN;

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
	fits_read_key(im->file, TDOUBLE, "BSCALE", &scale, NULL, &status);
	if (status == KEY_NO_EXIST)
		status = 0;
	fits_read_key(im->file, TDOUBLE, "BZERO", &zero, NULL, &status);
	if (status == KEY_NO_EXIST)
		status = 0;
	if (status)
		return status;
	fits_set_bscale(im->file, 1, 0, &status);
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
	if (status)
		return status;
	if (scale != 1 || zero != 0) {
		for (long i = 0; i < w * h; ++i) {
			if (type > 0 && isnan(im->data[i]))
				continue;
			double value = im->data[i] * scale;
			im->data[i] = value + zero;
		}
	}
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

int
write_fits(const char *path, const struct fits_image *im, const double *clean, const uint8_t *mask)
{
	fitsfile *out = NULL;
	int status = 0, close_status = 0;
	long axes[] = {im->axes[0], im->axes[1]}, n = axes[0] * axes[1];
	const char *keys[] = {"BSCALE", "BZERO", "BLANK"};

	/* Use literal disk paths; never accept CFITSIO's overwrite prefix. */
	if (!path[0] || path[0] == '!')
		return FILE_NOT_CREATED;
	fits_create_diskfile(&out, path, &status);
	if (status)
		return status;
	fits_copy_header(im->file, out, &status);
	fits_resize_img(out, DOUBLE_IMG, 2, axes, &status);
	for (unsigned i = 0; i < sizeof(keys) / sizeof(keys[0]) && !status; ++i) {
		fits_delete_key(out, keys[i], &status);
		if (status == KEY_NO_EXIST)
			status = 0;
	}
	fits_set_bscale(out, 1, 0, &status);
	fits_write_img(out, TDOUBLE, 1, n, (void *)clean, &status);
	fits_write_history(out, "Cosmic rays removed with cusmic", &status);
	fits_write_chksum(out, &status);
	fits_create_img(out, BYTE_IMG, 2, axes, &status);
	fits_update_key(out, TSTRING, "EXTNAME", "CRMASK", NULL, &status);
	fits_write_img(out, TBYTE, 1, n, (void *)mask, &status);
	fits_write_chksum(out, &status);
	if (status)
		fits_delete_file(out, &close_status);
	else
		fits_close_file(out, &close_status);
	return status ? status : close_status;
}
