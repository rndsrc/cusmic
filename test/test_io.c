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
#define _POSIX_C_SOURCE 200809L
#include "io.h"
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int
main(void)
{
	char path[] = "/tmp/cusmic-io-XXXXXX";
	struct fits_image raw = {0}, scaled = {0}, unsigned_im = {0};
	fitsfile *out = NULL;
	long axes[] = {2, 2}, blank = -32768;
	short counts[] = {1, -32768, 2, 3};
	double data[] = {-0.0, INFINITY, 0, 7};
	double bscale = 0.1, bzero = 10.2, unsigned_zero = 32768;
	uint64_t nan_bits = UINT64_C(0x7ff8000000001234);
	const uint64_t expected[] = {UINT64_C(0x4024999999999999),
		UINT64_C(0x7ff8000000000000), UINT64_C(0x4024cccccccccccc),
		UINT64_C(0x4025000000000000)};
	int fd, status = 0, close_status = 0, ret = 1;

	fd = mkstemp(path);
	if (fd < 0)
		return 1;
	close(fd);
	unlink(path);
	memcpy(&data[2], &nan_bits, sizeof(nan_bits));
	fits_create_diskfile(&out, path, &status);
	if (status)
		goto out;
	fits_create_img(out, DOUBLE_IMG, 2, axes, &status);
	fits_write_img(out, TDOUBLE, 1, 4, data, &status);
	fits_create_img(out, SHORT_IMG, 2, axes, &status);
	fits_update_key(out, TSTRING, "EXTNAME", "SCALED", NULL, &status);
	fits_write_img(out, TSHORT, 1, 4, counts, &status);
	fits_update_key(out, TDOUBLE, "BSCALE", &bscale, NULL, &status);
	fits_update_key(out, TDOUBLE, "BZERO", &bzero, NULL, &status);
	fits_update_key(out, TLONG, "BLANK", &blank, NULL, &status);
	fits_create_img(out, SHORT_IMG, 2, axes, &status);
	fits_update_key(out, TSTRING, "EXTNAME", "UNSIGNED", NULL, &status);
	fits_write_img(out, TSHORT, 1, 4, counts, &status);
	fits_update_key(out, TDOUBLE, "BZERO", &unsigned_zero, NULL, &status);
	fits_update_key(out, TLONG, "BLANK", &blank, NULL, &status);
	fits_close_file(out, &close_status);
	out = NULL;
	if (status || close_status || read_fits(path, NULL, &raw) ||
		read_fits(path, "SCALED", &scaled) ||
		read_fits(path, "UNSIGNED", &unsigned_im)) {
		fprintf(stderr, "could not write or read generated FITS case\n");
		goto out;
	}
	if (memcmp(raw.data, data, sizeof(data)) ||
		!isnan(unsigned_im.data[1]) || unsigned_im.data[0] != 32769 ||
		unsigned_im.data[2] != 32770 || unsigned_im.data[3] != 32771) {
		fprintf(stderr, "FITS special pixels, scaling or BLANK differ\n");
		goto out;
	}
	for (size_t i = 0; i < 4; ++i) {
		uint64_t bits;
		memcpy(&bits, scaled.data + i, sizeof(bits));
		if (bits != expected[i]) {
			fprintf(stderr, "scaled FITS pixel %zu has bits %016" PRIx64
				", expected %016" PRIx64 "\n", i, bits, expected[i]);
			goto out;
		}
	}
	puts("C FITS read preserves special pixels, scaling and BLANK");
	ret = 0;
out:
	if (out)
		fits_close_file(out, &close_status);
	close_fits(&raw);
	close_fits(&scaled);
	close_fits(&unsigned_im);
	unlink(path);
	return ret;
}
