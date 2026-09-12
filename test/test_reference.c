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
#include "io.h"
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int
main(int argc, char **argv)
{
	const char *dir = argc > 1 ? argv[1] : "test/data";
	const char *names[] = {"input.fits.gz", "error.fits.gz", "reference.fits.gz"};
	struct fits_image input = {0}, error = {0}, reference = {0}, flags = {0};
	struct fits_image *files[] = {&input, &error, &reference};
	struct cusmic_options o;
	struct cusmic_image im = {0};
	double *clean = NULL;
	uint8_t *mask = NULL;
	char path[1024], msg[256];
	int ret = 1, status;

	for (size_t i = 0; i < 3; ++i) {
		if (snprintf(path, sizeof(path), "%s/%s", dir, names[i]) >= (int)sizeof(path))
			goto out;
		status = read_fits(path, NULL, files[i]);
		if (status) {
			fits_report_error(stderr, status);
			goto out;
		}
	}
	if (snprintf(path, sizeof(path), "%s/reference.fits.gz", dir) >= (int)sizeof(path))
		goto out;
	status = read_fits(path, "CRMASK", &flags);
	if (status) {
		fits_report_error(stderr, status);
		goto out;
	}
	if (input.axes[0] != error.axes[0] || input.axes[1] != error.axes[1] ||
		input.axes[0] != reference.axes[0] || input.axes[1] != reference.axes[1] ||
		input.axes[0] != flags.axes[0] || input.axes[1] != flags.axes[1])
		goto out;

	size_t w = (size_t)input.axes[0], n = w * (size_t)input.axes[1];
	clean = malloc(n * sizeof(*clean));
	mask = malloc(n);
	if (!clean || !mask)
		goto out;
	im.data = input.data;
	im.error = error.data;
	im.width = w;
	im.height = (size_t)input.axes[1];
	cusmic_default_options(&o);
	o.contrast = 1;
	o.cr_threshold = o.neighbor_threshold = 5;
	if (remove_cosmics(&im, &o, clean, mask, msg, sizeof(msg))) {
		fprintf(stderr, "remove_cosmics: %s\n", msg);
		goto out;
	}

	for (size_t i = 0; i < n; ++i) {
		uint64_t got, want;
		memcpy(&got, clean + i, sizeof(got));
		memcpy(&want, reference.data + i, sizeof(want));
		double expected = reference.data[i];
		double tolerance = 32 * DBL_EPSILON * (1 + fabs(expected));
		if ((got != want && !(isfinite(clean[i]) && isfinite(expected) &&
			fabs(clean[i] - expected) <= tolerance)) ||
			mask[i] != (uint8_t)flags.data[i]) {
			fprintf(stderr, "reference differs at (%zu,%zu): pixels %.17g/%.17g"
				" (%016" PRIx64 "/%016" PRIx64 "), mask %u/%.0f\n",
				i % w, i / w, clean[i], expected, got, want, mask[i], flags.data[i]);
			goto out;
		}
	}
	puts("CUDA reference pixels are close and masks match exactly");
	ret = 0;
out:
	close_fits(&input);
	close_fits(&error);
	close_fits(&reference);
	close_fits(&flags);
	free(clean);
	free(mask);
	return ret;
}
