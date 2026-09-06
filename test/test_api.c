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
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int
main(void)
{
	struct cusmic_options o;
	double data[] = {1, -0.0, 3, 4};
	double before[4];
	double clean[] = {9, 9, 9, 9};
	uint8_t mask[] = {9, 9, 9, 9};
	char msg[128];
	uint64_t nan_bits = UINT64_C(0x7ff8000000001234);
	struct cusmic_image im = {.data = data, .width = 2, .height = 2};
	memcpy(&data[2], &nan_bits, sizeof(nan_bits));
	memcpy(before, data, sizeof(data));
	cusmic_default_options(&o);
	o.maxiter = 0;
	if (remove_cosmics(&im, &o, clean, mask, msg, sizeof(msg)) ||
		memcmp(clean, before, sizeof(data)) || memcmp(data, before, sizeof(data)) ||
		mask[0] || mask[1] || mask[2] || mask[3]) {
		fprintf(stderr, "zero-iteration copy changed pixels or mask: %s\n", msg);
		return 1;
	}

	o.cr_threshold = NAN;
	for (int i = 0; i < 4; ++i) {
		clean[i] = 9;
		mask[i] = 9;
	}
	if (remove_cosmics(&im, &o, clean, mask, msg, sizeof(msg)) != CUSMIC_INVALID ||
		clean[0] != 9 || clean[1] != 9 || clean[2] != 9 || clean[3] != 9 ||
		mask[0] != 9 || mask[1] != 9 || mask[2] != 9 || mask[3] != 9) {
		fprintf(stderr, "invalid call changed host outputs\n");
		return 1;
	}
	puts("C API preserves special pixels and host outputs on failure");
	return 0;
}
