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
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static void
help(void)
{
	puts("Usage: cusmic SOURCE OUTPUT [OPTIONS]\n"
	     "  --error FILE             Positive 1-sigma errors; overrides "
	     "gain/read noise\n"
	     "  --gain NUMBER            Electrons/ADU; required without "
	     "errors\n"
	     "  --readnoise NUMBER       Electrons (default: 0)\n"
	     "  --contrast NUMBER        Star rejection contrast (default: 3)\n"
	     "  --cr-threshold NUMBER    Cosmic threshold (default: 5)\n"
	     "  --neighbor-threshold N   Growth threshold (default: 3)\n"
	     "  --maxiter INTEGER        Maximum iterations (default: 4)\n"
	     "  --help                   Show this help\n\n"
	     "Reads one 2D image as float64; writes cleaned pixels and CRMASK.\n"
	     "Existing outputs are never overwritten.");
}

static void
parse_options(int argc, char **argv, struct cusmic_options *o, const char **error_path)
{
	static const struct option flags[] = {{"error", required_argument, NULL, 'e'},
		{"gain", required_argument, NULL, 'g'}, {"readnoise", required_argument, NULL, 'r'},
		{"contrast", required_argument, NULL, 'c'},
		{"cr-threshold", required_argument, NULL, 't'},
		{"neighbor-threshold", required_argument, NULL, 'u'},
		{"maxiter", required_argument, NULL, 'n'}, {"help", no_argument, NULL, 'h'},
		{NULL, 0, NULL, 0}};
	int opt;

	while ((opt = getopt_long(argc, argv, "", flags, NULL)) != -1) {
		if (opt == 'h') {
			help();
			exit(0);
		}
		if (opt == 'e') {
			*error_path = optarg;
			continue;
		}
		if (opt == '?')
			exit(2);
		char *end;
		errno = 0;
		double v = opt == 'n' ? strtol(optarg, &end, 10) : strtod(optarg, &end);
		if (errno || end == optarg || *end || !isfinite(v) || v < 0 ||
			(opt == 'g' && v == 0) || (opt == 'n' && v > INT_MAX)) {
			fprintf(stderr, "Invalid value: %s\n", optarg);
			exit(2);
		}
		switch (opt) {
		case 'g':
			o->gain = v;
			break;
		case 'r':
			o->readnoise = v;
			break;
		case 'c':
			o->contrast = v;
			break;
		case 't':
			o->cr_threshold = v;
			break;
		case 'u':
			o->neighbor_threshold = v;
			break;
		case 'n':
			o->maxiter = (int)v;
			break;
		}
	}
}

int
main(int argc, char **argv)
{
	struct cusmic_options o;
	struct cusmic_image im = {0};
	struct fits_image input = {0}, errors = {0};
	double *clean = NULL;
	uint8_t *mask = NULL;
	const char *error_path = NULL, *output_path;
	int status = 0, ret = 1;
	char msg[512];

	cusmic_default_options(&o);
	parse_options(argc, argv, &o, &error_path);
	if (argc - optind != 2) {
		help();
		return 2;
	}
	if (o.maxiter && !error_path && isnan(o.gain)) {
		fprintf(stderr, "Provide --error or --gain (optionally --readnoise)\n");
		return 2;
	}

	output_path = argv[optind + 1];
	if (output_path[0] == '!' || access(output_path, F_OK) == 0) {
		fprintf(stderr, "Choose a new output filename.\n");
		return 1;
	}

	status = read_fits(argv[optind], NULL, &input);
	if (status)
		goto out;
	if (error_path) {
		status = read_fits(error_path, NULL, &errors);
		if (status)
			goto out;
		if (input.axes[0] != errors.axes[0] || input.axes[1] != errors.axes[1]) {
			fprintf(stderr, "Error image must match the input shape.\n");
			goto out;
		}
	}

	long n = input.axes[0] * input.axes[1];
	clean = malloc((size_t)n * sizeof(double));
	mask = malloc((size_t)n);
	if (!clean || !mask) {
		status = MEMORY_ALLOCATION;
		goto out;
	}

	im.data = input.data;
	im.error = errors.data;
	im.width = (size_t)input.axes[0];
	im.height = (size_t)input.axes[1];
	if (remove_cosmics(&im, &o, clean, mask, msg, sizeof(msg))) {
		fprintf(stderr, "cusmic: %s\n", msg);
		goto out;
	}
	status = write_fits(output_path, &input, clean, mask);
	ret = status != 0;
	if (!ret) {
		size_t count = 0;
		for (long i = 0; i < n; ++i)
			count += mask[i] != 0;
		printf("Saved %s (%zu cosmic-ray pixels)\n", output_path, count);
	}
out:
	if (status)
		fits_report_error(stderr, status);
	close_fits(&input);
	close_fits(&errors);
	free(clean);
	free(mask);
	return ret;
}
