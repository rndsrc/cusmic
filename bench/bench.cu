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
#include "device.cuh"
#include "io.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>

using clock_type = std::chrono::steady_clock;

struct fits_pixels {
	fits_image im = {};

	explicit fits_pixels(const char *path, const char *ext = nullptr)
	{
		int status = read_fits(path, ext, &im);
		if (status) {
			char msg[FLEN_STATUS];
			fits_get_errstatus(status, msg);
			close_fits(&im);
			throw std::runtime_error(msg);
		}
	}
	~fits_pixels() { close_fits(&im); }
	fits_pixels(const fits_pixels &) = delete;
	fits_pixels &operator=(const fits_pixels &) = delete;
};

struct reference_check {
	bool exact = true;
	double max_abs_error = 0;
};

struct scene {
	fits_pixels input, error, expected, flags;
	cusmic_options options;
	size_t w, h, n;
	const char *mode;

	scene(const char *path, const char *noise, const char *reference)
	    : input(path), error(noise), expected(reference), flags(reference, "CRMASK"),
	      w(input.im.axes[0]), h(input.im.axes[1]), n(w * h)
	{
		mode = std::getenv("CUSMIC_REFERENCE");
		if (!mode)
			mode = "exact";
		if (std::strcmp(mode, "exact") && std::strcmp(mode, "close"))
			throw std::invalid_argument("CUSMIC_REFERENCE must be exact or close");
		for (const auto *im : {&error.im, &expected.im, &flags.im})
			if (im->axes[0] != long(w) || im->axes[1] != long(h))
				throw std::invalid_argument("reference shapes differ");
		cusmic_default_options(&options);
		options.contrast = 1;
		options.cr_threshold = 5;
		options.neighbor_threshold = 5;
		options.maxiter = 4;
	}

	std::vector<double>
	stack(int nf) const
	{
		std::vector<double> pixels(n * nf);
		for (int f = 0; f < nf; ++f)
			std::copy_n(input.im.data, n, pixels.data() + size_t(f) * n);
		return pixels;
	}

	reference_check
	check(const host_result &out, int nf, const host_result *first = nullptr) const
	{
		reference_check result;
		const double eps = 32 * std::numeric_limits<double>::epsilon();
		for (int f = 0; f < nf; ++f)
			for (size_t i = 0; i < n; ++i) {
				size_t at = size_t(f) * n + i;
				uint64_t got, want;
				std::memcpy(&got, out.clean.get() + at, sizeof(got));
				std::memcpy(&want, expected.im.data + i, sizeof(want));
				unsigned got_mask = out.mask[at];
				unsigned want_mask = unsigned(flags.im.data[i]);
				if (first && (std::memcmp(out.clean.get() + at, first->clean.get() + at,
						sizeof(double)) || got_mask != first->mask[at])) {
					char msg[100];
					std::snprintf(msg, sizeof(msg),
						"repeated output differs at frame %d, y %zu, x %zu",
						f, i / w, i % w);
					throw std::runtime_error(msg);
				}
				if (got == want && got_mask == want_mask)
					continue;
				result.exact = false;
				double actual = out.clean[at], expected_value = expected.im.data[i];
				double difference = std::abs(actual - expected_value);
				bool close = got == want ||
					(std::isfinite(actual) && std::isfinite(expected_value) &&
					 difference <= eps * (1 + std::abs(expected_value)));
				if (got_mask != want_mask || !close || !std::strcmp(mode, "exact")) {
					char msg[240];
					std::snprintf(msg, sizeof(msg),
						"%s reference, frame %d, y %zu, x %zu: pixel %.17g/%.17g "
						"(0x%016llx/0x%016llx), mask %u/%u",
						mode, f, i / w, i % w, actual, expected_value,
						(unsigned long long)got, (unsigned long long)want,
						got_mask, want_mask);
					throw std::runtime_error(msg);
				}
				if (std::isfinite(difference))
					result.max_abs_error = std::max(result.max_abs_error, difference);
			}
		return result;
	}
};

struct device_input {
	buffer<double> pixels, error;
	std::vector<cusmic_image> ims;

	device_input(const std::vector<double> &stack, const scene &ref, int nf)
	    : pixels(stack.size()), error(ref.n), ims(nf)
	{
		pixels.upload(stack.data(), stack.size());
		error.upload(ref.error.im.data, ref.n);
		for (int f = 0; f < nf; ++f)
			ims[f] = {pixels.data + size_t(f) * ref.n, error.data, nullptr, nullptr,
				nullptr, nullptr, ref.w, ref.h};
	}
};

struct device_result {
	buffer<double> clean;
	buffer<uint8_t> mask;
	explicit device_result(size_t n) : clean(n), mask(n) {}
};

static device_result
clean(const device_input &input, const cusmic_options &options, size_t total)
{
	device_result out(total);
	char msg[256];
	int status = remove_cosmics_batch_device(input.ims.data(), input.ims.size(), &options,
		out.clean.data, out.mask.data, nullptr, msg, sizeof(msg));
	if (status)
		throw std::runtime_error(msg);
	return out;
}

static host_result
download(const device_result &out, size_t n)
{
	host_result host(n);
	cuda_check(cudaMemcpy(host.clean.get(), out.clean.data, n * sizeof(double),
		cudaMemcpyDeviceToHost));
	cuda_check(cudaMemcpy(host.mask.get(), out.mask.data, n, cudaMemcpyDeviceToHost));
	return host;
}

template <class F>
static auto
timed(F call, double &ms) -> decltype(call())
{
	cuda_check(cudaStreamSynchronize(nullptr));
	auto start = clock_type::now();
	auto out = call();
	cuda_check(cudaStreamSynchronize(nullptr));
	ms = std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
	return out;
}

struct measurements {
	double first_ms;
	size_t detected;
	bool reference_exact;
	double max_abs_error;
	std::vector<double> upload_ms, clean_ms, download_ms, total_ms;
};

static measurements
benchmark(const scene &ref, int nf, int warmups, int repeats)
{
	auto stack = ref.stack(nf);
	size_t total = stack.size();
	measurements times;
	auto complete = [&] {
		device_input input(stack, ref, nf);
		return download(clean(input, ref.options, total), total);
	};

	auto start = clock_type::now();
	auto first = complete();
	times.first_ms = std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
	auto quality = ref.check(first, nf);
	times.reference_exact = quality.exact;
	times.max_abs_error = quality.max_abs_error;
	for (int i = 0; i < warmups; ++i)
		complete();

	double ms;
	for (int i = 0; i < repeats; ++i) {
		auto input = timed([&] { return device_input(stack, ref, nf); }, ms);
		times.upload_ms.push_back(ms);
	}

	device_input resident(stack, ref, nf);
	for (int i = 0; i < warmups; ++i)
		timed([&] { return clean(resident, ref.options, total); }, ms);
	for (int i = 0; i < repeats; ++i) {
		auto out = timed([&] { return clean(resident, ref.options, total); }, ms);
		times.clean_ms.push_back(ms);
		auto host = timed([&] { return download(out, total); }, ms);
		times.download_ms.push_back(ms);
		ref.check(host, nf, &first);
	}

	for (int i = 0; i < repeats; ++i) {
		auto host = timed(complete, ms);
		times.total_ms.push_back(ms);
		ref.check(host, nf, &first);
	}
	times.detected = std::count(first.mask.get(), first.mask.get() + total, uint8_t(1));
	return times;
}

static double
median(std::vector<double> values)
{
	std::sort(values.begin(), values.end());
	size_t n = values.size();
	return (values[(n - 1) / 2] + values[n / 2]) / 2;
}

static std::string
statistics(const std::vector<double> &samples)
{
	std::ostringstream out;
	out << std::setprecision(12) << "{\"median\":" << median(samples)
	    << ",\"minimum\":" << *std::min_element(samples.begin(), samples.end())
	    << ",\"maximum\":" << *std::max_element(samples.begin(), samples.end())
	    << ",\"samples\":[";
	for (size_t i = 0; i < samples.size(); ++i)
		out << (i ? "," : "") << samples[i];
	return out.str() + "]}";
}

static std::string
json_string(const char *text)
{
	std::ostringstream out;
	out << '"';
	for (const unsigned char *p = (const unsigned char *)text; *p; ++p) {
		if (*p == '"' || *p == '\\')
			out << '\\' << *p;
		else if (*p < 32)
			out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
			    << unsigned(*p);
		else
			out << *p;
	}
	return out.str() + '"';
}

static std::string
report(const scene &ref, const measurements &times, int nf, int warmups, int repeats)
{
	int device, runtime, driver;
	cudaDeviceProp gpu;
	cuda_check(cudaGetDevice(&device));
	cuda_check(cudaGetDeviceProperties(&gpu, device));
	cuda_check(cudaRuntimeGetVersion(&runtime));
	cuda_check(cudaDriverGetVersion(&driver));
	const char *revision = std::getenv("CUSMIC_REVISION");
	const auto &o = ref.options;
	std::ostringstream out;
	out << std::setprecision(12) << "{\"backend\":\"cuda\",\"dtype\":\"float64\",\"shape\":[";
	if (nf > 1)
		out << nf << ',';
	out << ref.h << ',' << ref.w << "],\"warmups\":" << warmups << ",\"repeats\":" << repeats
	    << ",\"first_result_ms\":" << times.first_ms
	    << ",\"host_memory\":\"pageable\",\"allocation\":\"ordinary calls\","
	       "\"detected_pixels\":" << times.detected << ",\"settings\":{\"contrast\":"
	    << o.contrast << ",\"cr_threshold\":" << o.cr_threshold
	    << ",\"neighbor_threshold\":" << o.neighbor_threshold << ",\"maxiter\":"
	    << o.maxiter << "},\"source_revision\":"
	    << json_string(revision ? revision : "unknown") << ",\"cusmic\":"
	    << json_string(cusmic_version()) << ",\"gpu\":" << json_string(gpu.name)
	    << ",\"cuda_runtime\":" << runtime << ",\"cuda_driver\":" << driver
	    << ",\"reference_mode\":" << json_string(ref.mode)
	    << ",\"reference_close\":true,\"reference_exact\":"
	    << (times.reference_exact ? "true" : "false")
	    << ",\"max_abs_error\":" << times.max_abs_error
	    << ",\"mask_disagreements\":0,\"milliseconds\":{"
	    << "\"upload_ms\":" << statistics(times.upload_ms)
	    << ",\"clean_ms\":" << statistics(times.clean_ms)
	    << ",\"download_ms\":" << statistics(times.download_ms)
	    << ",\"total_ms\":" << statistics(times.total_ms)
	    << "},\"frames_per_second\":" << 1000 * nf / median(times.total_ms) << '}';
	return out.str();
}

int
main(int argc, char **argv)
{
	const char *input = "test/data/input.fits.gz", *error = "test/data/error.fits.gz";
	const char *reference = "test/data/reference.fits.gz", *output = nullptr;
	int nf = 1, warmups = 4, repeats = 16;
	static const option flags[] = {{"input", required_argument, nullptr, 'i'},
		{"error", required_argument, nullptr, 'e'},
		{"reference", required_argument, nullptr, 'r'},
		{"frames", required_argument, nullptr, 'f'},
		{"warmups", required_argument, nullptr, 'w'},
		{"repeats", required_argument, nullptr, 'n'},
		{"output", required_argument, nullptr, 'o'},
		{"help", no_argument, nullptr, 'h'}, {nullptr, 0, nullptr, 0}};
	try {
		int opt;
		while ((opt = getopt_long(argc, argv, "", flags, nullptr)) != -1) {
			if (opt == 'h') {
				puts("Usage: bench [--input FITS] [--error FITS] [--reference FITS] "
				     "[--frames N] [--warmups N] [--repeats N] [--output JSONL]");
				return 0;
			}
			if (opt == 'i' || opt == 'e' || opt == 'r' || opt == 'o') {
				if (opt == 'i') input = optarg;
				if (opt == 'e') error = optarg;
				if (opt == 'r') reference = optarg;
				if (opt == 'o') output = optarg;
				continue;
			}
			if (opt == '?')
				return 1;
			char *end;
			long count = std::strtol(optarg, &end, 10);
			if (end == optarg || *end || count < 1 || count > 65535)
				throw std::invalid_argument("counts must be integers in 1..65535");
			if (opt == 'f') nf = int(count);
			if (opt == 'w') warmups = int(count);
			if (opt == 'n') repeats = int(count);
		}
		if (optind != argc)
			throw std::invalid_argument("unexpected argument");

		scene ref(input, error, reference);
		auto times = benchmark(ref, nf, warmups, repeats);
		auto line = report(ref, times, nf, warmups, repeats);
		puts(line.c_str());
		if (output) {
			std::ofstream file(output, std::ios::app);
			if (!file || !(file << line << '\n'))
				throw std::runtime_error("cannot write benchmark output");
		}
		return 0;
	} catch (const std::exception &e) {
		std::fprintf(stderr, "%s\n", e.what());
		return 1;
	}
}
