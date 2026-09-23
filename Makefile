GIT_TAG = $(shell git describe --tags --exact-match --match 'v[0-9]*' 2>/dev/null || echo v0.0.0.dev0)

.DEFAULT_GOAL := help

PYTHON ?= python3
PYTEST_ARGS ?=
BENCH_ARGS ?=
REFERENCE ?= exact
REFDIR ?= test/data
BUILD ?= build/cuda
CHECK_PREBUILT ?= 0

export PYTHONPATH := $(CURDIR)/mod:$(PYTHONPATH)
export CUSMIC_REFERENCE := $(REFERENCE)
export CHECK_PREBUILT

.PHONY: help build cuda check lint unit e2e ref bench container clean

help:
	@printf '%s\n' \
	    'Usage: make [target] [VARIABLE=value ...]' \
	    '' \
	    'Targets:' \
	    '  help        Show this help (default)' \
	    '  build       Compile the CUDA library and CLI; check Python syntax' \
	    '  lint        Check Python style with Ruff' \
	    '  check       Run lint, unit, and end-to-end checks' \
	    '  unit        Run Python and C/CUDA unit checks' \
	    '  e2e         Compare APIs and CLIs with reference images' \
	    '  bench       Benchmark CPU L.A.Cosmic, CuPy, and CUDA' \
	    '  ref         Generate reference FITS images in test/data/' \
	    '  container   Build all five container roles for CUDA 13' \
	    '  clean       Remove builds, benchmark results, and caches' \
	    '' \
	    'Options:' \
	    '  REFERENCE=exact|close   Pixel policy for checks/benchmarks (default: exact)' \
	    '  PYTEST_ARGS="..."      Extra pytest arguments for check/unit/e2e' \
	    '  BENCH_ARGS="..."       Benchmark sizes, repetitions, and output directory' \
	    '  REFDIR=PATH            Reference directory (default: test/data); no overwrite' \
	    '  CUDA_ARCHS="87 121"    GPU code targets for local or container builds' \
	    '  CUDA=12               Container CUDA profile (default: 13)' \
	    '  TARGET=full           Container role (default: all)' \
	    '  VERSION=0.0.0.dev0     Version (default: exact Git tag or 0.0.0.dev0)' \
	    '  PLATFORM=linux/amd64   Container platform (default: linux/arm64/v8)' \
	    '' \
	    'Examples:' \
	    '  make check REFERENCE=exact' \
	    '  make container TARGET=full VERSION=0.0.0.dev0' \
	    '' \
	    'See README.md, test/README.md, and bench/README.md for prerequisites.'

build: cuda
	$(PYTHON) -m compileall -q mod/cusmic

check:
	@status=0; \
	$(PYTHON) -m ruff check . || status=1; \
	sh test/check.sh all "$(PYTHON)" "$(NVCC)" "$(BUILD)" $(PYTEST_ARGS) || status=1; \
	exit $$status

lint:
	$(PYTHON) -m ruff check .

unit:
	sh test/check.sh unit "$(PYTHON)" "$(NVCC)" "$(BUILD)" $(PYTEST_ARGS)

e2e:
	sh test/check.sh e2e "$(PYTHON)" "$(NVCC)" "$(BUILD)" $(PYTEST_ARGS)

ref:
	$(PYTHON) test/mkref.py "$(REFDIR)"

bench:
	@sh tool/bench.sh "$(PYTHON)" "$(NVCC)" "$(BUILD)" $(BENCH_ARGS)

VERSION ?= $(patsubst v%,%,$(GIT_TAG))
CUDA ?= 13
PLATFORM ?= linux/arm64/v8
TARGET ?= all
CUDA_ARCHS ?=

container:
	sh tool/image.sh "$(VERSION)" "$(CUDA)" "$(PLATFORM)" "$(TARGET)" "$(CUDA_ARCHS)"

clean:
	sh tool/clean.sh

# CUDA keeps each float64 operation in reference order.
BIN ?= bin
CUDA_PATH ?= /usr/local/cuda
NVCC ?= $(CUDA_PATH)/bin/nvcc
CUDA_ARCH ?= 75
NVCCFLAGS ?= -O2
REVISION ?= $(shell git rev-parse HEAD 2>/dev/null || echo unknown)
CUDA_HEADERS = $(wildcard src/*.h src/*.cuh)
CUDA_GENCODE = $(foreach arch,$(or $(CUDA_ARCHS),$(CUDA_ARCH)),\
    -gencode arch=compute_$(arch),code=\"sm_$(arch),compute_$(arch)\")
CUDA_FLAGS = -std=c++14 --fmad=false --cudart=static \
    $(CUDA_GENCODE) \
    -Xcompiler=-fPIC,-Wall,-Wextra,-Werror,-ffp-contract=off \
    -DCUSMIC_VERSION='"$(VERSION)"'

.DELETE_ON_ERROR:
cuda: $(BUILD)/libcusmic.so $(BUILD)/libcusmic.a $(BIN)/cudasmic

$(BUILD):
	mkdir -p $@

$(BUILD)/api.o: src/api.cu $(CUDA_HEADERS) Makefile | $(BUILD)
	$(NVCC) $(CUDA_FLAGS) $(NVCCFLAGS) -Isrc -c $< -o $@

$(BUILD)/libcusmic.so: $(BUILD)/api.o
	$(NVCC) --shared --cudart=static $< -o $@
	strip --strip-unneeded $@

$(BUILD)/libcusmic.a: $(BUILD)/api.o
	$(AR) rcs $@ $<

# CFITSIO is used by the command, not the cleaning library.
CFLAGS ?= -O2
FITS_CFLAGS ?=
FITS_LIBS ?= -lcfitsio
WARN = -Wall -Wextra -Werror

$(BUILD)/io.o: src/io.c src/io.h Makefile | $(BUILD)
	$(CC) -std=c11 $(CFLAGS) -ffp-contract=off $(WARN) $(FITS_CFLAGS) -Isrc -c $< -o $@

$(BIN):
	mkdir -p $@

$(BIN)/cudasmic: src/main.c src/cusmic.h src/io.h $(BUILD)/io.o $(BUILD)/libcusmic.so | $(BIN)
	$(CC) -std=c11 $(CFLAGS) $(WARN) $(FITS_CFLAGS) -Isrc $< $(BUILD)/io.o \
	    -L$(BUILD) -lcusmic $(FITS_LIBS) -lm -Wl,-rpath,'$$ORIGIN/../$(BUILD)' -o $@
	strip --strip-unneeded $@

# Compare the C API with the saved float64 reference pixels and mask.
$(BUILD)/test_reference: test/test_reference.c src/cusmic.h src/io.h \
    $(BUILD)/io.o $(BUILD)/libcusmic.so
	$(CC) -std=c11 $(CFLAGS) $(WARN) $(FITS_CFLAGS) -Isrc $< $(BUILD)/io.o \
	    -L$(BUILD) -lcusmic $(FITS_LIBS) -lm -Wl,-rpath,'$$ORIGIN' -o $@

.PHONY: cuda-reference-check
cuda-reference-check: $(BUILD)/test_reference
	$(BUILD)/test_reference

# Invalid host calls must leave caller-owned outputs unchanged.
$(BUILD)/test_api: test/test_api.c src/cusmic.h $(BUILD)/libcusmic.so
	$(CC) -std=c11 $(CFLAGS) $(WARN) -Isrc $< -L$(BUILD) -lcusmic -lm \
	    -Wl,-rpath,'$$ORIGIN' -o $@

# Generated FITS cases need no CUDA device or saved fixture.
$(BUILD)/test_io: test/test_io.c src/io.h $(BUILD)/io.o
	$(CC) -std=c11 $(CFLAGS) $(WARN) $(FITS_CFLAGS) -Isrc $< \
	    $(BUILD)/io.o $(FITS_LIBS) -lm -o $@

# Distinct frames and caller-stream handoff use the public device C API.
$(BUILD)/test_batch: test/test_batch.cu src/cusmic.h $(CUDA_HEADERS) \
    $(BUILD)/libcusmic.so
	$(NVCC) $(CUDA_FLAGS) $(NVCCFLAGS) -Isrc $< -L$(BUILD) -lcusmic \
	    -Xlinker=-rpath,'$$ORIGIN' -o $@

# The CUDA benchmark uses the same FITS scene and timing fields as bench.py.
$(BUILD)/bench: bench/bench.cu src/cusmic.h src/io.h $(CUDA_HEADERS) \
    $(BUILD)/io.o $(BUILD)/libcusmic.so
	$(NVCC) $(CUDA_FLAGS) $(NVCCFLAGS) -Isrc $(FITS_CFLAGS) $< \
	    $(BUILD)/io.o -L$(BUILD) -lcusmic $(FITS_LIBS) \
	    -Xlinker=-rpath,'$$ORIGIN' -o $@
