GIT_TAG = $(shell git describe --tags --exact-match --match 'v[0-9]*' 2>/dev/null || echo v0.0.0)

PYTHON ?= python3
PYTEST_ARGS ?=
BENCH_ARGS ?=
REFDIR ?= test/data
BUILD ?= build/cuda
GPU_REQUIRED ?= 0
CHECK_PREBUILT ?= 0

export PYTHONPATH := $(CURDIR)/mod:$(PYTHONPATH)
export GPU_REQUIRED
export CHECK_PREBUILT

.PHONY: build check lint unit-test e2e-test unit e2e test mkref bench

build: cuda
	$(PYTHON) -m compileall -q mod/cusmic

check: lint
	sh test/check.sh all "$(PYTHON)" "$(NVCC)" "$(BUILD)" $(PYTEST_ARGS)

lint:
	$(PYTHON) -m ruff check .

unit-test:
	sh test/check.sh unit "$(PYTHON)" "$(NVCC)" "$(BUILD)" $(PYTEST_ARGS)

e2e-test:
	sh test/check.sh e2e "$(PYTHON)" "$(NVCC)" "$(BUILD)" $(PYTEST_ARGS)

unit: unit-test
e2e: e2e-test
test: check

mkref:
	$(PYTHON) test/mkref.py $(REFDIR)

bench: $(if $(filter 1,$(CHECK_PREBUILT)),,$(BUILD)/bench)
	$(PYTHON) -m bench.run $(BENCH_ARGS)

VERSION ?= $(patsubst v%,%,$(GIT_TAG))
CUDA ?= 13
PLATFORM ?= linux/arm64/v8
TARGET ?= all

.PHONY: image
image:
	sh tool/image.sh "$(VERSION)" "$(CUDA)" "$(PLATFORM)" "$(TARGET)"

# CUDA keeps each float64 operation in reference order.
BIN ?= bin
CUDA_PATH ?= /usr/local/cuda
NVCC ?= $(CUDA_PATH)/bin/nvcc
CUDA_ARCH ?= 75
CUDA_ARCHS ?= $(CUDA_ARCH)
NVCCFLAGS ?= -O2
REVISION ?= $(shell git rev-parse HEAD 2>/dev/null || echo unknown)
CUDA_HEADERS = $(wildcard src/*.h src/*.cuh)
CUDA_GENCODE = $(foreach arch,$(CUDA_ARCHS),\
    -gencode arch=compute_$(arch),code=\"sm_$(arch),compute_$(arch)\")
CUDA_FLAGS = -std=c++14 --fmad=false --cudart=static \
    $(CUDA_GENCODE) \
    -Xcompiler=-fPIC,-Wall,-Wextra,-Werror,-ffp-contract=off \
    -DCUSMIC_VERSION='"$(VERSION)"'

.DELETE_ON_ERROR:
.PHONY: cuda
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
	$(CC) -std=c11 $(CFLAGS) $(WARN) $(FITS_CFLAGS) -Isrc -c $< -o $@

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
