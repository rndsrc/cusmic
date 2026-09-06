GIT_TAG = $(shell git describe --tags --exact-match --match 'v[0-9]*' 2>/dev/null || echo v0.0.0)

PYTHON ?= python3
PYTEST_ARGS ?=
BENCH_ARGS ?=
REFDIR ?= test/data

export PYTHONPATH := $(CURDIR)/mod:$(PYTHONPATH)

.PHONY: check unit e2e test mkref bench

check:
	$(PYTHON) -m ruff check .

unit:
	$(PYTHON) -m pytest -q -rs -m 'not e2e' $(PYTEST_ARGS)

e2e:
	$(PYTHON) -m pytest -q -rs -m e2e $(PYTEST_ARGS)

test:
	$(PYTHON) -m pytest -q -rs $(PYTEST_ARGS)

mkref:
	$(PYTHON) test/mkref.py $(REFDIR)

bench:
	$(PYTHON) -m bench.bench $(BENCH_ARGS)

VERSION ?= $(patsubst v%,%,$(GIT_TAG))
CUDA ?= 13
PLATFORM ?= linux/arm64/v8
TARGET ?= api
TAG ?= $(VERSION)$(if $(filter full,$(TARGET)),,-$(TARGET))$(if $(filter 13,$(CUDA)),,-cuda$(CUDA))
IMAGE = rndsrc/$(if $(filter full,$(TARGET)),cusmic,cupysmic):$(TAG)

.PHONY: image
image:
	docker buildx build --load --platform $(PLATFORM) --target $(TARGET) \
	    --build-arg VERSION=$(VERSION) --build-arg CUDA=$(CUDA) -t $(IMAGE) .

# CUDA keeps each float64 operation in reference order.
BUILD ?= build/cuda
BIN ?= bin
CUDA_PATH ?= /usr/local/cuda
NVCC ?= $(CUDA_PATH)/bin/nvcc
CUDA_ARCH ?= 75
NVCCFLAGS ?= -O2
REVISION ?= $(shell git rev-parse HEAD 2>/dev/null || echo unknown)
CUDA_HEADERS = $(wildcard src/*.h src/*.cuh)
CUDA_FLAGS = -std=c++14 --fmad=false --cudart=static \
    -gencode arch=compute_$(CUDA_ARCH),code=\"sm_$(CUDA_ARCH),compute_$(CUDA_ARCH)\" \
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
