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
