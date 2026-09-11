# check=skip=InvalidDefaultArgInFrom
# Bake selects the CUDA toolkit.
ARG	CUDA_TOOLKIT

#------------------------------------------------------------------------------
# The toolkit is build-only. The host supplies the NVIDIA driver.
FROM	nvidia/cuda:${CUDA_TOOLKIT}-devel-ubuntu22.04 AS toolkit

#==============================================================================
# Compile CUDA C/C++ on Bookworm, matching the runtime's C and FITS libraries.
FROM	debian:bookworm-slim AS cuda-builder
ARG	VERSION
ARG	CUDA_ARCHS

RUN	apt-get update &&\
	apt-get install -y --no-install-recommends gcc-11 g++-11 make libcfitsio-dev &&\
	rm -rf /var/lib/apt/lists/*
COPY --from=toolkit	/usr/local/cuda/ /usr/local/cuda/

WORKDIR	/src
COPY	Makefile ./
COPY	src/ ./src/
COPY	test/test_api.c test/test_io.c test/test_reference.c test/test_batch.cu ./test/
COPY	bench/bench.cu ./bench/
RUN	make cuda build/cuda/test_api build/cuda/test_io build/cuda/test_batch build/cuda/test_reference build/cuda/bench \
	CC=/usr/bin/gcc-11 NVCC="/usr/local/cuda/bin/nvcc -ccbin=/usr/bin/g++-11" VERSION="$VERSION" CUDA_ARCHS="$CUDA_ARCHS"
RUN	./build/cuda/test_io

#------------------------------------------------------------------------------
# CuPy uses the same pinned toolkit version through CUDA runtime/NVRTC wheels.
FROM	python:3.13-slim-bookworm AS cupy-builder
ARG	VERSION
ARG	CUDA_MAJOR
ARG	CUDA_TOOLKIT
ARG	CUPY_VERSION

ENV	PIP_NO_CACHE_DIR=1 \
	PIP_DISABLE_PIP_VERSION_CHECK=1
RUN	python -m venv /opt/venv
RUN	/opt/venv/bin/python -m pip install --no-compile --only-binary=:all: \
	"cupy-cuda${CUDA_MAJOR}x==${CUPY_VERSION}" \
	"cuda-toolkit[cudart,nvrtc,cccl]==${CUDA_TOOLKIT}"

WORKDIR	/src
COPY	pyproject.toml README.md LICENSE ./
COPY	mod/cusmic/ ./mod/cusmic/
RUN	SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CUSMIC="$VERSION" \
	/opt/venv/bin/python -m pip install --no-deps --no-compile .

#------------------------------------------------------------------------------
FROM	cupy-builder AS cupy-cli-builder

RUN	/opt/venv/bin/python -m pip install --no-compile --only-binary=:all: \
	numpy astropy click

#------------------------------------------------------------------------------
FROM	cupy-cli-builder AS full-builder

RUN	/opt/venv/bin/python -m pip install --no-compile --only-binary=:all: \
	'pytest>=8.2' 'lacosmic==1.4.0' matplotlib jupyterlab

#==============================================================================
# Both runtime families are ordinary Bookworm images.
FROM	python:3.13-slim-bookworm AS cupy-runtime
ARG	REVISION

ENV	NVIDIA_DRIVER_CAPABILITIES=compute,utility \
	PYTHONDONTWRITEBYTECODE=1 \
	CUPY_CACHE_DIR=/tmp/cupy \
	CUSMIC_REVISION=$REVISION \
	PATH=/opt/venv/bin:$PATH

WORKDIR	/data

#------------------------------------------------------------------------------
FROM	cupy-runtime AS cupy-api

COPY --from=cupy-builder	/opt/venv/ /opt/venv/
CMD	["python", "-c", "import cusmic; print('cusmic', cusmic.__version__)"]

#------------------------------------------------------------------------------
FROM	cupy-runtime AS cupy-cli

COPY --from=cupy-cli-builder	/opt/venv/ /opt/venv/
RUN	cupysmic --help

ENTRYPOINT	["cupysmic"]
CMD	["--help"]

#------------------------------------------------------------------------------
FROM	debian:bookworm-slim AS cuda-api
ARG	REVISION

RUN	apt-get update &&\
	apt-get install -y --no-install-recommends libstdc++6 &&\
	rm -rf /var/lib/apt/lists/*
COPY --from=cuda-builder	/src/build/cuda/libcusmic.so /usr/local/lib/
COPY --from=cuda-builder	/src/build/cuda/libcusmic.a /usr/local/lib/
COPY	src/cusmic.h /usr/local/include/
COPY	LICENSE /usr/share/licenses/cusmic/LICENSE
RUN	ldconfig

ENV	NVIDIA_DRIVER_CAPABILITIES=compute,utility \
	CUSMIC_REVISION=$REVISION

WORKDIR	/data
CMD	["/bin/true"]

#------------------------------------------------------------------------------
FROM	cuda-api AS cuda-cli

RUN	apt-get update &&\
	apt-get install -y --no-install-recommends libcfitsio10 &&\
	rm -rf /var/lib/apt/lists/*
COPY --from=cuda-builder	/src/bin/cudasmic /usr/local/bin/

ENV	PATH=/usr/local/bin/:$PATH
RUN	cudasmic --help

ENTRYPOINT	["cudasmic"]
CMD	["--help"]

#------------------------------------------------------------------------------
# The full image runs exact checks and matched GPU benchmarks.
FROM	cupy-runtime AS full

COPY --from=full-builder	/opt/venv/ /opt/venv/
RUN	apt-get update &&\
	apt-get install -y --no-install-recommends libstdc++6 libcfitsio10 &&\
	rm -rf /var/lib/apt/lists/*

WORKDIR	/src
COPY	pyproject.toml ./
COPY	test/ ./test/
COPY	bench/ ./bench/
COPY	demo/ ./demo/
COPY	tool/report.sh ./tool/report.sh
COPY --from=cuda-builder	/src/build/cuda/ ./build/cuda/
COPY --from=cuda-builder	/src/bin/ ./bin/
COPY	src/cusmic.h /usr/local/include/
COPY	LICENSE /usr/share/licenses/cusmic/LICENSE

ENV	LD_LIBRARY_PATH=/src/build/cuda:/usr/local/lib \
	PATH=/src/bin:$PATH \
	CHECK_PREBUILT=1
ENTRYPOINT	["sh", "/src/tool/report.sh"]
