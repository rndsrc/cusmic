# Install dependencies before copying source so code edits reuse this layer.
FROM python:3.12-slim-trixie AS slim-builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends binutils
RUN python -m venv --without-pip /opt/venv
ARG CUPY_PACKAGE=cupy-cuda12x
ARG CUDA_VERSION=12.9.1
RUN pip --python /opt/venv/bin/python install --no-compile \
    "${CUPY_PACKAGE}==14.2.0" "cuda-toolkit[cudart,nvrtc,cccl]==${CUDA_VERSION}"

# Keep runtime headers and the standard NVRTC compiler with its builtins.
RUN find /opt/venv -type f -name 'libnvrtc.alt.so*' -delete && \
    find /opt/venv -type d \( -name tests -o -name __pycache__ \) -prune -exec rm -rf {} + && \
    find /opt/venv -type f -name '*.so*' -exec strip --strip-unneeded {} + &&\
    find /opt/venv -type f -name '*.a'   -delete

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY mod/cusmic/ ./mod/cusmic/
ARG VERSION
RUN SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CUSMIC="$VERSION" \
    pip wheel --no-deps --wheel-dir /wheels .
# CuPy is already installed as cupy-cuda12x; do not also resolve the cupy package.
RUN pip --python /opt/venv/bin/python install --no-deps --no-compile /wheels/*.whl

#------------------------------------------------------------------------------
FROM slim-builder AS cli-builder
RUN pip --python /opt/venv/bin/python install --no-compile --only-binary=:all: numpy astropy click
# Astropy imports its own test runner at startup; keep that directory.
RUN find /opt/venv -type d \( -name tests -o -name __pycache__ \) \
        ! -path '*/astropy/tests' -prune -exec rm -rf {} + && \
    find /opt/venv -type f -name '*.so*' -exec strip --strip-unneeded {} +
RUN /opt/venv/bin/python -B -m cusmic --help

#------------------------------------------------------------------------------
FROM cli-builder AS test-builder
RUN pip --python /opt/venv/bin/python install --no-compile 'pytest>=8.2' && \
    find /opt/venv -type f -name '*.so*' -exec strip --strip-unneeded {} +

#==============================================================================
# Final images contain only the Python base and a prepared environment.
FROM python:3.12-slim-bookworm AS base

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    CUPY_CACHE_DIR=/tmp/cupy \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /data

#------------------------------------------------------------------------------
FROM base AS cli
COPY --from=cli-builder /opt/venv /opt/venv
ENTRYPOINT ["python", "-m", "cusmic"]
CMD ["--help"]

#------------------------------------------------------------------------------
FROM base AS test
COPY --from=test-builder /opt/venv /opt/venv
WORKDIR /src
COPY pyproject.toml ./
COPY mod/cusmic/ ./mod/cusmic/
COPY test/ ./test/
COPY demo/ ./demo/
ENTRYPOINT ["/bin/sh"]
CMD ["demo/check-gpu.sh"]

#------------------------------------------------------------------------------
# Keep slim last so it is the default build target.
FROM base AS slim
COPY --from=slim-builder /opt/venv /opt/venv
ENTRYPOINT ["python"]
CMD ["-c", "import cusmic; print('cusmic', cusmic.__version__)"]
