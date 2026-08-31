# Install dependencies before copying source so code edits reuse this layer.
FROM python:3.12-slim-trixie AS slim-builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends binutils
RUN python -m venv --without-pip /opt/venv
RUN pip --python /opt/venv/bin/python install --no-compile \
    cupy-cuda12x==14.2.0 'cuda-toolkit[cudart,nvrtc,cccl]==12.9.1'

# Keep runtime headers and the standard NVRTC compiler with its builtins.
RUN rm -f /opt/venv/lib/python*/site-packages/nvidia/cuda_nvrtc/lib/*.alt.so.* && \
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

#==============================================================================
# Final images contain only the Python base and a prepared environment.
FROM python:3.12-slim-trixie AS base

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
# Keep slim last so it is the default build target.
FROM base AS slim
COPY --from=slim-builder /opt/venv /opt/venv
ENTRYPOINT ["python"]
CMD ["-c", "import cusmic; print('cusmic', cusmic.__version__)"]
