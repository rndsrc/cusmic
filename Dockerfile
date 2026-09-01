# Install dependencies before copying source so code edits reuse this layer.
FROM python:3.13-slim-trixie AS api-builder

ARG VERSION

ARG CUPY=14.2.0
ARG CUDA=13
ARG TK12=12.9.1
ARG TK13=13.0.2

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends binutils
RUN python -m venv --without-pip /opt/venv
RUN case "$CUDA" in \
        12) TK="$TK12" ;; \
        13) TK="$TK13" ;; \
        *) echo "CUDA must be 12 or 13" >&2; exit 1 ;; \
    esac && \
    pip --python /opt/venv/bin/python install --no-compile --only-binary=:all: \
        "cupy-cuda${CUDA}x==${CUPY}" "cuda-toolkit[cudart,nvrtc,cccl]==${TK}"

# Keep NVRTC, its builtins and headers for kernel compilation.
# Stripping wheel-vendored libraries can break their ELF alignment.
RUN find /opt/venv -type f \( -name 'libnvrtc.alt.so*' -o -name '*.a' \) -delete && \
    find /opt/venv -type d \( -name tests -o -name __pycache__ \) -prune -exec rm -rf {} + && \
    find /opt/venv -type f -name '*.so*' ! -path '*.libs/*' -exec strip --strip-unneeded {} +

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY mod/cusmic/ ./mod/cusmic/
RUN SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CUSMIC="$VERSION" \
    pip wheel --no-deps --wheel-dir /wheels .
# Use the installed CUDA-specific CuPy wheel, not the source-only cupy package.
RUN pip --python /opt/venv/bin/python install --no-deps --no-compile /wheels/*.whl

#------------------------------------------------------------------------------
FROM api-builder AS cli-builder

RUN pip --python /opt/venv/bin/python install --prefix /opt/cli --no-compile --only-binary=:all: astropy click
# Astropy imports its test runner at startup.
RUN find /opt/cli -type d \( -name tests -o -name __pycache__ \) \
        ! -path '*/astropy/tests' -prune -exec rm -rf {} + && \
    find /opt/cli -type f -name '*.so*' ! -path '*.libs/*' -exec strip --strip-unneeded {} +

#------------------------------------------------------------------------------
FROM cli-builder AS full-builder

ENV PYTHONPATH=/opt/cli/lib/python3.13/site-packages
RUN pip --python /opt/venv/bin/python install --prefix /opt/full --no-compile --only-binary=:all: \
    'pytest>=8.2' 'lacosmic==1.4.0' matplotlib jupyterlab
# Let ipykernel select the runtime interpreter instead of the builder's venv.
RUN rm -rf /opt/full/share/jupyter/kernels

#==============================================================================
# Only Python and the prepared packages enter the runtime images.
FROM gcr.io/distroless/python3-debian13:latest AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    CUPY_CACHE_DIR=/tmp/cupy \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
COPY --from=api-builder /opt/venv/lib/python3.13/site-packages /usr/local/lib/python3.13/dist-packages

WORKDIR /data

ENTRYPOINT ["/usr/bin/python3"]
CMD ["-c", "import cusmic; print('cusmic', cusmic.__version__)"]

#------------------------------------------------------------------------------
FROM runtime AS cli

COPY --from=cli-builder /opt/cli/lib/python3.13/site-packages /usr/local/lib/python3.13/dist-packages

RUN ["/usr/bin/python3", "-m", "cusmic", "--help"]

ENTRYPOINT ["/usr/bin/python3", "-m", "cusmic"]
CMD ["--help"]

#------------------------------------------------------------------------------
FROM cli AS full

COPY --from=full-builder /opt/full/lib/python3.13/site-packages /usr/local/lib/python3.13/dist-packages
COPY --from=full-builder /opt/full/share/jupyter /usr/share/jupyter
COPY --from=full-builder /opt/full/etc/jupyter /etc/jupyter

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY test/ ./test/
COPY bench/ ./bench/
COPY demo/ ./demo/

ENTRYPOINT ["/usr/bin/python3"]
CMD ["-m", "pytest", "-q", "-rs", "--require-gpu", "-p", "no:cacheprovider"]

#------------------------------------------------------------------------------
# Keep the API image last so it is the default build target.
FROM runtime AS api
