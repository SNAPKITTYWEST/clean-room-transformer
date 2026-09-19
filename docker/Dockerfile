# Reference image for running the runtime with no egress path.
#
#   build (needs network, once):
#     docker build -f docker/Dockerfile -t clean-room-transformer .
#
#   run (no network, read-only root, non-root user, no new privileges):
#     docker run --rm \
#       --network none \
#       --read-only \
#       --tmpfs /tmp:rw,noexec,nosuid,size=16m \
#       --cap-drop ALL \
#       --security-opt no-new-privileges \
#       --pids-limit 128 \
#       --memory 1g \
#       -v "$PWD/audit:/audit" \
#       clean-room-transformer python3 scripts/verification_report.py
#
# The build stage resolves dependencies; the runtime stage has no pip, no
# compiler and no package index configured, so nothing can be fetched after the
# image is sealed. `--network none` is what actually removes the network -- the
# code being free of socket imports (tests/test_isolation.py) is a separate,
# weaker guarantee, and neither one substitutes for the other.

FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .

# --require-hashes makes the resolver refuse anything but the pinned artifacts.
RUN pip install --no-cache-dir --require-hashes --no-deps \
        --target /deps -r requirements.txt


FROM python:3.11-slim AS runtime

# Remove the package manager from the sealed image.
RUN pip uninstall -y pip setuptools wheel 2>/dev/null || true; \
    rm -rf /usr/local/lib/python3.11/ensurepip \
           /usr/local/bin/pip* /usr/local/bin/easy_install*

ENV PYTHONPATH=/app/src:/deps \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# Single-threaded BLAS is not a performance choice: multi-threaded reductions
# sum in a nondeterministic order, which breaks bit-exact reproducibility.

COPY --from=builder /deps /deps
WORKDIR /app
COPY src ./src
COPY tests ./tests
COPY scripts ./scripts
COPY conftest.py requirements.txt requirements-dev.txt ./

RUN useradd --create-home --uid 10001 runtime \
    && mkdir -p /audit && chown runtime:runtime /audit
USER runtime

# The audit log is the only thing the container should ever write.
VOLUME ["/audit"]

CMD ["python3", "scripts/verification_report.py"]
