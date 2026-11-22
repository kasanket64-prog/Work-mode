# syntax=docker/dockerfile:1

FROM python:3.12-slim AS builder
ARG EXTRAS="video"
ENV PIP_NO_CACHE_DIR=1
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && if [ -n "${EXTRAS}" ]; then \
         /opt/venv/bin/pip install ".[${EXTRAS}]"; \
       else \
         /opt/venv/bin/pip install .; \
       fi

FROM python:3.12-slim AS runtime-base
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/hf \
    YOLO_CONFIG_DIR=/cache/yolo \
    FIELDNOTE_PORT=8780
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system fieldnote \
    && useradd --system --gid fieldnote --home-dir /app --shell /usr/sbin/nologin fieldnote \
    && mkdir -p /cache/hf /cache/yolo /data/workspace \
    && chown -R fieldnote:fieldnote /app /cache /data
COPY --from=builder /opt/venv /opt/venv
COPY configs ./configs
COPY examples ./examples
COPY pyproject.toml README.md ./
RUN chown -R fieldnote:fieldnote /app /opt/venv
EXPOSE 8780
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; port = os.environ.get('FIELDNOTE_PORT', '8780'); token = os.environ.get('FIELDNOTE_AUTH_TOKEN', ''); req = urllib.request.Request(f'http://127.0.0.1:{port}/api/system/status', headers={'Authorization': f'Bearer {token}'} if token else {}); urllib.request.urlopen(req, timeout=4).close()"]

FROM runtime-base AS cuda
ARG PYTORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu124"
USER root
RUN pip install --no-cache-dir --index-url "${PYTORCH_CUDA_INDEX}" torch torchvision
USER fieldnote
ENTRYPOINT ["sh", "-c", "if [ -n \"${FIELDNOTE_AUTH_TOKEN:-}\" ]; then exec camera-workspace --host 0.0.0.0 --port \"${FIELDNOTE_PORT:-8780}\" --data /data/workspace --warm-worker --strict-uploads --auth-token-env FIELDNOTE_AUTH_TOKEN \"$@\"; else exec camera-workspace --host 127.0.0.1 --port \"${FIELDNOTE_PORT:-8780}\" --data /data/workspace --warm-worker --strict-uploads \"$@\"; fi", "--"]
CMD []

FROM runtime-base AS cpu
USER fieldnote
ENTRYPOINT ["sh", "-c", "if [ -n \"${FIELDNOTE_AUTH_TOKEN:-}\" ]; then exec camera-workspace --host 0.0.0.0 --port \"${FIELDNOTE_PORT:-8780}\" --data /data/workspace --warm-worker --strict-uploads --auth-token-env FIELDNOTE_AUTH_TOKEN \"$@\"; else exec camera-workspace --host 127.0.0.1 --port \"${FIELDNOTE_PORT:-8780}\" --data /data/workspace --warm-worker --strict-uploads \"$@\"; fi", "--"]
CMD []
