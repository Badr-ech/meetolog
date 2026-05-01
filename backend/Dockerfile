# =============================================================================
# Meetolog Backend – Production Dockerfile (AWS ECS / Fargate)
#
# Multi-stage build that produces a single image used for both the API and
# the worker. The running role is chosen at deploy time via the SERVICE_TYPE
# environment variable (see start.sh).
#
# Runtime requirements: ffmpeg (audio chunking) + curl (HEALTHCHECK).
# The container holds zero persistent local state — all intermediate files
# go through Python's tempfile module.
# =============================================================================

# ── Stage 1: Wheel builder ──────────────────────────────────────────────────
FROM python:3.12-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# ── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --create-home appuser

WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* \
    && rm -rf /wheels \
    && pip cache purge

COPY --chown=appuser:appuser . .
RUN chmod +x /app/start.sh

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/app/start.sh"]
