# =============================================================================
# Meetolog Backend – Production Dockerfile (AWS ECS / Fargate)
#
# Single image for both the FastAPI API server and the background worker.
# The running role is selected at deploy time via the SERVICE_TYPE env var
# (parsed by start.sh) or by overriding CMD in the ECS Task Definition.
#
# Build optimisations
# -------------------
#   - Multi-stage build  (compile wheels in builder → copy into slim runtime)
#   - CPU-only PyTorch   (--extra-index-url → ~3 GB image savings)
#   - Non-root user      (ECS security best-practice)
#   - Whisper model baked into the image (avoids download on cold start)
#
# Runtime considerations
# ----------------------
#   - ffmpeg is required for audio chunking (split_audio_into_chunks)
#   - curl is required for the container-level HEALTHCHECK
#   - All intermediate files use Python tempfile — zero persistent local state
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

# Runtime system deps: ffmpeg (audio chunking) + curl (healthcheck)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (ECS Fargate best practice)
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --create-home appuser

WORKDIR /app

# Install Python packages from pre-compiled wheels
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* \
    && rm -rf /wheels \
    && pip cache purge

# Bake the default Whisper model into the image layer so ECS cold-starts
# never block on a network download.  The model files land inside the
# home directory cache (~/.cache/whisper/).  Switch to appuser first so
# the cached model is owned by the runtime user.
USER appuser
RUN python -c "import whisper; whisper.load_model('tiny')"
USER root

# Copy application source (owned by appuser for runtime writes to /tmp)
COPY --chown=appuser:appuser . .

# Ensure start.sh is executable
RUN chmod +x /app/start.sh

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: start the API server via start.sh (SERVICE_TYPE=web).
# ECS worker task definition overrides this with SERVICE_TYPE=worker.
ENTRYPOINT ["/app/start.sh"]
