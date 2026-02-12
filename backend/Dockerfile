# =============================================================================
# Meetolog Backend – Production Dockerfile (Optimized for Render)
# Supports both the FastAPI web server and the ARQ background worker
# via the SERVICE_TYPE environment variable.
#
# Image size optimizations:
# - CPU-only PyTorch (no CUDA = ~3GB savings)
# - Multi-stage build to remove build tools
# - Aggressive cache cleanup
# =============================================================================

FROM python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .

# Install Python packages (CPU-only torch to save ~3GB)
RUN pip install --no-cache-dir --upgrade pip \
    && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# =============================================================================
# Final stage - minimal runtime image
# =============================================================================
FROM python:3.12-slim

# Install only runtime dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy wheels from builder and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* \
    && rm -rf /wheels \
    && pip cache purge

# Copy application code
COPY . .

# Create directories for persistent storage
RUN mkdir -p /app/uploads /app/outputs

# Make start script executable
RUN chmod +x /app/start.sh

# Default environment
ENV SERVICE_TYPE=web \
    PORT=8000 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD if [ "$SERVICE_TYPE" = "web" ]; then \
            curl -f http://localhost:${PORT}/ || exit 1; \
        else \
            echo "worker healthcheck: pid 1 alive" && kill -0 1 || exit 1; \
        fi

ENTRYPOINT ["/app/start.sh"]
