# =============================================================================
# Meetolog Backend – Production Dockerfile
# Supports both the FastAPI web server and the ARQ background worker
# via the SERVICE_TYPE environment variable.
# =============================================================================

FROM python:3.12-slim AS base

# ---- System dependencies ----------------------------------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Working directory -------------------------------------------------------
WORKDIR /app

# ---- Python dependencies (cached layer) -------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- Application code --------------------------------------------------------
COPY . .

# ---- Directories for persistent storage -------------------------------------
# These should be backed by a persistent volume in production.
RUN mkdir -p /app/uploads /app/outputs

# ---- Entrypoint --------------------------------------------------------------
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Default environment
ENV SERVICE_TYPE=web \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD if [ "$SERVICE_TYPE" = "web" ]; then \
            curl -f http://localhost:${PORT}/ || exit 1; \
        else \
            echo "worker healthcheck: pid 1 alive" && kill -0 1 || exit 1; \
        fi

ENTRYPOINT ["/app/start.sh"]
