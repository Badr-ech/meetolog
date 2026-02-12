#!/usr/bin/env bash
# =============================================================================
# Meetolog – Container Entrypoint
#
# Launches either the FastAPI web server or the ARQ worker based on the
# SERVICE_TYPE environment variable.
#
#   SERVICE_TYPE=web     →  uvicorn  (default)
#   SERVICE_TYPE=worker  →  arq
# =============================================================================

set -euo pipefail

SERVICE_TYPE="${SERVICE_TYPE:-web}"
PORT="${PORT:-8000}"
RUN_WORKER="${RUN_WORKER:-false}"

echo "=========================================="
echo " Meetolog – Starting service: ${SERVICE_TYPE}"
echo "=========================================="

# If RUN_WORKER=true, start the ARQ worker in the background
# alongside the main service (used on Render where the API
# and worker share a single persistent disk).
if [ "${RUN_WORKER}" = "true" ] && [ "${SERVICE_TYPE}" = "web" ]; then
    echo "→ Starting ARQ worker in background (RUN_WORKER=true)..."
    arq app.worker.WorkerSettings &
    sleep 2
fi

case "${SERVICE_TYPE}" in
    web)
        echo "→ Launching Uvicorn on port ${PORT}..."
        exec uvicorn app.main:app \
            --host 0.0.0.0 \
            --port "${PORT}" \
            --workers 1 \
            --log-level info
        ;;
    worker)
        echo "→ Launching ARQ worker..."
        exec arq app.worker.WorkerSettings
        ;;
    *)
        echo "ERROR: Unknown SERVICE_TYPE '${SERVICE_TYPE}'."
        echo "       Valid values: web, worker"
        exit 1
        ;;
esac
