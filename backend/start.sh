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

echo "=========================================="
echo " Meetolog – Starting service: ${SERVICE_TYPE}"
echo "=========================================="

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
