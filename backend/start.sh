#!/usr/bin/env bash
# =============================================================================
# Meetolog – Container Entrypoint
#
# Launches either the FastAPI web server or the Postgres-polling background
# worker based on the SERVICE_TYPE environment variable.
#
#   SERVICE_TYPE=web     →  uvicorn  (default)
#   SERVICE_TYPE=worker  →  python -m app.worker
# =============================================================================

set -euo pipefail

SERVICE_TYPE="${SERVICE_TYPE:-web}"
PORT="${PORT:-8000}"

echo "Meetolog – starting service: ${SERVICE_TYPE}"

case "${SERVICE_TYPE}" in
    web)
        exec uvicorn app.main:app \
            --host 0.0.0.0 \
            --port "${PORT}" \
            --workers 1 \
            --log-level info
        ;;
    worker)
        exec python -m app.worker
        ;;
    *)
        echo "ERROR: Unknown SERVICE_TYPE '${SERVICE_TYPE}'." >&2
        echo "       Valid values: web, worker" >&2
        exit 1
        ;;
esac
