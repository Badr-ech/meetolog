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
        echo "Checking database migration state..."
        # If the alembic_version table is empty or missing, the database was
        # created before Alembic tracking began.  Stamp it at the last known
        # baseline so 'upgrade head' only applies new migrations.
        if ! alembic --config /app/alembic.ini current 2>/dev/null | grep -q '[a-f0-9]'; then
            echo "No migration history found — stamping baseline at c7d9e1f3a5b2"
            alembic --config /app/alembic.ini stamp c7d9e1f3a5b2
        fi
        echo "Applying pending migrations..."
        alembic --config /app/alembic.ini upgrade head
        echo "Migrations complete. Starting API server..."
        exec uvicorn app.main:app \
            --host 0.0.0.0 \
            --port "${PORT}" \
            --workers 1 \
            --log-level info
        ;;
    worker|splitter)
        exec python -m app.worker
        ;;
    chunk_worker)
        exec python -m app.worker
        ;;
    assembler)
        exec python -m app.worker
        ;;
    *)
        echo "ERROR: Unknown SERVICE_TYPE '${SERVICE_TYPE}'." >&2
        echo "       Valid values: web, worker" >&2
        exit 1
        ;;
esac
