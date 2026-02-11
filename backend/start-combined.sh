#!/bin/bash
set -e

echo "=========================================="
echo " Meetolog – Combined Deployment"
echo " Running API + Worker in single container"
echo "=========================================="

# Start ARQ worker in background
echo "→ Starting ARQ worker in background..."
arq app.worker.WorkerSettings &

# Give worker a moment to initialize
sleep 2

# Start API server in foreground
echo "→ Starting API server on port ${PORT:-8000}..."
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
