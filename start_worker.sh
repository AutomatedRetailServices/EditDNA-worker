#!/usr/bin/env bash
set -euo pipefail

# Asegura que REDIS_URL exista
REDIS_URL="${REDIS_URL:?Set REDIS_URL}"
QUEUE_NAME="${QUEUE_NAME:-default}"

# Estamos en /workspace/EditDNA-worker (el repo) por el comando de RunPod.
# PYTHONPATH debe apuntar al directorio actual (.)
export PYTHONPATH=".:${PYTHONPATH:-}"

echo ">>> PYTHONPATH = $PYTHONPATH"
echo ">>> Starting RQ worker on queue: $QUEUE_NAME (redis: $REDIS_URL)"

# Arranca el worker RQ que ejecuta tasks.job_render
exec rq worker -u "$REDIS_URL" --worker-ttl 1200 "$QUEUE_NAME"
