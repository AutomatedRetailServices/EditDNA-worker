#!/usr/bin/env bash
set -euo pipefail

REDIS_URL="${REDIS_URL:?Set REDIS_URL}"
QUEUE_NAME="${QUEUE_NAME:-default}"

# 👇 IMPORTANTÍSIMO: agrega el repo al PYTHONPATH
export PYTHONPATH="/workspace/EditDNA-worker:${PYTHONPATH:-}"

echo ">>> Starting RQ worker on queue: $QUEUE_NAME (redis: $REDIS_URL)"
exec rq worker "$QUEUE_NAME" --url "$REDIS_URL" --worker-ttl 1200
