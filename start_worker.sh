#!/usr/bin/env bash
set -euo pipefail
REDIS_URL="${REDIS_URL:?Set REDIS_URL}"
QUEUE_NAME="${QUEUE_NAME:-default}"
export PYTHONPATH="/app:${PYTHONPATH:-}"
exec rq worker -u "$REDIS_URL" --worker-ttl 1200 "$QUEUE_NAME"
