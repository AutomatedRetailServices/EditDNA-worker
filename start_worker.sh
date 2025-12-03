#!/usr/bin/env bash
set -euo pipefail

# ============================
#  Ensure REDIS_URL is defined
# ============================
REDIS_URL="${REDIS_URL:?REDIS_URL is required}"
QUEUE_NAME="${QUEUE_NAME:-default}"

# ============================================
#  PYTHONPATH must include the current project
#  (/workspace/EditDNA-worker)
# ============================================
export PYTHONPATH=".:${PYTHONPATH:-}"

echo ">>> PYTHONPATH = $PYTHONPATH"
echo ">>> Starting RQ worker on queue: $QUEUE_NAME  (redis: $REDIS_URL)"

# ===================================================
#  Start RQ worker — it MUST resolve tasks.job_render
# ===================================================
exec rq worker -u "$REDIS_URL" --worker-ttl 1200 "$QUEUE_NAME"
