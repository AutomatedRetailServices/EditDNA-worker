#!/usr/bin/env bash
set -euo pipefail

###############################################
#  EditDNA Worker — RQ Worker Launcher
#  Optimizado para RunPod (GPU / CPU pods)
###############################################

# --------- ENVIRONMENT VALIDATION ---------

# Redis connection URL (required)
REDIS_URL="${REDIS_URL:?ERROR: You MUST set REDIS_URL}"

# Queue name with fallback
QUEUE_NAME="${QUEUE_NAME:-default}"

# Job timeout (must match the WEB API)
# Default: 1800 sec (30 min)
JOB_TIMEOUT="${JOB_TIMEOUT:-1800}"

# TTL for results and failures
RESULT_TTL="${RESULT_TTL:-86400}"
FAILURE_TTL="${FAILURE_TTL:-86400}"

# --------- PYTHONPATH FIX ---------

# We FORCE Python to see worker/, tasks.py and pipeline.py
export PYTHONPATH="/workspace/EditDNA-worker:${PYTHONPATH:-}"

echo "=========================================="
echo "      EDITDNA WORKER STARTING"
echo "------------------------------------------"
echo " PYTHONPATH ....... $PYTHONPATH"
echo " REDIS_URL ........ $REDIS_URL"
echo " QUEUE_NAME ....... $QUEUE_NAME"
echo " JOB_TIMEOUT ...... $JOB_TIMEOUT"
echo " RESULT_TTL ....... $RESULT_TTL"
echo " FAILURE_TTL ...... $FAILURE_TTL"
echo "=========================================="
echo ""

# --------- START RQ WORKER ---------

exec rq worker \
  -u "$REDIS_URL" \
  --worker-ttl 3600 \
  --job-timeout "$JOB_TIMEOUT" \
  "$QUEUE_NAME"
