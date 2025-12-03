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

# Job timeout (MANTENIDO solo como referencia; el real vive en la API)
JOB_TIMEOUT="${JOB_TIMEOUT:-1800}"

# TTL for results and failures (solo informativo aquí)
RESULT_TTL="${RESULT_TTL:-86400}"
FAILURE_TTL="${FAILURE_TTL:-86400}"

# --------- PYTHONPATH FIX ---------

# VERY IMPORTANT: Para que Python encuentre worker/tasks.py y worker/pipeline.py
export PYTHONPATH="/workspace/EditDNA-worker:${PYTHONPATH:-}"

echo "=========================================="
echo "      EDITDNA WORKER STARTING"
echo "------------------------------------------"
echo " PYTHONPATH ....... $PYTHONPATH"
echo " REDIS_URL ........ $REDIS_URL"
echo " QUEUE_NAME ....... $QUEUE_NAME"
echo " JOB_TIMEOUT ...... $JOB_TIMEOUT  (handled by API)"
echo " RESULT_TTL ....... $RESULT_TTL"
echo " FAILURE_TTL ...... $FAILURE_TTL"
echo "=========================================="
echo ""

# --------- START RQ WORKER ---------
# ⚠️ NO SE INCLUYE --job-timeout AQUÍ.
#    El web API controla los timeouts reales.
exec rq worker \
  -u "$REDIS_URL" \
  --worker-ttl 3600 \
  "$QUEUE_NAME"
