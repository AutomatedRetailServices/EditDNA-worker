#!/usr/bin/env bash
set -euo pipefail

###############################################
#  EditDNA Worker — RQ Worker Launcher
#  Compatible con RQ oficial (sin flags inválidos)
###############################################

# Redis connection URL (REQUIRED)
REDIS_URL="${REDIS_URL:?ERROR: You MUST set REDIS_URL}"

# Queue name fallback
QUEUE_NAME="${QUEUE_NAME:-default}"

# --------- PYTHONPATH FIX ---------
# RunPod puts your repo in /workspace/EditDNA-worker/
export PYTHONPATH="/workspace/EditDNA-worker:${PYTHONPATH:-}"

echo "=========================================="
echo "      EDITDNA WORKER STARTING"
echo "------------------------------------------"
echo " PYTHONPATH ....... $PYTHONPATH"
echo " REDIS_URL ........ $REDIS_URL"
echo " QUEUE_NAME ....... $QUEUE_NAME"
echo "=========================================="
echo ""

# --------- START RQ WORKER (no job-timeout flags) ---------
# El timeout REAL viene desde EDITDNA-WEB cuando encola el job.
exec rq worker -u "$REDIS_URL" --worker-ttl 3600 "$QUEUE_NAME"
