#!/usr/bin/env bash
# start_worker.sh — RunPod-friendly worker launcher (no Docker-in-Docker)
set -euo pipefail

echo "== EditDNA worker boot =="

# -------- required env --------
: "${REDIS_URL:?Please export REDIS_URL before running}"
: "${S3_BUCKET:?Please export S3_BUCKET before running}"
: "${AWS_REGION:?Please export AWS_REGION before running}"
: "${AWS_ACCESS_KEY_ID:?Please export AWS_ACCESS_KEY_ID before running}"
: "${AWS_SECRET_ACCESS_KEY:?Please export AWS_SECRET_ACCESS_KEY before running}"

echo "Connecting Redis: ${REDIS_URL}"
echo "S3 Bucket: ${S3_BUCKET}"
command -v ffmpeg >/dev/null && ffmpeg -version | head -n1 || echo "ffmpeg not found (ensure FFMPEG_BIN=ffmpeg)"

# -------- code root detection --------
# Prefer /app if the repo is cloned there (RunPod best practice); fallback to script dir
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYROOT="/app"
if [[ ! -f "${PYROOT}/jobs.py" && ! -f "${PYROOT}/tasks.py" ]]; then
  PYROOT="${SCRIPT_DIR}"
fi
echo "Using code root: ${PYROOT}"

# -------- keep code fresh if repo is a git checkout --------
if [[ -d "${PYROOT}/.git" ]]; then
  echo "Updating from GitHub..."
  git -C "${PYROOT}" pull --rebase || true
fi

# -------- ensure minimal deps --------
if ! python3 -c "import rq, redis" >/dev/null 2>&1; then
  echo "Installing Python deps (rq, redis)..."
  python3 -m pip install --upgrade pip >/dev/null 2>&1 || true
  python3 -m pip install rq>=2.6.0 redis>=6.0 >/dev/null 2>&1
fi
# If your repo has a requirements.txt, try to install it
if [[ -f "${PYROOT}/requirements.txt" ]]; then
  echo "Installing requirements.txt..."
  python3 -m pip install -r "${PYROOT}/requirements.txt" >/dev/null 2>&1 || true
fi

# -------- launch worker (direct, no docker) --------
export PYTHONPATH="${PYROOT}"
echo "RQ worker starting..."
exec python3 - <<'PY'
import os, redis, sys
from rq import Worker, Queue
ru = os.environ['REDIS_URL']
conn = redis.from_url(ru)
q = Queue('default', connection=conn)
print("Listening on 'default' with RQ…", flush=True)
Worker([q], connection=conn).work(burst=False)
PY
