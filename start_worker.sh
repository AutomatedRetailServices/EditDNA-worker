##!/usr/bin/env bash
# start_worker.sh — RunPod-friendly worker launcher (no Docker-in-Docker)
set -euo pipefail
echo "== EditDNA worker boot =="

# -------- required env (minimal) --------
: "${REDIS_URL:?Please export REDIS_URL before running}"
: "${S3_BUCKET:?Please export S3_BUCKET before running}"

# Optional / sensible defaults
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-${AWS_REGION:-us-east-1}}"
export QUEUE="${QUEUE:-default}"
export ASR_ENABLED="${ASR_ENABLED:-0}"

echo "Connecting Redis: ${REDIS_URL}"
echo "S3 Bucket: ${S3_BUCKET}"
command -v ffmpeg >/dev/null && ffmpeg -version | head -n1 || echo "ffmpeg not found (set FFMPEG_BIN=ffmpeg)"

# -------- code root --------
# If RunPod Start Command cloned into /app, keep it; else fall back next to this script.
PYROOT="${CODE_ROOT:-/app}"
if [ ! -f "${PYROOT}/jobs.py" ]; then
  # try script directory as fallback
  HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PYROOT="${HERE}"
fi

export CODE_ROOT="$PYROOT"
export PYTHONPATH="${PYROOT}"

echo "Using code root: ${CODE_ROOT}"
echo "PYTHONPATH=${PYTHONPATH}"

# ---- boot debug (super helpful if imports fail) ----
python3 - <<'PY'
import sys, os, pathlib
print("[boot] sys.path:", sys.path, flush=True)
w = pathlib.Path(os.environ.get("PYTHONPATH","/app")) / "worker"
print("[boot] /app/worker exists:", w.exists(), flush=True)
try:
    print("[boot] worker contents:", [p.name for p in w.glob("*")][:15], flush=True)
except Exception as e:
    print("[boot] list worker failed:", e, flush=True)
PY

# -------- base deps --------
if ! python3 -c "import rq, redis, boto3" >/dev/null 2>&1; then
  echo "Installing base deps (rq, redis, boto3)..."
  python3 -m pip install --upgrade pip >/dev/null 2>&1 || true
  python3 -m pip install 'rq>=2.6.0' 'redis>=6.0' 'boto3>=1.28' >/dev/null 2>&1
fi

# repo deps
if [ -f "${PYROOT}/requirements.txt" ]; then
  echo "Installing requirements.txt..."
  python3 -m pip install -r "${PYROOT}/requirements.txt" >/dev/null 2>&1 || true
fi

# semantic deps (small) — always okay to install
if [ -f "${PYROOT}/requirements-semantic.txt" ]; then
  echo "Installing requirements-semantic.txt…"
  python3 -m pip install -r "${PYROOT}/requirements-semantic.txt" >/dev/null 2>&1 || true
fi

# optional: ASR deps (heavy, only if ASR_ENABLED=1)
if [[ "${ASR_ENABLED}" == "1" && -f "${PYROOT}/requirements-asr.txt" ]]; then
  echo "ASR_ENABLED=1 → installing ASR deps (Torch+Whisper)…"
  python3 -m pip install -r "${PYROOT}/requirements-asr.txt" >/dev/null 2>&1 || true
fi

# -------- launch worker --------
echo "RQ worker starting..."
exec python3 - <<'PY'
import os, redis
from rq import Worker, Queue
ru = os.environ['REDIS_URL']
qn = os.environ.get('QUEUE','default')
conn = redis.from_url(ru)
q = Queue(qn, connection=conn)
print(f"*** Listening on {qn}...", flush=True)
Worker([q], connection=conn).work(burst=False)
PY

