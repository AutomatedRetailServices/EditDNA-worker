#!/usr/bin/env bash
# start_worker.sh — RunPod-friendly worker launcher
set -euo pipefail

# --------- logging to persistent volume ----------
LOG="${WORKER_LOG:-/workspace/editdna/worker.log}"
mkdir -p "$(dirname "$LOG")" /workspace/editdna/tmp /workspace/editdna/pipcache || true
# send all stdout/stderr to both console and the log
exec > >(tee -a "$LOG") 2>&1

echo "== EditDNA worker boot =="

# --------- required/optional env ----------
: "${REDIS_URL:?Please export REDIS_URL before running}"
: "${S3_BUCKET:?Please export S3_BUCKET before running}"
# Region var name varies; accept either
AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-${AWS_REGION:-}}"
if [[ -z "${AWS_DEFAULT_REGION}" ]]; then
  echo "WARNING: AWS_DEFAULT_REGION/AWS_REGION not set; continuing."
fi
# AWS creds are optional (you might be using a role)
: "${AWS_ACCESS_KEY_ID:=}"
: "${AWS_SECRET_ACCESS_KEY:=}"

echo "Connecting Redis: ${REDIS_URL}"
echo "S3 Bucket: ${S3_BUCKET}"

# --------- code root / python path ----------
CODE_ROOT="${CODE_ROOT:-/app}"         # template symlinks /app -> /workspace/editdna/app
export PYTHONPATH="${PYTHONPATH:-$CODE_ROOT}"
export TMPDIR="/workspace/editdna/tmp"
export PIP_CACHE_DIR="/workspace/editdna/pipcache"

echo "Using code root: $CODE_ROOT"
echo "PYTHONPATH=$PYTHONPATH"

# ---- boot debug (very helpful if imports fail) ----
python3 - <<'PY'
import sys, os, pathlib
print("[boot] sys.path:", sys.path, flush=True)
root = pathlib.Path(os.environ.get("PYTHONPATH","/app"))
w = root / "worker"
print("[boot] worker dir exists:", w.exists(), flush=True)
try:
    print("[boot] worker contents:", sorted([p.name for p in w.glob("*")])[:20], flush=True)
except Exception as e:
    print("[boot] list worker failed:", e, flush=True)
PY

# --------- base deps ----------
if ! python3 -c "import rq, redis, boto3" >/dev/null 2>&1; then
  echo "Installing base deps (rq, redis, boto3)…"
  python3 -m pip install --upgrade pip --no-cache-dir || true
  python3 -m pip install --no-cache-dir 'rq>=2.6.0' 'redis>=6.0' 'boto3>=1.28'
fi

# repo deps
if [[ -f "$CODE_ROOT/requirements.txt" ]]; then
  echo "Installing requirements.txt…"
  python3 -m pip install --no-cache-dir -r "$CODE_ROOT/requirements.txt" || true
fi

# semantic deps (keep ready if file exists)
if [[ -f "$CODE_ROOT/requirements-semantic.txt" ]]; then
  echo "Installing requirements-semantic.txt…"
  python3 -m pip install --no-cache-dir -r "$CODE_ROOT/requirements-semantic.txt" || true
fi

# optional ASR deps (heavy)
if [[ "${ASR_ENABLED:-0}" == "1" && -f "$CODE_ROOT/requirements-asr.txt" ]]; then
  echo "ASR_ENABLED=1 → installing ASR deps (Torch+Whisper)…"
  python3 -m pip install --no-cache-dir -r "$CODE_ROOT/requirements-asr.txt" || true
fi

# --------- launch worker ----------
QUEUE="${QUEUE:-default}"
echo "RQ worker starting… (queue=${QUEUE})"
cd "$CODE_ROOT"

python3 - <<PY
import os, socket, redis
from rq import Worker, Queue
ru = os.environ["REDIS_URL"]
qname = os.environ.get("QUEUE","default")
conn = redis.from_url(ru)
q = Queue(qname, connection=conn)
name = f"editdna@{socket.gethostname()}"
print(f"*** Listening on {qname} as {name} …", flush=True)
Worker([q], connection=conn, name=name).work(burst=False)
PY
