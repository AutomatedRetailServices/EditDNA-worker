#!/usr/bin/env bash

# bootstrap.sh — idempotent pod bootstrapper
set -euo pipefail
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/workspace/pip-cache}
echo "== Bootstrap: ensuring git =="
command -v git >/dev/null 2>&1 || { apt-get update -y && apt-get install -y git; }

WORK=/workspace/editdna
REPO=AutomatedRetailServices/EditDNA-worker
CODE="$WORK/app"

mkdir -p "$WORK"
if [ -d "$CODE/.git" ]; then
  echo "== Refreshing repo =="
  git -C "$CODE" fetch --depth=1 origin main
  git -C "$CODE" reset --hard FETCH_HEAD
else
  echo "== Cloning repo =="
  rm -rf "$CODE"
  git clone --depth=1 "git@github.com:${REPO}.git" "$CODE" || \
  git clone --depth=1 "https://github.com/${REPO}.git" "$CODE"
fi

rm -rf /app && ln -sfn "$CODE" /app
export PYTHONPATH=/app

git -C "$CODE" remote set-url origin "git@github.com:${REPO}.git" || true

echo "== /app contents =="
ls -la /app | head -n 50 || true
echo "== /app/worker =="
ls -la /app/worker || true

if [ -f /app/requirements.txt ]; then python3 -m pip install -r /app/requirements.txt || true; fi
if [ -f /app/requirements-semantic.txt ]; then python3 -m pip install -r /app/requirements-semantic.txt || true; fi
if [[ "${ASR_ENABLED:-0}" == "1" && -f /app/requirements-asr.txt ]]; then python3 -m pip install -r /app/requirements-asr.txt || true; fi

exec bash /app/start_worker.sh
