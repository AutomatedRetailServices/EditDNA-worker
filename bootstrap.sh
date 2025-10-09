#!/usr/bin/env bash
# bootstrap.sh — idempotent pod bootstrapper
set -euo pipefail

echo "== Bootstrap: ensuring git =="
command -v git >/dev/null 2>&1 || { apt-get update -y && apt-get install -y git; }

# persistent workspace
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
  # try SSH first (if you added the deploy key), fall back to HTTPS
  git clone --depth=1 "git@github.com:${REPO}.git" "$CODE" || \
  git clone --depth=1 "https://github.com/${REPO}.git" "$CODE"
fi

# make code visible at /app and to Python
rm -rf /app && ln -sfn "$CODE" /app
export PYTHONPATH=/app

# (optional) flip remote to SSH so you can push from the pod
git -C "$CODE" remote set-url origin "git@github.com:${REPO}.git" || true

# show key bits
echo "== /app =="
ls -la /app
echo "== /app/worker =="
ls -la /app/worker || true

# print the env knobs you care about (add/remove as needed)
echo "MAX_TAKE_SEC=${MAX_TAKE_SEC:-unset}  MAX_DURATION_SEC=${MAX_DURATION_SEC:-unset}  MIN_TAKE_SEC=${MIN_TAKE_SEC:-unset}"
echo "W_SEM=${W_SEM:-unset} W_FACE=${W_FACE:-unset} W_SCENE=${W_SCENE:-unset} W_VTX=${W_VTX:-unset}"
echo "SEM_DUP_THRESHOLD=${SEM_DUP_THRESHOLD:-unset}  SEM_MERGE_SIM=${SEM_MERGE_SIM:-unset}  VIZ_MERGE_SIM=${VIZ_MERGE_SIM:-unset}"
echo "MERGE_MAX_CHAIN=${MERGE_MAX_CHAIN:-unset}  SEM_FILLER_MAX_RATE=${SEM_FILLER_MAX_RATE:-unset}"
echo "SLOT_REQUIRE_PRODUCT=${SLOT_REQUIRE_PRODUCT:-unset}  SLOT_REQUIRE_OCR_CTA=${SLOT_REQUIRE_OCR_CTA:-unset}"
echo "QUEUE=${QUEUE:-unset}  ASR_ENABLED=${ASR_ENABLED:-unset}"

# install deps (light + conditional heavy)
if [ -f /app/requirements.txt ]; then
  python3 -m pip install -r /app/requirements.txt || true
fi
if [ -f /app/requirements-semantic.txt ]; then
  python3 -m pip install -r /app/requirements-semantic.txt || true
fi
if [[ "${ASR_ENABLED:-0}" == "1" && -f /app/requirements-asr.txt ]]; then
  python3 -m pip install -r /app/requirements-asr.txt || true
fi

# launch worker (blocks)
exec bash /app/start_worker.sh
