#!/usr/bin/env bash
set -euo pipefail

echo "🚀 entrypoint.sh — EditDNA WORKER boot"

# Basic dirs
WORKROOT="/workspace/editdna"
APPDIR="$WORKROOT/app"
mkdir -p "$WORKROOT"

# Asegura git + ffmpeg (por si la imagen viene pelada)
if ! command -v git >/dev/null 2>&1; then
  echo "ℹ️ Installing git..."
  apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ℹ️ Installing ffmpeg..."
  apt-get update && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
fi

# -------- clone/refresh (idempotent) --------
REPO_URL="${REPO_URL:-https://github.com/AutomatedRetailServices/EditDNA-worker}"
BRANCH="${BRANCH:-main}"

if [ -d "$APPDIR/.git" ]; then
  echo "🔄 Refreshing repo at $APPDIR"
  git -C "$APPDIR" fetch --depth=1 origin "$BRANCH" || true
  git -C "$APPDIR" reset --hard "origin/$BRANCH" || true
else
  echo "⬇️  Cloning $REPO_URL → $APPDIR"
  rm -rf "$APPDIR"
  git clone --depth=1 -b "$BRANCH" "$REPO_URL" "$APPDIR"
fi

# Symlink /app para PYTHONPATH
rm -rf /app && ln -sfn "$APPDIR" /app
export PYTHONPATH="/app:${PYTHONPATH:-}"

# -------- Python deps --------
python -m pip install --upgrade pip
pip install --no-cache-dir -r /app/requirements.txt || true

echo "📂 /app tree (top):"
ls -la /app | head -n 80 || true

# -------- RQ WORKER --------
REDIS_URL="${REDIS_URL:?Set REDIS_URL env}"
QUEUE_NAME="${QUEUE_NAME:-default}"

echo "🧰 Starting RQ worker on queue=$QUEUE_NAME"
exec rq worker -u "$REDIS_URL" --worker-ttl 1200 "$QUEUE_NAME"

