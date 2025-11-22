#!/usr/bin/env bash
set -euo pipefail

echo "🚀 entrypoint.sh — EditDNA boot"

# ====== Paths básicos ======
WORKROOT="/workspace/editdna"
APPDIR="$WORKROOT/app"
mkdir -p "$WORKROOT"

# ====== Asegurar git + ffmpeg (por si la imagen viene pelada) ======
if ! command -v git >/dev/null 2>&1; then
  echo "ℹ️ Installing git..."
  apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ℹ️ Installing ffmpeg..."
  apt-get update && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
fi

# ====== Clonar / refrescar repo (idempotente) ======
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

# ====== Symlink /app y PYTHONPATH ======
rm -rf /app && ln -sfn "$APPDIR" /app
export PYTHONPATH="/app:${PYTHONPATH:-}"

# ====== DEBUG: comprobar que tasks.job_render existe ======
python - << 'EOF'
import sys, importlib

print("🔎 PYTHONPATH:", sys.path)
try:
    m = importlib.import_module("tasks")
    print("✅ tasks module importado:", m)
    print("   file:", getattr(m, "__file__", None))
    print("✅ has job_render:", hasattr(m, "job_render"))
except Exception as e:
    print("❌ No se pudo importar 'tasks':", repr(e))
EOF

# ====== Dependencias Python ======
python3 -m pip install --upgrade pip
pip install --no-cache-dir -r /app/requirements.txt || true

echo "📂 /app tree (top):"
ls -la /app | head -n 80 || true

# ====== Modo: API web o worker ======
MODE="${MODE:-worker}"   # set MODE=web on Render; MODE=worker on RunPod

if [ "$MODE" = "web" ]; then
  echo "🌐 Starting API → uvicorn app:app --host 0.0.0.0 --port 8000"
  exec uvicorn app:app --host 0.0.0.0 --port 8000
else
  # Worker RQ por defecto
  REDIS_URL="${REDIS_URL:?Set REDIS_URL env}"
  QUEUE_NAME="${QUEUE_NAME:-default}"
  echo "🧰 Starting RQ worker on queue=$QUEUE_NAME"
  exec rq worker -u "$REDIS_URL" --worker-ttl 1200 "$QUEUE_NAME"
fi
