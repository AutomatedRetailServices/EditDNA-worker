#!/usr/bin/env bash
set -euo pipefail

echo "🚀 entrypoint.sh — EditDNA WORKER boot"

# Aquí ya estamos dentro de una imagen que YA tiene el repo copiado en /app
WORKDIR="/app"
cd "$WORKDIR"

# Por si acaso, mostrar árbol básico
echo "📂 /app tree:"
ls -la /app | head -n 80 || true

# Aseguramos ffmpeg instalado (por si la imagen base cambia en el futuro)
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ℹ️ Installing ffmpeg..."
  apt-get update && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
fi

# Variables de entorno críticas
REDIS_URL="${REDIS_URL:?Set REDIS_URL env}"
QUEUE_NAME="${QUEUE_NAME:-default}"

echo "🧪 Sanity check: import tasks"
python - << 'PY'
try:
    import tasks  # noqa
    print("✅ imported tasks successfully")
except Exception as e:
    print("❌ failed to import tasks:", e)
PY

echo "🧰 Starting RQ worker on queue=$QUEUE_NAME"
exec rq worker -u "$REDIS_URL" --worker-ttl 1200 "$QUEUE_NAME"
