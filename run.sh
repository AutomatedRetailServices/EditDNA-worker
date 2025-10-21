#!/usr/bin/env bash
set -euo pipefail

echo "🚀 EditDNA container boot"

APP_DIR="/opt/app"
cd "$APP_DIR"

export PYTHONPATH="$APP_DIR"
python3 -m pip install --upgrade pip >/dev/null 2>&1 || true

# Install deps if we baked requirements into the image, this is a no-op.
if [ -f requirements.txt ]; then
  pip3 install --no-cache-dir -r requirements.txt || true
fi

# Optional background worker (only if REDIS_URL is present)
if [ -n "${REDIS_URL:-}" ]; then
  echo "🧰 Starting RQ worker in background..."
  ( rq worker default \
      --url "$REDIS_URL" \
      --serializer rq.serializers.DefaultSerializer \
      --worker-ttl 600 \
      --job-monitoring-interval 10 \
      2>&1 | sed -u "s/^/[rq] /" ) &
else
  echo "ℹ️  REDIS_URL not set; skipping background worker."
fi

# Start the FastAPI app in the foreground (keeps container alive)
echo "🌐 Starting FastAPI (Uvicorn) on :8000"
exec uvicorn app:app --host 0.0.0.0 --port 8000
