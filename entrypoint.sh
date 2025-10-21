#!/bin/bash
set -euo pipefail
echo "🚀 Starting EditDNA API (entrypoint.sh)"

WORKDIR="/workspace/editdna"
mkdir -p "$WORKDIR"
cd "$WORKDIR" || { echo "❌ $WORKDIR not found"; exit 1; }

# Ensure Python path and deps
export PYTHONPATH="$WORKDIR"
python3 -m pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt || true
pip install --no-cache-dir fastapi uvicorn || true

# Check what to run
if grep -q "uvicorn.run" app.py 2>/dev/null; then
  echo "▶️ Running python3 app.py"
  exec python3 app.py
else
  echo "▶️ Running uvicorn app:app --host 0.0.0.0 --port 8000"
  exec uvicorn app:app --host 0.0.0.0 --port 8000
fi
