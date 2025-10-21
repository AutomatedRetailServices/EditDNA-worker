#!/bin/bash
set -euo pipefail

WORKDIR="/workspace/editdna"
cd "$WORKDIR"

# Python deps
python3 -m pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt || true
pip install --no-cache-dir fastapi uvicorn || true

export PYTHONPATH="$WORKDIR"

# Prefer app.py if it contains a runner, else direct uvicorn
if grep -q "uvicorn.run" app.py 2>/dev/null; then
  echo "▶️  python3 app.py"
  exec python3 app.py
else
  echo "▶️  uvicorn app:app --host 0.0.0.0 --port 8000"
  exec uvicorn app:app --host 0.0.0.0 --port 8000
fi
