#!/usr/bin/env bash
set -e

echo "🚀 entrypoint.sh — EditDNA boot"

REPO_URL="https://github.com/AutomatedRetailServices/EditDNA-worker"
REPO_DIR="/workspace/editdna/app"

echo ">>> STEP 1: make sure /workspace/editdna exists"
mkdir -p /workspace/editdna
cd /workspace/editdna

echo ">>> STEP 2: clone or update worker repo"
if [ -d "$REPO_DIR/.git" ]; then
  echo "repo already exists, pulling latest..."
  cd "$REPO_DIR"
  git pull --rebase || git pull
else
  echo "⬇️  Cloning $REPO_URL → $REPO_DIR"
  git clone "$REPO_URL" "$REPO_DIR"
  cd "$REPO_DIR"
fi

echo ">>> STEP 3: install python deps (requirements.txt)"
# usamos siempre un cwd que SÍ existe para que pip no rompa con FileNotFoundError
pip install --no-cache-dir -r requirements.txt

echo ">>> STEP 4: start RQ worker"
export PYTHONPATH="$REPO_DIR"
# si ya tienes start_worker.sh en el repo, usamos eso:
if [ -f "start_worker.sh" ]; then
  chmod +x start_worker.sh
  exec bash start_worker.sh
else
  # fallback simple: arranca un worker rq estándar
  echo "⚠️  start_worker.sh no existe, arrancando rq worker por defecto..."
  exec python -m rq.cli worker default
fi
