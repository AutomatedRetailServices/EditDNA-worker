#!/usr/bin/env bash
set -euo pipefail

echo "🚀 entrypoint.sh — EditDNA WORKER boot"

# -------- PATHS BÁSICOS --------
WORKROOT="/workspace/editdna"
APPDIR="$WORKROOT/app"
mkdir -p "$WORKROOT"

# -------- SISTEMA: git + ffmpeg (si faltan) --------
NEED_APT=0

if ! command -v git >/dev/null 2>&1; then
  NEED_APT=1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  NEED_APT=1
fi

if [ "$NEED_APT" -eq 1 ]; then
  echo "ℹ️ Installing system packages (git/ffmpeg)..."
  apt-get update
  if ! command -v git >/dev/null 2>&1; then
    apt-get install -y --no-install-recommends git
  fi
  if ! command -v ffmpeg >/dev/null 2>&1; then
    apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 libgl1
  fi
  rm -rf /var/lib/apt/lists/*
fi

# -------- CLONE / REFRESH RE
