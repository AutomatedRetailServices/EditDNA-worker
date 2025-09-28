#!/usr/bin/env bash
# start_worker.sh — EditDNA worker launcher (repo version)
set -Eeuo pipefail

# -------- Config you can override via env or .env -------------
IMAGE="${IMAGE:-editdna/worker:latest}"
NAME="${NAME:-editdna-worker}"
SESSIONS_DIR="${SESSIONS_DIR:-/data/editdna/sessions}"
USE_GPU="${USE_GPU:-1}"
FUNNEL_COUNTS="${FUNNEL_COUNTS:-1,1,1,1}"

# Required runtime envs (export or use .env)
: "${REDIS_URL:?Please export REDIS_URL before running}"
: "${S3_BUCKET:?Please export S3_BUCKET before running}"
: "${AWS_REGION:?Please export AWS_REGION before running}"
: "${AWS_ACCESS_KEY_ID:?Please export AWS_ACCESS_KEY_ID before running}"
: "${AWS_SECRET_ACCESS_KEY:?Please export AWS_SECRET_ACCESS_KEY before running}"

# Tuning (defaults = your baseline)
ASR_ENABLED="${ASR_ENABLED:-1}"
W_AUDIO="${W_AUDIO:-0.30}"; W_SCENE="${W_SCENE:-0.20}"; W_SPEECH="${W_SPEECH:-0.50}"
W_FACE="${W_FACE:-0.20}"; W_FLUENCY="${W_FLUENCY:-0.35}"
FACE_MIN_SIZE="${FACE_MIN_SIZE:-0.08}"; FACE_CENTER_TOL="${FACE_CENTER_TOL:-0.35}"
FLUENCY_MIN_WPM="${FLUENCY_MIN_WPM:-95}"; FLUENCY_FILLER_PENALTY="${FLUENCY_FILLER_PENALTY:-0.65}"
VETO_MIN_SCORE="${VETO_MIN_SCORE:-0.40}"; GRACE_SEC="${GRACE_SEC:-0.6}"; MAX_BAD_SEC="${MAX_BAD_SEC:-1.2}"

# V2 feature flags (optional)
V2_SLOT_SCORER="${V2_SLOT_SCORER:-0}"
V2_PROXY_FUSE="${V2_PROXY_FUSE:-0}"
V2_VARIANT_EXPAND="${V2_VARIANT_EXPAND:-0}"
V2_CAPTIONER="${V2_CAPTIONER:-0}"
CAPTIONS="${CAPTIONS:-off}"        # off|soft|hard
MAX_DURATION_SEC="${MAX_DURATION_SEC:-0}"  # 0 = unlimited
MIN_TAKE_SEC="${MIN_TAKE_SEC:-1.0}"

# Load .env if present (same folder as script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.env"
  set +a
fi
# --------------------------------------------------------------

echo "== Preparing session dir =="
sudo mkdir -p "$SESSIONS_DIR" || mkdir -p "$SESSIONS_DIR"
sudo chmod 777 "$SESSIONS_DIR" || true

# Choose docker command (with sudo fallback)
if docker info >/dev/null 2>&1; then
  DOCKER="docker"
elif sudo -n docker info >/dev/null 2>&1; then
  DOCKER="sudo docker"
else
  echo "!! Docker not accessible. Ensure daemon is running or run this script with sudo."
  exit 1
fi

echo "== Restarting $NAME =="
$DOCKER rm -f "$NAME" >/dev/null 2>&1 || true

RUN_FLAGS=(
  run -d --name "$NAME" --restart unless-stopped
  -v "${SESSIONS_DIR}:/sessions"
  -e SESSION_ROOT="/sessions"
  -e ASR_ENABLED="${ASR_ENABLED}"
  -e W_AUDIO="${W_AUDIO}" -e W_SCENE="${W_SCENE}" -e W_SPEECH="${W_SPEECH}"
  -e W_FACE="${W_FACE}" -e W_FLUENCY="${W_FLUENCY}"
  -e FACE_MIN_SIZE="${FACE_MIN_SIZE}" -e FACE_CENTER_TOL="${FACE_CENTER_TOL}"
  -e FLUENCY_MIN_WPM="${FLUENCY_MIN_WPM}" -e FLUENCY_FILLER_PENALTY="${FLUENCY_FILLER_PENALTY}"
  -e VETO_MIN_SCORE="${VETO_MIN_SCORE}" -e GRACE_SEC="${GRACE_SEC}" -e MAX_BAD_SEC="${MAX_BAD_SEC}"
  -e FUNNEL_COUNTS="${FUNNEL_COUNTS}"
  -e REDIS_URL="${REDIS_URL}"
  -e S3_BUCKET="${S3_BUCKET}"
  -e AWS_REGION="${AWS_REGION}"
  -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}"
  -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}"
  -e V2_SLOT_SCORER="${V2_SLOT_SCORER}"
  -e V2_PROXY_FUSE="${V2_PROXY_FUSE}"
  -e V2_VARIANT_EXPAND="${V2_VARIANT_EXPAND}"
  -e V2_CAPTIONER="${V2_CAPTIONER}"
  -e CAPTIONS="${CAPTIONS}"
  -e MAX_DURATION_SEC="${MAX_DURATION_SEC}"
  -e MIN_TAKE_SEC="${MIN_TAKE_SEC}"
  --log-opt max-size=10m --log-opt max-file=5
)

# Optional session token
if [[ -n "${AWS_SESSION_TOKEN:-}" ]]; then
  RUN_FLAGS+=(-e AWS_SESSION_TOKEN="${AWS_SESSION_TOKEN}")
fi

# GPU
if [[ "${USE_GPU}" == "1" ]]; then
  RUN_FLAGS+=(--gpus all)
fi

# Image last
RUN_FLAGS+=("$IMAGE")

echo "== Launching worker container =="
$DOCKER "${RUN_FLAGS[@]}"

echo "== Started '$NAME'. Tail logs with: $DOCKER logs -f $NAME"
