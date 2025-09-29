#!/usr/bin/env bash
set -euo pipefail

: "${REDIS_URL:?REDIS_URL is required (e.g. redis://:pass@host:port/0)}"

exec "$@" -u "$REDIS_URL"
