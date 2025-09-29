#!/usr/bin/env bash
set -euo pipefail

# Fail fast if REDIS_URL not set
: "${REDIS_URL:?REDIS_URL is required (e.g. redis://:pass@host:port/0)}"

# Exec rqworker with the provided URL and args
exec "$@" -u "$REDIS_URL"
