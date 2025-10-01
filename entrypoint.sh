#!/usr/bin/env bash
set -euo pipefail

cd /app
: "${REDIS_URL:?Missing REDIS_URL}"

export PYTHONPATH=/app

exec rq worker default \
  --url "$REDIS_URL" \
  --serializer rq.serializers.DefaultSerializer \
  --worker-ttl 600 \
  --job-monitoring-interval 10
