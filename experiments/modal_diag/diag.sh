#!/usr/bin/env bash
# Approved cleanup (2026-10-11): stop the idle leftover engine-temperature app (0 tasks), then
# list engine-temperature apps to confirm. Starts nothing, deploys nothing.
set -u
modal app stop --yes ap-pvf4sZtpKo9uKjXiDITkgc 2>&1 | tail -2
modal app list --json 2>/dev/null | python3 -c '
import json,sys
for a in json.load(sys.stdin):
    if "temperature" in json.dumps(a): print("APP", a.get("app_id"), a.get("state"), "tasks", a.get("tasks"), "stopped", a.get("stopped_at"))'
