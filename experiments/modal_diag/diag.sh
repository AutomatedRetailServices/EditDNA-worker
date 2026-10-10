#!/usr/bin/env bash
# Read-only: what happened to the engine-temperature Modal runs? Lists apps and prints only
# filtered error lines (cut to 220 chars). Starts nothing, deploys nothing, costs nothing.
set -u
modal app list --json > apps.json 2>/dev/null || modal app list > apps.txt
python3 - <<'PY'
import json, os
apps = json.load(open("apps.json")) if os.path.exists("apps.json") else []
rows = [a for a in apps if "temperature" in json.dumps(a)]
for a in rows:
    print("APP", {k: a.get(k) for k in a if k.lower() in ("app id", "app_id", "description", "name", "state", "created at", "created_at", "stopped at", "stopped_at", "tasks")})
open("ids.txt", "w").write("\n".join(str(a.get("App ID") or a.get("app_id") or "") for a in rows))
PY
while read -r id; do
  [ -z "$id" ] && continue
  echo "=== $id"
  timeout 90 modal app logs "$id" 2>&1 \
    | grep -iE "error|exception|traceback|timeout|timed out|killed|memory|failed|anthropic http|stopp|retr" \
    | cut -c1-220 | sort | uniq -c | sort -rn | head -25
done < ids.txt
