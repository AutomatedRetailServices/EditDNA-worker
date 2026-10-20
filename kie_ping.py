"""Read-only: one tiny kie.ai Claude call and the account credit balance, to see why every call got HTTP 429.
Prints only status codes and kie.ai's own message (never the key)."""
import json
import os

import modal

app = modal.App("cutsell-kie-ping")
secret = modal.Secret.from_name("cutsell-worker")
image = modal.Image.debian_slim(python_version="3.11").pip_install("requests")


@app.function(image=image, secrets=[secret], timeout=120)
def ping():
    import requests
    key = os.environ.get("KIE_API_KEY", "").strip()
    h = {"Authorization": "Bearer " + key, "content-type": "application/json"}
    out = {"key_present": bool(key), "key_len": len(key)}
    try:
        r = requests.get("https://api.kie.ai/api/v1/chat/credit", headers=h, timeout=30)
        out["credit"] = {"status": r.status_code, "body": r.text[:300]}
    except Exception as exc:
        out["credit"] = repr(exc)[:200]
    for model in ("claude-sonnet-5-5", "claude-sonnet-4-6"):
        try:
            r = requests.post("https://api.kie.ai/claude/v1/messages", headers=h, timeout=60, json={
                "model": model, "max_tokens": 50, "messages": [{"role": "user", "content": "Reply with the word OK."}]})
            out[model] = {"status": r.status_code, "headers": {k: v for k, v in r.headers.items() if k.lower().startswith(("retry", "x-ratelimit"))},
                          "body": r.text[:400]}
        except Exception as exc:
            out[model] = repr(exc)[:200]
    return out


@app.local_entrypoint()
def main():
    out = ping.remote()
    with open("engine_check.json", "w") as f:
        json.dump(out, f)
