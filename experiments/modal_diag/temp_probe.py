"""One tiny Anthropic call (5 output tokens) with and without temperature=0 on the production model.
Prints only the HTTP status and Anthropic's error type/message. No keys, no video text."""
import json, os
import modal

app = modal.App("cutsell-temperature-probe")
image = modal.Image.debian_slim().pip_install("requests")


@app.function(image=image, secrets=[modal.Secret.from_name("cutsell-worker")], timeout=120)
def probe():
    import requests
    key = os.environ["ANTHROPIC_API_KEY"].strip()
    out = []
    for extra in ({}, {"temperature": 0.0}, {"temperature": 1.0}, {"top_k": 1}):
        body = {"model": "claude-sonnet-5-5", "max_tokens": 5, "messages": [{"role": "user", "content": "Say OK"}], **extra}
        r = requests.post("https://api.anthropic.com/v1/messages", json=body, timeout=60,
                          headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        err = r.json().get("error", {}) if r.status_code != 200 else {}
        out.append(f"{json.dumps(extra) or '{}'} -> HTTP {r.status_code} {err.get('type', '')} {str(err.get('message', ''))[:200]}")
    return out


@app.local_entrypoint()
def main():
    for line in probe.remote():
        print("PROBE", line)
