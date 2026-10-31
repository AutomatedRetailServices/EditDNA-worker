"""One-off (PO 31 Oct): does the engine's model accept temperature 0 with its default (adaptive) thinking?
Two tiny calls, prints only the HTTP status and the API's error type/message (no key, no prompt). < 1 cent."""
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install("requests>=2.32")
app = modal.App("cutsell-diag-temperature")


@app.function(image=image, secrets=[modal.Secret.from_name("cutsell-worker")], timeout=120, cpu=0.25)
def probe():
    import os, requests
    key = os.environ["ANTHROPIC_API_KEY"].strip()
    model = os.environ.get("CUTSELL_SIMPLE_ENGINE_MODEL", "").strip() or "claude-sonnet-5-5"
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    base = {"model": model, "max_tokens": 64, "messages": [{"role": "user", "content": "Reply with the word ok."}]}
    for name, extra in (("temperature0", {"temperature": 0}),
                        ("temperature0_thinking_off", {"temperature": 0, "thinking": {"type": "disabled"}})):
        r = requests.post("https://api.anthropic.com/v1/messages", json={**base, **extra}, headers=headers, timeout=60)
        detail = ""
        if r.status_code != 200:
            try:
                err = r.json().get("error", {})
                detail = f"{err.get('type')}: {str(err.get('message'))[:200]}"
            except Exception:
                detail = r.text[:200]
        print(f"PROBE {name} model={model} status={r.status_code} {detail}")


@app.local_entrypoint()
def main():
    probe.remote()
