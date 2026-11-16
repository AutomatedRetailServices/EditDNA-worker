"""Does Claude Sonnet 5.5 work through kie.ai today? (PO 16 Nov 2026: "revisa si el motor de sonnet funciona ya en kie")

PAID but tiny: 3 short calls (~US$0.01 total). Runs on Modal because the kie.ai key is only in the Modal secret.
It sends the SAME request shape the simple engine sends (cutsell_worker/simple_engine/llm.py: model, adaptive
thinking by default, output_config.effort, output_config.format json_schema, a cached first block) and checks what
the 19 Oct provider code checked: answer arrives, valid JSON, thinking happened at high effort, cache marks honoured.
Prints only sizes, flags and timings -- never the key.
"""
from __future__ import annotations

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install("requests>=2.31")
app = modal.App("cutsell-kie-claude-probe")
URL = "https://api.kie.ai/claude/v1/messages"
MODEL = "claude-sonnet-5-5"


@app.function(image=image, secrets=[modal.Secret.from_name("cutsell-worker")], timeout=600)
def probe() -> list[dict]:
    import json
    import os
    import time

    import requests

    key = os.environ.get("KIE_API_KEY", "").strip()
    if not key:
        return [{"error": "KIE_API_KEY missing"}]
    headers = {"Authorization": "Bearer " + key, "content-type": "application/json", "anthropic-version": "2023-06-01"}
    fixed = ("You edit short selling videos. Rules: keep complete sentences, drop false starts, keep prices. " * 40)
    words = "hola chicas hoy les traigo este serum este serum cuesta veinte dolares link en mi bio"
    schema = {"type": "object", "properties": {"keep": {"type": "array", "items": {"type": "string"}},
                                                "reason": {"type": "string"}},
              "required": ["keep", "reason"], "additionalProperties": False}
    cases = [
        ("high effort + json schema + cache", {"effort": "high", "format": {"type": "json_schema", "schema": schema}}),
        ("same again (cache read?)", {"effort": "high", "format": {"type": "json_schema", "schema": schema}}),
        ("low effort, no schema", {"effort": "low"}),
    ]
    rows = []
    for name, output_config in cases:
        body = {"model": MODEL, "max_tokens": 4000, "output_config": output_config,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": fixed, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "Transcript: " + words + "\nReturn JSON {keep: [phrases kept], reason}."}]}]}
        row = {"case": name}
        started = time.monotonic()
        try:
            r = requests.post(URL, json=body, headers=headers, timeout=(10, 120))
            row["http"] = r.status_code
            row["seconds"] = round(time.monotonic() - started, 1)
            data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            if r.status_code != 200 or not isinstance(data.get("content"), list):
                row["error"] = str(data.get("error") or data.get("msg") or data)[:200]
            else:
                text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
                usage = data.get("usage") or {}
                try:
                    json.loads(text)
                    row["json_ok"] = True
                except ValueError:
                    row["json_ok"] = False
                row.update(model=data.get("model"), stop=data.get("stop_reason"),
                           thinking_blocks=sum(1 for b in data["content"] if b.get("type") == "thinking"),
                           in_tok=usage.get("input_tokens"), out_tok=usage.get("output_tokens"),
                           cache_write=usage.get("cache_creation_input_tokens"), cache_read=usage.get("cache_read_input_tokens"),
                           answer_chars=len(text), credits=data.get("credits_consumed") or usage.get("credits_consumed"))
        except Exception as exc:                      # noqa: BLE001
            row["error"] = exc.__class__.__name__
        rows.append(row)
    return rows


@app.local_entrypoint()
def main():
    for row in probe.remote():
        print("RESULT", row)
