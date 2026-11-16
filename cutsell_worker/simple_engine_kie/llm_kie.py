"""Claude Sonnet 5.5 through kie.ai, shaped for what kie.ai does differently (measured 16 Nov 2026):

  - it ignores output_config.format (the JSON schema)  -> the schema is written into the question itself and the JSON
    is taken out of the answer (code fences, notes before it); one re-ask on kie.ai if it is still not JSON;
  - it ignores prompt caching and adds text of its own -> nothing to do here; the comparison measures the real cost;
  - it is slower (~15 s a call)                         -> measured too.
Same signature as simple_engine.llm.call_anthropic: prompt in, (text, usage) out. Reads simple_engine.llm (effort per
step, Prompt parts) without changing it. Never logs the prompt or the key.
"""
from __future__ import annotations

import json
import os
import re
import time

import requests

from cutsell_worker.simple_engine import llm as _base

KIE_URL = "https://api.kie.ai/claude/v1/messages"
KEY_ENV = "KIE_API_KEY"
CREDIT_USD = 0.005          # kie.ai: US$5 = 1,000 credits (16 Nov 2026)
TIMEOUT = (10, 300)


def _schema_note(schema: dict | None) -> str:
    if not schema:
        return ""
    return ("\n\nAnswer with ONLY one JSON object (no code fences, no text before or after it) that matches this "
            "JSON schema exactly:\n" + json.dumps(schema, separators=(",", ":")))


def extract_json(text: str) -> str | None:
    """The first complete JSON object or array in the answer (fences and notes around it removed), else None."""
    cleaned = re.sub(r"```(?:json)?", "", text or "")
    for match in re.finditer(r"[\{\[]", cleaned):
        try:
            obj, end = json.JSONDecoder().raw_decode(cleaned[match.start():])
        except ValueError:
            continue
        if isinstance(obj, (dict, list)):
            return json.dumps(obj, ensure_ascii=False)
    return None


def _blocks(prompt, extra: str) -> list | str:
    parts = list(getattr(prompt, "parts", None) or [str(prompt)])
    parts[-1] = parts[-1] + extra
    return [{"type": "text", "text": p} for p in parts]


def call_kie(prompt, *, max_tokens: int = 16000, attempts: int = 3, session=None, sleep=time.sleep) -> tuple[str, dict]:
    key = os.environ.get(KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(f"{KEY_ENV} missing")
    client = session or requests
    schema = getattr(prompt, "schema", None)
    effort = _base.effort_for()
    headers = {"Authorization": "Bearer " + key, "content-type": "application/json", "anthropic-version": "2023-06-01"}
    note = _schema_note(schema)
    total = {"input_tokens": 0, "output_tokens": 0, "credits": 0.0, "calls": 0, "reasks": 0, "seconds": 0.0}
    last = "no answer"
    for attempt in range(attempts):
        body = {"model": _base.model_name(), "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": _blocks(prompt, note)}]}
        if effort:
            body["output_config"] = {"effort": effort}
        started = time.monotonic()
        try:
            response = client.post(KIE_URL, json=body, headers=headers, timeout=TIMEOUT)
        except requests.RequestException as exc:
            last = exc.__class__.__name__
            sleep(3 * (attempt + 1))
            continue
        total["seconds"] += time.monotonic() - started
        total["calls"] += 1
        try:
            data = json.loads(response.text or "")
        except ValueError:
            data = {}
        if response.status_code != 200 or not isinstance(data.get("content"), list):
            last = f"HTTP {response.status_code}"
            sleep(3 * (attempt + 1))
            continue
        usage = data.get("usage") or {}
        total["input_tokens"] += int(usage.get("input_tokens") or 0) + int(usage.get("cache_creation_input_tokens") or 0)
        total["output_tokens"] += int(usage.get("output_tokens") or 0)
        total["credits"] += float(data.get("credits_consumed") or usage.get("credits_consumed") or 0)
        text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
        if schema:
            found = extract_json(text)
            if found is None:
                last = "not JSON"
                total["reasks"] += 1
                note = _schema_note(schema) + "\nYour previous answer was not valid JSON. Answer again with ONLY the JSON."
                continue
            text = found
        return text, {
            "provider": "kie", "step": _base._STEP.get() or "other", "effort": effort or "default(high)",
            "input_tokens": total["input_tokens"], "output_tokens": total["output_tokens"],
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
            "credits": round(total["credits"], 4), "cost_usd": round(total["credits"] * CREDIT_USD, 6),
            "calls": total["calls"], "reasks": total["reasks"], "seconds": round(total["seconds"], 1),
            "stop_reason": data.get("stop_reason"), "answer_chars": len(text),
        }
    raise RuntimeError(f"kie.ai call failed: {last}")
