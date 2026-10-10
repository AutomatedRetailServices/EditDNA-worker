"""Anthropic call for the simple engine. Only env var NAMES live here; the key stays in the host's secrets."""
from __future__ import annotations

import os
import time

import requests

DEFAULT_MODEL = "claude-sonnet-5-5"
API_URL = "https://api.anthropic.com/v1/messages"


def model_name() -> str:
    return os.environ.get("CUTSELL_SIMPLE_ENGINE_MODEL", "").strip() or DEFAULT_MODEL


# Same transcript -> same cut. With the model's default sampling, borderline lines (a filler, a
# retake) flip from run to run. 0 asks the model for its most likely answer every time.
DEFAULT_TEMPERATURE = 0.0


def temperature() -> float | None:
    """CUTSELL_SIMPLE_ENGINE_TEMPERATURE overrides; "default" leaves it to the model (no parameter sent)."""
    raw = os.environ.get("CUTSELL_SIMPLE_ENGINE_TEMPERATURE", "").strip().lower()
    if raw == "default":
        return None
    if not raw:
        return DEFAULT_TEMPERATURE
    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise ValueError("CUTSELL_SIMPLE_ENGINE_TEMPERATURE must be between 0 and 1, or 'default'")
    return value


def call_anthropic(prompt: str, *, max_tokens: int = 16000, attempts: int = 4) -> tuple[str, dict]:
    """One user message in, (text, usage) out. Retries transient failures; never logs the prompt or the key."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY missing")
    body = {"model": model_name(), "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
    temp = temperature()
    if temp is not None:
        body["temperature"] = temp
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.post(API_URL, json=body, headers=headers, timeout=(20, 600))
            if response.status_code == 200:
                data = response.json()
                text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
                return text, dict(data.get("usage") or {})
            if response.status_code in (400, 401, 403, 404):
                raise RuntimeError(f"Anthropic HTTP {response.status_code}")
            last = RuntimeError(f"Anthropic HTTP {response.status_code}")
        except requests.RequestException as exc:
            last = exc
        if attempt < attempts - 1:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Anthropic call failed: {last}")
