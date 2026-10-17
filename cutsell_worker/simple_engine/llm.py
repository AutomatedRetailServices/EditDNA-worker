"""Anthropic call for the simple engine. Only env var NAMES live here; the key stays in the host's secrets."""
from __future__ import annotations

import contextlib
import contextvars
import os
import time

import requests

DEFAULT_MODEL = "claude-sonnet-5-5"
API_URL = "https://api.anthropic.com/v1/messages"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
STEPS = ("pass", "decision", "haul_review", "sale_review", "fixwords")
_STEP: contextvars.ContextVar = contextvars.ContextVar("cutsell_simple_engine_step", default="")


@contextlib.contextmanager
def step(name: str):
    """Marks which engine step the calls inside belong to (only used to pick that step's thinking effort)."""
    token = _STEP.set(name)
    try:
        yield
    finally:
        _STEP.reset(token)


def effort_for(name: str = "") -> str:
    """Thinking effort for a step from CUTSELL_SIMPLE_ENGINE_EFFORT: "medium" (every step) or
    "pass=medium,fixwords=low" (only those steps). Unset / unknown -> "" = the model default (high), as in 1.2.1."""
    raw = os.environ.get("CUTSELL_SIMPLE_ENGINE_EFFORT", "").strip().lower()
    if not raw:
        return ""
    if "=" not in raw:
        return raw if raw in EFFORTS else ""
    for part in raw.split(","):
        key, _, value = part.partition("=")
        if key.strip() == (name or _STEP.get()) and value.strip() in EFFORTS:
            return value.strip()
    return ""


def model_name() -> str:
    return os.environ.get("CUTSELL_SIMPLE_ENGINE_MODEL", "").strip() or DEFAULT_MODEL


def call_anthropic(prompt: str, *, max_tokens: int = 16000, attempts: int = 4) -> tuple[str, dict]:
    """One user message in, (text, usage) out. Retries transient failures; never logs the prompt or the key."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY missing")
    body = {"model": model_name(), "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
    effort = effort_for()
    if effort:
        body["output_config"] = {"effort": effort}
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
