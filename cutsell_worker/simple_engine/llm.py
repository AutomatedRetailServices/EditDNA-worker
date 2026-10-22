"""Anthropic call for the simple engine. Only env var NAMES live here; the key stays in the host's secrets."""
from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import time

import requests

DEFAULT_MODEL = "claude-sonnet-5-5"
API_URL = "https://api.anthropic.com/v1/messages"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
STEPS = ("pass", "decision", "haul_review", "sale_review", "fixwords")
_STEP: contextvars.ContextVar = contextvars.ContextVar("cutsell_simple_engine_step", default="")
_LOG = logging.getLogger("cutsell.simple_engine.llm")


@contextlib.contextmanager
def step(name: str):
    """Marks which engine step the calls inside belong to (only used to pick that step's thinking effort)."""
    token = _STEP.set(name)
    try:
        yield
    finally:
        _STEP.reset(token)


# Measured on the Gold (2026-10-17, 26 videos x 2 runs): less thinking in the pass step and the haul review keeps the
# cut quality; the decision, the sale review and the word fix stay at the model default (high). A lower effort on the
# word fix wrote a wrong brand name into a caption, and a lower effort on the decision lost sales lines.
DEFAULT_EFFORT = "pass=medium,haul_review=low"


def effort_for(name: str = "") -> str:
    """Thinking effort for a step from CUTSELL_SIMPLE_ENGINE_EFFORT (else DEFAULT_EFFORT): "medium" (every step) or
    "pass=medium,fixwords=low" (only those steps). "high" = every step at the model default, exactly as in 1.2.1
    (the rollback without a redeploy). A step not listed / an unknown value -> "" = the model default (high)."""
    raw = (os.environ.get("CUTSELL_SIMPLE_ENGINE_EFFORT", "").strip() or DEFAULT_EFFORT).lower()
    if raw == "high":
        return ""
    if "=" not in raw:
        return raw if raw in EFFORTS else ""
    for part in raw.split(","):
        key, _, value = part.partition("=")
        if key.strip() == (name or _STEP.get()) and value.strip() in EFFORTS:
            return value.strip()
    return ""


class Prompt(str):
    """A prompt that is still a plain string (tests and replays see the same text) but remembers its parts: the fixed
    instructions of a step, then the video's own text, then the run tag. call_anthropic sends the parts as separate
    content blocks and marks the fixed instructions for prompt caching (identical for every video, so the next video
    within 5 minutes reads them at the cache price). The text the model reads is exactly the same."""

    def __new__(cls, *parts: str, schema: dict | None = None):
        obj = super().__new__(cls, "".join(parts))
        obj.parts = tuple(p for p in parts if p)
        obj.schema = schema
        return obj


def _content(prompt):
    parts = getattr(prompt, "parts", None)
    if not parts or not cache_enabled():
        return str(prompt)
    blocks = [{"type": "text", "text": p} for p in parts]
    blocks[0]["cache_control"] = cache_control()
    return blocks


def cache_control() -> dict:
    """The breakpoint on the fixed instructions: 5 minutes (the default) or 1 hour when few videos are being cut."""
    return {"type": "ephemeral", "ttl": "1h"} if cache_ttl() == "1h" else {"type": "ephemeral"}


def json_enabled() -> bool:
    """CUTSELL_SIMPLE_ENGINE_JSON=0 lets the model answer in free text again (rollback without a redeploy)."""
    return os.environ.get("CUTSELL_SIMPLE_ENGINE_JSON", "1").strip() != "0"


_CACHE_ON: contextvars.ContextVar = contextvars.ContextVar("cutsell_simple_engine_cache", default=True)
_CACHE_TTL: contextvars.ContextVar = contextvars.ContextVar("cutsell_simple_engine_cache_ttl", default="5m")


@contextlib.contextmanager
def cache_scope(on: bool = True, ttl: str | None = None):
    """Prompt caching for the calls inside. Product Owner (2026-10-22): caching is on for EVERY video (Single, Combine,
    Bulk, Mix; every user), 1 hour while few videos are being cut and 5 minutes when many are (see cache_policy).
    Before (2026-10-19) it was on only for the videos of a Bulk send."""
    token = _CACHE_ON.set(bool(on))
    ttl_token = _CACHE_TTL.set(ttl if ttl in ("5m", "1h") else _CACHE_TTL.get())
    try:
        yield
    finally:
        _CACHE_TTL.reset(ttl_token)
        _CACHE_ON.reset(token)


def cache_enabled() -> bool:
    """On unless a cache_scope(False) says otherwise. CUTSELL_SIMPLE_ENGINE_CACHE=0 turns it off everywhere (no redeploy)."""
    return _CACHE_ON.get() and os.environ.get("CUTSELL_SIMPLE_ENGINE_CACHE", "1").strip() != "0"


def cache_ttl() -> str:
    return _CACHE_TTL.get()


# Claude Sonnet 5.5, USD per million tokens (platform.claude.com/docs/en/about-claude/pricing, read 2026-10-22).
PRICES = {"claude-sonnet-5-5": {"input": 2.0, "output": 10.0, "write_5m": 2.5, "write_1h": 4.0, "read": 0.10}}


def call_cost(usage: dict, model: str | None = None) -> dict:
    """What one call cost with the cache, and what the same tokens would have cost without it (all at the input price)."""
    price = PRICES.get(model or model_name(), PRICES["claude-sonnet-5-5"])
    inp = int(usage.get("input_tokens") or 0)
    out = int(usage.get("output_tokens") or 0)
    read = int(usage.get("cache_read_input_tokens") or 0)
    created = usage.get("cache_creation") or {}
    w1h = int(created.get("ephemeral_1h_input_tokens") or 0)
    w5m = int(created.get("ephemeral_5m_input_tokens") or 0)
    if not created:
        w5m = int(usage.get("cache_creation_input_tokens") or 0)
    actual = (inp * price["input"] + out * price["output"] + read * price["read"] + w5m * price["write_5m"]
              + w1h * price["write_1h"]) / 1e6
    without = ((inp + read + w5m + w1h) * price["input"] + out * price["output"]) / 1e6
    return {"usd": round(actual, 6), "usd_without_cache": round(without, 6), "read_tokens": read,
            "write_5m_tokens": w5m, "write_1h_tokens": w1h}


def cache_report(usages: list[dict], *, ttl: str | None = None, videos_last_hour: int | None = None) -> dict:
    """Per video: was the cache used, and how much did it save (negative = the write premium cost more)."""
    total = {"usd": 0.0, "usd_without_cache": 0.0, "read_tokens": 0, "write_5m_tokens": 0, "write_1h_tokens": 0}
    for usage in usages or []:
        for key, value in call_cost(usage).items():
            total[key] += value
    total["usd"] = round(total["usd"], 6)
    total["usd_without_cache"] = round(total["usd_without_cache"], 6)
    total["saved_usd"] = round(total["usd_without_cache"] - total["usd"], 6)
    total["cache_used"] = total["read_tokens"] > 0
    total["cache_on"] = cache_enabled()
    total["ttl"] = ttl or cache_ttl()
    if videos_last_hour is not None:
        total["videos_last_hour"] = videos_last_hour
    return total


def model_name() -> str:
    return os.environ.get("CUTSELL_SIMPLE_ENGINE_MODEL", "").strip() or DEFAULT_MODEL


def call_anthropic(prompt, *, max_tokens: int = 16000, attempts: int = 4) -> tuple[str, dict]:
    """One user message in, (text, usage) out. Retries transient failures; never logs the prompt or the key."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY missing")
    body = {"model": model_name(), "max_tokens": max_tokens, "messages": [{"role": "user", "content": _content(prompt)}]}
    effort = effort_for()
    if effort:
        body["output_config"] = {"effort": effort}
    schema = getattr(prompt, "schema", None)
    if schema and json_enabled():
        # Structured output: the visible answer is only the JSON (no notes before it). Thinking is not affected.
        body.setdefault("output_config", {})["format"] = {"type": "json_schema", "schema": schema}
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    last: Exception | None = None
    json_refused = False
    for attempt in range(attempts):
        try:
            response = requests.post(API_URL, json=body, headers=headers, timeout=(20, 600))
            if response.status_code == 200:
                data = response.json()
                text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
                usage = dict(data.get("usage") or {})
                # Measurement only (the call itself is unchanged): which step, the max_tokens ceiling, the thinking
                # setting (adaptive thinking is the model default; effort "" = the model default, high) and how big the
                # visible answer is. output_tokens includes the thinking; the thinking text itself is not returned.
                usage.update({
                    "step": _STEP.get() or "other", "max_tokens": max_tokens,
                    "thinking": "adaptive", "effort": effort or "default(high)",
                    "thinking_blocks": sum(1 for b in data.get("content", []) if b.get("type") == "thinking"),
                    "answer_chars": len(text), "stop_reason": data.get("stop_reason"),
                    "json_format": "format" in body.get("output_config", {}), "json_refused": json_refused,
                    "cache": cache_enabled(), "cache_ttl": cache_ttl() if cache_enabled() else None,
                })
                _LOG.info("simple_engine_llm_call %s", json.dumps({k: usage.get(k) for k in (
                    "step", "input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
                    "max_tokens", "thinking", "effort", "answer_chars", "stop_reason", "cache", "cache_ttl", "json_format")}))
                return text, usage
            if response.status_code == 400 and "format" in body.get("output_config", {}):
                # The JSON format was refused: ask again in free text (the parsers read both) and say so in the logs.
                _LOG.warning("simple_engine_llm_json_refused step=%s; asking again without the JSON format", _STEP.get())
                body["output_config"].pop("format")
                if not body["output_config"]:
                    body.pop("output_config")
                json_refused = True
                continue
            if response.status_code in (400, 401, 403, 404):
                raise RuntimeError(f"Anthropic HTTP {response.status_code}")
            last = RuntimeError(f"Anthropic HTTP {response.status_code}")
        except requests.RequestException as exc:
            last = exc
        if attempt < attempts - 1:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Anthropic call failed: {last}")
