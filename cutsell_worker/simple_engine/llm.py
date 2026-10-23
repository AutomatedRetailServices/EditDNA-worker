"""Anthropic call for the simple engine. Only env var NAMES live here; the key stays in the host's secrets."""
from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import time
import uuid

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
    normal = (inp * price["input"] + out * price["output"] + read * price["read"] + w5m * price["write_5m"]
              + w1h * price["write_1h"]) / 1e6
    without = ((inp + read + w5m + w1h) * price["input"] + out * price["output"]) / 1e6
    # Batch API: every price at 50 % (the cache prices too; the discounts stack).
    factor = 0.5 if usage.get("batch") else 1.0
    return {"usd": round(normal * factor, 6), "usd_without_cache": round(without * factor, 6),
            "usd_normal_api": round(normal, 6), "read_tokens": read, "write_5m_tokens": w5m, "write_1h_tokens": w1h,
            "batch_calls": 1 if usage.get("batch") else 0, "batch_fallbacks": 1 if usage.get("batch_fallback") else 0,
            "batch_wait_s": float(usage.get("batch_wait_s") or 0.0)}


def cache_report(usages: list[dict], *, ttl: str | None = None, videos_last_hour: int | None = None) -> dict:
    """Per video: was the cache used, and how much did it save (negative = the write premium cost more)."""
    total = {"usd": 0.0, "usd_without_cache": 0.0, "usd_normal_api": 0.0, "read_tokens": 0, "write_5m_tokens": 0,
             "write_1h_tokens": 0, "batch_calls": 0, "batch_fallbacks": 0, "batch_wait_s": 0.0}
    for usage in usages or []:
        for key, value in call_cost(usage).items():
            total[key] += value
    for key in ("usd", "usd_without_cache", "usd_normal_api", "batch_wait_s"):
        total[key] = round(total[key], 6 if key != "batch_wait_s" else 1)
    total["saved_by_batch_usd"] = round(total["usd_normal_api"] - total["usd"], 6)
    total["saved_usd"] = round(total["usd_without_cache"] - total["usd"], 6)
    total["cache_used"] = total["read_tokens"] > 0
    total["cache_on"] = cache_enabled()
    total["ttl"] = ttl or cache_ttl()
    if videos_last_hour is not None:
        total["videos_last_hour"] = videos_last_hour
    return total


def model_name() -> str:
    return os.environ.get("CUTSELL_SIMPLE_ENGINE_MODEL", "").strip() or DEFAULT_MODEL


# ---- Batch API for Bulk and Mix (PO 2026-10-23) -------------------------------------------------------------------
# The calls of a video sent in a Bulk send (or a Mix) go through the Message Batches API: 50 % of the price, same model,
# same request body (the cut is decided by exactly the same text). A batch usually ends within minutes; the video never
# waits more than BATCH_WAIT_MIN in all: past its deadline the pending batch is cancelled and that call (and the next
# ones) go through the normal API at once. Single and Combine never use it.
BATCH_URL = API_URL + "/batches"
BATCH_WAIT_MIN = 30.0
_BATCH_DEADLINE: contextvars.ContextVar = contextvars.ContextVar("cutsell_simple_engine_batch_deadline", default=None)


@contextlib.contextmanager
def batch_scope(on: bool, *, minutes: float | None = None):
    """Inside: calls go through the Batch API until `minutes` from now (default BATCH_WAIT_MIN). CUTSELL_ENGINE_BATCH=0
    turns it off everywhere (no redeploy)."""
    wait = BATCH_WAIT_MIN if minutes is None else float(minutes)
    deadline = time.time() + wait * 60 if on and batch_allowed() else None
    token = _BATCH_DEADLINE.set(deadline)
    try:
        yield
    finally:
        _BATCH_DEADLINE.reset(token)


def batch_allowed() -> bool:
    return os.environ.get("CUTSELL_ENGINE_BATCH", "1").strip() != "0"


def batch_deadline() -> float | None:
    return _BATCH_DEADLINE.get()


def via_batch(body: dict, headers: dict, *, post=None, get=None, sleep=time.sleep, now=time.time) -> tuple[dict | None, str]:
    """One request through the Batch API. Returns (message, "batch") or (None, why) -- the caller then asks the normal
    API. Never raises."""
    deadline = _BATCH_DEADLINE.get()
    if deadline is None or now() >= deadline:
        return None, "deadline_passed" if deadline is not None else "off"
    post = post or requests.post
    get = get or requests.get
    custom_id = "cs-" + uuid.uuid4().hex[:24]
    try:
        created = post(BATCH_URL, json={"requests": [{"custom_id": custom_id, "params": body}]}, headers=headers,
                       timeout=(20, 60))
        if created.status_code != 200:
            return None, f"create_http_{created.status_code}"
        batch = created.json()
        batch_id = batch["id"]
        started = now()
        while batch.get("processing_status") != "ended":
            if now() >= deadline:
                try:
                    post(f"{BATCH_URL}/{batch_id}/cancel", headers=headers, timeout=(20, 60))
                except Exception:
                    pass
                _LOG.warning("simple_engine_batch_timeout step=%s waited_s=%.0f", _STEP.get(), now() - started)
                return None, "timeout"
            sleep(5.0 if now() - started < 60 else 15.0)
            polled = get(f"{BATCH_URL}/{batch_id}", headers=headers, timeout=(20, 60))
            if polled.status_code == 200:
                batch = polled.json()
        results = get(batch["results_url"], headers=headers, timeout=(20, 120))
        if results.status_code != 200:
            return None, f"results_http_{results.status_code}"
        for line in results.text.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("custom_id") != custom_id:
                continue
            result = row.get("result") or {}
            if result.get("type") == "succeeded":
                message = dict(result.get("message") or {})
                message["_batch_wait_s"] = round(now() - started, 1)
                return message, "batch"
            return None, f"result_{result.get('type')}"
        return None, "result_missing"
    except Exception as exc:
        _LOG.warning("simple_engine_batch_failed %s", exc.__class__.__name__)
        return None, f"error_{exc.__class__.__name__}"


def _answer(data: dict, body: dict, max_tokens: int, effort: str, json_refused: bool, *, batched: bool) -> tuple[str, dict]:
    text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
    usage = dict(data.get("usage") or {})
    # Measurement only (the call itself is unchanged): which step, the max_tokens ceiling, the thinking setting
    # (adaptive thinking is the model default; effort "" = the model default, high) and how big the visible answer is.
    # output_tokens includes the thinking; the thinking text itself is not returned.
    usage.update({
        "step": _STEP.get() or "other", "max_tokens": max_tokens,
        "thinking": "adaptive", "effort": effort or "default(high)",
        "thinking_blocks": sum(1 for b in data.get("content", []) if b.get("type") == "thinking"),
        "answer_chars": len(text), "stop_reason": data.get("stop_reason"),
        "json_format": "format" in body.get("output_config", {}), "json_refused": json_refused,
        "cache": cache_enabled(), "cache_ttl": cache_ttl() if cache_enabled() else None,
        "batch": batched, "batch_wait_s": data.get("_batch_wait_s"),
    })
    _LOG.info("simple_engine_llm_call %s", json.dumps({k: usage.get(k) for k in (
        "step", "input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
        "max_tokens", "thinking", "effort", "answer_chars", "stop_reason", "cache", "cache_ttl", "json_format",
        "batch", "batch_wait_s")}))
    return text, usage


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
    batch_note = None
    if _BATCH_DEADLINE.get() is not None:
        message, batch_note = via_batch(body, headers)
        if message is not None:
            return _answer(message, body, max_tokens, effort, json_refused, batched=True)
    for attempt in range(attempts):
        try:
            response = requests.post(API_URL, json=body, headers=headers, timeout=(20, 600))
            if response.status_code == 200:
                data = response.json()
                text, usage = _answer(data, body, max_tokens, effort, json_refused, batched=False)
                if batch_note is not None:
                    usage["batch_fallback"] = batch_note
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
