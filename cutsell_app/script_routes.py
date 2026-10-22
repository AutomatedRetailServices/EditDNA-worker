"""Teleprompter · "✦ Split for Mix": Claude Haiku splits the creator's script into the Mix parts Hook / Body / Benefit
/ CTA, using HER words only (never invents a sentence). The app sends each part to its Mix section.

POST /v1/scripts/split {user_id, script, product?} → {"parts": {"hook", "body", "benefit", "cta"}, "cost_usd"}
One small Haiku call (about 600 tokens in, 300 out ≈ $0.0002). No storage: the script stays on the iPhone.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/v1/scripts", tags=["scripts"])
_LOG = logging.getLogger("cutsell.script_split")

MODEL = "claude-haiku-5-5"
API_URL = "https://api.anthropic.com/v1/messages"
PRICE_PER_MTOK = {"input": 0.10, "output": 0.50}
PARTS = ("hook", "body", "benefit", "cta")
SCHEMA = {
    "type": "object",
    "properties": {name: {"type": "string"} for name in PARTS},
    "required": list(PARTS),
    "additionalProperties": False,
}
PROMPT = """You split a short-video sales script into the 4 parts of a CutSell Mix.
- hook: the opening line(s) that grab attention.
- body: what the product is / what it does.
- benefit: why it is worth it (results, proof, feelings).
- cta: the call to action (buy, tap, link, offer, urgency).
Rules: use ONLY the creator's own sentences, copied exactly, in their order and language. Never add, rewrite or
translate words. Every sentence goes to exactly one part. A part with nothing for it is "".
Product (may be empty): {product}
Script:
<<<
{script}
>>>
Answer with the JSON object only."""


class SplitRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=200)
    script: str = Field(min_length=1, max_length=6000)
    product: str | None = Field(default=None, max_length=200)


def _ask(text: str, call=None):
    body = {"model": MODEL, "max_tokens": 1500, "messages": [{"role": "user", "content": text}],
            "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}}}
    if call:
        return call(body)
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return None, {}, "no_key"
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    error = None
    for attempt in range(2):
        try:
            response = requests.post(API_URL, json=body, headers=headers, timeout=(10, 45))
            if response.status_code == 200:
                data = response.json()
                answer = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
                try:
                    return json.loads(answer), dict(data.get("usage") or {}), None
                except ValueError:
                    error = "json_broken"
            else:
                error = f"http_{response.status_code}"
                if response.status_code in (400, 401, 403, 404):
                    break
        except requests.RequestException as exc:
            error = "network_" + exc.__class__.__name__.lower()
        if attempt == 0:
            time.sleep(1)
    return None, {}, error


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def keep_her_words(parts: dict, script: str) -> dict:
    """Drops any sentence Haiku wrote that is not in her script (never invent speech)."""
    source = " ".join(_words(script))
    clean = {}
    for name in PARTS:
        kept = []
        for sentence in re.split(r"(?<=[.!?¡¿…])\s+", str(parts.get(name) or "").strip()):
            words = _words(sentence)
            if words and " ".join(words) in source:
                kept.append(sentence.strip())
        clean[name] = " ".join(kept)
    return clean


def split(script: str, product: str | None = None, call=None) -> dict:
    answer, usage, error = _ask(PROMPT.format(product=product or "", script=script.strip()), call=call)
    cost = (int(usage.get("input_tokens") or 0) * PRICE_PER_MTOK["input"]
            + int(usage.get("output_tokens") or 0) * PRICE_PER_MTOK["output"]) / 1e6
    if not isinstance(answer, dict):
        return {"parts": None, "error": error or "no_answer", "cost_usd": round(cost, 6)}
    parts = keep_her_words(answer, script)
    if not any(parts.values()):
        return {"parts": None, "error": "nothing_kept", "cost_usd": round(cost, 6)}
    return {"parts": parts, "error": None, "cost_usd": round(cost, 6)}


@router.post("/split")
def split_route(payload: SplitRequest):
    result = split(payload.script, payload.product)
    _LOG.info("script_split user=%s error=%s cost=%s", payload.user_id[:8], result["error"], result["cost_usd"])
    if result["parts"] is None:
        raise HTTPException(status_code=503, detail=f"split_unavailable:{result['error']}")
    return {"parts": result["parts"], "cost_usd": result["cost_usd"]}
