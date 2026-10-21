"""Strict-but-tolerant JSON extraction for OpenAI provider adapters.

Models may wrap an otherwise valid JSON object in Markdown fences or a small amount
of surrounding prose. We tolerate only that presentation layer, then the caller
still validates every field/id/range before applying provider output.
"""
from __future__ import annotations

import json


def parse_json_object(text: str) -> dict:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("provider returned empty output")

    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(raw[start : end + 1])

    if not isinstance(value, dict):
        raise ValueError("provider output must be a JSON object")
    return value
