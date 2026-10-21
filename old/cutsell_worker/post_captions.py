"""The posting caption of a Ready video: written ONCE by Claude Haiku 5.5 from what the video says and its product.

A separate step after the cut (the cut engine, its prompts and its calls are not touched; nothing here can change or
delay a cut). Short, English, 3-5 hashtags, never a price, discount or promise the video does not say. The creator
can edit it; her version is never overwritten. Every call's tokens and cost are kept with the caption."""
from __future__ import annotations

import json
import logging
import os
import re
import time

import requests

_LOG = logging.getLogger("cutsell.post_captions")
MODEL = "claude-haiku-5-5"
API_URL = "https://api.anthropic.com/v1/messages"
PRICE_PER_MTOK = {"input": 0.10, "output": 0.50}           # Haiku 5.5 (Anthropic price list)
MAX_WORDS = 600

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["caption", "hashtags"],
          "properties": {"caption": {"type": "string"}, "hashtags": {"type": "array", "items": {"type": "string"}}}}

PROMPT = """Write the TikTok caption for this TikTok Shop creator's video, in English.
- One or two short sentences (under 150 characters), in her voice, about what the video shows. One emoji at most.
- Use only what she says in the video (below) and the product name. NEVER write a price, a discount, a percentage,
  "sale", "free", a deadline or any promise she does not say in the video. If she says it, you may use it.
- 3 to 5 hashtags: the brand, the product type, and #tiktokshop. Lowercase, no spaces.
Return only the JSON: {"caption": "...", "hashtags": ["#...", ...]}"""

_MONEY = re.compile(r"(\$\s?\d|\d+\s?%|\bpercent\b|\bfree\b|\bsale\b|\bdiscount|\boff\b|\bdeal\b|\bcoupon|\bcode\b)", re.I)


def spoken_text(draft: dict) -> str:
    """The words that stay in the video (the kept clips), in order."""
    clips = [c for c in (draft.get("selected") or []) if isinstance(c, dict)]
    clips.sort(key=lambda c: (int(c.get("source_order") or 0), float(c.get("start") or 0)))
    words = [str(w.get("text") or "").strip() for c in clips for w in (c.get("words") or []) if isinstance(w, dict)]
    return " ".join(w for w in words if w)[:6000]


def _usage_cost(usage: dict) -> float:
    return (int(usage.get("input_tokens") or 0) * PRICE_PER_MTOK["input"]
            + int(usage.get("output_tokens") or 0) * PRICE_PER_MTOK["output"]) / 1e6


def _ask(text: str, call=None) -> tuple[dict | None, dict, str | None]:
    body = {"model": MODEL, "max_tokens": 2000, "messages": [{"role": "user", "content": text}],
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
            response = requests.post(API_URL, json=body, headers=headers, timeout=(10, 60))
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
            time.sleep(2)
    return None, {}, error


def write(spoken: str, *, brand: str | None, product: str | None, call=None) -> dict:
    """-> {"caption", "log": {model, tokens_in, tokens_out, cost_usd, error}}; caption None when it failed."""
    if not spoken.strip():
        return {"caption": None, "log": {"model": MODEL, "error": "no_words", "cost_usd": 0.0}}
    words = spoken.split()
    text = (PROMPT + f"\n\nPRODUCT: {brand or '?'} · {product or '?'}\n\nWHAT SHE SAYS:\n"
            + " ".join(words[:MAX_WORDS]))
    answer, usage, error = _ask(text, call)
    caption = None
    if answer and str(answer.get("caption") or "").strip():
        caption = clean(str(answer["caption"]), [str(h) for h in answer.get("hashtags") or []], spoken)
    return {"caption": caption, "log": {"model": MODEL, "tokens_in": int(usage.get("input_tokens") or 0),
                                        "tokens_out": int(usage.get("output_tokens") or 0),
                                        "cost_usd": round(_usage_cost(usage), 6), "error": error}}


def clean(sentence: str, hashtags: list[str], spoken: str) -> str:
    """The rule as code too: a money / offer word that is not in what she says is dropped from the sentence."""
    said = spoken.lower()
    parts = []
    for piece in re.split(r"(?<=[.!?])\s+", sentence.strip()):
        bad = [m.group(0) for m in _MONEY.finditer(piece) if m.group(0).lower().strip() not in said]
        if not bad:
            parts.append(piece)
    sentence = " ".join(parts).strip()
    tags = []
    for tag in hashtags:
        tag = "#" + re.sub(r"[^a-z0-9]", "", tag.lower().lstrip("#"))
        if len(tag) > 1 and tag not in tags:
            tags.append(tag)
    if "#tiktokshop" not in tags:
        tags.append("#tiktokshop")
    return (sentence + " " + " ".join(tags[:5])).strip()


def caption_once(*, user_id: str, project_id: str, draft: dict | None = None, client=None, call=None) -> dict:
    """Writes the caption of one Ready video if it has none yet (never over hers). Never raises."""
    try:
        from . import post_plan, product_catalog
        if os.environ.get("CUTSELL_POST_CAPTIONS", "1").strip() == "0":
            return {"status": "off"}
        entry = post_plan.video(user_id, project_id, client=client)
        if entry.get("caption"):
            return {"status": "exists"}
        if draft is None:
            from .draft_store import get_draft_snapshot
            draft = get_draft_snapshot(user_id=user_id, project_id=project_id).get("draft") or {}
        data = product_catalog.load(user_id, client=client)
        link = data["videos"].get(project_id) or {}
        product = data["products"].get(link.get("product_id") or "") or {}
        brand = (data["brands"].get(product.get("brand_id") or "") or {}).get("name")
        result = write(spoken_text(draft), brand=brand, product=product.get("name"), call=call)
        if result["caption"]:
            post_plan.set_caption(user_id, project_id, result["caption"], source="ai", log=result["log"], client=client)
        _LOG.info("post_caption %s", json.dumps(result["log"]))
        return {"status": "written" if result["caption"] else "failed", "log": result["log"]}
    except Exception as exc:
        _LOG.warning("post_caption_failed %s", exc.__class__.__name__)
        return {"status": "failed", "error": exc.__class__.__name__}


def run_caption_job(payload: dict) -> dict:
    """Worker job: captions for Ready videos that have none (the week was built before they were written)."""
    user_id = str(payload.get("user_id") or "")
    out = {}
    for project_id in payload.get("project_ids") or []:
        out[project_id] = caption_once(user_id=user_id, project_id=str(project_id))
    return {"captions": out}
