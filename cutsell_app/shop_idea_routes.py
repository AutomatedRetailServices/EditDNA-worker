"""Shop board · "My idea" (PO 18 Nov 2026: "up load an idea es para replicar EXACT SAME THING").

She uploads a screenshot of a board she likes; Claude Sonnet 5.5 looks at it once and returns an exact MAP of it: the
background colours, every text (its words, place, size, colour, kind of font), and every product place (what kind of
product, where, how big, how turned). The iPhone builds HER board from that map with HER products in those places and
saves it as one of her styles. Figma S3i-C 3078:418 / S3i-D 3078:499.

POST /v1/shop/idea {user_id, image_b64 (JPEG/PNG, max ~1.5 MB)} -> {"style": {...}, "cost_usd"}
One Sonnet call (~1,600 tokens in, ~900 out: about 1-2 cents). Nothing is stored: not the picture, not the map
(PO: "cargando esto a su memoria del celular no a nuestro server cloud"). The image is never logged.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
import time
from datetime import date

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/v1/shop", tags=["shop"])
_LOG = logging.getLogger("cutsell.shop_idea")

MODEL = "claude-sonnet-5-5"
API_URL = "https://api.anthropic.com/v1/messages"
PRICE_PER_MTOK = {"input": 2.0, "output": 10.0}          # cutsell_worker/simple_engine/llm.py PRICES
MAX_IMAGE_BYTES = 1_600_000
DAILY_LIMIT = 30                                          # ideas per creator per day (each one costs money)
KINDS = ("top", "bottom", "dress", "outerwear", "shoes", "bag", "accessory", "jewelry", "beauty", "home", "other")
FONTS = ("serif", "script", "sans", "display", "condensed", "handwritten", "mono")
ROLES = ("title", "subtitle", "brand", "price", "other")

_BOX = {"x": {"type": "number"}, "y": {"type": "number"}, "w": {"type": "number"}, "h": {"type": "number"}}
SCHEMA = {
    "type": "object",
    "properties": {
        "background": {"type": "object", "properties": {"top": {"type": "string"}, "bottom": {"type": "string"}},
                       "required": ["top", "bottom"], "additionalProperties": False},
        "texts": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"}, "role": {"type": "string", "enum": list(ROLES)}, **_BOX,
            "color": {"type": "string"}, "font": {"type": "string", "enum": list(FONTS)},
            "weight": {"type": "string", "enum": ["light", "regular", "bold", "black"]},
            "align": {"type": "string", "enum": ["left", "center", "right"]}, "uppercase": {"type": "boolean"},
            "rotation": {"type": "number"}},
            "required": ["text", "role", "x", "y", "w", "h", "color", "font", "weight", "align", "uppercase", "rotation"],
            "additionalProperties": False}},
        "slots": {"type": "array", "items": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": list(KINDS)}, **_BOX, "rotation": {"type": "number"},
            "front": {"type": "integer"}},
            "required": ["kind", "x", "y", "w", "h", "rotation", "front"], "additionalProperties": False}},
    },
    "required": ["background", "texts", "slots"], "additionalProperties": False,
}

PROMPT = """This picture is a vertical "shop board": products on a background, with a title and maybe a brand line,
made for TikTok / Instagram. A creator wants to copy it EXACTLY with her own products. Describe it as a precise map.

All places are fractions of the picture: x, y = the top-left corner, w, h = the size (0 to 1). If the picture is not
9:16, map it as if it were stretched to 9:16.
- background: the colour at the top and at the bottom (#RRGGBB; the same twice for a flat colour).
- texts: EVERY piece of text, word for word as written (keep its language and capitals), its box, colour (#RRGGBB),
  the kind of font (serif, script, sans, display, condensed, handwritten, mono), its weight, alignment, whether it is
  all capitals, and its tilt in degrees (clockwise +). role: title (the biggest words), subtitle, brand (a handle or
  shop name), price, other.
- slots: EVERY product, one each: what kind it is (top, bottom, dress, outerwear, shoes, bag, accessory, jewelry,
  beauty, home, other), the box around the product itself (tight), its tilt in degrees, and "front": 1 for the one
  drawn on top of the others, 2 next, and so on.
Do not invent anything that is not in the picture. Answer with the JSON object only."""


class IdeaRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=200)
    image_b64: str = Field(min_length=100, max_length=2_200_000)


_COUNTS: dict[tuple[str, str], int] = {}
_COUNTS_LOCK = threading.Lock()


def _allowed(user_id: str) -> bool:
    key = (user_id, date.today().isoformat())
    with _COUNTS_LOCK:
        for old in [k for k in _COUNTS if k[1] != key[1]]:
            _COUNTS.pop(old, None)
        if _COUNTS.get(key, 0) >= DAILY_LIMIT:
            return False
        _COUNTS[key] = _COUNTS.get(key, 0) + 1
        return True


def _media_type(raw: bytes) -> str | None:
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    return None


def _ask(image_b64: str, media_type: str, call=None):
    body = {"model": MODEL, "max_tokens": 3000,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
                {"type": "text", "text": PROMPT}]}],
            "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}}}
    if call:
        return call(body)
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return None, {}, "no_key"
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    error = None
    for attempt in range(2):
        try:
            response = requests.post(API_URL, json=body, headers=headers, timeout=(10, 90))
            if response.status_code == 200:
                data = response.json()
                answer = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
                try:
                    return json.loads(answer), dict(data.get("usage") or {}), None
                except ValueError:
                    error = "json_broken"
            else:
                error = f"http_{response.status_code}"
                if response.status_code in (400, 401, 403, 404, 413):
                    break
        except requests.RequestException as exc:
            error = "network_" + exc.__class__.__name__.lower()
        if attempt == 0:
            time.sleep(1)
    return None, {}, error


_HEX = re.compile(r"^#?([0-9a-fA-F]{6})$")


def _hex(value, default: str) -> str:
    m = _HEX.match(str(value or "").strip())
    return f"#{m.group(1).upper()}" if m else default


def _unit(value, default: float = 0.0) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 4)
    except (TypeError, ValueError):
        return default


def _turn(value) -> float:
    try:
        return round(max(-180.0, min(180.0, float(value))), 1)
    except (TypeError, ValueError):
        return 0.0


def _box(item: dict) -> dict | None:
    x, y, w, h = (_unit(item.get(k)) for k in ("x", "y", "w", "h"))
    w, h = min(w, 1 - x), min(h, 1 - y)
    return {"x": x, "y": y, "w": round(w, 4), "h": round(h, 4)} if w >= 0.02 and h >= 0.01 else None


def clean(answer: dict) -> dict | None:
    """The map as the iPhone reads it: every value checked and bounded; empty = None."""
    if not isinstance(answer, dict):
        return None
    background = answer.get("background") if isinstance(answer.get("background"), dict) else {}
    top = _hex(background.get("top"), "#F4EFE6")
    style = {"background": {"top": top, "bottom": _hex(background.get("bottom"), top)}, "texts": [], "slots": []}
    for item in (answer.get("texts") or [])[:12]:
        if not isinstance(item, dict) or not str(item.get("text") or "").strip():
            continue
        box = _box(item)
        if not box:
            continue
        style["texts"].append({
            "text": str(item["text"]).strip()[:120], "role": item.get("role") if item.get("role") in ROLES else "other",
            **box, "color": _hex(item.get("color"), "#0B1020"),
            "font": item.get("font") if item.get("font") in FONTS else "sans",
            "weight": item.get("weight") if item.get("weight") in ("light", "regular", "bold", "black") else "regular",
            "align": item.get("align") if item.get("align") in ("left", "center", "right") else "center",
            "uppercase": bool(item.get("uppercase")), "rotation": _turn(item.get("rotation"))})
    for n, item in enumerate((answer.get("slots") or [])[:12]):
        if not isinstance(item, dict):
            continue
        box = _box(item)
        if not box:
            continue
        try:
            front = int(item.get("front") or n + 1)
        except (TypeError, ValueError):
            front = n + 1
        style["slots"].append({"kind": item.get("kind") if item.get("kind") in KINDS else "other", **box,
                               "rotation": _turn(item.get("rotation")), "front": max(1, min(99, front))})
    return style if style["slots"] else None


def read_idea(image_b64: str, call=None) -> dict:
    try:
        raw = base64.b64decode(image_b64, validate=True)
    except (ValueError, TypeError):
        return {"style": None, "error": "bad_image", "cost_usd": 0.0}
    media = _media_type(raw)
    if not media or len(raw) > MAX_IMAGE_BYTES:
        return {"style": None, "error": "bad_image", "cost_usd": 0.0}
    answer, usage, error = _ask(image_b64, media, call=call)
    cost = (int(usage.get("input_tokens") or 0) * PRICE_PER_MTOK["input"]
            + int(usage.get("output_tokens") or 0) * PRICE_PER_MTOK["output"]) / 1e6
    style = clean(answer) if answer is not None else None
    if style is None:
        return {"style": None, "error": error or "no_products_found", "cost_usd": round(cost, 6)}
    return {"style": style, "error": None, "cost_usd": round(cost, 6)}


@router.post("/idea")
def idea_route(payload: IdeaRequest):
    if not _allowed(payload.user_id):
        raise HTTPException(status_code=429, detail="idea_daily_limit")
    result = read_idea(payload.image_b64)
    # Never log the picture or what it says: only counts and the cost.
    style = result["style"] or {}
    _LOG.info("shop_idea user=%s error=%s slots=%s texts=%s cost=%s", payload.user_id[:8], result["error"],
              len(style.get("slots", [])), len(style.get("texts", [])), result["cost_usd"])
    if result["style"] is None:
        status = 422 if result["error"] in ("bad_image", "no_products_found") else 503
        raise HTTPException(status_code=status, detail=f"idea_unavailable:{result['error']}")
    return {"style": result["style"], "cost_usd": result["cost_usd"]}
