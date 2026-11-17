"""Smart zoom eyes (PO 17 Nov 2026: "lo de la imagen se haga con haiku").

The AI that cuts the video (Sonnet) only reads WORDS, so it knows WHEN to zoom and what she talks about, but not
where things are in the picture. Here Claude Haiku 5.5 looks at one small frame per zoom moment -- all of a video's
frames in ONE call (~US$0.001 per video) -- together with what she is saying right then, and answers like a human
editor: what to zoom into (her face, the product in her hand, her arm when she talks about her skin...) and where it is
(a box). zoom_framing turns that box into how deep the zoom goes and where it aims.

No answer (no key, an error, CUTSELL_ZOOM_VISION=0) = the OpenCV face finder, as before. Never logs the prompt or key.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import subprocess
import time
from typing import Callable, Optional

import requests

MODEL = "claude-haiku-5-5"
API_URL = "https://api.anthropic.com/v1/messages"
#: Claude Haiku 5.5, US$ per million tokens (prompts under 100K tokens; ours are ~10K).
PRICE_IN, PRICE_OUT = 0.10, 0.50
FRAME_WIDTH = 384
MAX_FRAMES = 16
_LOG = logging.getLogger("cutsell.zoom_vision")

PROMPT = """You are a dynamic human video editor for short vertical selling videos (TikTok / Reels). For each numbered
frame below you get what the creator is saying at that moment and why the cut engine wants a zoom there. Decide, like
an editor who understands the talk, WHAT the zoom should go into, and give where it is in that frame:

- she talks about the product, its price, "look at this", shows it -> the PRODUCT itself (the bottle, jar, item in her
  hand or on the table), tightly;
- she talks about a part of her body (skin, arm, face, lips, eyes, hair, hands, legs) or shows the product on it ->
  THAT part, where she is touching / showing it (if she rubs her arm while saying "skin", it is her ARM, not her face);
- a hook, a benefit, a story, a call to action, a list point -> her FACE (from the chin to the top of the head);
- nothing sensible to zoom into -> "none".

Answer ONLY JSON: {"frames": [{"n": 1, "target": "product|face|arm|skin|lips|eyes|hair|hands|legs|body|none",
"box": [left, top, right, bottom]}, ...]} with the box as fractions of the frame width / height (0 to 1). The box is
the thing to fill the screen with, a little loose (not cut). One entry per frame, in order."""

SCHEMA = {
    "type": "object",
    "properties": {"frames": {"type": "array", "items": {
        "type": "object",
        "properties": {"n": {"type": "integer"}, "target": {"type": "string"},
                       "box": {"type": "array", "items": {"type": "number"}}},
        "required": ["n", "target", "box"], "additionalProperties": False}}},
    "required": ["frames"], "additionalProperties": False,
}


def enabled() -> bool:
    return os.environ.get("CUTSELL_ZOOM_VISION", "1").strip() != "0" and bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


def grab_jpeg(path: str, t: float, *, width: int = FRAME_WIDTH) -> Optional[bytes]:
    """One small JPEG frame at source second t (rotation applied by ffmpeg)."""
    try:
        done = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, float(t)):.3f}", "-i", str(path), "-frames:v", "1",
             "-vf", f"scale={width}:-2", "-q:v", "5", "-f", "image2", "-c:v", "mjpeg", "-"],
            capture_output=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 and done.stdout[:2] == b"\xff\xd8" else None


def _box(value) -> Optional[tuple[float, float, float, float]]:
    try:
        left, top, right, bottom = (max(0.0, min(1.0, float(v))) for v in value[:4])
    except (TypeError, ValueError):
        return None
    if right - left < 0.03 or bottom - top < 0.03:
        return None
    return left, top, right, bottom


def parse(text: str, count: int) -> list[Optional[dict]]:
    """The answer -> one {"target", "box"} (or None) per frame, in order."""
    match = re.search(r"\{.*\}", text or "", re.S)
    try:
        frames = json.loads(match.group(0))["frames"] if match else []
    except (ValueError, KeyError, TypeError):
        frames = []
    out: list[Optional[dict]] = [None] * count
    for item in frames if isinstance(frames, list) else []:
        try:
            n = int(item.get("n")) - 1
        except (TypeError, ValueError, AttributeError):
            continue
        target = str(item.get("target") or "").strip().lower()
        box = _box(item.get("box") or [])
        if 0 <= n < count and target and target != "none" and box:
            out[n] = {"target": target, "box": box}
    return out


def look(frames: list[tuple[bytes, str]], *, post: Callable = requests.post, attempts: int = 2) -> tuple[list, dict]:
    """[(jpeg, what she says + why a zoom)] -> ([{"target", "box"} or None per frame], usage). One call."""
    if not frames:
        return [], {}
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY missing")
    content: list[dict] = [{"type": "text", "text": PROMPT}]
    for n, (jpeg, context) in enumerate(frames[:MAX_FRAMES], start=1):
        content.append({"type": "text", "text": f"Frame {n}: {context}"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": base64.b64encode(jpeg).decode("ascii")}})
    body = {"model": MODEL, "max_tokens": 1500, "messages": [{"role": "user", "content": content}],
            "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}}}
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    last = None
    for attempt in range(attempts):
        try:
            response = post(API_URL, json=body, headers=headers, timeout=(20, 120))
            if response.status_code == 400 and "output_config" in body:
                body.pop("output_config")            # no JSON format: ask in free text (parse reads both)
                continue
            if response.status_code == 200:
                data = response.json()
                text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
                usage = dict(data.get("usage") or {})
                usage["usd"] = round((int(usage.get("input_tokens") or 0) * PRICE_IN
                                      + int(usage.get("output_tokens") or 0) * PRICE_OUT) / 1e6, 6)
                usage["model"] = MODEL
                return parse(text, len(frames[:MAX_FRAMES])) + [None] * max(0, len(frames) - MAX_FRAMES), usage
            last = RuntimeError(f"Anthropic HTTP {response.status_code}")
        except requests.RequestException as exc:
            last = exc
        if attempt < attempts - 1:
            time.sleep(3)
    raise RuntimeError(f"zoom vision failed: {last}")


def context_for(moment: dict, words: list[dict]) -> str:
    """What she says around the moment (about 1.5 s before to the end of her sentence) and why the engine zooms."""
    s = float(moment["s"])
    end = max(float(moment.get("e") or 0.0), s + 1.5)
    said = " ".join(str(w["w"]) for w in words if s - 1.5 <= float(w["s"]) <= end)[:240]
    why = str(moment.get("kind") or "")
    if moment.get("part"):
        why += f" ({moment['part']})"
    return f'she says "{said}" -- zoom reason: {why}'
