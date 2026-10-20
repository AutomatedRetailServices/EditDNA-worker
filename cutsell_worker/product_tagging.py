"""Brand and product of a video, read by Claude Haiku 5.5 AFTER the cut (a separate step: the cut engine, its prompts
and its calls are not touched, and nothing here can change or delay the cut).

1. Text: Haiku reads the transcript already made for the cut (no new Deepgram call) plus the creator's own brand and
   product names (so it reuses them) and an optional Photos-album hint -> {brand, product, category, confidence,
   evidence, product_moment_seconds}.
2. Photo, only when the text is not enough (no brand/product, or confidence < 0.7): 1-2 frames from the original video
   at the moment she shows / talks about the product (else 2 frames from the first 10 s) are read by Haiku together
   with that phrase. Nothing is uploaded; the frame used becomes the product's small thumbnail.
3. Still < 0.7: the video stays Uncategorized with the answer kept as a suggestion (product_catalog).

Every function here returns a result and never raises: a failure leaves the video Uncategorized.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import requests

_LOG = logging.getLogger("cutsell.product_tagging")
MODEL = "claude-haiku-5-5"
API_URL = "https://api.anthropic.com/v1/messages"
PRICE_PER_MTOK = {"input": 0.10, "output": 0.50, "cache_write": 0.125, "cache_read": 0.01}   # Haiku 5.5, <= 100k
MIN_CONFIDENCE = 0.7
MAX_TRANSCRIPT_WORDS = 3000
FRAME_LONG_EDGE = 768
DEICTIC = re.compile(r"\b(this|these|it|look|here|mira|este|esta|esto|estos|estas|aquí|aqui)\b", re.I)

_NULLABLE_STR = {"anyOf": [{"type": "string"}, {"type": "null"}]}
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["brand", "product", "category", "confidence", "evidence", "product_moment_seconds"],
    "properties": {
        "brand": _NULLABLE_STR, "product": _NULLABLE_STR, "category": _NULLABLE_STR,
        "confidence": {"type": "number"}, "evidence": _NULLABLE_STR,
        "product_moment_seconds": {"anyOf": [{"type": "number"}, {"type": "null"}]},
    },
}

TEXT_PROMPT = """You organize a TikTok Shop creator's videos by BRAND and PRODUCT. Below is the transcript of ONE video
(times in seconds). Tell which brand and which product this video is about: the main item she sells, reviews or shows.
- Write the brand and the product with their official spelling (the speech recognizer can mishear them).
- If one of HER EXISTING NAMES below is the same brand or the same product, use exactly that name.
- The product name is the product line without the brand (e.g. brand "CeraVe", product "Hydrating Cleanser").
- If she never names the brand or the product, return null for it. Never guess a brand from the category alone.
- confidence: 0 to 1, how sure you are of BOTH brand and product from what she says.
- evidence: the exact words from the transcript that name it (copied), or null.
- product_moment_seconds: the time of the line where she shows or presents the product ("look at this", "this one",
  "mira", "the cleanser"...), or null.
- category: one or two words (Skincare, Makeup, Fragrance, Haircare, Fashion, Shoes, Jewelry, Home, Kitchen, Tech,
  Supplements, Kids, Pets, Other), or null.
Return only the JSON."""

PHOTO_PROMPT = """These are frames from a TikTok Shop creator's video, taken when she shows the product. Read the
product's label, packaging or logo. Tell the BRAND and the PRODUCT (product line without the brand), official
spelling. If one of HER EXISTING NAMES below is the same, use exactly that name. If you cannot read it, return null.
confidence 0 to 1 for both. evidence: the words you read on the product (or the phrase she says), else null.
product_moment_seconds: null. category: one or two words, or null. Return only the JSON."""


def _known_text(known: list[dict]) -> str:
    rows = [f"- {k.get('brand') or '?'} · {k.get('product') or '(brand only)'}" for k in (known or [])[:80]]
    return "HER EXISTING NAMES:\n" + ("\n".join(rows) if rows else "(none yet)")


def transcript_lines(draft: dict) -> tuple[str, list[dict]]:
    """Every word of the video (kept and hidden parts) in source time, from the stored draft: no new transcription.
    -> (text with a time per line, words of the first source [{w, s}])."""
    clips = [c for c in (draft.get("selected") or []) + (draft.get("alternates") or []) if isinstance(c, dict)]
    clips.sort(key=lambda c: (int(c.get("source_order") or 0), float(c.get("start") or 0)))
    lines, first_words, count = [], [], 0
    for clip in clips:
        words = [w for w in clip.get("words") or [] if isinstance(w, dict) and str(w.get("text") or "").strip()]
        if not words:
            continue
        if int(clip.get("source_order") or 0) == 0:
            first_words += [{"w": str(w["text"]), "s": float(w.get("start") or 0)} for w in words]
        prefix = f"[{float(words[0].get('start') or 0):.1f}s]" + (f" (clip {int(clip.get('source_order') or 0) + 1})"
                                                                   if int(clip.get("source_order") or 0) else "")
        text = " ".join(str(w["text"]) for w in words)
        lines.append(f"{prefix} {text}")
        count += len(words)
        if count >= MAX_TRANSCRIPT_WORDS:
            break
    return "\n".join(lines), first_words


def _cost(usage: dict) -> float:
    return ((usage.get("input_tokens") or 0) * PRICE_PER_MTOK["input"]
            + (usage.get("output_tokens") or 0) * PRICE_PER_MTOK["output"]
            + (usage.get("cache_creation_input_tokens") or 0) * PRICE_PER_MTOK["cache_write"]
            + (usage.get("cache_read_input_tokens") or 0) * PRICE_PER_MTOK["cache_read"]) / 1e6


def _ask_haiku(content: list[dict], *, timeout: float = 60.0) -> tuple[dict | None, dict, str | None]:
    """-> (answer, usage, error). One retry on a transient failure."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return None, {}, "no_key"
    body = {"model": MODEL, "max_tokens": 3000, "messages": [{"role": "user", "content": content}],
            "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}}}
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    error = None
    for attempt in range(2):
        try:
            response = requests.post(API_URL, json=body, headers=headers, timeout=(10, timeout))
            if response.status_code == 200:
                data = response.json()
                usage = dict(data.get("usage") or {})
                text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
                try:
                    return json.loads(text), usage, None
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


def _clean(answer: dict | None) -> dict:
    answer = answer or {}
    def text(value):
        value = str(value).strip() if value not in (None, "") else None
        return value if value and value.lower() not in ("null", "none", "unknown", "n/a") else None
    try:
        confidence = max(0.0, min(1.0, float(answer.get("confidence") or 0)))
    except (TypeError, ValueError):
        confidence = 0.0
    moment = answer.get("product_moment_seconds")
    try:
        moment = float(moment) if moment is not None else None
    except (TypeError, ValueError):
        moment = None
    return {"brand": text(answer.get("brand")), "product": text(answer.get("product")),
            "category": text(answer.get("category")), "confidence": round(confidence, 2),
            "evidence": text(answer.get("evidence")), "product_moment_seconds": moment}


def classify_text(transcript: str, known: list[dict], album_hint: str | None = None) -> tuple[dict, dict]:
    hint = f"\nPHOTOS ALBUM THIS VIDEO CAME FROM: \"{album_hint}\" (often the brand or product name)." if album_hint else ""
    content = [{"type": "text", "text": TEXT_PROMPT + "\n\n" + _known_text(known) + hint + "\n\nTRANSCRIPT:\n" + transcript}]
    answer, usage, error = _ask_haiku(content)
    return _clean(answer), {"usage": usage, "error": error}


def moment_for_photo(result: dict, first_words: list[dict], duration: float) -> list[float]:
    """Seconds of the 1-2 frames to read: Haiku's product moment, else the first 'this / look / mira...' word, else
    two frames inside the first 10 s."""
    moment = result.get("product_moment_seconds")
    if moment is None:
        hit = next((w for w in first_words if DEICTIC.search(w["w"])), None)
        moment = hit["s"] if hit else None
    end = max(0.5, float(duration or 10.0) - 0.3)
    if moment is None:
        return [round(min(2.0, end), 2), round(min(6.0, end), 2)]
    moment = max(0.0, min(float(moment), end))
    return sorted({round(moment + 0.5, 2) if moment + 0.5 <= end else round(moment, 2),
                   round(min(moment + 1.5, end), 2)})


def grab_frames(video_path: str, seconds: list[float], directory: str) -> list[bytes]:
    """JPEG frames (long edge FRAME_LONG_EDGE) read straight from the original file; nothing is uploaded."""
    frames = []
    for index, t in enumerate(seconds):
        out = str(Path(directory) / f"frame-{index}.jpg")
        scale = (f"scale='if(gt(iw,ih),{FRAME_LONG_EDGE},-2)':'if(gt(iw,ih),-2,{FRAME_LONG_EDGE})'")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(video_path), "-frames:v", "1",
                        "-vf", scale, "-q:v", "4", out], capture_output=True, timeout=60)
        if Path(out).exists() and Path(out).stat().st_size > 0:
            frames.append(Path(out).read_bytes())
    return frames


def thumbnail_jpeg(frame: bytes, directory: str) -> str | None:
    """A small (160 px wide) copy of the frame, base64, kept as the product's thumbnail."""
    src, out = Path(directory) / "thumb-src.jpg", Path(directory) / "thumb.jpg"
    src.write_bytes(frame)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", "scale=160:-2", "-q:v", "6", str(out)],
                   capture_output=True, timeout=30)
    return base64.b64encode(out.read_bytes()).decode() if out.exists() else None


def classify_photo(frames: list[bytes], phrase: str | None, known: list[dict]) -> tuple[dict, dict]:
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                            "data": base64.b64encode(f).decode()}} for f in frames]
    said = f"\nWHAT SHE SAYS AT THAT MOMENT: \"{phrase}\"" if phrase else ""
    content.append({"type": "text", "text": PHOTO_PROMPT + "\n\n" + _known_text(known) + said})
    answer, usage, error = _ask_haiku(content)
    return _clean(answer), {"usage": usage, "error": error}


def needs_photo(result: dict) -> bool:
    return not (result.get("brand") and result.get("product")) or float(result.get("confidence") or 0) < MIN_CONFIDENCE


def _phrase_at(first_words: list[dict], seconds: float | None) -> str | None:
    if seconds is None or not first_words:
        return None
    near = [w["w"] for w in first_words if seconds - 2.0 <= w["s"] <= seconds + 4.0]
    return " ".join(near)[:200] or None


def tag(draft: dict, known: list[dict], *, video_path: str | None = None, duration: float | None = None,
        album_hint: str | None = None) -> dict:
    """The whole step for one video. Never raises. Result: brand, product, category, confidence, evidence,
    used_photo, needs_photo (photo wanted but the video is not here yet), thumb_jpeg_b64, log {tokens, cost_usd,
    seconds, errors}."""
    started = time.time()
    log = {"calls": [], "errors": []}
    try:
        transcript, first_words = transcript_lines(draft)
        if not transcript.strip():
            result = _clean(None)
            log["errors"].append("no_transcript")
        else:
            result, info = classify_text(transcript, known, album_hint)
            log["calls"].append({"kind": "text", **info["usage"]})
            if info["error"]:
                log["errors"].append("text_" + info["error"])
        result.update({"used_photo": False, "needs_photo": False, "thumb_jpeg_b64": None})
        if needs_photo(result) and transcript.strip():
            if video_path and Path(video_path).exists():
                photo_result, photo_info, thumb = _photo_step(result, first_words, known, video_path, duration)
                log["calls"].append({"kind": "photo", **photo_info["usage"]})
                if photo_info["error"]:
                    log["errors"].append("photo_" + photo_info["error"])
                if photo_result is not None:
                    better = (photo_result.get("brand") and photo_result.get("product")
                              and photo_result["confidence"] >= result["confidence"])
                    if better:
                        evidence = photo_result.get("evidence")
                        result.update({k: photo_result[k] for k in ("brand", "product", "category", "confidence")})
                        result["evidence"] = evidence or result.get("evidence")
                    result["used_photo"] = True
                    result["thumb_jpeg_b64"] = thumb
            else:
                result["needs_photo"] = True        # the video is still uploading: the follow-up job reads it
    except Exception as exc:                        # never block or fail the cut
        result = {**_clean(None), "used_photo": False, "needs_photo": False, "thumb_jpeg_b64": None}
        log["errors"].append("crash_" + exc.__class__.__name__)
    tokens_in = sum(int(c.get("input_tokens") or 0) for c in log["calls"])
    tokens_out = sum(int(c.get("output_tokens") or 0) for c in log["calls"])
    result["log"] = {"model": MODEL, "tokens_in": tokens_in, "tokens_out": tokens_out,
                     "cost_usd": round(sum(_cost(c) for c in log["calls"]), 6), "calls": len(log["calls"]),
                     "seconds": round(time.time() - started, 2), "errors": log["errors"]}
    _LOG.info("product_tagging %s", json.dumps({k: result.get(k) for k in ("brand", "product", "confidence", "used_photo",
                                                                           "needs_photo")} | result["log"]))
    return result


def _photo_step(result, first_words, known, video_path, duration):
    with tempfile.TemporaryDirectory(prefix="cutsell-tag-") as directory:
        seconds = moment_for_photo(result, first_words, float(duration or 0) or _duration(video_path))
        frames = grab_frames(video_path, seconds, directory)
        if not frames:
            return None, {"usage": {}, "error": "no_frames"}, None
        phrase = _phrase_at(first_words, result.get("product_moment_seconds") if result.get("product_moment_seconds")
                            is not None else seconds[0])
        photo_result, info = classify_photo(frames, phrase, known)
        thumb = thumbnail_jpeg(frames[0], directory)
        return photo_result, info, thumb


def _duration(path: str) -> float:
    try:
        from .media_probe import probe_media
        return float(probe_media(path).duration_sec)
    except Exception:
        return 10.0


def album_hint_of(sources) -> str | None:
    """The Photos album the creator picked the video from (sent by the app in metadata.album_hint): a free hint."""
    for source in sources or []:
        meta = getattr(source, "metadata", None) if not isinstance(source, dict) else source.get("metadata")
        if isinstance(meta, dict) and str(meta.get("album_hint") or "").strip():
            return str(meta["album_hint"]).strip()[:80]
    return None


def enabled() -> bool:
    """CUTSELL_PRODUCT_TAGGING=0 turns the step off (rollback without a redeploy)."""
    return os.environ.get("CUTSELL_PRODUCT_TAGGING", "1").strip() != "0"


def tag_and_record(*, user_id: str, project_id: str, draft: dict, sources, video_path: str | None,
                   duration: float | None = None, client=None) -> dict:
    """Worker entry point, called after the cut is saved and announced. Never raises."""
    if not enabled():
        return {"status": "off"}
    try:
        from . import product_catalog
        known = product_catalog.known_names(product_catalog.load(user_id, client=client))
        result = tag(draft, known, video_path=video_path, duration=duration, album_hint=album_hint_of(sources))
        entry = product_catalog.record_ai_result(user_id, project_id, result, client=client)
        return {"status": "done", "brand": result.get("brand"), "product": result.get("product"),
                "confidence": result.get("confidence"), "used_photo": result.get("used_photo"),
                "needs_photo": result.get("needs_photo"), "placed": bool(entry.get("product_id")),
                "log": result.get("log")}
    except Exception as exc:
        _LOG.warning("product_tagging_failed %s", exc.__class__.__name__)
        return {"status": "failed", "error": exc.__class__.__name__}


def photo_followup(*, user_id: str, project_id: str, video_path: str, client=None) -> dict:
    """Audio first: the video arrived after the cut. If the text step asked for a photo, read it now. Never raises."""
    if not enabled():
        return {"status": "off"}
    try:
        from . import product_catalog
        from .draft_store import get_draft_snapshot
        entry = product_catalog.video_entry(user_id, project_id, client=client) or {}
        suggestion = entry.get("suggestion") or {}
        if entry.get("source") == "user" or not suggestion.get("needs_photo"):
            return {"status": "not_needed"}
        snapshot = get_draft_snapshot(user_id=user_id, project_id=project_id)
        draft = (snapshot.get("undo") or [snapshot.get("draft")])[0] or snapshot.get("draft") or {}
        known = product_catalog.known_names(product_catalog.load(user_id, client=client))
        result = tag(draft, known, video_path=video_path, album_hint=album_hint_of(snapshot.get("sources") or []))
        product_catalog.record_ai_result(user_id, project_id, result, client=client)
        return {"status": "done", "used_photo": result.get("used_photo"), "confidence": result.get("confidence"),
                "log": result.get("log")}
    except Exception as exc:
        _LOG.warning("product_tagging_followup_failed %s", exc.__class__.__name__)
        return {"status": "failed", "error": exc.__class__.__name__}
