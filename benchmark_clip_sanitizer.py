"""Benchmark-only clip cleanup used before historical comparisons.

This keeps production rendering behavior unchanged while making benchmark output
faithful to the requested feature flags and robust to ASR timestamp artifacts.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


MIN_FRAGMENT_SECONDS = 0.40
_TERMINAL_PUNCTUATION_RE = re.compile(r"^(.+?[.!?])(?:\s+.+)?$", re.DOTALL)


def _duration(clip: Dict[str, Any]) -> float:
    try:
        return max(0.0, float(clip.get("end", 0.0)) - float(clip.get("start", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _normalized_text(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"\.{2,}", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _trim_after_complete_sentence(clip: Dict[str, Any]) -> Dict[str, Any]:
    """Remove a trailing incomplete phrase after a completed sentence.

    Word timestamps are used when available so clip timing remains aligned with
    the retained sentence.
    """
    text = str(clip.get("text") or "").strip()
    match = _TERMINAL_PUNCTUATION_RE.match(text)
    if not match:
        return clip

    retained = match.group(1).strip()
    if retained == text:
        return clip

    words = list(clip.get("words") or [])
    retained_words: List[Dict[str, Any]] = []
    for word in words:
        retained_words.append(word)
        if str(word.get("word") or "").strip().endswith((".", "?", "!")):
            break

    cleaned = dict(clip)
    cleaned["text"] = retained
    if retained_words:
        cleaned["words"] = retained_words
        try:
            cleaned["end"] = float(retained_words[-1].get("end", cleaned.get("end", 0.0)))
            cleaned["source_end"] = cleaned["end"]
        except (TypeError, ValueError):
            pass
    return cleaned


def _remove_disabled_v2_metadata(clip: Dict[str, Any]) -> None:
    """Make disabled Semantic V2 unambiguously absent from benchmark output."""
    meta = clip.setdefault("meta", {})
    meta.pop("semantic_v2", None)


def sanitize_benchmark_result(
    result: Dict[str, Any], *, use_semantic_v2: bool
) -> Dict[str, Any]:
    """Return benchmark results without ASR micro-fragments or false V2 usage."""
    clips = list(result.get("clips") or [])
    cleaned: List[Dict[str, Any]] = []
    previous_norm = ""
    previous_end = None

    for raw in clips:
        clip = _trim_after_complete_sentence(dict(raw))
        text = str(clip.get("text") or "").strip()
        norm = _normalized_text(text)
        duration = _duration(clip)

        if not use_semantic_v2:
            _remove_disabled_v2_metadata(clip)

        # Drop timestamp artifacts and punctuation-only remnants. This targets
        # impossible edit units such as 20–260 ms repeated Whisper fragments.
        if duration < MIN_FRAGMENT_SECONDS:
            continue
        if not norm:
            continue

        # Drop contiguous duplicate fragments while preserving legitimate later
        # repeats elsewhere in a recording.
        try:
            start = float(clip.get("start", 0.0))
        except (TypeError, ValueError):
            start = 0.0
        contiguous = previous_end is not None and start - previous_end <= 0.35
        if contiguous and norm == previous_norm:
            continue

        cleaned.append(clip)
        previous_norm = norm
        try:
            previous_end = float(clip.get("end", start))
        except (TypeError, ValueError):
            previous_end = start

    result["clips"] = cleaned
    result["benchmark_sanitization"] = {
        "input_clip_count": len(clips),
        "output_clip_count": len(cleaned),
        "minimum_fragment_seconds": MIN_FRAGMENT_SECONDS,
        "semantic_v2_requested": bool(use_semantic_v2),
    }
    return result
