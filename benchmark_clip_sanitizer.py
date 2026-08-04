"""Benchmark-only clip cleanup used before historical comparisons."""

from __future__ import annotations

import re
from typing import Any, Dict, List


SHORT_FRAGMENT_SECONDS = 0.40


def _duration(clip: Dict[str, Any]) -> float:
    try:
        return max(0.0, float(clip.get("end", 0.0)) - float(clip.get("start", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _normalized_text(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"\.{2,}", " ", text)
    # Preserve letters and digits from every Unicode script while treating
    # punctuation, symbols, and separators as whitespace.
    text = "".join(char if char.isalnum() else " " for char in text)
    return " ".join(text.split())


def _is_incomplete_fragment(text: str) -> bool:
    """Return true only for text with explicit evidence of truncation."""
    stripped = text.strip()
    if not _normalized_text(stripped):
        return True
    # Ellipsis is the concrete ASR artifact observed in Bloppers1. Do not infer
    # incompleteness from word count or ASCII punctuation because brief valid
    # utterances such as "Yes", "Buy now", or "好！" must be preserved.
    return stripped.endswith("...") or stripped.endswith("…")


def _trim_incomplete_tail(clip: Dict[str, Any]) -> Dict[str, Any]:
    """Trim only a demonstrably incomplete tail after the last complete sentence."""
    text = str(clip.get("text") or "").strip()
    punctuation_positions = [i for i, char in enumerate(text) if char in ".?!。？！"]
    if not punctuation_positions:
        return clip

    for boundary in reversed(punctuation_positions):
        retained = text[: boundary + 1].strip()
        remainder = text[boundary + 1 :].strip()
        if not remainder or not _is_incomplete_fragment(remainder):
            continue

        words = list(clip.get("words") or [])
        if not words:
            # Without word timestamps timing cannot be adjusted safely.
            return clip

        retained_words: List[Dict[str, Any]] = []
        for word in words:
            retained_words.append(word)
            if str(word.get("word") or "").strip().endswith((".", "?", "!", "。", "？", "！")):
                candidate_text = "".join(str(item.get("word") or "") for item in retained_words).strip()
                if candidate_text == retained:
                    break

        candidate_text = "".join(str(item.get("word") or "") for item in retained_words).strip()
        if candidate_text != retained:
            # Ellipsis dots are also punctuation positions. If one of those
            # boundaries cannot align to the timestamped words, keep scanning
            # backward until the preceding real sentence boundary is found.
            continue

        cleaned = dict(clip)
        cleaned["text"] = retained
        cleaned["words"] = retained_words
        try:
            cleaned["end"] = float(retained_words[-1].get("end", cleaned.get("end", 0.0)))
            cleaned["source_end"] = cleaned["end"]
        except (TypeError, ValueError):
            pass
        return cleaned

    return clip


def sanitize_benchmark_result(result: Dict[str, Any], *, use_semantic_v2: bool) -> Dict[str, Any]:
    """Remove only proven ASR artifacts after pipeline analysis."""
    clips = list(result.get("clips") or [])
    cleaned: List[Dict[str, Any]] = []
    previous_norm = ""
    previous_end = None

    for raw in clips:
        clip = _trim_incomplete_tail(dict(raw))
        text = str(clip.get("text") or "").strip()
        norm = _normalized_text(text)
        duration = _duration(clip)

        if not norm:
            continue

        try:
            start = float(clip.get("start", 0.0))
        except (TypeError, ValueError):
            start = 0.0
        contiguous = previous_end is not None and start - previous_end <= 0.35
        duplicate_artifact = contiguous and norm == previous_norm
        incomplete_artifact = duration < SHORT_FRAGMENT_SECONDS and _is_incomplete_fragment(text)

        # Short duration alone is not enough: preserve valid brief utterances.
        if duplicate_artifact or incomplete_artifact:
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
        "short_fragment_seconds": SHORT_FRAGMENT_SECONDS,
        "semantic_v2_requested": bool(use_semantic_v2),
    }
    return result
