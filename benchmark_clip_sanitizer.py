"""Benchmark-only clip cleanup used before historical comparisons."""

from __future__ import annotations

import re
from typing import Any, Dict, List


SHORT_FRAGMENT_SECONDS = 0.40
COMMON_ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc",
}
COMMON_BUSINESS_SUFFIXES = {"inc", "ltd", "corp", "co", "llc"}


def _duration(clip: Dict[str, Any]) -> float:
    try:
        return max(0.0, float(clip.get("end", 0.0)) - float(clip.get("start", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _normalized_text(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"\.{2,}", " ", text)
    text = "".join(char if char.isalnum() else " " for char in text)
    return " ".join(text.split())


def _duplicate_key(value: Any) -> str:
    """Normalize whitespace only so punctuation and capitalization stay meaningful."""
    return " ".join(str(value or "").split())


def _is_incomplete_fragment(text: str) -> bool:
    """Return true only for text with explicit evidence of truncation."""
    stripped = text.strip()
    if not _normalized_text(stripped):
        return True
    return stripped.endswith("...") or stripped.endswith("…")


def _business_suffix_has_incomplete_lowercase_tail(text: str, boundary: int) -> bool:
    """Return true when a suffix period is followed by an explicitly incomplete tail."""
    remainder = text[boundary + 1 :].strip()
    return _is_incomplete_fragment(remainder)


def _is_false_sentence_boundary(text: str, boundary: int) -> bool:
    """Reject periods used by abbreviations, initials, decimals, ellipses, or suffixes."""
    if text[boundary] != ".":
        return False

    if boundary > 0 and text[boundary - 1] == ".":
        return True
    if boundary + 1 < len(text) and text[boundary + 1] == ".":
        return True
    if (
        boundary > 0
        and boundary + 1 < len(text)
        and text[boundary - 1].isdigit()
        and text[boundary + 1].isdigit()
    ):
        return True

    prefix = text[:boundary]
    match = re.search(r"([\w]+)$", prefix, flags=re.UNICODE)
    token = match.group(1).casefold() if match else ""
    if token in COMMON_ABBREVIATIONS or (len(token) == 1 and token.isalpha()):
        return True
    return token in COMMON_BUSINESS_SUFFIXES and _business_suffix_has_incomplete_lowercase_tail(text, boundary)


def _trim_incomplete_tail(clip: Dict[str, Any]) -> Dict[str, Any]:
    """Trim only a demonstrably incomplete tail after the last complete sentence."""
    text = str(clip.get("text") or "").strip()
    punctuation_positions = [
        i
        for i, char in enumerate(text)
        if char in ".?!。？！" and not _is_false_sentence_boundary(text, i)
    ]
    if not punctuation_positions:
        return clip

    for boundary in reversed(punctuation_positions):
        retained = text[: boundary + 1].strip()
        remainder = text[boundary + 1 :].strip()
        if not remainder or not _is_incomplete_fragment(remainder):
            continue

        words = list(clip.get("words") or [])
        if not words:
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
    previous_duplicate_key = ""
    previous_end = None

    for raw in clips:
        clip = _trim_incomplete_tail(dict(raw))
        text = str(clip.get("text") or "").strip()
        norm = _normalized_text(text)
        duplicate_key = _duplicate_key(text)
        duration = _duration(clip)

        if not norm:
            continue

        try:
            start = float(clip.get("start", 0.0))
        except (TypeError, ValueError):
            start = 0.0
        try:
            end = float(clip.get("end", start))
        except (TypeError, ValueError):
            end = start

        contiguous = previous_end is not None and start - previous_end <= 0.35
        duplicate_artifact = contiguous and duplicate_key == previous_duplicate_key
        incomplete_artifact = duration < SHORT_FRAGMENT_SECONDS and _is_incomplete_fragment(text)

        if duplicate_artifact:
            previous_end = end
            continue
        if incomplete_artifact:
            continue

        cleaned.append(clip)
        previous_duplicate_key = duplicate_key
        previous_end = end

    result["clips"] = cleaned
    result["benchmark_sanitization"] = {
        "input_clip_count": len(clips),
        "output_clip_count": len(cleaned),
        "short_fragment_seconds": SHORT_FRAGMENT_SECONDS,
        "semantic_v2_requested": bool(use_semantic_v2),
    }
    return result
