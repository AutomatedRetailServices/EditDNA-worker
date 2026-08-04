"""Benchmark-only clip cleanup used before historical comparisons."""

from __future__ import annotations

import re
from typing import Any, Dict, List


SHORT_FRAGMENT_SECONDS = 0.40
TITLE_ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st"}
BUSINESS_SUFFIX_ABBREVIATIONS = {"inc", "ltd", "llc", "corp", "co"}


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


def _is_incomplete_fragment(text: str) -> bool:
    stripped = text.strip()
    normalized = _normalized_text(stripped)
    if not normalized:
        return True
    if stripped.endswith("...") or stripped.endswith("…"):
        return True
    words = normalized.split()
    if len(words) <= 2 and not stripped.endswith((".", "?", "!")):
        return True
    return False


def _is_sentence_boundary(text: str, position: int) -> bool:
    """Return whether punctuation marks a genuine sentence boundary."""
    punctuation = text[position]
    if punctuation in "?!":
        return True

    # Ellipses and decimal points do not finish a sentence on their own.
    if ((position > 0 and text[position - 1] == ".")
            or (position + 1 < len(text) and text[position + 1] == ".")):
        return False
    if (position > 0 and position + 1 < len(text)
            and text[position - 1].isdigit() and text[position + 1].isdigit()):
        return False

    token_match = re.search(r"([A-Za-z]+)$", text[:position])
    token = token_match.group(1).casefold() if token_match else ""
    if token in TITLE_ABBREVIATIONS or len(token) == 1:
        return False

    if token in BUSINESS_SUFFIX_ABBREVIATIONS:
        following_word = re.search(r"[A-Za-z]+", text[position + 1 :])
        # A lowercase continuation ("Acme Inc. to ...") proves this period is
        # part of the suffix. Preserve an uppercase continuation as a possible
        # new sentence rather than weakening genuine sentence boundaries.
        if following_word and following_word.group(0)[0].islower():
            return False

    return True


def _trim_incomplete_tail(clip: Dict[str, Any]) -> Dict[str, Any]:
    """Trim only a demonstrably incomplete tail after the last complete sentence."""
    text = str(clip.get("text") or "").strip()
    punctuation_positions = [
        i for i, char in enumerate(text)
        if char in ".?!" and _is_sentence_boundary(text, i)
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
            # Without word timestamps timing cannot be adjusted safely.
            return clip

        retained_words: List[Dict[str, Any]] = []
        for word in words:
            retained_words.append(word)
            if str(word.get("word") or "").strip().endswith((".", "?", "!")):
                candidate_text = "".join(str(item.get("word") or "") for item in retained_words).strip()
                if candidate_text == retained:
                    break

        candidate_text = "".join(str(item.get("word") or "") for item in retained_words).strip()
        if candidate_text != retained:
            return clip

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
