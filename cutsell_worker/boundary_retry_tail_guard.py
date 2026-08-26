"""Trim only short trailing retry tails that are covered by the next selected delivery.

Complete-idea recovery may occasionally absorb the beginning of the next thought when the
original clip boundary touches that thought's first word. This guard does not infer a cut
from silence or punctuation alone. It trims only when all of the following are true:
- the expanded left clip contains a completed sentence followed by a short trailing clause;
- the next selected clip is nearby in the same source;
- the short trailing clause is strongly covered by that next selected delivery;
- numeric facts and negation in the trailing clause are preserved by the next delivery.

Ambiguous or unique speech fails open and is kept. No benchmark phrases, clip ids, or
source timestamps are hardcoded.
"""
from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+(?:[-–][0-9]+)?%?", re.IGNORECASE)
_STOP = frozenset({
    "a", "al", "and", "as", "at", "con", "de", "del", "el", "en", "for", "from",
    "in", "la", "las", "lo", "los", "me", "of", "on", "or", "para", "por", "que",
    "se", "the", "to", "un", "una", "with", "y",
})
_NEGATION = frozenset({"no", "not", "never", "nunca", "sin", "without"})


def _canon(token: str) -> str:
    raw = unicodedata.normalize("NFKD", str(token or "").casefold())
    return "".join(char for char in raw if not unicodedata.combining(char))


def _lexeme(token: str) -> str:
    """Small inflection normalizer used only for lexical retry coverage."""
    token = _canon(token)
    if len(token) >= 5 and token.isalpha() and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_lexeme(token) for token in _TOKEN_RE.findall(str(text or "")))


def _content(text: str) -> set[str]:
    return {token for token in _tokens(text) if len(token) >= 3 and token not in _STOP}


def _critical(text: str) -> set[str]:
    out: set[str] = set()
    for raw_token in _TOKEN_RE.findall(str(text or "")):
        token = _canon(raw_token)
        if token in _NEGATION:
            out.add("__negation__")
        if any(ch.isdigit() for ch in token):
            out.add(token)
    return out


def _semantic_coverage(tail_text: str, next_text: str) -> tuple[int, float, int]:
    tail = _content(tail_text)
    if not tail:
        return 0, 0.0, 0
    shared = len(tail & _content(next_text))
    return shared, shared / max(1, len(tail)), len(tail)


def install_boundary_retry_tail_guard() -> None:
    from . import final_boundary_authority as authority

    original = authority._reconcile_same_source_overlaps
    if getattr(original, "_cutsell_boundary_retry_tail_guard", False):
        return

    def protected(originals, expanded, source_map):
        output, rows = original(originals, expanded, source_map)
        output = list(output)
        extra_rows: list[dict] = []

        for index in range(len(output) - 1):
            left = output[index]
            right = output[index + 1]
            left_orig = originals[index]
            right_orig = originals[index + 1]
            if left.source_asset_id != right.source_asset_id:
                continue
            if left_orig.clip_id == right_orig.clip_id:
                continue
            if float(right_orig.start) - float(left_orig.end) > 24.0:
                continue

            words = tuple(left.words)
            boundary_candidates = [
                i for i, word in enumerate(words[:-1]) if authority._terminal(word)
            ]
            if not boundary_candidates:
                continue
            boundary_index = boundary_candidates[-1]
            tail_words = words[boundary_index + 1 :]
            if not tail_words or len(tail_words) > 9:
                continue
            tail_start = float(tail_words[0].start)
            tail_end = float(tail_words[-1].end)
            tail_duration = max(0.0, tail_end - tail_start)
            if tail_duration > 3.5:
                continue

            tail_text = " ".join(str(word.text).strip() for word in tail_words).strip()
            shared, coverage, content_count = _semantic_coverage(tail_text, right.text)
            # For a very short tail, two independent shared content tokens are already
            # strong retry evidence. Longer tails keep the stricter proportional gate.
            short_tail_covered = content_count <= 4 and shared >= 2 and coverage >= 0.50
            long_tail_covered = shared >= 2 and coverage >= 0.55
            if not (short_tail_covered or long_tail_covered):
                continue
            if not _critical(tail_text).issubset(_critical(right.text)):
                continue

            safe_end = float(words[boundary_index].end)
            if safe_end >= float(left.end) - 0.05:
                continue

            source_words = source_map.get(left.source_asset_id, ())
            trimmed = authority._rebuild_clip(left, source_words, float(left.start), safe_end)
            if float(trimmed.end) > safe_end + 0.08:
                continue

            output[index] = trimmed
            extra_rows.append({
                "action": "trim_retry_covered_trailing_clause",
                "left_clip_id": left.clip_id,
                "right_clip_id": right.clip_id,
                "original_end": round(float(left.end), 3),
                "result_end": round(float(trimmed.end), 3),
                "removed_tail_sec": round(max(0.0, float(left.end) - float(trimmed.end)), 3),
                "tail_word_count": len(tail_words),
                "tail_content_token_count": content_count,
                "tail_shared_content_tokens": shared,
                "tail_semantic_coverage": round(coverage, 3),
                "critical_preserved": True,
            })

        return output, list(rows) + extra_rows

    protected._cutsell_boundary_retry_tail_guard = True
    authority._reconcile_same_source_overlaps = protected
