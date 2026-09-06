"""Deterministic fail-safe for incomplete retries when Hybrid windows are unavailable.

Hybrid deliberately fails open when an editorial window times out or exhausts its test
budget. That is correct for uncertain audience-facing material, but an undecided,
incomplete take can still be safely removed when a nearby later complete delivery covers
the same idea and preserves critical facts.

This fallback is intentionally narrow:
- it activates only when requested Hybrid windows exceed available windows;
- it considers only kept takes that received no semantic decision at all;
- the candidate must be incomplete_idea=False;
- a later kept complete take in the same source must strongly cover the candidate;
- numbers and negation must be preserved;
- ambiguous cases fail open and remain kept.

No benchmark phrases, timestamps, clip ids, or Gold data are embedded here.
"""
from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+(?:[-–][0-9]+)?%?", re.IGNORECASE)
_STOP = frozenset({
    "a", "al", "and", "as", "at", "con", "de", "del", "el", "en", "for", "from",
    "in", "la", "las", "lo", "los", "me", "of", "on", "or", "para", "por", "que",
    "se", "the", "to", "un", "una", "with", "y", "yo", "mi", "mis", "su", "sus",
})
_NEGATION = frozenset({"no", "not", "never", "nunca", "sin", "without"})


def _canon(token: str) -> str:
    raw = unicodedata.normalize("NFKD", str(token or "").casefold())
    return "".join(char for char in raw if not unicodedata.combining(char))


def _lexeme(token: str) -> str:
    token = _canon(token)
    if len(token) >= 5 and token.isalpha() and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _content(text: str) -> set[str]:
    return {
        token for token in (_lexeme(item) for item in _TOKEN_RE.findall(str(text or "")))
        if len(token) >= 3 and token not in _STOP
    }


def _critical(text: str) -> set[str]:
    out: set[str] = set()
    for raw in _TOKEN_RE.findall(str(text or "")):
        token = _canon(raw)
        if token in _NEGATION:
            out.add("__negation__")
        if any(ch.isdigit() for ch in token):
            out.add(token)
    return out


def _coverage(left_text: str, right_text: str) -> tuple[int, float]:
    left = _content(left_text)
    if not left:
        return 0, 0.0
    shared = len(left & _content(right_text))
    return shared, shared / max(1, len(left))


def apply_unavailable_retry_fallback(result, source_takes):
    """Core transform, extracted for direct use by composite_resolver.py.

    Identical logic to what used to live only inside this module's
    install-time monkeypatch closure -- see D-023. ``install_hybrid_
    unavailable_retry_fallback`` below now delegates here so its own
    (monkeypatch-based) tests keep working unchanged.
    """
    if result.requested_chunk_count <= result.available_chunk_count:
        return result
    if not result.kept:
        return result

    decided_ids = {str(clip_id) for clip_id, _label, _confidence in result.semantic_decisions}
    kept = tuple(sorted(result.kept, key=lambda take: (take.source_order, take.start, take.end, take.clip_id)))
    removed_ids: set[str] = set()
    fallback_rows: list[dict] = []

    for candidate in kept:
        if candidate.clip_id in decided_ids:
            continue
        if bool(candidate.complete_idea):
            continue
        own_content = _content(candidate.text)
        if len(own_content) < 2:
            continue

        best = None
        best_shared = 0
        best_coverage = 0.0
        for replacement in kept:
            if replacement.clip_id == candidate.clip_id:
                continue
            if replacement.source_asset_id != candidate.source_asset_id:
                continue
            if replacement.start <= candidate.end:
                continue
            delay = float(replacement.start) - float(candidate.end)
            if delay > 30.0:
                continue
            if not bool(replacement.complete_idea):
                continue
            if not _critical(candidate.text).issubset(_critical(replacement.text)):
                continue
            shared, coverage = _coverage(candidate.text, replacement.text)
            if shared < 2:
                continue
            required = 0.72 if len(own_content) <= 4 else 0.62
            if coverage < required:
                continue
            if coverage > best_coverage or (coverage == best_coverage and shared > best_shared):
                best = replacement
                best_shared = shared
                best_coverage = coverage

        if best is None:
            continue

        removed_ids.add(candidate.clip_id)
        fallback_rows.append({
            "clip_id": candidate.clip_id,
            "replacement_clip_id": best.clip_id,
            "reason": "hybrid_unavailable_incomplete_retry_covered_by_later_complete_delivery",
            "shared_content_tokens": best_shared,
            "coverage": round(best_coverage, 4),
            "requested_chunks": result.requested_chunk_count,
            "available_chunks": result.available_chunk_count,
        })

    if not removed_ids:
        return result

    survivors = tuple(take for take in kept if take.clip_id not in removed_ids)
    deleted_ids = {take.clip_id for take in result.deleted} | removed_ids
    deleted = tuple(take for take in source_takes if take.clip_id in deleted_ids)
    diagnostics = tuple(result.diagnostics) + ({
        "hybrid_unavailable_retry_fallback": fallback_rows,
        "deleted_ids": sorted(removed_ids),
    },)
    return type(result)(
        kept=survivors,
        deleted=deleted,
        requested_chunk_count=result.requested_chunk_count,
        available_chunk_count=result.available_chunk_count,
        diagnostics=diagnostics,
        semantic_decisions=result.semantic_decisions,
    )


def install_hybrid_unavailable_retry_fallback() -> None:
    from . import hybrid_session_cleanup

    original = hybrid_session_cleanup.apply_hybrid_session_cleanup
    if getattr(original, "_cutsell_hybrid_unavailable_retry_fallback", False):
        return

    def apply_with_unavailable_retry_fallback(*args, **kwargs):
        if args:
            source_takes = tuple(args[0])
            result = original(source_takes, *args[1:], **kwargs)
        else:
            source_takes = tuple(kwargs.get("takes") or ())
            call_kwargs = dict(kwargs)
            call_kwargs["takes"] = source_takes
            result = original(**call_kwargs)
        return apply_unavailable_retry_fallback(result, source_takes)

    apply_with_unavailable_retry_fallback._cutsell_hybrid_unavailable_retry_fallback = True
    hybrid_session_cleanup.apply_hybrid_session_cleanup = apply_with_unavailable_retry_fallback
