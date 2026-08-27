"""Final Hybrid guard for performance-only restores that are actually retries.

Composite Best Take may conservatively restore a semantically failed complete delivery
when the only delete evidence is performance/reset evidence and the delivery carries
unique words. That is useful for complementary story material, but it must not resurrect
an earlier failed retry when a later authoritative winner begins with the same strong
opening. In that pattern the unique words belong to the losing attempt, not to a separate
story beat.

This guard runs after Composite Best Take, reads only Composite's own restore audit, and
removes a restored candidate only when:
- Composite explicitly restored it as performance-only;
- Hybrid labeled it failed with high confidence;
- the named peer is a high-confidence winner/keep;
- both deliveries share the same strong content opening;
- the candidate precedes the peer in the same source.

Ambiguity fails open. No benchmark ids, timestamps, phrases, or Human Gold data are
embedded here.
"""
from __future__ import annotations

import re
import sys
import unicodedata

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+(?:[-–][0-9]+)?%?", re.IGNORECASE)
_STOP = frozenset({
    "a", "al", "and", "are", "as", "at", "be", "but", "by", "como", "con", "de", "del",
    "el", "en", "es", "esta", "este", "for", "from", "in", "is", "it", "la", "las", "lo",
    "los", "me", "mi", "mis", "of", "on", "or", "para", "pero", "por", "porque", "que",
    "se", "so", "su", "sus", "that", "the", "this", "to", "un", "una", "was", "we",
    "with", "y", "yo",
})


def _canon(token: str) -> str:
    raw = unicodedata.normalize("NFKD", str(token or "").casefold())
    return "".join(ch for ch in raw if not unicodedata.combining(ch))


def _ordered_content(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in (_canon(item) for item in _TOKEN_RE.findall(str(text or "")))
        if len(token) >= 3 and token not in _STOP
    )


def same_strong_opening(left_text: str, right_text: str, *, width: int = 4) -> bool:
    left = _ordered_content(left_text)
    right = _ordered_content(right_text)
    if len(left) < width + 2 or len(right) < width + 2:
        return False
    return left[:width] == right[:width]


def _semantic_map(rows) -> dict[str, tuple[str, float]]:
    best: dict[str, tuple[str, float]] = {}
    for clip_id, label, confidence in rows:
        clip_id = str(clip_id)
        confidence = float(confidence)
        current = best.get(clip_id)
        if current is None or confidence > current[1]:
            best[clip_id] = (str(label), confidence)
    return best


def _composite_performance_restore_rows(diagnostics):
    rows = []
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict):
            continue
        section = diagnostic.get("hybrid_composite_best_take")
        if not isinstance(section, dict):
            continue
        restored = section.get("restored_performance_only")
        if isinstance(restored, list):
            rows.extend(item for item in restored if isinstance(item, dict))
    return rows


def revoke_retry_like_performance_restores(result, source_takes):
    rows = _composite_performance_restore_rows(result.diagnostics)
    if not rows:
        return result

    by_id = {take.clip_id: take for take in source_takes}
    semantic = _semantic_map(result.semantic_decisions)
    remove_ids: set[str] = set()
    audit: list[dict] = []

    for row in rows:
        clip_id = str(row.get("clip_id") or "")
        peer_id = str(row.get("peer_clip_id") or "")
        candidate = by_id.get(clip_id)
        peer = by_id.get(peer_id)
        if candidate is None or peer is None:
            continue
        if candidate.source_asset_id != peer.source_asset_id:
            continue
        if float(candidate.end) > float(peer.start):
            continue

        label, confidence = semantic.get(clip_id, ("", 0.0))
        peer_label, peer_confidence = semantic.get(peer_id, ("", 0.0))
        if label != "failed" or confidence < 0.80:
            continue
        if peer_label not in {"winner", "keep"} or peer_confidence < 0.85:
            continue
        if not same_strong_opening(candidate.text, peer.text):
            continue

        remove_ids.add(clip_id)
        audit.append({
            "clip_id": clip_id,
            "peer_clip_id": peer_id,
            "reason": "revoke_performance_restore_same_strong_opening_as_later_winner",
            "semantic_confidence": round(confidence, 4),
            "peer_confidence": round(peer_confidence, 4),
        })

    if not remove_ids:
        return result

    kept = tuple(take for take in result.kept if take.clip_id not in remove_ids)
    deleted_ids = {take.clip_id for take in result.deleted} | remove_ids
    deleted = tuple(take for take in source_takes if take.clip_id in deleted_ids)
    diagnostics = tuple(result.diagnostics) + ({
        "hybrid_performance_retry_restore_guard": audit,
        "deleted_ids": sorted(remove_ids),
    },)
    return type(result)(
        kept=kept,
        deleted=deleted,
        requested_chunk_count=result.requested_chunk_count,
        available_chunk_count=result.available_chunk_count,
        diagnostics=diagnostics,
        semantic_decisions=result.semantic_decisions,
    )


def install_hybrid_performance_retry_restore_guard() -> None:
    from . import hybrid_session_cleanup

    original = hybrid_session_cleanup.apply_hybrid_session_cleanup
    if getattr(original, "_cutsell_hybrid_performance_retry_restore_guard", False):
        return

    def apply_with_performance_retry_restore_guard(*args, **kwargs):
        if args:
            source_takes = tuple(args[0])
            result = original(source_takes, *args[1:], **kwargs)
        else:
            source_takes = tuple(kwargs.get("takes") or ())
            call_kwargs = dict(kwargs)
            call_kwargs["takes"] = source_takes
            result = original(**call_kwargs)
        return revoke_retry_like_performance_restores(result, source_takes)

    apply_with_performance_retry_restore_guard._cutsell_hybrid_performance_retry_restore_guard = True
    hybrid_session_cleanup.apply_hybrid_session_cleanup = apply_with_performance_retry_restore_guard

    pipeline_module = sys.modules.get(f"{__package__}.pipeline")
    if pipeline_module is not None:
        pipeline_module.apply_hybrid_session_cleanup = apply_with_performance_retry_restore_guard
