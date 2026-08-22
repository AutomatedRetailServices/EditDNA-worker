"""Reconcile retries when a failed restart fragment sits between two selected deliveries.

Round 7 exposed a real talking-head structure that a direct retry_setup-only rule cannot
see: selected attempt A -> failed restart fragment -> selected clean retake B.  The failed
bridge is itself strong recording-process evidence that A and B are competing attempts.

This pass is deliberately conservative.  It removes A only when:
- the bridge is already semantically failed/BTS with high confidence;
- the bridge is temporally between A and B in the same source;
- A and B substantially overlap in content;
- the bridge substantially overlaps A (a restart/prefix of that delivery);
- B is not semantically failed/BTS; and
- watch/listen evidence independently prefers B (visual fumble/distraction/naturalness).

It does not infer retries from topic similarity alone and never fabricates speech.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from .contracts import CandidateTake

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+", re.IGNORECASE)
_STOP = frozenset({
    "a","al","and","are","as","at","be","but","by","como","con","cuando","de","del",
    "el","en","es","esta","este","for","from","fue","in","is","it","la","las","lo",
    "los","me","mi","mis","of","on","or","para","pero","por","porque","que","se",
    "si","so","su","sus","that","the","this","to","un","una","was","we","with",
    "y","yo",
})


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def _concept(token: str) -> str:
    value = "".join(
        ch for ch in unicodedata.normalize("NFKD", str(token or "").casefold())
        if not unicodedata.combining(ch)
    )
    if len(value) >= 7 and value.endswith("es"):
        value = value[:-2]
    elif len(value) >= 6 and value.endswith("s"):
        value = value[:-1]
    return value


def _content(text: str) -> set[str]:
    return {
        _concept(token) for token in _tokens(text)
        if len(token) >= 3 and token not in _STOP and _concept(token)
    }


def _coverage(left_text: str, right_text: str) -> tuple[int, float, float]:
    left = _content(left_text)
    right = _content(right_text)
    if not left or not right:
        return 0, 0.0, 0.0
    shared = len(left & right)
    return shared, shared / max(1, len(left)), shared / max(1, len(right))


def _semantic_map(decisions: Iterable[tuple[str, str, float]]) -> dict[str, tuple[str, float]]:
    return {str(cid): (str(label), float(conf)) for cid, label, conf in decisions}


def _visual_prefers_later(earlier: CandidateTake, later: CandidateTake) -> bool:
    left = earlier.signals
    right = later.signals
    if left is None or right is None:
        return False
    if float(left.visual_fumble) >= max(0.55, float(right.visual_fumble) + 0.12):
        return True
    if float(left.distraction_risk) >= float(right.distraction_risk) + 0.18:
        return True
    if (
        float(left.expression_naturalness) + 0.18 <= float(right.expression_naturalness)
        and float(left.gesture_naturalness) + 0.12 <= float(right.gesture_naturalness)
    ):
        return True
    return False


def collapse_failed_bridge_retries(
    kept: Iterable[CandidateTake],
    deleted: Iterable[CandidateTake],
    semantic_decisions: Iterable[tuple[str, str, float]],
    *,
    maximum_pair_gap_sec: float = 20.0,
    maximum_bridge_side_gap_sec: float = 4.0,
) -> tuple[tuple[CandidateTake, ...], tuple[CandidateTake, ...], tuple[dict, ...]]:
    kept_tuple = tuple(sorted(kept, key=lambda t: (t.source_order, t.start, t.end, t.clip_id)))
    deleted_tuple = tuple(sorted(deleted, key=lambda t: (t.source_order, t.start, t.end, t.clip_id)))
    semantic = _semantic_map(semantic_decisions)
    removed_ids: set[str] = set()
    diagnostics: list[dict] = []

    for index, earlier in enumerate(kept_tuple):
        if earlier.clip_id in removed_ids:
            continue
        for later in kept_tuple[index + 1:]:
            if later.source_asset_id != earlier.source_asset_id:
                continue
            pair_gap = float(later.start) - float(earlier.end)
            if pair_gap < 0:
                continue
            if pair_gap > maximum_pair_gap_sec:
                break

            shared, earlier_cov, later_cov = _coverage(earlier.text, later.text)
            if shared < 3 or max(earlier_cov, later_cov) < 0.55:
                continue

            later_label, later_conf = semantic.get(later.clip_id, ("", 0.0))
            if later_label in {"failed", "bts"} and later_conf >= 0.80:
                continue
            if not _visual_prefers_later(earlier, later):
                continue

            bridges = []
            for bridge in deleted_tuple:
                if bridge.source_asset_id != earlier.source_asset_id:
                    continue
                if bridge.start < earlier.end or bridge.end > later.start:
                    continue
                left_gap = float(bridge.start) - float(earlier.end)
                right_gap = float(later.start) - float(bridge.end)
                if left_gap > maximum_bridge_side_gap_sec or right_gap > maximum_bridge_side_gap_sec:
                    continue
                bridge_label, bridge_conf = semantic.get(bridge.clip_id, ("", 0.0))
                if bridge_label not in {"failed", "bts"} or bridge_conf < 0.80:
                    continue
                bridge_shared, bridge_cov, _ = _coverage(bridge.text, earlier.text)
                if bridge_shared < 3 or bridge_cov < 0.60:
                    continue
                bridges.append((bridge_conf, bridge_cov, bridge_shared, -left_gap-right_gap, bridge))
            if not bridges:
                continue

            bridge_conf, bridge_cov, bridge_shared, _, bridge = max(bridges, key=lambda item: item[:4])
            removed_ids.add(earlier.clip_id)
            earlier_label, earlier_conf = semantic.get(earlier.clip_id, ("", 0.0))
            diagnostics.append({
                "reason": "selected_attempt_yields_across_failed_restart_bridge",
                "removed_clip_id": earlier.clip_id,
                "failed_bridge_clip_id": bridge.clip_id,
                "winner_clip_id": later.clip_id,
                "earlier_label": earlier_label,
                "earlier_confidence": round(earlier_conf, 4),
                "bridge_confidence": round(bridge_conf, 4),
                "later_label": later_label,
                "later_confidence": round(later_conf, 4),
                "shared_content_tokens": shared,
                "earlier_coverage": round(earlier_cov, 4),
                "later_coverage": round(later_cov, 4),
                "bridge_shared_content_tokens": bridge_shared,
                "bridge_coverage": round(bridge_cov, 4),
                "removed_text": earlier.text,
                "bridge_text": bridge.text,
                "winner_text": later.text,
            })
            break

    survivors = tuple(t for t in kept_tuple if t.clip_id not in removed_ids)
    removed = tuple(t for t in kept_tuple if t.clip_id in removed_ids)
    return survivors, removed, tuple(diagnostics)


def install_failed_bridge_retry_integrity() -> None:
    from . import hybrid_session_cleanup

    original = hybrid_session_cleanup.apply_hybrid_session_cleanup
    if getattr(original, "_cutsell_failed_bridge_retry_integrity", False):
        return

    def apply_with_failed_bridge_retry_integrity(*args, **kwargs):
        source_takes = tuple(args[0]) if args else tuple(kwargs.get("takes") or ())
        result = original(*args, **kwargs)
        if not result.kept or not result.semantic_decisions:
            return result
        kept, removed, diagnostics = collapse_failed_bridge_retries(
            result.kept,
            result.deleted,
            result.semantic_decisions,
        )
        if not diagnostics:
            return result
        deleted_ids = {take.clip_id for take in result.deleted}
        deleted_ids.update(take.clip_id for take in removed)
        deleted = tuple(take for take in source_takes if take.clip_id in deleted_ids)
        return type(result)(
            kept=kept,
            deleted=deleted,
            requested_chunk_count=result.requested_chunk_count,
            available_chunk_count=result.available_chunk_count,
            diagnostics=tuple(result.diagnostics) + ({"failed_bridge_retry_integrity": list(diagnostics)},),
            semantic_decisions=result.semantic_decisions,
        )

    apply_with_failed_bridge_retry_integrity._cutsell_failed_bridge_retry_integrity = True
    hybrid_session_cleanup.apply_hybrid_session_cleanup = apply_with_failed_bridge_retry_integrity
