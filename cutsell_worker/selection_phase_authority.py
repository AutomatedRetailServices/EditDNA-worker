"""Explicit final Selection authority for Universal Clean Cut.

Critical Selection ownership must not depend on import-time wrapper order. This module
runs the final semantic membership passes in a fixed order immediately before speech
recovery and the Selection freeze:

1. stabilize complementary information-preserving retry families;
2. trim a proven internal spoken retake only when the later delivery fully recovers it;
3. arbitrate selected retry losers using final Hybrid + physical evidence;
4. keep semantic alternates available for manual SWAP rather than discarding them;
5. enforce final deterministic guards for already-proven covered failures and
   consensus alternate bridges.

No Boundary operation is installed or invoked here. Ambiguity fails open.
"""
from __future__ import annotations

from dataclasses import replace
import re
import unicodedata

from .final_selection_retry_arbiter import apply_final_selection_retry_arbiter
from .post_selection_complementary_family_stabilizer import (
    apply_post_selection_complementary_family_stabilizer,
)
from .post_selection_internal_retake_trim import trim_selected_internal_retakes

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+(?:[-–][0-9]+)?%?", re.IGNORECASE)
_STOP = frozenset({
    "a", "al", "and", "are", "as", "at", "be", "but", "by", "como", "con", "de", "del",
    "el", "en", "es", "esta", "este", "for", "from", "in", "is", "it", "la", "las", "lo",
    "los", "me", "mi", "mis", "of", "on", "or", "para", "pero", "por", "porque", "que",
    "se", "so", "su", "sus", "that", "the", "this", "to", "un", "una", "was", "we",
    "with", "y", "yo",
})
_NEGATION = frozenset({"no", "not", "never", "nunca", "sin", "without", "ni"})


def _canon(token: str) -> str:
    raw = unicodedata.normalize("NFKD", str(token or "").casefold())
    return "".join(ch for ch in raw if not unicodedata.combining(ch))


def _concept(token: str) -> str:
    value = _canon(token)
    if len(value) >= 7 and value.endswith("es"):
        value = value[:-2]
    elif len(value) >= 5 and value.endswith("s") and not value.endswith("ss"):
        value = value[:-1]
    return value


def _content(text: str) -> set[str]:
    return {
        concept
        for raw in _TOKEN_RE.findall(str(text or ""))
        for concept in (_concept(raw),)
        if len(concept) >= 3 and concept not in _STOP
    }


def _critical(text: str) -> set[str]:
    out = set()
    for raw in _TOKEN_RE.findall(str(text or "")):
        token = _canon(raw)
        if token in _NEGATION:
            out.add("__negation__")
        if any(ch.isdigit() for ch in token):
            out.add(token)
    return out


def _hybrid_votes(diagnostics: dict) -> dict[str, list[tuple[str, float]]]:
    votes: dict[str, list[tuple[str, float]]] = {}
    for chunk in diagnostics.get("hybrid_editorial_chunks") or ():
        if not isinstance(chunk, dict):
            continue
        for row in chunk.get("decisions") or ():
            if not isinstance(row, dict) or not row.get("clip_id") or not row.get("label"):
                continue
            try:
                confidence = float(row.get("confidence") or 0.0)
            except (TypeError, ValueError):
                continue
            votes.setdefault(str(row["clip_id"]), []).append((str(row["label"]), confidence))
    return votes


def _strongest(votes: dict[str, list[tuple[str, float]]], clip_id: str, labels: set[str]) -> float:
    return max(
        (confidence for label, confidence in votes.get(str(clip_id), ()) if label in labels),
        default=0.0,
    )


def _iter_dicts(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_dicts(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_dicts(child)


def authoritative_failed_retry_ids(selected, diagnostics: dict):
    """Remove only failures another deterministic authority already proved covered.

    Hybrid can fail-open a short failed prefix and a later soft-restore can reinsert it.
    When cross-group retry integrity has already proved that an authoritative later
    delivery covers the failed clip and preserves critical facts, the final Selection
    authority must honor that proof instead of allowing a resurrection race.
    """
    ordered = tuple(sorted(selected, key=lambda c: (c.source_order, float(c.start), float(c.end), c.clip_id)))
    selected_by_id = {clip.clip_id: clip for clip in ordered}
    votes = _hybrid_votes(diagnostics)
    remove: set[str] = set()
    audit: list[dict] = []

    for row in _iter_dicts(diagnostics.get("hybrid_editorial_chunks") or ()):
        if row.get("reason") != "cross_group_semantic_retry_covered_by_authoritative_delivery":
            continue
        clip_id = str(row.get("clip_id") or "")
        peer_id = str(row.get("strongest_peer_clip_id") or "")
        clip = selected_by_id.get(clip_id)
        peer = selected_by_id.get(peer_id)
        if clip is None or peer is None or clip.source_asset_id != peer.source_asset_id:
            continue
        if float(peer.start) < float(clip.end):
            continue
        failed_confidence = max(
            float(row.get("semantic_confidence") or 0.0),
            _strongest(votes, clip_id, {"failed", "bts"}),
        )
        winner_confidence = _strongest(votes, peer_id, {"winner", "keep"})
        coverage = float(row.get("coverage") or row.get("strongest_peer_coverage") or 0.0)
        content_count = int(row.get("content_token_count") or 0)
        shared = int(row.get("shared_union") or row.get("strongest_shared") or 0)
        critical_preserved = bool(row.get("critical_preserved"))
        if failed_confidence < 0.80 or winner_confidence < 0.85 or not critical_preserved:
            continue
        if content_count <= 4:
            covered = shared >= 2 and coverage >= 0.60
        else:
            covered = shared >= 4 and coverage >= 0.75
        if not covered:
            continue
        remove.add(clip_id)
        audit.append({
            "clip_id": clip_id,
            "winner_clip_id": peer_id,
            "reason": "authoritatively_covered_failed_retry_removed_at_final_selection",
            "failed_confidence": round(failed_confidence, 4),
            "winner_confidence": round(winner_confidence, 4),
            "coverage": round(coverage, 4),
            "shared_content_tokens": shared,
            "critical_preserved": True,
        })
    return remove, audit


def redundant_alternate_bridge_ids(selected, diagnostics: dict):
    """Identify consensus alternates that only bridge two already-selected deliveries.

    This is intentionally narrower than generic duplicate removal. The middle clip must
    receive at least two independent high-confidence ``alternate`` votes, no strong
    winner/keep vote, be short and temporally sandwiched, and have its informational
    content substantially covered by BOTH neighboring selected deliveries. Numeric or
    negated facts must also survive in the neighbors. The clip remains a SWAP alternate.
    """
    ordered = tuple(sorted(selected, key=lambda c: (c.source_order, float(c.start), float(c.end), c.clip_id)))
    votes = _hybrid_votes(diagnostics)
    remove: set[str] = set()
    audit: list[dict] = []

    for index in range(1, len(ordered) - 1):
        left, middle, right = ordered[index - 1], ordered[index], ordered[index + 1]
        if not (left.source_asset_id == middle.source_asset_id == right.source_asset_id):
            continue
        alternate_votes = [
            confidence for label, confidence in votes.get(middle.clip_id, ())
            if label == "alternate" and confidence >= 0.75
        ]
        if len(alternate_votes) < 2:
            continue
        if _strongest(votes, middle.clip_id, {"winner", "keep"}) >= 0.80:
            continue
        if max(0.0, float(middle.end) - float(middle.start)) > 5.0:
            continue
        left_gap = float(middle.start) - float(left.end)
        right_gap = float(right.start) - float(middle.end)
        if left_gap < 0.0 or left_gap > 2.5 or right_gap < 0.0 or right_gap > 10.0:
            continue
        left_strength = _strongest(votes, left.clip_id, {"winner", "keep"})
        right_strength = max(
            _strongest(votes, right.clip_id, {"winner", "keep"}),
            _strongest(votes, right.clip_id, {"alternate", "failed"}),
        )
        if left_strength < 0.90 or right_strength < 0.75:
            continue

        middle_content = _content(middle.text)
        left_content = _content(left.text)
        right_content = _content(right.text)
        if len(middle_content) < 4:
            continue
        left_shared = middle_content & left_content
        right_shared = middle_content & right_content
        union_shared = middle_content & (left_content | right_content)
        coverage = len(union_shared) / max(1, len(middle_content))
        if len(left_shared) < 2 or len(right_shared) < 2 or coverage < 0.60:
            continue
        if not _critical(middle.text).issubset(_critical(left.text + " " + right.text)):
            continue

        remove.add(middle.clip_id)
        audit.append({
            "clip_id": middle.clip_id,
            "left_clip_id": left.clip_id,
            "right_clip_id": right.clip_id,
            "reason": "consensus_alternate_bridge_covered_by_neighboring_selected_deliveries",
            "alternate_vote_count": len(alternate_votes),
            "best_alternate_confidence": round(max(alternate_votes), 4),
            "left_strength": round(left_strength, 4),
            "right_strength": round(right_strength, 4),
            "left_shared_content_tokens": len(left_shared),
            "right_shared_content_tokens": len(right_shared),
            "union_coverage": round(coverage, 4),
            "left_gap_sec": round(left_gap, 3),
            "right_gap_sec": round(right_gap, 3),
        })
    return remove, audit


def _apply_internal_retake_trim(draft):
    diagnostics = dict(draft.diagnostics or {})
    selected, audit = trim_selected_internal_retakes(draft.selected, diagnostics)
    if not audit:
        return draft
    diagnostics["post_selection_internal_retake_trim"] = list(audit)
    return replace(draft, selected=selected, diagnostics=diagnostics)


def _restore_semantic_alternates_to_swap(before_arbiter, after_arbiter):
    """Move physical/semantic alternates out of Selected but keep them swappable."""
    diagnostics = dict(after_arbiter.diagnostics or {})
    audit = diagnostics.get("final_selection_retry_arbiter") or ()
    swap_ids = {
        str(row.get("clip_id"))
        for row in audit
        if isinstance(row, dict)
        and float(row.get("alternate_confidence") or 0.0) >= 0.60
        and float(row.get("failed_confidence") or 0.0) < 0.80
        and row.get("clip_id")
    }
    if not swap_ids:
        return after_arbiter

    removed_by_id = {
        clip.clip_id: clip
        for clip in before_arbiter.selected
        if clip.clip_id in swap_ids
    }
    if not removed_by_id:
        return after_arbiter

    existing_alt_ids = {clip.clip_id for clip in after_arbiter.alternates}
    alternates = list(after_arbiter.alternates)
    for clip_id, clip in removed_by_id.items():
        if clip_id not in existing_alt_ids:
            alternates.append(replace(clip, selected=False))
    alternates.sort(key=lambda c: (c.source_order, float(c.start), float(c.end), c.clip_id))

    discarded = tuple(
        clip for clip in after_arbiter.discarded
        if clip.clip_id not in swap_ids
    )
    diagnostics["selection_swap_preserved_alternates"] = [
        {
            "clip_id": clip_id,
            "reason": "semantic_alternate_removed_from_auto_edit_but_preserved_for_swap",
        }
        for clip_id in sorted(removed_by_id)
    ]
    return replace(
        after_arbiter,
        alternates=tuple(alternates),
        discarded=discarded,
        diagnostics=diagnostics,
    )


def _apply_final_retry_guards(draft):
    diagnostics = dict(draft.diagnostics or {})
    failed_ids, failed_audit = authoritative_failed_retry_ids(draft.selected, diagnostics)
    bridge_ids, bridge_audit = redundant_alternate_bridge_ids(draft.selected, diagnostics)
    if not failed_ids and not bridge_ids:
        return draft

    selected_by_id = {clip.clip_id: clip for clip in draft.selected}
    remove_ids = failed_ids | bridge_ids
    selected = tuple(clip for clip in draft.selected if clip.clip_id not in remove_ids)

    discarded = list(draft.discarded)
    existing_discarded = {clip.clip_id for clip in discarded}
    for clip_id in sorted(failed_ids):
        clip = selected_by_id.get(clip_id)
        if clip is not None and clip_id not in existing_discarded:
            discarded.append(replace(clip, selected=False))
    discarded.sort(key=lambda c: (c.source_order, float(c.start), float(c.end), c.clip_id))

    alternates = list(draft.alternates)
    existing_alternates = {clip.clip_id for clip in alternates}
    for clip_id in sorted(bridge_ids):
        clip = selected_by_id.get(clip_id)
        if clip is not None and clip_id not in existing_alternates:
            alternates.append(replace(clip, selected=False))
    alternates.sort(key=lambda c: (c.source_order, float(c.start), float(c.end), c.clip_id))

    diagnostics["selection_final_retry_guards"] = [*failed_audit, *bridge_audit]
    return replace(
        draft,
        selected=selected,
        alternates=tuple(alternates),
        discarded=tuple(discarded),
        diagnostics=diagnostics,
    )


def apply_selection_phase_authority(draft):
    """Execute final semantic Selection deterministically in one explicit location."""
    input_selected_count = len(draft.selected)

    draft = apply_post_selection_complementary_family_stabilizer(draft)
    draft = _apply_internal_retake_trim(draft)

    before_arbiter = draft
    draft = apply_final_selection_retry_arbiter(draft)
    draft = _restore_semantic_alternates_to_swap(before_arbiter, draft)
    draft = _apply_final_retry_guards(draft)

    diagnostics = dict(draft.diagnostics or {})
    diagnostics["selection_phase_authority"] = {
        "status": "executed",
        "input_selected_count": input_selected_count,
        "output_selected_count": len(draft.selected),
        "alternate_count": len(draft.alternates),
        "discarded_count": len(draft.discarded),
        "ordered_passes": [
            "post_selection_complementary_family_stabilizer",
            "post_selection_internal_retake_trim",
            "final_selection_retry_arbiter",
            "swap_alternate_preservation",
            "selection_final_retry_guards",
        ],
    }
    return replace(draft, diagnostics=diagnostics)
