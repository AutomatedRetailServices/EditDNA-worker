"""Final retry arbitration after Composite and post-selection timeline construction.

This is intentionally late. Hybrid and Composite may generate candidates and composite
structures first; only after interior splits/handoffs are stable do we remove an earlier
selected delivery that is clearly the losing retry of a later selected winner.

The authority requires all of the following:
- the earlier selected clip has strong failed/bts evidence and no strong winner/keep evidence;
- a later selected clip in the same source has strong winner/keep evidence;
- both begin with the same strong content opening;
- the earlier clip precedes the later clip.

This avoids the previous failure mode where an early Hybrid guard changed composite
construction and accidentally removed useful interior material. Ambiguity fails open.
"""
from __future__ import annotations

from dataclasses import replace
import re
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


def _semantic_strengths(diagnostics: dict) -> dict[str, dict[str, float]]:
    strengths: dict[str, dict[str, float]] = {}

    def add(clip_id, label, confidence):
        if not clip_id or not label:
            return
        try:
            conf = float(confidence)
        except (TypeError, ValueError):
            return
        bucket = strengths.setdefault(str(clip_id), {})
        label = str(label)
        bucket[label] = max(conf, bucket.get(label, 0.0))

    def visit(value):
        if isinstance(value, dict):
            if value.get("clip_id") and value.get("label"):
                add(value.get("clip_id"), value.get("label"), value.get("confidence"))
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(diagnostics.get("hybrid_editorial_chunks") or ())
    return strengths


def losing_retry_ids(selected, diagnostics: dict, *, failed_min=0.80, winner_min=0.85, maximum_gap_sec=45.0):
    selected = tuple(sorted(selected, key=lambda c: (c.source_order, float(c.start), float(c.end), c.clip_id)))
    strengths = _semantic_strengths(diagnostics)
    remove = set()
    audit = []

    for earlier in selected:
        labels = strengths.get(str(earlier.clip_id)) or {}
        failed = max(labels.get("failed", 0.0), labels.get("bts", 0.0))
        own_winner = max(labels.get("winner", 0.0), labels.get("keep", 0.0))
        if failed < failed_min or own_winner >= 0.90:
            continue

        candidates = []
        for later in selected:
            if later.clip_id == earlier.clip_id:
                continue
            if later.source_asset_id != earlier.source_asset_id:
                continue
            if float(later.start) < float(earlier.end):
                continue
            gap = float(later.start) - float(earlier.end)
            if gap > maximum_gap_sec:
                continue
            peer_labels = strengths.get(str(later.clip_id)) or {}
            peer_winner = max(peer_labels.get("winner", 0.0), peer_labels.get("keep", 0.0))
            if peer_winner < winner_min:
                continue
            if not same_strong_opening(earlier.text, later.text):
                continue
            candidates.append((peer_winner, -gap, later))

        if not candidates:
            continue
        candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
        best = candidates[0]
        if len(candidates) > 1 and candidates[1][:2] == best[:2]:
            continue
        later = best[2]
        remove.add((earlier.clip_id, float(earlier.start), float(earlier.end)))
        audit.append({
            "clip_id": earlier.clip_id,
            "later_winner_clip_id": later.clip_id,
            "reason": "final_selection_same_opening_failed_retry_yields_to_later_winner",
            "failed_confidence": round(failed, 4),
            "later_winner_confidence": round(best[0], 4),
            "gap_sec": round(float(later.start) - float(earlier.end), 3),
        })
    return remove, audit


def apply_final_selection_retry_arbiter(draft):
    diagnostics = dict(draft.diagnostics or {})
    remove, audit = losing_retry_ids(draft.selected, diagnostics)
    if not remove:
        return draft
    selected = tuple(
        clip for clip in draft.selected
        if (clip.clip_id, float(clip.start), float(clip.end)) not in remove
    )
    removed = [
        clip for clip in draft.selected
        if (clip.clip_id, float(clip.start), float(clip.end)) in remove
    ]
    discarded = list(draft.discarded)
    discarded.extend(replace(clip, selected=False) for clip in removed)
    discarded.sort(key=lambda c: (c.source_order, float(c.start), float(c.end), c.clip_id))
    diagnostics["final_selection_retry_arbiter"] = audit
    return replace(draft, selected=selected, discarded=tuple(discarded), diagnostics=diagnostics)


def install_final_selection_retry_arbiter() -> None:
    from . import pipeline

    original = pipeline.build_flow_b_draft
    if getattr(original, "_cutsell_final_selection_retry_arbiter", False):
        return

    def build_with_final_selection_retry_arbiter(*args, **kwargs):
        result = original(*args, **kwargs)
        repaired = apply_final_selection_retry_arbiter(result.draft)
        if repaired is result.draft:
            return result
        return replace(result, draft=repaired)

    build_with_final_selection_retry_arbiter._cutsell_final_selection_retry_arbiter = True
    pipeline.build_flow_b_draft = build_with_final_selection_retry_arbiter
