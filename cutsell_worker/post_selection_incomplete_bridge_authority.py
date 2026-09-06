"""Final physical retry authority after all semantic selection passes.

Some semantic passes can discard the clean later delivery before the existing
incomplete-bridge authority sees the final draft. This module repairs only the narrow
case where three consecutive reconstructed attempts are complete -> incomplete ->
complete, Clean Cut independently rejected the middle attempt with high confidence, and
the later complete delivery covers the same message without losing critical facts.

No source timestamps, phrases, clip ids, or benchmark-specific facts are encoded here.
Ambiguous cases fail open.
"""
from __future__ import annotations

from dataclasses import replace
import re
import unicodedata

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_STOP = frozenset({
    "a", "al", "and", "are", "as", "at", "be", "but", "by", "como", "con", "cuando",
    "de", "del", "el", "en", "es", "esta", "este", "for", "from", "fue", "in", "is",
    "it", "la", "las", "lo", "los", "me", "mi", "mis", "of", "on", "or", "para",
    "pero", "por", "porque", "que", "se", "si", "so", "su", "sus", "that", "the",
    "this", "to", "un", "una", "was", "we", "with", "y", "yo",
})
_NEGATION = frozenset({
    "no", "not", "never", "nunca", "sin", "without", "nadie", "ningun", "ningún",
    "ninguna", "ninguno", "nobody", "none", "neither",
})


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def _concept(token: str) -> str:
    value = "".join(
        char for char in unicodedata.normalize("NFKD", str(token or "").casefold())
        if not unicodedata.combining(char)
    )
    if len(value) >= 7 and value.endswith("es"):
        value = value[:-2]
    elif len(value) >= 6 and value.endswith("s"):
        value = value[:-1]
    return value


def _content(text: str) -> set[str]:
    return {
        _concept(token)
        for token in _tokens(text)
        if len(token) >= 3 and token not in _STOP and _concept(token)
    }


def _coverage(left_text: str, right_text: str) -> tuple[int, float, float]:
    left, right = _content(left_text), _content(right_text)
    if not left or not right:
        return 0, 0.0, 0.0
    shared = len(left & right)
    return shared, shared / len(left), shared / len(right)


def _critical(text: str) -> set[str]:
    raw = str(text or "")
    facts = {f"num:{m.group(0).replace(',', '.')}" for m in _NUMBER_RE.finditer(raw)}
    if any(token in _NEGATION for token in _tokens(raw)):
        facts.add("__negation__")
    return facts


def _attempt_rows(diagnostics: dict) -> list[dict]:
    value = diagnostics.get("attempt_reconstruction") or {}
    rows = value.get("attempts") if isinstance(value, dict) else None
    if rows is None and isinstance(value, dict):
        rows = value.get("takes")
    if rows is None and isinstance(value, list):
        rows = value
    return [row for row in (rows or ()) if isinstance(row, dict) and row.get("clip_id")]


def _clean_cut_map(diagnostics: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in diagnostics.get("clean_cut_decisions") or ():
        if isinstance(row, dict) and row.get("clip_id"):
            out[str(row["clip_id"])] = row
    return out


def _bridge_is_independently_rejected(bridge_id: str, clean_cut: dict[str, dict]) -> bool:
    row = clean_cut.get(bridge_id) or {}
    reason = str(row.get("reason") or "").lower()
    return bool(
        row.get("keep") is False
        and float(row.get("confidence") or 0.0) >= 0.88
        and any(token in reason for token in ("incomplete", "retry", "restart", "reset"))
    )


def _repair(draft):
    diagnostics = dict(draft.diagnostics or {})
    attempts = _attempt_rows(diagnostics)
    if len(attempts) < 3:
        return draft

    selected_by_id = {clip.clip_id: clip for clip in draft.selected}
    discarded_by_id = {clip.clip_id: clip for clip in draft.discarded}
    clean_cut = _clean_cut_map(diagnostics)
    replacements: list[tuple[str, str, str, dict]] = []

    for index in range(len(attempts) - 2):
        early_row, bridge_row, late_row = attempts[index : index + 3]
        early_id = str(early_row.get("clip_id") or "")
        bridge_id = str(bridge_row.get("clip_id") or "")
        late_id = str(late_row.get("clip_id") or "")
        if not early_id or not bridge_id or not late_id:
            continue
        if not bool(early_row.get("complete_idea")) or bool(bridge_row.get("complete_idea")) or not bool(late_row.get("complete_idea")):
            continue
        if early_id not in selected_by_id or bridge_id not in discarded_by_id or late_id not in discarded_by_id:
            continue
        if not _bridge_is_independently_rejected(bridge_id, clean_cut):
            continue

        early = selected_by_id[early_id]
        bridge = discarded_by_id[bridge_id]
        late = discarded_by_id[late_id]
        if early.source_asset_id != bridge.source_asset_id or early.source_asset_id != late.source_asset_id:
            continue
        gap1 = float(bridge.start) - float(early.end)
        gap2 = float(late.start) - float(bridge.end)
        if gap1 < 0.0 or gap1 > 8.0 or gap2 < 0.0 or gap2 > 8.0:
            continue

        shared, early_cov, late_cov = _coverage(early.text, late.text)
        if shared < 3 or early_cov < 0.45 or late_cov < 0.45:
            continue
        bridge_shared, bridge_cov, _ = _coverage(bridge.text, early.text)
        if bridge_shared < 1 or bridge_cov < 0.40:
            continue
        if not _critical(early.text).issubset(_critical(late.text)):
            continue

        replacements.append((early_id, bridge_id, late_id, {
            "reason": "promote_discarded_complete_retry_across_rejected_incomplete_bridge",
            "early_clip_id": early_id,
            "bridge_clip_id": bridge_id,
            "late_clip_id": late_id,
            "shared_content_tokens": shared,
            "early_coverage": round(early_cov, 4),
            "late_coverage": round(late_cov, 4),
            "bridge_coverage": round(bridge_cov, 4),
            "bridge_clean_cut_confidence": round(float(clean_cut[bridge_id].get("confidence") or 0.0), 4),
        }))

    if not replacements:
        return draft

    # Multiple disjoint triples are allowed; conflicting triples fail open by accepting
    # the first source-ordered replacement only for any clip id.
    used: set[str] = set()
    audit: list[dict] = []
    selected = list(draft.selected)
    discarded = list(draft.discarded)
    for early_id, bridge_id, late_id, row in replacements:
        if early_id in used or bridge_id in used or late_id in used:
            continue
        used.update((early_id, bridge_id, late_id))
        early = next((clip for clip in selected if clip.clip_id == early_id), None)
        late = next((clip for clip in discarded if clip.clip_id == late_id), None)
        if early is None or late is None:
            continue
        selected = [clip for clip in selected if clip.clip_id != early_id]
        selected.append(replace(late, selected=True))
        selected.sort(key=lambda clip: (clip.source_order, float(clip.start), float(clip.end)))
        discarded = [clip for clip in discarded if clip.clip_id != late_id]
        if not any(clip.clip_id == early_id for clip in discarded):
            discarded.append(replace(early, selected=False))
        discarded.sort(key=lambda clip: (clip.source_order, float(clip.start), float(clip.end)))
        audit.append(row)

    if not audit:
        return draft
    diagnostics["post_selection_incomplete_bridge_authority"] = audit
    return replace(draft, selected=tuple(selected), discarded=tuple(discarded), diagnostics=diagnostics)


def install_post_selection_incomplete_bridge_authority() -> None:
    from . import pipeline

    original = pipeline.build_flow_b_draft
    if getattr(original, "_cutsell_post_selection_incomplete_bridge_authority", False):
        return

    def build_with_authority(*args, **kwargs):
        result = original(*args, **kwargs)
        repaired = _repair(result.draft)
        if repaired is result.draft:
            return result
        return replace(result, draft=repaired)

    build_with_authority._cutsell_post_selection_incomplete_bridge_authority = True
    pipeline.build_flow_b_draft = build_with_authority
