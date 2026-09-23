"""Deterministically stabilize complementary retry families after Hybrid arbitration.

This authority runs after Hybrid/Composite semantic conflict resolution but before
post-selection interior boundary splitting. It repairs one narrow structural failure:
a concise discarded delivery contributes unique audience-facing information, while a
selected monolithic retry is largely redundant with a later selected winner.

When the concise delivery + later winner preserve the monolithic delivery's critical
facts and cover most of its content, prefer that complementary pair. This is Selection
logic, not Boundary logic. No benchmark ids, timestamps, source ids, or transcript
phrases are embedded here. Ambiguity fails open.
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
_EXPLICIT_NEGATION = frozenset({"no", "not", "never", "nunca", "sin", "without"})


def _canon(token: str) -> str:
    raw = unicodedata.normalize("NFKD", str(token or "").casefold())
    return "".join(ch for ch in raw if not unicodedata.combining(ch))


def _lexeme(token: str) -> str:
    token = _canon(token)
    if len(token) >= 5 and token.isalpha() and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _content(text: str) -> set[str]:
    return {
        token
        for token in (_lexeme(item) for item in _TOKEN_RE.findall(str(text or "")))
        if len(token) >= 3 and token not in _STOP
    }


def _critical(text: str) -> set[str]:
    out: set[str] = set()
    for raw in _TOKEN_RE.findall(str(text or "")):
        token = _canon(raw)
        if token in _EXPLICIT_NEGATION:
            out.add("__negation__")
        if any(ch.isdigit() for ch in token):
            out.add(token)
    return out


def _duration(clip) -> float:
    return max(0.0, float(clip.end) - float(clip.start))


def _ordered(clips):
    return tuple(sorted(clips, key=lambda c: (c.source_order, float(c.start), float(c.end), c.clip_id)))


def complementary_family_swaps(
    selected,
    discarded,
    *,
    maximum_adjacent_gap_sec: float = 4.0,
    minimum_redundant_overlap: float = 0.45,
    minimum_monolith_coverage: float = 0.60,
    minimum_discarded_family_overlap: float = 0.35,
    minimum_unique_fraction: float = 0.15,
    maximum_concise_ratio: float = 0.75,
):
    selected = _ordered(selected)
    discarded = _ordered(discarded)
    swaps = []

    for concise in discarded:
        concise_content = _content(concise.text)
        if len(concise_content) < 4 or _duration(concise) < 1.5:
            continue

        for index in range(len(selected) - 1):
            monolith = selected[index]
            later = selected[index + 1]
            if not (
                concise.source_asset_id == monolith.source_asset_id == later.source_asset_id
                and concise.source_order == monolith.source_order == later.source_order
            ):
                continue
            if float(concise.end) > float(monolith.start):
                continue
            left_gap = float(monolith.start) - float(concise.end)
            right_gap = float(later.start) - float(monolith.end)
            if left_gap < 0 or right_gap < 0:
                continue
            if left_gap > maximum_adjacent_gap_sec or right_gap > maximum_adjacent_gap_sec:
                continue
            if _duration(concise) > (_duration(monolith) * maximum_concise_ratio):
                continue

            monolith_content = _content(monolith.text)
            later_content = _content(later.text)
            if len(monolith_content) < 6 or len(later_content) < 5:
                continue

            shared_monolith_later = monolith_content & later_content
            redundant_overlap = len(shared_monolith_later) / max(1, min(len(monolith_content), len(later_content)))
            if len(shared_monolith_later) < 4 or redundant_overlap < minimum_redundant_overlap:
                continue

            family_union = monolith_content | later_content
            concise_family_overlap = len(concise_content & family_union) / max(1, len(concise_content))
            if len(concise_content & family_union) < 2 or concise_family_overlap < minimum_discarded_family_overlap:
                continue

            concise_unique = concise_content - later_content
            unique_fraction = len(concise_unique) / max(1, len(concise_content))
            if not concise_unique or unique_fraction < minimum_unique_fraction:
                continue

            replacement_union = concise_content | later_content
            monolith_coverage = len(monolith_content & replacement_union) / max(1, len(monolith_content))
            if monolith_coverage < minimum_monolith_coverage:
                continue
            if not _critical(monolith.text).issubset(_critical(concise.text + " " + later.text)):
                continue

            swaps.append({
                "restore": concise,
                "suppress": monolith,
                "later": later,
                "reason": "concise_complementary_delivery_plus_later_winner_replaces_redundant_monolith",
                "left_gap_sec": round(left_gap, 3),
                "right_gap_sec": round(right_gap, 3),
                "redundant_overlap": round(redundant_overlap, 4),
                "monolith_coverage": round(monolith_coverage, 4),
                "concise_family_overlap": round(concise_family_overlap, 4),
                "concise_unique_fraction": round(unique_fraction, 4),
                "concise_unique_tokens": sorted(concise_unique),
            })
    return swaps


def apply_post_selection_complementary_family_stabilizer(draft):
    swaps = complementary_family_swaps(draft.selected, draft.discarded)
    if not swaps:
        return draft

    # Avoid conflicting edits: one restore/suppress pair per clip and fail open on ambiguity.
    by_restore = {}
    by_suppress = {}
    for row in swaps:
        by_restore.setdefault(row["restore"].clip_id, []).append(row)
        by_suppress.setdefault(row["suppress"].clip_id, []).append(row)
    chosen = [
        row for row in swaps
        if len(by_restore[row["restore"].clip_id]) == 1
        and len(by_suppress[row["suppress"].clip_id]) == 1
    ]
    if not chosen:
        return draft

    restore_ids = {row["restore"].clip_id for row in chosen}
    suppress_ids = {row["suppress"].clip_id for row in chosen}

    selected = [clip for clip in draft.selected if clip.clip_id not in suppress_ids]
    selected.extend(replace(clip, selected=True) for clip in draft.discarded if clip.clip_id in restore_ids)
    selected.sort(key=lambda c: (c.source_order, float(c.start), float(c.end), c.clip_id))

    discarded = [clip for clip in draft.discarded if clip.clip_id not in restore_ids]
    discarded.extend(replace(clip, selected=False) for clip in draft.selected if clip.clip_id in suppress_ids)
    discarded.sort(key=lambda c: (c.source_order, float(c.start), float(c.end), c.clip_id))

    diagnostics = dict(draft.diagnostics or {})
    diagnostics["post_selection_complementary_family_stabilizer"] = [
        {
            "restored_clip_id": row["restore"].clip_id,
            "suppressed_clip_id": row["suppress"].clip_id,
            "later_winner_clip_id": row["later"].clip_id,
            "reason": row["reason"],
            "left_gap_sec": row["left_gap_sec"],
            "right_gap_sec": row["right_gap_sec"],
            "redundant_overlap": row["redundant_overlap"],
            "monolith_coverage": row["monolith_coverage"],
            "concise_family_overlap": row["concise_family_overlap"],
            "concise_unique_fraction": row["concise_unique_fraction"],
            "concise_unique_tokens": row["concise_unique_tokens"],
        }
        for row in chosen
    ]
    return replace(draft, selected=tuple(selected), discarded=tuple(discarded), diagnostics=diagnostics)


def install_post_selection_complementary_family_stabilizer() -> None:
    from . import pipeline

    original = pipeline.build_flow_b_draft
    if getattr(original, "_cutsell_post_selection_complementary_family_stabilizer", False):
        return

    def build_with_complementary_family_stabilizer(*args, **kwargs):
        result = original(*args, **kwargs)
        repaired = apply_post_selection_complementary_family_stabilizer(result.draft)
        if repaired is result.draft:
            return result
        return replace(result, draft=repaired)

    build_with_complementary_family_stabilizer._cutsell_post_selection_complementary_family_stabilizer = True
    pipeline.build_flow_b_draft = build_with_complementary_family_stabilizer
