"""Trim only a redundant suffix fragment when a later selected delivery takes over.

After Best Take and interior performance splitting, one logical selected take may appear as
multiple DraftClip fragments. If only its final fragment substantially repeats a later
selected complete delivery, while earlier sibling fragments carry information not covered
by that later delivery, a human editor can hand off at the internal reset: keep the unique
prefix fragments and remove only the redundant suffix.

This module is benchmark-agnostic and is installed separately after validation.
"""
from __future__ import annotations

from dataclasses import replace
import re
import unicodedata
from typing import Iterable

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+(?:[-–][0-9]+)?%?", re.IGNORECASE)
_STOP = frozenset({
    "a", "al", "and", "are", "as", "at", "be", "but", "by", "como", "con", "de", "del",
    "el", "en", "es", "esta", "este", "for", "from", "in", "is", "it", "la", "las", "lo",
    "los", "me", "mi", "mis", "of", "on", "or", "para", "pero", "por", "porque", "que",
    "se", "so", "su", "sus", "that", "the", "this", "to", "un", "una", "was", "we",
    "with", "y", "yo",
})
_NEGATIONS = frozenset({"no", "not", "never", "nunca", "sin", "without"})


def _canon(token: str) -> str:
    raw = unicodedata.normalize("NFKD", str(token or "").casefold())
    return "".join(ch for ch in raw if not unicodedata.combining(ch))


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_canon(item) for item in _TOKEN_RE.findall(str(text or "")))


def _content(text: str) -> set[str]:
    return {token for token in _tokens(text) if len(token) >= 3 and token not in _STOP}


def _critical(text: str) -> set[str]:
    out = set()
    for token in _tokens(text):
        if token in _NEGATIONS:
            out.add("__negation__")
        if any(ch.isdigit() for ch in token):
            out.add(token)
    return out


def suffix_handoff_relation(
    suffix,
    later,
    prefix_siblings: Iterable,
    *,
    maximum_gap_sec: float = 15.0,
    minimum_shared_tokens: int = 5,
    minimum_suffix_coverage: float = 0.52,
    minimum_later_unique_tokens: int = 2,
    minimum_prefix_unique_tokens: int = 2,
) -> dict | None:
    if suffix.source_asset_id != later.source_asset_id or suffix.clip_id == later.clip_id:
        return None
    gap = float(later.start) - float(suffix.end)
    if gap < 0.0 or gap > maximum_gap_sec:
        return None
    suffix_content = _content(suffix.text)
    later_content = _content(later.text)
    if len(suffix_content) < 5 or len(later_content) < 5:
        return None
    shared = len(suffix_content & later_content)
    coverage = shared / max(1, len(suffix_content))
    if shared < minimum_shared_tokens or coverage < minimum_suffix_coverage:
        return None
    if not _critical(suffix.text).issubset(_critical(later.text)):
        return None
    later_unique = later_content - suffix_content
    if len(later_unique) < minimum_later_unique_tokens:
        return None

    siblings = tuple(prefix_siblings)
    if not siblings:
        return None
    prefix_content: set[str] = set()
    for sibling in siblings:
        if sibling.clip_id != suffix.clip_id or sibling.source_asset_id != suffix.source_asset_id:
            return None
        if float(sibling.start) >= float(suffix.start):
            return None
        prefix_content.update(_content(sibling.text))
    prefix_unique = prefix_content - later_content
    if len(prefix_unique) < minimum_prefix_unique_tokens:
        return None
    return {
        "suffix_clip_id": suffix.clip_id,
        "suffix_start": round(float(suffix.start), 3),
        "suffix_end": round(float(suffix.end), 3),
        "later_clip_id": later.clip_id,
        "reason": "selected_composite_suffix_yields_to_later_selected_delivery",
        "shared_content_tokens": shared,
        "suffix_coverage_by_later": round(coverage, 4),
        "later_unique_content_tokens": sorted(later_unique),
        "preserved_prefix_unique_content_tokens": sorted(prefix_unique),
        "gap_sec": round(gap, 3),
    }


def trim_redundant_composite_suffixes(draft):
    selected = tuple(sorted(draft.selected, key=lambda c: (c.source_order, c.start, c.end, c.clip_id)))
    by_clip: dict[str, list] = {}
    for clip in selected:
        by_clip.setdefault(clip.clip_id, []).append(clip)

    remove_keys: set[tuple[str, float, float]] = set()
    audit: list[dict] = []
    for clip_id, siblings in by_clip.items():
        siblings = sorted(siblings, key=lambda c: (c.start, c.end))
        if len(siblings) < 2:
            continue
        suffix = siblings[-1]
        prefix = siblings[:-1]
        later_options = [
            clip for clip in selected
            if clip.clip_id != clip_id
            and clip.source_asset_id == suffix.source_asset_id
            and clip.start >= suffix.end
        ]
        relations = []
        for later in later_options:
            relation = suffix_handoff_relation(suffix, later, prefix)
            if relation is not None:
                relations.append((relation["suffix_coverage_by_later"], -relation["gap_sec"], later, relation))
        if not relations:
            continue
        relations.sort(key=lambda row: (row[0], row[1]), reverse=True)
        best = relations[0]
        if len(relations) > 1 and relations[1][:2] == best[:2]:
            continue
        remove_keys.add((suffix.clip_id, float(suffix.start), float(suffix.end)))
        audit.append(best[3])

    if not remove_keys:
        return draft
    new_selected = tuple(
        clip for clip in draft.selected
        if (clip.clip_id, float(clip.start), float(clip.end)) not in remove_keys
    )
    removed = [
        clip for clip in draft.selected
        if (clip.clip_id, float(clip.start), float(clip.end)) in remove_keys
    ]
    new_discarded = list(draft.discarded)
    for clip in removed:
        new_discarded.append(replace(clip, selected=False))
    new_discarded.sort(key=lambda c: (c.source_order, c.start, c.end, c.clip_id))
    diagnostics = dict(draft.diagnostics or {})
    diagnostics["post_selection_composite_handoff_trim"] = audit
    return replace(
        draft,
        selected=new_selected,
        discarded=tuple(new_discarded),
        diagnostics=diagnostics,
    )


def install_post_selection_composite_handoff_trim() -> None:
    from . import pipeline

    original = pipeline.build_flow_b_draft
    if getattr(original, "_cutsell_post_selection_composite_handoff_trim", False):
        return

    def build_with_composite_handoff_trim(*args, **kwargs):
        result = original(*args, **kwargs)
        repaired = trim_redundant_composite_suffixes(result.draft)
        if repaired is result.draft:
            return result
        return replace(result, draft=repaired)

    build_with_composite_handoff_trim._cutsell_post_selection_composite_handoff_trim = True
    pipeline.build_flow_b_draft = build_with_composite_handoff_trim
