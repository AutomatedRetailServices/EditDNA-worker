"""Final composer guard against Frankenstein selection across retry groups.

A later failed retry can be segmented into two selected pieces: an open prefix in one
singleton group plus a continuation that wins a separate retry group.  If that continuation
already has an earlier complete peer in its deterministic retry group, and the earlier peer
covers the combined later pieces, the complete earlier delivery is safer and more faithful.

This guard changes selection only after deterministic grouping has already proven the
continuation and earlier peer compete for the same idea.  It never groups by topic alone.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Tuple

from .contracts import CandidateTake, SemanticLabel, TakeGroup

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_OPEN_TAIL = frozenset({
    "a","al","and","as","at","because","but","by","con","como","cuando","de","del",
    "el","en","for","from","if","in","into","la","las","los","o","of","on","or",
    "para","pero","por","porque","que","si","sin","so","than","that","the","to",
    "un","una","unos","unas","when","which","while","who","with","without","y",
})
_STOP = frozenset({
    "a","al","and","are","as","at","be","but","by","como","con","cuando","de","del",
    "el","en","es","esta","este","for","from","fue","in","is","it","la","las","lo",
    "los","me","mi","mis","of","on","or","para","pero","por","porque","que","se",
    "si","so","su","sus","that","the","this","to","un","una","was","we","with",
    "y","yo",
})
_NEGATION = frozenset({
    "no","not","never","nunca","sin","without","nadie","ningun","ningún","ninguna",
    "ninguno","nobody","none","neither",
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


def _critical(text: str) -> set[str]:
    raw = str(text or "")
    # Individual numeric values make ASR formatting 5-10 / 5 -10 equivalent.
    out = {f"num:{value}" for value in _NUMBER_RE.findall(raw)}
    if any(token in _NEGATION for token in _tokens(raw)):
        out.add("__negation__")
    return out


def _is_open(take: CandidateTake) -> bool:
    tokens = _tokens(take.text)
    if not tokens:
        return False
    return (
        not take.complete_idea
        or tokens[-1] in _OPEN_TAIL
        or str(take.text or "").strip().endswith((",", ":", ";", "-", "–", "—"))
    )


def reconcile_split_retry_selection(
    selected: Iterable[CandidateTake],
    takes: Iterable[CandidateTake],
    groups: Iterable[TakeGroup],
    *,
    maximum_piece_gap_sec: float = 4.0,
    maximum_prior_gap_sec: float = 45.0,
) -> Tuple[CandidateTake, ...]:
    selected_tuple = tuple(sorted(selected, key=lambda t: (t.source_order, t.start, t.end, t.clip_id)))
    take_tuple = tuple(takes)
    group_tuple = tuple(groups)
    by_id = {take.clip_id: take for take in take_tuple}
    group_by_selected = {group.selected_clip_id: group for group in group_tuple}
    remove_ids: set[str] = set()
    restore_ids: set[str] = set()

    for index, prefix in enumerate(selected_tuple[:-1]):
        if prefix.clip_id in remove_ids or not _is_open(prefix):
            continue
        continuation = selected_tuple[index + 1]
        if continuation.clip_id in remove_ids or continuation.source_asset_id != prefix.source_asset_id:
            continue
        piece_gap = float(continuation.start) - float(prefix.end)
        if piece_gap < 0 or piece_gap > maximum_piece_gap_sec:
            continue

        group = group_by_selected.get(continuation.clip_id)
        if group is None or len(group.candidate_ids) < 2:
            continue

        combined_text = f"{prefix.text} {continuation.text}".strip()
        combined_critical = _critical(combined_text)
        options = []
        for peer_id in group.candidate_ids:
            if peer_id == continuation.clip_id:
                continue
            peer = by_id.get(peer_id)
            if peer is None or peer.source_asset_id != prefix.source_asset_id:
                continue
            if peer.end > prefix.start or not peer.complete_idea:
                continue
            prior_gap = float(prefix.start) - float(peer.end)
            if prior_gap > maximum_prior_gap_sec:
                continue
            shared, combined_cov, peer_cov = _coverage(combined_text, peer.text)
            if shared < 7 or combined_cov < 0.35 or peer_cov < 0.35:
                continue
            peer_critical = _critical(peer.text)
            # Critical numbers/negation must remain compatible.  Either side may omit
            # a number because ASR can drop punctuation/percent symbols, but conflicts
            # are not allowed when both sides express critical facts.
            if combined_critical and peer_critical:
                nums_combined = {x for x in combined_critical if x.startswith("num:")}
                nums_peer = {x for x in peer_critical if x.startswith("num:")}
                if nums_combined and nums_peer and nums_combined != nums_peer:
                    continue
                if ("__negation__" in combined_critical) != ("__negation__" in peer_critical):
                    continue
            options.append((combined_cov, peer_cov, shared, -prior_gap, peer))
        if not options:
            continue

        _, _, _, _, peer = max(options, key=lambda item: item[:4])
        remove_ids.update({prefix.clip_id, continuation.clip_id})
        restore_ids.add(peer.clip_id)

    if not remove_ids and not restore_ids:
        return selected_tuple

    chosen = [take for take in selected_tuple if take.clip_id not in remove_ids]
    chosen_ids = {take.clip_id for take in chosen}
    for take in take_tuple:
        if take.clip_id in restore_ids and take.clip_id not in chosen_ids:
            chosen.append(take)
            chosen_ids.add(take.clip_id)
    return tuple(sorted(chosen, key=lambda t: (t.source_order, t.start, t.end, t.clip_id)))


def install_split_retry_selection_integrity() -> None:
    from . import composer

    original = composer.compose_selected
    if getattr(original, "_cutsell_split_retry_selection_integrity", False):
        return

    def compose_selected_with_split_retry_integrity(
        takes: Iterable[CandidateTake],
        groups: Iterable[TakeGroup],
        labels: Iterable[SemanticLabel],
    ) -> Tuple[CandidateTake, ...]:
        take_tuple = tuple(takes)
        group_tuple = tuple(groups)
        label_tuple = tuple(labels)
        selected = original(take_tuple, group_tuple, label_tuple)
        return reconcile_split_retry_selection(selected, take_tuple, group_tuple)

    compose_selected_with_split_retry_integrity._cutsell_split_retry_selection_integrity = True
    composer.compose_selected = compose_selected_with_split_retry_integrity
