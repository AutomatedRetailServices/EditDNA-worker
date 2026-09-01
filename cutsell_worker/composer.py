"""Flexible composer preserving creator/source order by default."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

from .contracts import CandidateTake, SemanticLabel, TakeGroup


def compose_selected(
    takes: Iterable[CandidateTake],
    groups: Iterable[TakeGroup],
    labels: Iterable[SemanticLabel],
) -> Tuple[CandidateTake, ...]:
    """Select one winner per retry group and keep every ungrouped valid take.

    Retry groups exist only to collapse competing deliveries of the same idea. A take
    that does not belong to any retry group is independent audience-facing material and
    must remain selected unless an earlier cleanup stage explicitly removed it. Semantic
    labels are intentionally unused as deletion authority.
    """
    take_map: Dict[str, CandidateTake] = {take.clip_id: take for take in takes}
    group_tuple = tuple(groups)
    grouped_ids = {
        clip_id
        for group in group_tuple
        for clip_id in group.candidate_ids
    }
    selected_ids = {group.selected_clip_id for group in group_tuple}
    selected_ids.update(
        clip_id for clip_id in take_map
        if clip_id not in grouped_ids
    )
    selected = [take_map[clip_id] for clip_id in selected_ids if clip_id in take_map]
    return tuple(sorted(selected, key=lambda take: (take.source_order, take.start, take.end, take.clip_id)))
