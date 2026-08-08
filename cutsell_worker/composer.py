"""Flexible composer preserving creator/source order by default."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

from .contracts import CandidateTake, SemanticLabel, TakeGroup


def compose_selected(
    takes: Iterable[CandidateTake],
    groups: Iterable[TakeGroup],
    labels: Iterable[SemanticLabel],
) -> Tuple[CandidateTake, ...]:
    """Select one take per group and preserve natural source/time order.

    Semantic labels are intentionally unused as deletion authority.
    """
    take_map: Dict[str, CandidateTake] = {take.clip_id: take for take in takes}
    selected_ids = {group.selected_clip_id for group in groups}
    selected = [take_map[clip_id] for clip_id in selected_ids if clip_id in take_map]
    return tuple(sorted(selected, key=lambda take: (take.source_order, take.start, take.end, take.clip_id)))
