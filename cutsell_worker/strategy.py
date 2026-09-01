"""Automatic editing-strategy selection from semantic and visual evidence."""
from __future__ import annotations

from collections import Counter
from typing import Iterable

from .contracts import CandidateTake, EditStrategy, SemanticLabel, SemanticRole


def _average(values: list[float], default: float = 0.5) -> float:
    return sum(values) / len(values) if values else default


def choose_strategy(
    labels: Iterable[SemanticLabel],
    takes: Iterable[CandidateTake] = (),
) -> EditStrategy:
    """Describe the dominant creator style without imposing a rigid template.

    Strategy is advisory: it may guide ranking/composition later, but it is never
    deletion authority and never forces a commercial funnel.
    """
    label_tuple = tuple(labels)
    take_tuple = tuple(takes)
    counts = Counter(label.role for label in label_tuple)
    total = sum(counts.values()) or 1

    face_values = [take.signals.face_visibility for take in take_tuple if take.signals is not None]
    product_values = [take.signals.product_visibility for take in take_tuple if take.signals is not None]
    face_visibility = _average(face_values)
    product_visibility = _average(product_values, default=0.0)

    # A creator who is mostly off-camera with product/visual footage is best treated
    # as faceless/voiceover even when the spoken commercial labels are mixed.
    if take_tuple and face_visibility <= 0.25 and product_visibility >= 0.35:
        return EditStrategy.FACELESS

    if counts[SemanticRole.STORY] / total >= 0.45:
        return EditStrategy.STORYTELLING

    if counts[SemanticRole.PROOF] >= 2 and counts[SemanticRole.STORY] >= 1:
        return EditStrategy.TESTIMONIAL

    # Strong product visibility plus feature/benefit language is a demonstration,
    # not merely a generic direct-sales clip.
    if (
        product_visibility >= 0.60
        and counts[SemanticRole.FEATURES] + counts[SemanticRole.BENEFITS] >= 1
    ):
        return EditStrategy.DEMO

    # Educational clips tend to explain a problem/feature/benefit chain without
    # relying on proof-heavy testimonial language or an explicit sales CTA.
    explanatory = counts[SemanticRole.PROBLEM] + counts[SemanticRole.FEATURES] + counts[SemanticRole.BENEFITS]
    if explanatory >= max(2, total // 2) and counts[SemanticRole.CTA] == 0 and counts[SemanticRole.PROOF] == 0:
        return EditStrategy.EDUCATIONAL

    if counts[SemanticRole.FEATURES] + counts[SemanticRole.BENEFITS] >= max(2, total // 2):
        return EditStrategy.DIRECT_SALES

    return EditStrategy.MIXED
