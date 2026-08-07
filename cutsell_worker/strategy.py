"""Automatic editing-strategy selection from descriptive semantic labels."""
from collections import Counter
from typing import Iterable

from .contracts import EditStrategy, SemanticLabel, SemanticRole


def choose_strategy(labels: Iterable[SemanticLabel]) -> EditStrategy:
    counts = Counter(label.role for label in labels)
    total = sum(counts.values()) or 1
    if counts[SemanticRole.STORY] / total >= 0.45:
        return EditStrategy.STORYTELLING
    if counts[SemanticRole.PROOF] >= 2 and counts[SemanticRole.STORY] >= 1:
        return EditStrategy.TESTIMONIAL
    if counts[SemanticRole.FEATURES] + counts[SemanticRole.BENEFITS] >= max(2, total // 2):
        return EditStrategy.DIRECT_SALES
    return EditStrategy.MIXED
