"""Group repeated valid attempts by normalized semantic idea."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Iterable, Tuple

from .contracts import CandidateTake


def semantic_key(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9áéíóúñü]+", " ", text.casefold())
    tokens = [token for token in normalized.split() if token]
    return " ".join(tokens[:18])


def group_takes(takes: Iterable[CandidateTake]) -> Dict[str, Tuple[CandidateTake, ...]]:
    groups = defaultdict(list)
    for take in takes:
        key = semantic_key(take.text)
        if not key:
            key = f"__silent__:{take.clip_id}"
        groups[key].append(take)
    return {key: tuple(values) for key, values in groups.items()}
