"""Conservative retry grouping for valid takes.

The creator may repeat the same idea with small wording changes. Grouping is fuzzy
enough to recognize those retries, but deliberately refuses to cluster short or
weakly-overlapping phrases just because they share a commercial role.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, Iterable, Tuple

from .contracts import CandidateTake


def semantic_key(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9áéíóúñü]+", " ", text.casefold())
    tokens = [token for token in normalized.split() if token]
    return " ".join(tokens[:18])


def retry_similarity(left: str, right: str) -> float:
    a = semantic_key(left)
    b = semantic_key(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    tokens_a = a.split()
    tokens_b = b.split()
    if min(len(tokens_a), len(tokens_b)) < 3:
        return 0.0
    set_a, set_b = set(tokens_a), set(tokens_b)
    containment = len(set_a & set_b) / max(1, min(len(set_a), len(set_b)))
    sequence = SequenceMatcher(None, a, b).ratio()
    # Both lexical containment and ordering matter. A generic shared phrase alone
    # is not enough to make two sales ideas alternate takes.
    if containment < 0.60:
        return 0.0
    return round(0.55 * sequence + 0.45 * containment, 4)


def group_takes(
    takes: Iterable[CandidateTake],
    *,
    similarity_threshold: float = 0.72,
) -> Dict[str, Tuple[CandidateTake, ...]]:
    ordered = sorted(takes, key=lambda take: (take.source_order, take.start, take.end, take.clip_id))
    clusters: list[list[CandidateTake]] = []

    for take in ordered:
        best_index = None
        best_score = 0.0
        for index, cluster in enumerate(clusters):
            # Compare with the first and latest attempts; this catches small retry
            # drift while avoiding broad transitive chains across unrelated ideas.
            representatives = (cluster[0], cluster[-1]) if len(cluster) > 1 else (cluster[0],)
            score = max(retry_similarity(take.text, item.text) for item in representatives)
            if score > best_score:
                best_score, best_index = score, index
        if best_index is not None and best_score >= similarity_threshold:
            clusters[best_index].append(take)
        else:
            clusters.append([take])

    output: Dict[str, Tuple[CandidateTake, ...]] = {}
    for cluster in clusters:
        base = semantic_key(cluster[0].text) or f"__silent__:{cluster[0].clip_id}"
        key = base
        suffix = 1
        while key in output:
            suffix += 1
            key = f"{base} #{suffix}"
        output[key] = tuple(cluster)
    return output
