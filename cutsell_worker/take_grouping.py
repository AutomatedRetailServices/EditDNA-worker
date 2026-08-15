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
    # Short phrases are too semantically dense for fuzzy grouping: changing one
    # token can change the creator's meaning or CTA. Require exact equality for
    # three-word-or-shorter attempts; a separate source/time-aware path below may
    # still join an exact-prefix false start to its nearby longer retry.
    if min(len(tokens_a), len(tokens_b)) <= 3:
        return 0.0
    set_a, set_b = set(tokens_a), set(tokens_b)
    containment = len(set_a & set_b) / max(1, min(len(set_a), len(set_b)))
    sequence = SequenceMatcher(None, a, b).ratio()
    if containment < 0.60:
        return 0.0
    return round(0.55 * sequence + 0.45 * containment, 4)


def _gap_between(left: CandidateTake, right: CandidateTake) -> float:
    if left.end <= right.start:
        return right.start - left.end
    if right.end <= left.start:
        return left.start - right.end
    return 0.0


def _safe_short_prefix_retry(
    left: CandidateTake,
    right: CandidateTake,
    *,
    maximum_gap_sec: float = 12.0,
) -> bool:
    """Join only a 2-3 word exact-prefix false start to a nearby longer retry.

    One-word reactions are never grouped through this exception.  The match must be
    same-source, temporally close, and the longer attempt must contain at least five
    words so a short standalone phrase is not clustered just because it shares a
    generic opening.
    """
    if left.source_asset_id != right.source_asset_id:
        return False
    if _gap_between(left, right) > maximum_gap_sec:
        return False
    left_tokens = semantic_key(left.text).split()
    right_tokens = semantic_key(right.text).split()
    if not left_tokens or not right_tokens:
        return False
    short, long = (left_tokens, right_tokens) if len(left_tokens) <= len(right_tokens) else (right_tokens, left_tokens)
    if not 2 <= len(short) <= 3 or len(long) < 5:
        return False
    return long[:len(short)] == short


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
            representatives = (cluster[0], cluster[-1]) if len(cluster) > 1 else (cluster[0],)
            score = 0.0
            for item in representatives:
                candidate_score = retry_similarity(take.text, item.text)
                if _safe_short_prefix_retry(take, item):
                    candidate_score = max(candidate_score, 1.0)
                score = max(score, candidate_score)
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
