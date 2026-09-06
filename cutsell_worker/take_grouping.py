"""Conservative retry grouping for valid takes.

The creator may repeat the same idea with small wording changes. Grouping is fuzzy
enough to recognize those retries, but deliberately refuses to cluster short or
weakly-overlapping phrases just because they share a commercial role.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, Iterable, Mapping, Tuple

from .contracts import CandidateTake

_CONTRACTION_SUFFIXES = frozenset({"m", "re", "ve", "ll", "d", "s", "t"})


def semantic_key(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9áéíóúñü]+", " ", text.casefold())
    tokens = [token for token in normalized.split() if token]
    return " ".join(tokens[:18])


def _natural_tokens(text: str) -> tuple[str, ...]:
    """Preserve semantic keys while counting common contractions as one word."""
    raw = semantic_key(text).split()
    output: list[str] = []
    index = 0
    while index < len(raw):
        token = raw[index]
        if index + 1 < len(raw) and raw[index + 1] in _CONTRACTION_SUFFIXES:
            output.append(token + "'" + raw[index + 1])
            index += 2
            continue
        output.append(token)
        index += 1
    return tuple(output)


def retry_similarity(left: str, right: str) -> float:
    a = semantic_key(left)
    b = semantic_key(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    tokens_a = a.split()
    tokens_b = b.split()
    # Short phrases are too semantically dense for fuzzy grouping. Count common
    # contractions as one natural word so punctuation normalization cannot make a
    # three-word phrase accidentally enter the fuzzy path.
    if min(len(_natural_tokens(left)), len(_natural_tokens(right))) <= 3:
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
    """Join only a 2-3 word exact-prefix false start to a nearby longer retry."""
    if left.source_asset_id != right.source_asset_id:
        return False
    if _gap_between(left, right) > maximum_gap_sec:
        return False
    left_tokens = _natural_tokens(left.text)
    right_tokens = _natural_tokens(right.text)
    if not left_tokens or not right_tokens:
        return False
    short, long = (left_tokens, right_tokens) if len(left_tokens) <= len(right_tokens) else (right_tokens, left_tokens)
    if not 2 <= len(short) <= 3 or len(long) < 5:
        return False
    return long[:len(short)] == short


# D-097.A (retry-family completeness): SAME-OPENING RESTART evidence.
#
# Run 34008386434 (D-096 collision C-1): a failed take, its abandoned
# restart and the clean retry all began with the same four words within a
# few seconds of each other, yet `retry_similarity` scored the failed take
# vs the clean retry 0.0 -- its 0.60 word-containment floor is defeated by
# a self-correction that rewrites the middle of the sentence ("hablé con"
# -> "cambié de", "todos los test" -> "un test de todo") -- and the
# semantic arbiter, asked about the pair, declined. A creator restarting a
# sentence from its first words within seconds IS recording-process
# evidence of a retry, independent of what any semantic judge thinks about
# the factual detail that changed: the two deliveries must compete (the
# corrected one should win) rather than both play. Two deterministic shapes,
# both bounded by temporal adjacency and shared content beyond the opening
# so a shared discourse connector ("y después de eso ...") never qualifies:
# (1) RESTART -- both takes long enough to carry content after the opening,
#     and their post-opening content overlaps materially;
# (2) ABANDONED START -- the shorter take is at most half the longer one,
#     shares the opening and at least one content word beyond it (the
#     "Al terminar mi contrato le pedía a mi ginecóloga" shape, too short
#     for (1) and not an exact prefix for `_safe_short_prefix_retry`).
# No Video00 wording is referenced; thresholds are structural.
_RESTART_OPENING_TOKENS = 4
_RESTART_MAXIMUM_GAP_SEC = 8.0
_RESTART_MINIMUM_REMAINDER_OVERLAP = 0.40
_RESTART_MINIMUM_SHARED_CONTENT = 2
_RESTART_STOP = frozenset({
    # es
    "que", "con", "por", "para", "una", "uno", "unos", "unas", "los", "las", "del", "les",
    "como", "pero", "muy", "más", "mas", "sin", "sobre", "entre", "hasta", "desde", "ella",
    "ellos", "ellas", "eso", "esa", "ese", "esto", "esta", "este", "aquí", "ahí", "allí",
    "era", "fue", "son", "está", "esta", "hay", "había", "tenía", "tener", "sea", "ser",
    # en
    "the", "and", "that", "this", "with", "for", "was", "were", "are", "have", "has", "had",
    "you", "your", "our", "they", "them", "there", "here", "then", "than", "but", "not",
    "just", "very", "also", "into", "from", "what", "when", "which", "who",
})


def _restart_content(tokens: tuple[str, ...]) -> set[str]:
    return {token for token in tokens if len(token) >= 3 and token not in _RESTART_STOP}


def same_opening_restart(
    left: CandidateTake,
    right: CandidateTake,
    *,
    maximum_gap_sec: float = _RESTART_MAXIMUM_GAP_SEC,
    opening_tokens: int = _RESTART_OPENING_TOKENS,
    minimum_remainder_overlap: float = _RESTART_MINIMUM_REMAINDER_OVERLAP,
    minimum_shared_content: int = _RESTART_MINIMUM_SHARED_CONTENT,
) -> str | None:
    """Return the deterministic restart evidence kind ("same_opening_restart"
    or "same_opening_abandoned_start") joining two takes, or None. See the
    D-097.A module comment above for the rationale and bounds."""
    if left.source_asset_id != right.source_asset_id:
        return None
    if _gap_between(left, right) > maximum_gap_sec:
        return None
    left_tokens = _natural_tokens(left.text)
    right_tokens = _natural_tokens(right.text)
    if min(len(left_tokens), len(right_tokens)) < opening_tokens + 2:
        return None
    if left_tokens[:opening_tokens] != right_tokens[:opening_tokens]:
        return None
    left_rest = _restart_content(left_tokens[opening_tokens:])
    right_rest = _restart_content(right_tokens[opening_tokens:])
    if not left_rest or not right_rest:
        return None
    shared = left_rest & right_rest
    smaller = min(len(left_rest), len(right_rest))
    if len(shared) >= minimum_shared_content and len(shared) / smaller >= minimum_remainder_overlap:
        return "same_opening_restart"
    short, long = (left_tokens, right_tokens) if len(left_tokens) <= len(right_tokens) else (right_tokens, left_tokens)
    if shared and len(short) * 2 <= len(long):
        return "same_opening_abandoned_start"
    return None


# D-097.12 (stomach family, bounded encargo): an ABANDONED ATTEMPT completed
# by a later retry the creator self-corrects with different wording all the
# way through (not just the first four words `same_opening_restart` requires).
#
# RAW 34043967265/34045158712/34047064840/34048444463 (D-097.9-.11): "Tuve
# problemas estomacales ... y me diagnosticaron con..." (grammatically open
# tail, R15) and "Tuve problemas de digestión ... dijeron que tenía
# gastritis." share only a 2-word opening -- the creator rewrote "estomacales"
# to "de digestión" and "diagnosticaron con" to "dijeron que tenía gastritis"
# -- so `same_opening_restart`'s 4-token exact-prefix requirement never
# fires, and the semantic arbiter answered this exact pair SAME 0.95 once and
# NOT-same 0.85/0.90 twice on "incomplete vs complete", the one reason the
# module comment above already rules must never gate a retry family. An
# EARLIER take that is provably incomplete (open tail / trailing off, not a
# label or position alone) is recording-process evidence independent of the
# arbiter's answer -- but a 2-word opening is common enough ("Tuve
# problemas...") that it must never join on the opening alone: the SAME
# gastritis pair vs an unrelated "Tuve problemas de estómago ... no hay que
# preguntar." aside shares the identical 2-word opening and ZERO further
# content, and must not merge (an independent sentence sharing only a topic
# opener, D-020). The gate is therefore the SAME shared-content-beyond-the-
# opening test `same_opening_restart` already uses, just at a shorter
# opening and a wider gap (the creator's aside intervenes between the
# abandoned attempt and its completion in the source recording).
_INCOMPLETE_RETRY_OPENING_TOKENS = 2
_INCOMPLETE_RETRY_MAXIMUM_GAP_SEC = 20.0
_INCOMPLETE_RETRY_MINIMUM_TOKENS = 5
_INCOMPLETE_RETRY_MINIMUM_SHARED_CONTENT = 2


def incomplete_attempt_completed_by_retry(
    left: CandidateTake,
    right: CandidateTake,
    *,
    maximum_gap_sec: float = _INCOMPLETE_RETRY_MAXIMUM_GAP_SEC,
    opening_tokens: int = _INCOMPLETE_RETRY_OPENING_TOKENS,
    minimum_tokens: int = _INCOMPLETE_RETRY_MINIMUM_TOKENS,
    minimum_shared_content: int = _INCOMPLETE_RETRY_MINIMUM_SHARED_CONTENT,
) -> str | None:
    """Return `"incomplete_attempt_completed_by_retry"` when an earlier,
    grammatically incomplete take is completed by a later delivery sharing
    its opening and real content beyond it, or None. Order-independent:
    the chronologically earlier take must be the incomplete one. See the
    D-097.12 module comment above."""
    if left.source_asset_id != right.source_asset_id:
        return None
    if _gap_between(left, right) > maximum_gap_sec:
        return None
    earlier, later = (left, right) if left.start <= right.start else (right, left)
    if earlier.complete_idea or not later.complete_idea:
        return None
    earlier_tokens = _natural_tokens(earlier.text)
    later_tokens = _natural_tokens(later.text)
    if min(len(earlier_tokens), len(later_tokens)) < minimum_tokens:
        return None
    if earlier_tokens[:opening_tokens] != later_tokens[:opening_tokens]:
        return None
    earlier_rest = _restart_content(earlier_tokens[opening_tokens:])
    later_rest = _restart_content(later_tokens[opening_tokens:])
    if len(earlier_rest & later_rest) >= minimum_shared_content:
        return "incomplete_attempt_completed_by_retry"
    return None


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
                if _safe_short_prefix_retry(take, item) or same_opening_restart(take, item):
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


# D-100 (D-099 Gap #1, bounded encargo): CONFIRMED MULTIMODAL
# RECORDING-BEHAVIOR CORROBORATION.
#
# D-099's dataflow trace found that `performance_confirmation.py` already
# promotes dense local visual candidates (MediaPipe reset/break evidence
# cross-checked against a nearby lexically-similar take) into CONFIRMED
# `wrong_take`/`retry_setup` events -- real evidence the creator visibly
# reset, stumbled, or interrupted a delivery -- but this module never
# received any of it: every restart-evidence rule above is lexical only.
# D-097.12's stomach fix needed a brand-new lexical rule (opening-token
# window + shared-content-beyond-opening) precisely because of this
# blindness. This rule closes that ONE plumbing gap by letting a
# CONFIRMED event corroborate a WEAKER lexical link than the rules above
# require -- never replacing them, never running when they already fired
# (see `reconcile_semantic_idea_equivalence`'s merge loop: this is tried
# only after `same_opening_restart`, `_safe_short_prefix_retry` and
# `incomplete_attempt_completed_by_retry` all return None).
#
# Deliberately NOT a standalone trigger -- three independent safety gates,
# each proven necessary by a specific negative control (see
# tests/test_cutsell_d100_multimodal_retry_corroboration.py):
#   1. REAL shared topical content beyond stopwords must exist between the
#      two takes (D-020: an independent sentence sharing nothing but a
#      confirmed nearby reset must never merge on recording behavior
#      alone -- confirmed evidence with no textual link is not evidence
#      of a retry, it is evidence of *something* happening nearby).
#   2. The two takes must be asymmetric in completeness (one incomplete,
#      one complete) -- the same incomplete-attempt-completed-by-retry
#      shape D-097.12 already established as the safe retry pattern. Two
#      genuinely complementary COMPLETE statements (the D-020 composite
#      case) are excluded by this gate even when they share real content
#      and a confirmed event happens to sit between them.
#   3. The confirmed event must sit at the physical BOUNDARY between the
#      two takes, not merely somewhere in the source -- a reset far from
#      this specific transition is not evidence about THIS pair.
_MULTIMODAL_CORROBORATION_KINDS = frozenset({"wrong_take", "retry_setup"})
_MULTIMODAL_CORROBORATION_MAXIMUM_GAP_SEC = 10.0
_MULTIMODAL_CORROBORATION_MINIMUM_TOKENS = 4
_MULTIMODAL_CORROBORATION_MINIMUM_SHARED_CONTENT = 1
_MULTIMODAL_CORROBORATION_MINIMUM_OVERLAP_RATIO = 0.25
_MULTIMODAL_CORROBORATION_EVENT_BEFORE_SEC = 1.0
_MULTIMODAL_CORROBORATION_EVENT_AFTER_SEC = 2.0

# One confirmed-event record: (kind, start, end). Plain tuples, not
# `TemporalEvent` -- this module stays a pure lexical/text module with no
# dependency on `whole_video_analysis.py`; the caller (`take_grouping_
# provider.reconcile_semantic_idea_equivalence`) is responsible for
# narrowing `WholeVideoContext` down to this shape (see `whole_video_
# analysis.confirmed_recording_behavior_events`).
ConfirmedEvent = Tuple[str, float, float]


def _confirmed_event_at_boundary(
    events: Iterable[ConfirmedEvent],
    *,
    boundary_start: float,
    boundary_end: float,
) -> ConfirmedEvent | None:
    for event in events:
        kind, start, end = event
        if kind not in _MULTIMODAL_CORROBORATION_KINDS:
            continue
        if end >= boundary_start and start <= boundary_end:
            return event
    return None


def multimodal_corroborated_retry(
    left: CandidateTake,
    right: CandidateTake,
    confirmed_events: Mapping[str, Tuple[ConfirmedEvent, ...]] | None,
    *,
    maximum_gap_sec: float = _MULTIMODAL_CORROBORATION_MAXIMUM_GAP_SEC,
    minimum_tokens: int = _MULTIMODAL_CORROBORATION_MINIMUM_TOKENS,
    minimum_shared_content: int = _MULTIMODAL_CORROBORATION_MINIMUM_SHARED_CONTENT,
    minimum_overlap_ratio: float = _MULTIMODAL_CORROBORATION_MINIMUM_OVERLAP_RATIO,
) -> Tuple[str, ConfirmedEvent] | None:
    """Return `("multimodal_corroborated_retry", confirmed_event)` when a
    CONFIRMED recording-behavior event (`wrong_take`/`retry_setup`, from
    `performance_confirmation.py`) sits at the physical boundary between
    `left` and `right` AND they share real (weaker-than-the-lexical-rules-
    above) topical content with an incomplete/complete asymmetry, or
    `None`. See the module comment above for why each gate exists and
    which negative control it protects. `confirmed_events` empty or
    `None` returns `None` immediately -- zero behavior change when no
    multimodal evidence is supplied (backward compatibility)."""
    if not confirmed_events:
        return None
    if left.source_asset_id != right.source_asset_id:
        return None
    if _gap_between(left, right) > maximum_gap_sec:
        return None
    events = confirmed_events.get(left.source_asset_id)
    if not events:
        return None
    earlier, later = (left, right) if left.start <= right.start else (right, left)
    if earlier.complete_idea and later.complete_idea:
        return None  # gate 2: no asymmetry -- protects complementary-COMPLETE pairs
    earlier_tokens = _natural_tokens(earlier.text)
    later_tokens = _natural_tokens(later.text)
    if min(len(earlier_tokens), len(later_tokens)) < minimum_tokens:
        return None
    earlier_content = _restart_content(earlier_tokens)
    later_content = _restart_content(later_tokens)
    if not earlier_content or not later_content:
        return None
    shared = earlier_content & later_content
    smaller = min(len(earlier_content), len(later_content))
    if len(shared) < minimum_shared_content or len(shared) / smaller < minimum_overlap_ratio:
        return None  # gate 1: no real shared topic -- protects unrelated pairs
    found = _confirmed_event_at_boundary(
        events,
        boundary_start=earlier.end - _MULTIMODAL_CORROBORATION_EVENT_BEFORE_SEC,
        boundary_end=later.start + _MULTIMODAL_CORROBORATION_EVENT_AFTER_SEC,
    )
    if found is None:
        return None  # gate 3: no confirmed event at THIS boundary
    return "multimodal_corroborated_retry", found
