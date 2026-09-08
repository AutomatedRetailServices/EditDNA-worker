"""Provider boundary for semantic retry/take grouping."""
from __future__ import annotations

from collections import Counter

import re
from dataclasses import dataclass
from typing import Mapping, Protocol, Tuple

from .contracts import CandidateTake
from .providers import ProviderStatus
from .semantic_atom_importance import _clause_has_any
from .semantic_idea_equivalence import (
    IdeaEquivalencePair,
    IdeaEquivalenceRequest,
    SemanticEquivalenceArbiter,
    SemanticEquivalenceGatePolicy,
    safe_check_idea_equivalence,
    same_idea_by_pair_index,
)
from .take_grouping import (
    _safe_short_prefix_retry,
    group_takes,
    incomplete_attempt_completed_by_retry,
    multimodal_corroborated_retry,
    retry_similarity,
    same_opening_restart,
    semantic_key,
)


def _attempt_relationship_authority():
    """Lazy import (D-158): `attempt_relationship_authority.py` imports
    `watch_listen_understanding.py`, which imports `attempt_reconstruction.
    py`, which imports `session_boundaries.py`, which imports THIS module
    for `TakeGroupingProvider` -- a module-level import here would be a
    genuine import cycle. Deferring it to call time (Python caches the
    module after the first successful import either way) is the standard,
    minimal fix; it changes no behavior and adds no new dependency edge at
    import time."""
    from . import attempt_relationship_authority
    return attempt_relationship_authority

# General (English + Spanish) "this is a new/additional item, not a restatement"
# discourse markers -- a candidate pair where exactly ONE side carries one of
# these is EVIDENCE the marked side may be introducing a distinct point,
# regardless of what the arbiter said. Found via an offline audit of RAW
# 33432104336: the semantic-equivalence arbiter confirmed "Otro sintoma era
# que me salian espinillas ..." ("ANOTHER symptom was that I also got
# pimples...") as the same idea as an earlier, unrelated pimples mention --
# the discourse marker itself ("otro sintoma" / "another symptom") is the
# speaker explicitly signalling a DISTINCT, additional point, which a coarse
# topical-similarity judgment can miss. General across any talking-head
# content that enumerates multiple points -- no Video00 fact, phrase, or
# literal transcript text is hardcoded, only the generic connector.
# Deliberately excludes "otra vez"/"another time"/"again", which mean
# REPETITION of the same thing, not a new one.
#
# D-048 FIX 1: this evidence alone is no longer sufficient to override a
# high-confidence same-idea verdict -- see _marked_side_diverges_in_content's
# own docstring below for why (D-047 Case 1: the marker can just as easily
# open a RESTATEMENT of the same specific content, e.g. "Otro sintoma era
# que me salian espinillas ... detras de la oreja ... cuello ... alergia..."
# repeating the exact same symptom/location as an unmarked prior mention).
# The marker vocabulary itself is UNCHANGED by this fix -- "tambien"/"also"
# were deliberately evaluated and left out: unlike the deliberate, specific
# "otro sintoma"/"another symptom" phrasing, they are extremely common
# ordinary connectors that show up on either side of countless unrelated
# pairs, so even content-gated they would flag far more pairs than the
# guard is meant to examine at all -- an unrelated behavior change, not a
# refinement of this one. "on top of that"/"an additional"/"one more thing"
# already cover the same general "additive framing" category without that
# collision risk.
_DISTINCT_ADDITION_MARKERS = (
    "otro sintoma", "otro síntoma", "otra cosa", "otro problema", "otro punto",
    "otra situacion", "otra situación", "otro detalle", "otro aspecto",
    "another symptom", "another issue", "another problem", "another thing",
    "a different issue", "a different problem", "an additional", "one more thing",
    "on top of that",
)


def _has_distinct_addition_marker(text: str) -> bool:
    return _clause_has_any(text, _DISTINCT_ADDITION_MARKERS)


# D-048 FIX 1: minimal local content-token machinery, mirroring
# final_sibling_grouping._content (not imported directly -- that module
# imports FROM this one, so importing back would be circular; this is the
# same small, order-independent bag-of-words helper other cutsell_worker
# modules each keep a local copy of for the same reason).
_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+", re.IGNORECASE)
_CONTENT_STOP = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "i",
    "in", "is", "it", "its", "me", "my", "of", "on", "or", "so", "that", "the",
    "this", "to", "was", "we", "with", "you", "your", "also", "another", "other",
    "one", "more", "thing", "different", "additional", "top", "problem", "issue",
    "al", "como", "con", "cuando", "de", "del", "el", "en", "es", "esta",
    "este", "la", "las", "le", "les", "lo", "los", "me", "mi", "mis", "o", "para",
    "pero", "por", "porque", "que", "se", "si", "sin", "su", "sus", "un", "una",
    "unos", "unas", "y", "yo", "tambien", "también", "otro", "otra", "sintoma",
    "síntoma", "cosa", "situacion", "situación", "detalle", "aspecto", "punto",
})


def _content_tokens(text: str) -> frozenset[str]:
    """Order-independent, marker/connector-stripped content vocabulary of one
    candidate's text. Excludes the addition-marker words themselves (and
    their own immediate generic frame: "another", "one more thing", "otro
    problema", etc.) alongside ordinary stopwords, so two candidates that
    share only the discourse marker's own generic scaffolding never register
    as content overlap -- overlap must come from the actual claim."""
    return frozenset(
        token for token in (t.casefold() for t in _TOKEN_RE.findall(str(text or "")))
        if len(token) >= 3 and token not in _CONTENT_STOP
    )


# D-048 FIX 1 thresholds -- derived from the two calibration shapes named in
# the D-047 directive, not picked arbitrarily:
#   founding D-039 case (genuinely distinct: "manchas rojas en la piel del
#   BRAZO" vs "manchas rojas en la piel de la PIERNA" -- arm vs leg): shared
#   content tokens = 3.
#   D-047 Case 1 false positive (genuine retry: bad pimples monolith vs its
#   "Otro sintoma" restatement, same symptom + same "detras de la oreja"/
#   "cuello" location): shared content tokens = 9, coverage (shared /
#   shorter side's own content vocabulary) ~0.60-0.64.
# The floor sits strictly between the two: high enough that the founding
# case (3 shared, generic-only overlap) still blocks, low enough that a
# same-specific-content retry (9 shared, including the distinguishing nouns)
# does not. See test_cutsell_d048_fix1_distinct_addition_guard.py for the
# full regression suite locking both ends and the boundary.
_DIVERGENCE_MIN_SHARED_CONTENT = 6
_DIVERGENCE_MIN_SHARED_COVERAGE = 0.55


def _marked_side_diverges_in_content(left_text: str, right_text: str) -> bool:
    """D-048 FIX 1: is there enough of the marked candidate's OWN specific
    content -- beyond the addition marker's own generic scaffolding --
    absent from the other candidate to trust the marker as real evidence of
    a distinct point?

    The original D-039 guard blocked purely on "exactly one side carries the
    marker," which cannot tell a marker introducing a genuinely new fact
    (the founding audit: two different body-part locations) apart from a
    marker prefacing a near-identical restatement of the same specific
    claim (D-047 Case 1: the same symptom AND the same location, just
    reformulated) -- a coarse topical read from the arbiter can call both
    cases "same idea," and the marker alone cannot separate them either.

    Returns True (divergence -- the D-039 block should still apply) when
    shared specific content is too thin to prove the marked side is a
    restatement; False (no meaningful divergence -- treat the marker as
    discourse framing only, let the arbiter's same-idea verdict stand) when
    the two share substantial specific content including whatever concrete
    nouns/details carry the claim, not just the generic setup.
    """
    left = _content_tokens(left_text)
    right = _content_tokens(right_text)
    if len(left) < 3 or len(right) < 3:
        # Too little content on one side to prove real overlap either way --
        # fail toward the guard's original, protective behavior.
        return True
    shared = left & right
    if len(shared) < _DIVERGENCE_MIN_SHARED_CONTENT:
        return True
    coverage = len(shared) / min(len(left), len(right))
    return coverage < _DIVERGENCE_MIN_SHARED_COVERAGE


# D-083: DISTINCT-IDEA RETRY GROUPING SAFETY (within-group weak-pair gate)
#
# `_marked_side_diverges_in_content` above only ever guards
# `reconcile_semantic_idea_equivalence`'s CROSS-group merges.
# `split_incohesive_retry_groups` below -- the WITHIN-group cohesion pass --
# has no equivalent safety net at all: any arbiter "same_idea=True" verdict
# for a within-group weak pair is trusted unconditionally, marker or no
# marker. Live audit of the D-082 stability battery (docs/
# CUTSELL_DECISIONS.md D-083) found a baseline group bundling one back-acne
# mention with three separate hormonal-pimples mentions (none of the pairs
# scoring above `_provider_members_compatible`'s own threshold, so all six
# became weak pairs sent to this function's own arbiter call). One of those
# pimples mentions carried the exact "otro sintoma"/"another symptom" marker
# this module already treats as distinct-addition evidence elsewhere -- but
# because `split_incohesive_retry_groups` never applies that check itself,
# a coarse same-topic confirmation involving the marked mention had nothing
# to override it here, unlike the identical scenario in
# `reconcile_semantic_idea_equivalence`.
#
# FIX: apply the exact same marker-gated `_marked_side_diverges_in_content`
# override to this function's own arbiter confirmations, for consistency
# between the two cohesion passes. A broader, unconditional content-overlap
# floor on EVERY confirmation (marked or not) was evaluated and rejected: it
# fails at both ends against this module's own regression contract --
# `test_arbiter_confirmed_retry_stays_grouped`'s true-retry paraphrase pair
# ("I had seasonal back acne ... an ointment" vs "Every season I would get
# back breakouts ... an ointment for it") scores LOWER lexical/claim overlap
# by every measure tried (raw shared tokens, coverage, claim-level Dice)
# than the specific unmarked pimples pair this fix must NOT merge -- proving
# no fixed lexical-overlap threshold can separate "same proposition, very
# different words" (trust the arbiter -- exactly what an LLM arbiter exists
# to catch) from "different proposition, overlapping topic vocabulary"
# (don't) in general. The marker is a genuine, narrow, non-lexical signal;
# widening the override beyond it re-introduces exactly the false-positive
# class this module's own paraphrase-retry contract already forbids. An
# unmarked within-group weak pair (e.g. two pimples mentions neither of
# which carries a discourse marker) is therefore still governed by the
# arbiter's own judgment alone here, same as before this fix -- see D-083's
# own decision log entry for the honest scope of what this closes and what
# it does not.
def _within_group_arbiter_confirmation_diverges(
    take_map: dict[str, CandidateTake], left_id: str, right_id: str,
) -> bool:
    left_text = take_map[left_id].text
    right_text = take_map[right_id].text
    left_marked = _has_distinct_addition_marker(left_text)
    right_marked = _has_distinct_addition_marker(right_text)
    return left_marked != right_marked and _marked_side_diverges_in_content(left_text, right_text)


@dataclass(frozen=True)
class TakeGroupingProviderResult:
    groups: Tuple[Tuple[str, ...], ...]
    status: ProviderStatus
    reason: str = ""


class TakeGroupingProvider(Protocol):
    def group(
        self,
        takes: Tuple[CandidateTake, ...],
        context_text: str = "",
    ) -> TakeGroupingProviderResult: ...


def _baseline_groups(takes: Tuple[CandidateTake, ...]) -> Tuple[Tuple[str, ...], ...]:
    grouped = group_takes(takes)
    return tuple(tuple(item.clip_id for item in members) for members in grouped.values())


def _provider_members_compatible(left: CandidateTake, right: CandidateTake) -> bool:
    if left.source_asset_id != right.source_asset_id:
        return False
    score = retry_similarity(left.text, right.text)
    gap = max(0.0, max(left.start, right.start) - min(left.end, right.end))
    if gap <= 8.0:
        return score >= 0.72
    return score >= 0.82


def _constrain_provider_group(
    group: Tuple[str, ...],
    take_map: dict[str, CandidateTake],
) -> Tuple[Tuple[str, ...], ...]:
    """Split provider groups using complete-link retry compatibility."""
    members = [take_map[clip_id] for clip_id in group if clip_id in take_map]
    members.sort(key=lambda take: (take.source_order, take.start, take.end, take.clip_id))
    if len(members) <= 1:
        return (tuple(take.clip_id for take in members),) if members else ()

    clusters: list[list[CandidateTake]] = []
    for take in members:
        placed = False
        for cluster in clusters:
            if all(_provider_members_compatible(take, existing) for existing in cluster):
                cluster.append(take)
                placed = True
                break
        if not placed:
            clusters.append([take])
    return tuple(tuple(take.clip_id for take in cluster) for cluster in clusters)


def _repair_groups(
    groups: Tuple[Tuple[str, ...], ...],
    takes: Tuple[CandidateTake, ...],
) -> tuple[Tuple[Tuple[str, ...], ...], bool]:
    natural_ids = tuple(take.clip_id for take in takes)
    take_map = {take.clip_id: take for take in takes}
    allowed = set(natural_ids)
    seen: set[str] = set()
    repaired = False
    normalized: list[Tuple[str, ...]] = []

    for raw_group in groups:
        kept: list[str] = []
        for raw_id in raw_group:
            clip_id = str(raw_id)
            if clip_id not in allowed or clip_id in seen:
                repaired = True
                continue
            seen.add(clip_id)
            kept.append(clip_id)
        if kept:
            constrained = _constrain_provider_group(tuple(kept), take_map)
            if len(constrained) > 1:
                repaired = True
            normalized.extend(group for group in constrained if group)
        elif raw_group:
            repaired = True

    for clip_id in natural_ids:
        if clip_id not in seen:
            normalized.append((clip_id,))
            seen.add(clip_id)
            repaired = True

    return tuple(normalized), repaired


def _group_gap(
    left_group: Tuple[str, ...],
    right_group: Tuple[str, ...],
    take_map: dict[str, CandidateTake],
) -> float:
    gaps = []
    for left_id in left_group:
        left = take_map[left_id]
        for right_id in right_group:
            right = take_map[right_id]
            if left.source_asset_id != right.source_asset_id:
                continue
            gaps.append(max(0.0, max(left.start, right.start) - min(left.end, right.end)))
    return min(gaps) if gaps else float("inf")


def _reconcile_similarity_threshold(group_gap_sec: float) -> float:
    if group_gap_sec <= 8.0:
        return 0.80
    if group_gap_sec <= 30.0:
        return 0.90
    return 0.97


def _is_prefix_fragment(fragment: CandidateTake, reference: CandidateTake) -> bool:
    fragment_tokens = semantic_key(fragment.text).split()
    reference_tokens = semantic_key(reference.text).split()
    if not fragment_tokens or len(fragment_tokens) > 8:
        return False
    return reference_tokens[: len(fragment_tokens)] == fragment_tokens


def _is_material_prefix_fragment(fragment: CandidateTake, reference: CandidateTake) -> bool:
    """Recognize exact false-start prefixes even when completeness heuristics overstate them.

    ``complete_idea`` intentionally fails open for longer speech, so a seven-word
    false start can be marked complete merely because it crossed the length threshold.
    For retry reconciliation only, an exact prefix can be treated as non-substantive
    when it is materially shorter in both words and time. The fragment remains in the
    group for Best Take; this only prevents it from vetoing two strong full retries.
    """
    if not _is_prefix_fragment(fragment, reference):
        return False
    fragment_tokens = semantic_key(fragment.text).split()
    reference_tokens = semantic_key(reference.text).split()
    return (
        len(reference_tokens) - len(fragment_tokens) >= 3
        and fragment.duration_sec + 0.75 <= reference.duration_sec
    )


def _substantive_reconcile_members(members: list[CandidateTake]) -> list[CandidateTake]:
    """Exclude only structural prefix debris from complete-link retry comparison.

    A provider may already place a false-start prefix beside its full attempt. That
    debris should remain in the group for Best Take, but must not veto reconciliation
    with another near-identical full attempt. Because ``complete_idea`` is deliberately
    fail-open for longer speech, a materially shorter exact prefix may be neutralized
    here even when that heuristic marked it complete.
    """
    substantive: list[CandidateTake] = []
    for candidate in members:
        prefix_debris = any(
            other.clip_id != candidate.clip_id
            and (
                (not candidate.complete_idea and _is_prefix_fragment(candidate, other))
                or _is_material_prefix_fragment(candidate, other)
            )
            for other in members
        )
        if not prefix_debris:
            substantive.append(candidate)
    return substantive or members


def _groups_should_reconcile(
    left_group: Tuple[str, ...],
    right_group: Tuple[str, ...],
    take_map: dict[str, CandidateTake],
) -> bool:
    """Use group-level timing and complete-link over substantive retry attempts."""
    left_members = []
    right_members = []
    for clip_id in left_group:
        take = take_map.get(clip_id)
        if take is None:
            return False
        left_members.append(take)
    for clip_id in right_group:
        take = take_map.get(clip_id)
        if take is None:
            return False
        right_members.append(take)
    if not left_members or not right_members:
        return False
    source_ids = {take.source_asset_id for take in (*left_members, *right_members)}
    if len(source_ids) != 1:
        return False

    threshold = _reconcile_similarity_threshold(_group_gap(left_group, right_group, take_map))
    left_substantive = _substantive_reconcile_members(left_members)
    right_substantive = _substantive_reconcile_members(right_members)
    return all(
        retry_similarity(left.text, right.text) >= threshold
        for left in left_substantive
        for right in right_substantive
    )


def _reconcile_missed_retries(
    groups: Tuple[Tuple[str, ...], ...],
    takes: Tuple[CandidateTake, ...],
) -> tuple[Tuple[Tuple[str, ...], ...], bool]:
    if len(groups) <= 1:
        return groups, False
    take_map = {take.clip_id: take for take in takes}
    merged: list[list[str]] = []
    changed = False

    for group in groups:
        target_index = None
        for index, existing in enumerate(merged):
            if _groups_should_reconcile(tuple(existing), group, take_map):
                target_index = index
                break
        if target_index is None:
            merged.append(list(group))
        else:
            merged[target_index].extend(group)
            changed = True

    ordered_groups: list[Tuple[str, ...]] = []
    for group in merged:
        unique = {clip_id for clip_id in group}
        ordered = sorted(
            unique,
            key=lambda clip_id: (
                take_map[clip_id].source_order,
                take_map[clip_id].start,
                take_map[clip_id].end,
                clip_id,
            ),
        )
        ordered_groups.append(tuple(ordered))
    ordered_groups.sort(
        key=lambda group: (
            take_map[group[0]].source_order,
            take_map[group[0]].start,
            take_map[group[0]].end,
            group[0],
        )
    )
    return tuple(ordered_groups), changed


def _extend_adjacent_retry_groups(
    groups: Tuple[Tuple[str, ...], ...],
    takes: Tuple[CandidateTake, ...],
    *,
    max_gap_sec: float = 8.0,
    minimum_similarity: float = 0.93,
) -> tuple[Tuple[Tuple[str, ...], ...], bool]:
    """Extend a validated retry group only into the next near-verbatim attempt.

    This is intentionally narrower than general reconciliation. It may look past one
    singleton false-start when that false-start is an exact lexical prefix of the
    group's latest substantive attempt. The next substantive singleton must then be
    highly similar (>=0.93) and within eight seconds. This recovers serial retries
    without reintroducing broad transitive/topic chaining.
    """
    take_map = {take.clip_id: take for take in takes}
    ordered = tuple(sorted(takes, key=lambda item: (item.source_order, item.start, item.end, item.clip_id)))
    position = {take.clip_id: index for index, take in enumerate(ordered)}
    group_lists = [list(group) for group in groups]
    membership = {clip_id: index for index, group in enumerate(group_lists) for clip_id in group}
    changed = False

    for group_index, group in enumerate(group_lists):
        if len(group) < 2:
            continue
        members = sorted(
            (take_map[clip_id] for clip_id in group if clip_id in take_map),
            key=lambda item: (item.source_order, item.start, item.end, item.clip_id),
        )
        if not members:
            continue
        anchor = members[-1]
        anchor_pos = position.get(anchor.clip_id)
        if anchor_pos is None:
            continue

        pending_prefix: CandidateTake | None = None
        for candidate in ordered[anchor_pos + 1 :]:
            if candidate.source_asset_id != anchor.source_asset_id:
                break
            if candidate.start - anchor.end > max_gap_sec:
                break
            candidate_group_index = membership.get(candidate.clip_id)
            if candidate_group_index == group_index:
                continue
            if candidate_group_index is None or len(group_lists[candidate_group_index]) != 1:
                break

            if pending_prefix is None and len(candidate.text.split()) <= 3 and _is_prefix_fragment(candidate, anchor):
                pending_prefix = candidate
                continue

            if retry_similarity(anchor.text, candidate.text) < minimum_similarity:
                break

            if pending_prefix is not None:
                prefix_group_index = membership[pending_prefix.clip_id]
                group_lists[prefix_group_index].remove(pending_prefix.clip_id)
                group.append(pending_prefix.clip_id)
                membership[pending_prefix.clip_id] = group_index
            group_lists[candidate_group_index].remove(candidate.clip_id)
            group.append(candidate.clip_id)
            membership[candidate.clip_id] = group_index
            changed = True
            break

    normalized = []
    for group in group_lists:
        if not group:
            continue
        unique = sorted(
            set(group),
            key=lambda clip_id: (
                take_map[clip_id].source_order,
                take_map[clip_id].start,
                take_map[clip_id].end,
                clip_id,
            ),
        )
        normalized.append(tuple(unique))
    normalized.sort(
        key=lambda group: (
            take_map[group[0]].source_order,
            take_map[group[0]].start,
            take_map[group[0]].end,
            group[0],
        )
    )
    return tuple(normalized), changed


def _absorb_interstitial_retry_debris(
    groups: Tuple[Tuple[str, ...], ...],
    takes: Tuple[CandidateTake, ...],
    *,
    max_retry_span_sec: float = 15.0,
    max_fragment_sec: float = 2.5,
    max_fragment_words: int = 5,
    max_edge_gap_sec: float = 3.0,
) -> tuple[Tuple[Tuple[str, ...], ...], bool]:
    """Fold short incomplete speech trapped inside or directly beside a retry envelope."""
    take_map = {take.clip_id: take for take in takes}
    ordered = tuple(sorted(takes, key=lambda item: (item.source_order, item.start, item.end, item.clip_id)))
    group_lists = [list(group) for group in groups]
    membership = {clip_id: index for index, group in enumerate(group_lists) for clip_id in group}
    changed = False

    for group_index, group in enumerate(tuple(tuple(item) for item in group_lists)):
        if len(group) < 2:
            continue
        members = [take_map[clip_id] for clip_id in group if clip_id in take_map]
        members.sort(key=lambda item: (item.source_order, item.start, item.end, item.clip_id))
        for left, right in zip(members, members[1:]):
            if left.source_asset_id != right.source_asset_id:
                continue
            if right.start - left.end > max_retry_span_sec:
                continue
            for candidate in ordered:
                if membership.get(candidate.clip_id) == group_index:
                    continue
                if candidate.source_asset_id != left.source_asset_id:
                    continue
                if candidate.start < left.end or candidate.end > right.start:
                    continue
                if candidate.duration_sec > max_fragment_sec:
                    continue
                if len(candidate.text.split()) > max_fragment_words or candidate.complete_idea:
                    continue
                old_index = membership.get(candidate.clip_id)
                if old_index is None or len(group_lists[old_index]) != 1:
                    continue
                group_lists[old_index].remove(candidate.clip_id)
                group_lists[group_index].append(candidate.clip_id)
                membership[candidate.clip_id] = group_index
                changed = True

        first = members[0]
        first_pos = next((index for index, item in enumerate(ordered) if item.clip_id == first.clip_id), None)
        if first_pos is not None and first_pos > 0:
            candidate = ordered[first_pos - 1]
            old_index = membership.get(candidate.clip_id)
            gap = first.start - candidate.end
            if (
                old_index is not None
                and old_index != group_index
                and len(group_lists[old_index]) == 1
                and candidate.source_asset_id == first.source_asset_id
                and 0.0 <= gap <= max_edge_gap_sec
                and candidate.duration_sec <= max_fragment_sec
                and len(candidate.text.split()) <= max_fragment_words + 2
                and _is_prefix_fragment(candidate, first)
            ):
                group_lists[old_index].remove(candidate.clip_id)
                group_lists[group_index].append(candidate.clip_id)
                membership[candidate.clip_id] = group_index
                changed = True

    normalized = []
    for group in group_lists:
        if not group:
            continue
        unique = sorted(
            set(group),
            key=lambda clip_id: (
                take_map[clip_id].source_order,
                take_map[clip_id].start,
                take_map[clip_id].end,
                clip_id,
            ),
        )
        normalized.append(tuple(unique))
    normalized.sort(
        key=lambda group: (
            take_map[group[0]].source_order,
            take_map[group[0]].start,
            take_map[group[0]].end,
            group[0],
        )
    )
    return tuple(normalized), changed


def _cross_group_candidate_pairs(
    groups: Tuple[Tuple[str, ...], ...],
    take_map: dict[str, CandidateTake],
    *,
    maximum_gap_sec: float,
) -> tuple[tuple[int, int, str, str], ...]:
    pairs: list[tuple[int, int, str, str]] = []
    for left_index in range(len(groups)):
        for right_index in range(left_index + 1, len(groups)):
            left_group, right_group = groups[left_index], groups[right_index]
            if _group_gap(left_group, right_group, take_map) > maximum_gap_sec:
                continue
            for left_id in left_group:
                left_take = take_map.get(left_id)
                if left_take is None or len(semantic_key(left_take.text).split()) <= 3:
                    continue
                for right_id in right_group:
                    right_take = take_map.get(right_id)
                    if right_take is None or len(semantic_key(right_take.text).split()) <= 3:
                        continue
                    if left_take.source_asset_id != right_take.source_asset_id:
                        continue
                    pairs.append((left_index, right_index, left_id, right_id))
    return tuple(pairs)


def _raw_content_overlap(left_text: str, right_text: str) -> float:
    """Unfloored word-containment overlap, for RANKING only -- never a hard
    accept/reject gate. retry_similarity() deliberately floors to 0.0 below
    0.60 containment (see reconcile_semantic_idea_equivalence's docstring for
    why that makes it useless as an eligibility gate); this raw score keeps
    the same low-overlap paraphrases distinguishable from zero-overlap
    unrelated text purely as a priority signal."""
    # D-097.8 (R9): CONTENT tokens (the D-048 stoplist above), not the raw
    # semantic_key words -- with stopwords counted, two adjacent sentences
    # of one narrative ("en la ... de ... que se ...") scored 0.6 overlap
    # and outranked a real retry pair whose shared content was the actual
    # claim vocabulary.
    left_tokens = _content_tokens(left_text)
    right_tokens = _content_tokens(right_text)
    if not left_tokens or not right_tokens:
        return 0.0
    shared = len(left_tokens & right_tokens)
    return shared / max(1, min(len(left_tokens), len(right_tokens)))


def _continuation_or_restart_bonus(left_take: CandidateTake, right_take: CandidateTake) -> float:
    """Boost pairs carrying existing continuation/restart evidence this
    codebase already computes on every CandidateTake: an incomplete delivery
    (complete_idea=False) or an exact lexical prefix relationship is a strong,
    general prior that two takes are the same attempt at different points of
    completion -- not a new heuristic, just reusing fields/helpers this module
    already has for other purposes."""
    bonus = 0.0
    if not left_take.complete_idea or not right_take.complete_idea:
        bonus += 0.25
    if _is_prefix_fragment(left_take, right_take) or _is_prefix_fragment(right_take, left_take):
        bonus += 0.25
    return bonus


_PROXIMITY_RANK_WEIGHT = 0.25
# D-097.9 (R12): the ONE authority over the arbiter budget order.
_PAIR_ORDER_AUTHORITY = "take_grouping_provider._rank_candidate_pairs:content_overlap_v2+group_cap"
# D-097.9 (R12): fairness bound INSIDE the ranking authority. In the first
# pass a group may take part in at most this many asked pairs; pairs a
# capped group would push over the bound are deferred behind every
# first-pass pair (still in score order) so a dense neighbourhood of
# mutually similar takes cannot spend the whole budget before a distinct
# paraphrase pair elsewhere is asked (the D-042 concern the retired
# coverage-first wrapper served) -- without ever promoting a zero-evidence
# pair over one with content or restart evidence (the way that wrapper
# spent RAW 34045158712's budget on hair loss <-> stomach aside, 0.02).
_PAIR_BUDGET_PER_GROUP_CAP = 2


def _pair_priority_score(
    left_take: CandidateTake, right_take: CandidateTake, *, gap_sec: float,
) -> float:
    """Composite ranking score for one candidate pair: temporal proximity +
    raw lexical/topical overlap + continuation/restart evidence. General and
    reusable -- no per-video tuning, no hardcoded thresholds beyond what the
    eligibility gate already enforces. Used only to decide WHICH eligible
    pairs get asked about first when there are more than the batch budget
    allows; never used to decide same_idea itself (that stays the arbiter's
    job, or the existing lexical reconciliation's)."""
    proximity = 1.0 / (1.0 + max(0.0, gap_sec))
    overlap = _raw_content_overlap(left_take.text, right_take.text)
    # D-097.8 (R9): proximity is a tie-break, not the lead signal. With a
    # full weight, every ADJACENT pair of different sentences (nodule ->
    # biopsy result, advice -> "no, no, no") scored ~0.7-1.0 on proximity
    # alone and outranked a real retry 14 s away with half its content in
    # common (RAW 34043967265: 55 candidate pairs, 14 asked, 9 of them
    # sequential-narrative neighbours the arbiter rejected, while the
    # abandoned stomach attempt <-> clean gastritis delivery pair -- shared
    # opening, "endoscopía", "donde" -- was never asked; the two abandoned
    # attempts formed their own family and its incomplete winner played).
    # Content overlap and restart/continuation evidence decide the order;
    # proximity only separates otherwise-equal pairs.
    return overlap + _continuation_or_restart_bonus(left_take, right_take) + _PROXIMITY_RANK_WEIGHT * proximity


def _rank_candidate_pairs(
    pairs: tuple[tuple[int, int, str, str], ...],
    take_map: dict[str, CandidateTake],
) -> tuple[tuple[int, int, str, str], ...]:
    """Sort eligible candidate pairs by priority, highest first, so a fixed
    per-request pair budget spends its slots on the pairs most likely to be
    real retries instead of whichever happened to be enumerated first.

    This directly addresses the root cause an offline audit of a real run
    found: _cross_group_candidate_pairs enumerates ALL eligible group-index
    pairs in plain chronological order with no priority, so on any video
    dense enough to exceed the batch cap, coverage became a function of
    "where in iteration order did this pair land" rather than "how likely is
    this to be a real duplicate" -- pairs later in a long video were
    systematically less likely to ever be proposed to the arbiter at all,
    regardless of how obvious a retry they were. Ranking does not remove the
    batch cap or make this pairwise discovery exhaustive; it makes the
    truncation that DOES happen non-arbitrary.
    """
    return tuple(pair for pair, _deferred in _rank_candidate_pairs_with_marks(pairs, take_map))


def _rank_candidate_pairs_with_marks(
    pairs: tuple[tuple[int, int, str, str], ...],
    take_map: dict[str, CandidateTake],
    *,
    per_group_cap: int = _PAIR_BUDGET_PER_GROUP_CAP,
) -> tuple[tuple[tuple[int, int, str, str], bool], ...]:
    """Score order with the D-097.9 per-group fairness bound: first-pass
    pairs (score order, no group beyond `per_group_cap` pairs) followed by
    the deferred pairs (score order). The mark says whether a pair was
    deferred by the cap -- recorded in the run diagnostics so a score out of
    descending order is explained by this authority, never by a wrapper."""
    scored = [
        (_pair_priority_score(take_map[left_id], take_map[right_id], gap_sec=_group_gap(
            (left_id,), (right_id,), take_map,
        )), pair)
        for pair in pairs
        for left_index, right_index, left_id, right_id in (pair,)
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    usage: Counter = Counter()
    accepted: list[tuple[tuple[int, int, str, str], bool]] = []
    deferred: list[tuple[tuple[int, int, str, str], bool]] = []
    for _score, pair in scored:
        left_index, right_index = int(pair[0]), int(pair[1])
        if per_group_cap > 0 and (usage[left_index] >= per_group_cap or usage[right_index] >= per_group_cap):
            deferred.append((pair, True))
            continue
        accepted.append((pair, False))
        usage[left_index] += 1
        usage[right_index] += 1
    return tuple(accepted + deferred)


def _watch_listen_family_evidence_summary(
    enabled: bool, conflict_blocked: list[dict], final_relations: list,
) -> dict:
    """Tail-safe, counts-only D-158 summary -- matches the D-119/D-125/
    D-152/D-155/D-157 compact-summary pattern (no transcript, no per-pair
    reason string dumped here; those live only in `watch_listen_conflict_
    blocked`'s own rows, which are themselves bounded to clip ids + short
    enum/string fields, never transcript text)."""
    if not enabled:
        return {"status": "disabled"}
    if not final_relations:
        return {"status": "no_pairs_evaluated"}
    m = _attempt_relationship_authority()
    return {
        "status": "evaluated",
        "conflict_blocked_count": len(conflict_blocked),
        **m.attempt_relationship_diagnostics(final_relations),
    }


def reconcile_semantic_idea_equivalence(
    groups: Tuple[Tuple[str, ...], ...],
    takes: Tuple[CandidateTake, ...],
    arbiter: SemanticEquivalenceArbiter | None,
    *,
    policy: SemanticEquivalenceGatePolicy = SemanticEquivalenceGatePolicy(),
    maximum_gap_sec: float = 30.0,
    protected_ids: frozenset[str] = frozenset(),
    confirmed_recording_evidence: Mapping[str, Tuple[Tuple[str, float, float], ...]] | None = None,
    watch_listen_spans_by_id: Mapping[str, object] | None = None,
) -> tuple[Tuple[Tuple[str, ...], ...], dict]:
    """Merge groups the lexical layer left separate only when a narrow
    semantic arbiter is confident they are recording attempts of the same
    intended idea. Phase 2 of the architecture rebalance.

    Eligibility is temporal/structural, not a retry_similarity score band:
    a genuine paraphrase pair can score exactly 0.0 on that function's
    word-containment floor -- identical to genuinely unrelated text -- so
    no numeric similarity threshold reliably separates "ambiguous" from
    "definitely distinct" here (confirmed against real paraphrase fixtures;
    see semantic_idea_equivalence tests). A pair is eligible when both
    groups are (a) still separate after the full existing lexical
    reconciliation above, (b) from the same source, (c) within this
    module's own existing 30-second outer reconcile breakpoint
    (_reconcile_similarity_threshold's widest tier -- reused, not
    invented), and (d) both sides longer than retry_similarity's own
    existing short-phrase floor (<=3 tokens is already "not fuzzy-
    comparable" there).

    ``protected_ids`` (D-025): clip ids CompositeResolver already marked as
    an accepted composite's pieces (pipeline.py's ``composite_split_ids``,
    forced into singleton groups by ``apply_composite_group_split`` right
    before this call). Once CompositeResolver has decided two deliveries
    are complementary halves of one composite, this step -- running its
    OWN, separate arbiter call, with no knowledge of that decision -- must
    never re-merge them (which collapses the composite back into an
    ordinary one-winner retry contest and can discard both pieces if a
    third clip wins that contest) or merge either piece into any other
    group. RAW 33366538992 hit exactly this: the two accepted composite
    pieces for the pimples/rash idea were both re-merged with an unrelated
    third clip by this step's own arbiter call, that third clip won the
    resulting contest, and neither composite piece survived to the final
    KEEP sequence, despite CompositeResolver's decision record showing them
    accepted. A protected clip is filtered out of candidate-pair generation
    entirely -- it is not merely "unlikely to merge", it cannot be proposed
    as a candidate pair at all here.

    Fails open throughout: a pair the arbiter did not confidently confirm
    as the same idea leaves both groups exactly as they were.
    """
    if len(groups) < 2:
        return groups, {"status": "not_requested", "candidate_pair_count": 0, "merged_pair_count": 0}

    take_map = {take.clip_id: take for take in takes}
    candidate_pairs = _cross_group_candidate_pairs(groups, take_map, maximum_gap_sec=maximum_gap_sec)
    if protected_ids:
        candidate_pairs = tuple(
            pair for pair in candidate_pairs
            if pair[2] not in protected_ids and pair[3] not in protected_ids
        )
    if not candidate_pairs:
        return groups, {"status": "no_eligible_pairs", "candidate_pair_count": 0, "merged_pair_count": 0}

    # Union-find over group indices: if any member of group A is confirmed
    # the same idea as any member of group B, the two contests are one
    # retry family and their whole groups merge.
    parent = list(range(len(groups)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    audit: list[dict] = []
    distinct_addition_blocked: list[dict] = []
    merged_count = 0
    # D-158 Phase C: structured Attempt Relationship consumption of D-157's
    # Watch+Listen Understanding V1 -- OFF by default
    # (CUTSELL_WATCH_LISTEN_FAMILY_EVIDENCE_ENABLED), and a total no-op
    # whenever `watch_listen_spans_by_id` is not supplied or carries no
    # evidence for a given pair (fail-open; see attempt_relationship_
    # authority.py's own module docstring for the full authority order and
    # truth table). Can only WITHHOLD a merge this function's own existing
    # rules already decided on -- never creates one from Watch+Listen
    # evidence alone.
    _wla = _attempt_relationship_authority()
    watch_listen_enabled = _wla.watch_listen_family_evidence_enabled()
    watch_listen_conflict_blocked: list[dict] = []
    final_relations: list = []

    # D-097.2 (RAW 34029861712, D-096 C-1 again one tier later): a cross-
    # group candidate pair that carries DETERMINISTIC restart evidence
    # (`take_grouping.same_opening_restart` / `_safe_short_prefix_retry` --
    # same source, seconds apart, same opening words, shared content beyond
    # the opening) is recording-process evidence of a retry and merges here
    # without the arbiter, exactly as D-097.A already ruled for the cohesion
    # pass. The run asked the arbiter about an abandoned "same sentence,
    # restarted" take and its clean retry and it answered NOT-same-idea
    # because the abandoned take was an "incomplete fragment" -- the one
    # reason that must never keep a failed attempt OUT of its family (both
    # then played back to back). The D-083 marker gate is kept: a marked
    # side with real content divergence is still blocked. The pair is not
    # spent on the arbiter's bounded request either.
    restart_merged: list[dict] = []
    remaining_pairs: list[tuple] = []
    for pair in candidate_pairs:
        left_group_index, right_group_index, left_id, right_id = pair
        left_take, right_take = take_map[left_id], take_map[right_id]
        restart_kind = same_opening_restart(left_take, right_take)
        if restart_kind is None and _safe_short_prefix_retry(left_take, right_take):
            restart_kind = "safe_short_prefix_retry"
        if restart_kind is None:
            restart_kind = incomplete_attempt_completed_by_retry(left_take, right_take)
        # D-100 (D-099 Gap #1, bounded encargo): only tried once every
        # existing lexical rule above has already declined this pair --
        # confirmed multimodal evidence CORROBORATES a weaker lexical link
        # the deterministic rules above intentionally do not accept on
        # their own; it never runs ahead of, or in place of, them. See
        # `take_grouping.multimodal_corroborated_retry`'s own module
        # comment for the three safety gates.
        corroboration = None
        if restart_kind is None and confirmed_recording_evidence:
            corroborated = multimodal_corroborated_retry(left_take, right_take, confirmed_recording_evidence)
            if corroborated is not None:
                restart_kind, corroboration = corroborated
        if restart_kind is None:
            remaining_pairs.append(pair)
            continue
        left_marked = _has_distinct_addition_marker(left_take.text)
        right_marked = _has_distinct_addition_marker(right_take.text)
        if left_marked != right_marked and _marked_side_diverges_in_content(left_take.text, right_take.text):
            distinct_addition_blocked.append({
                "left_clip_id": left_id, "right_clip_id": right_id,
                "confidence": 1.0, "reason": f"restart_evidence:{restart_kind}", "source": "restart_evidence",
            })
            remaining_pairs.append(pair)
            continue
        if watch_listen_enabled and watch_listen_spans_by_id:
            wl_relations = _wla.attempt_relation_hypotheses_for_pair(watch_listen_spans_by_id, left_id, right_id)
            final = _wla.resolve_final_attempt_relation(
                would_merge=True, would_merge_source=_wla.RELATION_SOURCE_DETERMINISTIC_RESTART,
                watch_listen_relations=wl_relations,
            )
            final_relations.append(final)
            if not final.would_merge:
                watch_listen_conflict_blocked.append({
                    "left_clip_id": left_id, "right_clip_id": right_id,
                    "watch_listen_relation_evaluated": final.watch_listen_relation_evaluated,
                    "proposition_evidence_status": "merge_proposed",
                    "attempt_relation_final": final.relation,
                    "attempt_relation_source": final.source,
                    "attempt_relation_conflict": final.conflict,
                    "watch_listen_hypothesis_used": final.watch_listen_supported,
                    "semantic_relation_used": _wla.RELATION_SOURCE_DETERMINISTIC_RESTART,
                    "family_membership_action": final.family_membership_action,
                    "family_membership_reason": final.reason,
                })
                remaining_pairs.append(pair)
                continue
        union(left_group_index, right_group_index)
        merged_count += 1
        row = {
            "left_clip_id": left_id, "right_clip_id": right_id,
            "confidence": 1.0, "reason": f"deterministic restart evidence ({restart_kind}); arbiter not consulted",
            "accepted_by": restart_kind,
        }
        if corroboration is not None:
            event_kind, event_start, event_end = corroboration
            # Observability (D-100): make it possible to tell lexical vs.
            # multimodal evidence apart in the audit trail, and which
            # confirmed event/source range corroborated the merge, without
            # ever exposing benchmark-specific transcript text.
            row["corroborating_evidence"] = "confirmed_multimodal_event"
            row["corroborating_event_kind"] = event_kind
            row["corroborating_event_range"] = [round(event_start, 3), round(event_end, 3)]
        audit.append(row)
        restart_merged.append(row)

    ranked_pair_budget: list[dict] = []
    if arbiter is None:
        if merged_count == 0:
            # D-158: this pre-existing early exit (no arbiter, no restart-
            # evidence merge succeeded) must still surface the Watch+Listen
            # evidence summary -- otherwise a merge this function's own
            # deterministic-restart rule proposed and D-158 withheld would
            # silently vanish from observability. Same tail-safe helper used
            # by both later return paths.
            return groups, {
                "status": "not_requested", "candidate_pair_count": len(candidate_pairs), "merged_pair_count": 0,
                "watch_listen_family_evidence": _watch_listen_family_evidence_summary(
                    watch_listen_enabled, watch_listen_conflict_blocked, final_relations,
                ),
            }
        result = None
        truncated: tuple = ()
        decisions: dict = {}
    else:
        # Priority-ranked, not appearance-ordered: see _rank_candidate_pairs's
        # docstring for the root-cause finding this fixes. The full eligible set
        # is still bounded by the same structural gates above; only the order in
        # which the batch budget below gets spent changes.
        marked = _rank_candidate_pairs_with_marks(tuple(remaining_pairs), take_map) if remaining_pairs else ()
        ranked_pairs = tuple(pair for pair, _deferred in marked)
        truncated = tuple(ranked_pairs[: policy.max_pairs_per_request])
        # D-097.9 (R12): record the order the budget was actually spent in,
        # with the score that produced it and the cap mark, so a run can
        # show WHICH authority ordered the pairs (a foreign re-order is
        # visible as scores out of descending order without a cap mark).
        ranked_pair_budget = [
            {
                "left_clip_id": left_id, "right_clip_id": right_id,
                "priority_score": round(_pair_priority_score(
                    take_map[left_id], take_map[right_id], gap_sec=_group_gap((left_id,), (right_id,), take_map),
                ), 4),
                "group_cap_deferred": deferred,
            }
            for (_, _, left_id, right_id), deferred in marked[: policy.max_pairs_per_request]
        ]
        if truncated:
            request = IdeaEquivalenceRequest(pairs=tuple(
                IdeaEquivalencePair(left_text=take_map[left_id].text, right_text=take_map[right_id].text)
                for _, _, left_id, right_id in truncated
            ))
            result = safe_check_idea_equivalence(arbiter, request, policy)
            decisions = same_idea_by_pair_index(result)
        else:
            result = None
            decisions = {}

    # D-097.A observability: a pair the arbiter answered NOT-same-idea used to
    # vanish from the record entirely (only merges were traced), so a run
    # could not show WHY two takes never became one family.
    arbiter_rejected_pairs: list[dict] = []
    for pair_index, (left_group_index, right_group_index, left_id, right_id) in enumerate(truncated):
        decision = decisions.get(pair_index)
        if decision is None:
            continue  # fail-open: arbiter unavailable/declined -> preserve separate
        same_idea, confidence, reason = decision
        if not same_idea:
            arbiter_rejected_pairs.append({
                "left_clip_id": left_id, "right_clip_id": right_id,
                "confidence": round(float(confidence), 4), "reason": str(reason)[:200],
            })
            continue
        # General override, independent of the arbiter: exactly one side
        # explicitly signals "this is a new/additional point" (see
        # _DISTINCT_ADDITION_MARKERS's own docstring for the real audit
        # finding this guards against -- a coarse topical-similarity
        # judgment can still say "same idea" for two mentions of the same
        # general subject even when one of them is explicitly introducing a
        # DIFFERENT item). A speaker's own discourse marker is EVIDENCE of
        # distinctness -- but D-048 FIX 1 (D-047 Case 1: the arbiter
        # confirmed same_idea at 0.95 confidence for a marked side that
        # shared the exact symptom AND location with the unmarked side --
        # the marker was a narrative restart, not a new point) showed marker
        # presence alone is too coarse: it cannot tell a marker introducing a
        # genuinely new fact apart from one prefacing a near-identical
        # restatement. Only override the arbiter when the marked pairing
        # ALSO shows real content divergence -- see
        # _marked_side_diverges_in_content's own docstring. Arbiter
        # confidence is deliberately not weighed here either way (per
        # D-048's directive: supporting evidence only, never sole authority
        # in either direction) -- content divergence alone decides.
        left_marked = _has_distinct_addition_marker(take_map[left_id].text)
        right_marked = _has_distinct_addition_marker(take_map[right_id].text)
        if left_marked != right_marked and _marked_side_diverges_in_content(
            take_map[left_id].text, take_map[right_id].text,
        ):
            distinct_addition_blocked.append({
                "left_clip_id": left_id,
                "right_clip_id": right_id,
                "confidence": round(confidence, 4),
                "reason": reason,
            })
            continue
        if watch_listen_enabled and watch_listen_spans_by_id:
            wl_relations = _wla.attempt_relation_hypotheses_for_pair(watch_listen_spans_by_id, left_id, right_id)
            final = _wla.resolve_final_attempt_relation(
                would_merge=True, would_merge_source=_wla.RELATION_SOURCE_SEMANTIC_PROVIDER,
                watch_listen_relations=wl_relations,
            )
            final_relations.append(final)
            if not final.would_merge:
                watch_listen_conflict_blocked.append({
                    "left_clip_id": left_id, "right_clip_id": right_id,
                    "watch_listen_relation_evaluated": final.watch_listen_relation_evaluated,
                    "proposition_evidence_status": "merge_proposed",
                    "attempt_relation_final": final.relation,
                    "attempt_relation_source": final.source,
                    "attempt_relation_conflict": final.conflict,
                    "watch_listen_hypothesis_used": final.watch_listen_supported,
                    "semantic_relation_used": _wla.RELATION_SOURCE_SEMANTIC_PROVIDER,
                    "family_membership_action": final.family_membership_action,
                    "family_membership_reason": final.reason,
                })
                continue
        union(left_group_index, right_group_index)
        merged_count += 1
        audit.append({
            "left_clip_id": left_id,
            "right_clip_id": right_id,
            "confidence": round(confidence, 4),
            "reason": reason,
        })

    if merged_count == 0:
        return groups, {
            "status": (
                "checked_no_merge" if (result is not None and result.available)
                else ("arbiter_unavailable" if result is not None else "no_eligible_pairs")
            ),
            "distinct_addition_blocked": distinct_addition_blocked,
            "provider": result.provider if result is not None else None,
            "candidate_pair_count": len(candidate_pairs),
            "checked_pair_count": len(truncated),
            "ranked_pair_budget": ranked_pair_budget,
            "pair_order_authority": _PAIR_ORDER_AUTHORITY,
            "merged_pair_count": 0,
            "restart_evidence_merges": restart_merged,
            "arbiter_rejected_pairs": arbiter_rejected_pairs,
            "arbiter_rejected_pair_count": len(arbiter_rejected_pairs),
            "watch_listen_family_evidence": _watch_listen_family_evidence_summary(
                watch_listen_enabled, watch_listen_conflict_blocked, final_relations,
            ),
        }

    clusters: dict[int, list[str]] = {}
    for index, group in enumerate(groups):
        clusters.setdefault(find(index), []).extend(group)
    merged_groups = tuple(tuple(members) for members in clusters.values())

    return merged_groups, {
        "status": "applied",
        "provider": result.provider if result is not None else "deterministic_restart_evidence",
        "model": result.model if result is not None else None,
        "candidate_pair_count": len(candidate_pairs),
        "checked_pair_count": len(truncated),
        "ranked_pair_budget": ranked_pair_budget,
        "pair_order_authority": _PAIR_ORDER_AUTHORITY,
        "merged_pair_count": merged_count,
        "merges": audit,
        "restart_evidence_merges": restart_merged,
        "distinct_addition_blocked": distinct_addition_blocked,
        "arbiter_rejected_pairs": arbiter_rejected_pairs,
        "arbiter_rejected_pair_count": len(arbiter_rejected_pairs),
        "watch_listen_family_evidence": _watch_listen_family_evidence_summary(
            watch_listen_enabled, watch_listen_conflict_blocked, final_relations,
        ),
    }


# ---------------------------------------------------------------------------
# D-058 Phase 1: DISTINCT-IDEA GROUPING SAFETY
# ---------------------------------------------------------------------------
#
# Root defect (docs/CUTSELL_DECISIONS.md D-057's forensic on the D-056.6
# pimples/acne shape): a deterministic multi-member `take_judge_groups` group
# is treated as ONE mutually-exclusive retry family by construction, with no
# re-validation of that assumption anywhere downstream. `_repair_groups`'s
# own `_constrain_provider_group` complete-link check only ever runs on the
# provider's RAW output, before `_extend_adjacent_retry_groups`/`_absorb_
# interstitial_retry_debris` (which can add members by adjacency/prefix-
# debris alone, no content re-check) and before `reconcile_semantic_idea_
# equivalence` (which only ever MERGES groups, never re-examines an already-
# multi-member group's own internal cohesion). The live shape: a deterministic
# group bundled "back acne treated with resorcina" together with "hormonal
# pimples behind the ear/neck" -- two distinct symptom beats sharing only
# temporal proximity and generic skin-symptom vocabulary -- and the semantic
# arbiter, when it separately evaluated OTHER pairs in this exact run, never
# confirmed these two as the same idea (its own merges list named only the
# genuine back-acne retry pair). Forcing them to compete for one winner
# guaranteed one beat would be silently discarded regardless of which member
# actually won that contest.
#
# FIX: one additional, final cohesion-validation pass over whatever groups
# `reconcile_semantic_idea_equivalence` produced (the single well-defined
# choke point that function's own docstring already establishes) -- run once,
# after all merging is done and before Best Take ranking ever sees a group.
# For every already-multi-member group, every pair of members must show
# EITHER (a) strong deterministic lexical evidence of being the same retry
# (`_provider_members_compatible`'s own existing complete-link threshold --
# reused verbatim, not reinvented) OR (b) explicit arbiter confirmation of
# same intended idea for that SPECIFIC pair (the same `IdeaEquivalenceRequest`/
# `safe_check_idea_equivalence` contract `reconcile_semantic_idea_equivalence`
# already uses, batched and budget-truncated the same way). Neither temporal
# proximity nor raw topic/vocabulary overlap alone is ever sufficient by
# construction -- this function's evidence check is text-only (no timestamps,
# no gap comparison beyond what `_provider_members_compatible` itself already
# requires). A pair with neither kind of evidence is split apart -- one
# connected component of confirmed-cohesive pairs per resulting group,
# members with no confirmed edge to anyone become their own singleton group
# -- so both beats get an independent chance to survive Selection rather than
# one silently losing a contest it was never actually part of. Fails open in
# the content-preserving direction throughout, matching every other gate in
# this module: uncertain evidence never merges, and here it also never
# forces an artificial contest.
#
# `protected_ids` (D-025, same contract as `reconcile_semantic_idea_
# equivalence`): CompositeResolver-accepted composite pieces are exempt --
# this pass must never re-examine or split a decision that authority already
# made.

def _within_group_candidate_pairs(
    group: Tuple[str, ...],
    take_map: dict[str, CandidateTake],
    *,
    protected_ids: frozenset[str],
) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    members = [clip_id for clip_id in group if clip_id in take_map and clip_id not in protected_ids]
    for i in range(len(members)):
        left_id = members[i]
        left_take = take_map[left_id]
        if len(semantic_key(left_take.text).split()) <= 3:
            continue
        for j in range(i + 1, len(members)):
            right_id = members[j]
            right_take = take_map[right_id]
            if len(semantic_key(right_take.text).split()) <= 3:
                continue
            pairs.append((left_id, right_id))
    return tuple(pairs)


def _cohesive_components(
    group: Tuple[str, ...],
    cohesive_pairs: frozenset[frozenset[str]],
    protected_ids: frozenset[str],
) -> tuple[Tuple[str, ...], ...]:
    """Connected components of `group` under the `cohesive_pairs` edge set.

    A protected id (already an accepted composite piece) is never split off
    from the rest of its original group -- it keeps whatever membership
    upstream authority already assigned it, exactly like every other guard
    in this module treats `protected_ids`.

    Superseded within `split_incohesive_retry_groups` by
    `_bridge_aware_components` (D-085) -- kept as a standalone, independently
    testable primitive (plain pairwise union-find, no bridge sensitivity)
    since nothing about it is wrong on its own; it simply isn't sufficient by
    itself any more for the one caller that used to rely on it exclusively.
    """
    parent = {clip_id: clip_id for clip_id in group}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for pair in cohesive_pairs:
        left_id, right_id = tuple(pair)
        if left_id in parent and right_id in parent:
            union(left_id, right_id)
    if protected_ids:
        protected_in_group = [clip_id for clip_id in group if clip_id in protected_ids]
        for anchor, other in zip(protected_in_group, protected_in_group[1:]):
            union(anchor, other)

    clusters: dict[str, list[str]] = {}
    for clip_id in group:
        clusters.setdefault(find(clip_id), []).append(clip_id)
    return tuple(tuple(members) for members in clusters.values())


# ---------------------------------------------------------------------------
# D-085: BRIDGE-AWARE RETRY FAMILY COHESION
# ---------------------------------------------------------------------------
#
# Root defect (docs/CUTSELL_DECISIONS.md D-084's forensic, live evidence
# recovered from 4 independent Modal runs): `_cohesive_components` above is
# plain union-find over an unordered edge set -- ANY accepted pairwise edge
# can transitively connect two full components, and nothing ever re-checks
# that the resulting merged component still represents ONE shared audience-
# facing proposition. D-084 proved the exact live shape: a true back-acne
# subcluster (acne1<->acne2, confidence 0.95) and a true ear/neck-pimples
# subcluster (monolith<->restatement, confidence 0.98) were bridged by TWO
# independent, unmarked, weaker (0.80/0.85) pairwise confirmations
# (acne2<->monolith, acne2<->short) that D-083's marker gate never touches
# (neither side of either pair carries a `_DISTINCT_ADDITION_MARKERS`
# marker) -- and this exact bridge recurred, at the same ~0.85 confidence,
# in all 4 independently-executed runs (D-058/D-083 already-fixed
# regressions plus this one): a reproducible, deterministic grouping defect,
# not one-off LLM noise.
#
# SESSION CLUSTER vs RETRY FAMILY (D-085 formalization): a "session cluster"
# -- the output of `_baseline_groups`/`_repair_groups`/`reconcile_semantic_
# idea_equivalence`, i.e. everything upstream of this module's own final
# cohesion pass -- is only ever a bounded NEIGHBORHOOD of clips worth
# comparing; temporal/topical proximity that put clips in one cluster is
# NEVER by itself evidence they belong in one RETRY FAMILY (a mutually-
# exclusive set of realizations competing to express ONE audience-facing
# proposition). Confusing the two is exactly D-084's root cause.
#
# FIX: order-independent, bridge-sensitive component construction.
# 1. Every candidate edge (deterministic lexical match, OR arbiter same_idea
#    confirmation surviving the existing D-083 marker gate -- unchanged,
#    still the ONLY thing that can produce an edge at all) is processed in a
#    fixed, input-order-independent sequence: deterministic edges first,
#    then semantic edges by descending confidence, tie-broken by clip-id
#    pair -- so the same edge set always yields the same families regardless
#    of `weak_pairs`/dict/group iteration order (D-085 Section 2; see the
#    dedicated permutation-invariance tests).
# 2. An edge is a BRIDGE the moment either endpoint's current component
#    (before this edge) already has >=2 members -- i.e. it would join two
#    already-established components, or attach another member to one that
#    already exists, rather than a first-time singleton<->singleton merge.
#    A non-bridge ("internal/simple") edge is accepted exactly as before
#    (D-058/D-083 behavior byte-for-byte unchanged for the common case: two
#    still-unattached clips merging into a fresh pair).
# 3. A bridge is NEVER accepted on the strength of its own triggering pair's
#    same_idea/confidence alone (D-084's own explicit warning: two
#    independent pairwise mistakes are not component-level proof, and one is
#    even less so). It must additionally clear `_evaluate_bridge_cohesion`:
#    a bounded, component-level question posed to the SAME already-
#    configured `SemanticEquivalenceArbiter` (no new provider/model/
#    authority) built from BOTH components' own member texts rather than
#    just the two touching clips, PLUS a deterministic `any_pair_contradicts`
#    safety net across every cross-component member pair (the same
#    contradiction primitive D-082 already trusts for exactly this "never
#    silently merge a genuine contradiction" role). Absent/malformed/
#    declined/low-confidence responses fail closed -- the bridge is
#    rejected, never merged.
#
# Grouping-only, exactly like D-083: this changes only which candidates end
# up competing for one BestTake winner, never who wins that contest
# (BestTake/Resolver authority, D-081/D-082 fallback ladder, StoryValidator,
# Freeze, Boundary, Render/QC, Human Choice/SWAP -- all untouched).

_BRIDGE_MIN_COHESION_CONFIDENCE = 0.90
_BRIDGE_PROBE_MAX_MEMBERS_PER_SIDE = 3


@dataclass(frozen=True)
class _RetryEdge:
    """One candidate cohesion edge inside a single already-multi-member
    baseline group. `confidence` is 1.0 for deterministic evidence (which
    has no natural confidence score of its own but must still sort ahead of
    every semantic edge, per D-085 Section 2's precedence order)."""
    left_id: str
    right_id: str
    evidence: str  # "deterministic" | "semantic"
    confidence: float
    reason: str


def _edge_sort_key(edge: _RetryEdge) -> tuple:
    """D-085 Section 2: deterministic edges first, then semantic edges by
    descending confidence, then a stable clip-id tie-break -- independent of
    whatever order `edges` was originally built in."""
    evidence_rank = 0 if edge.evidence == "deterministic" else 1
    return (evidence_rank, -edge.confidence, edge.left_id, edge.right_id)


def _component_probe_text(take_map: dict[str, CandidateTake], member_ids: Tuple[str, ...]) -> str:
    """D-085 Section 6: a bounded, deterministic textual stand-in for one
    whole component, used only to pose the component-level cohesion question
    -- built from up to `_BRIDGE_PROBE_MAX_MEMBERS_PER_SIDE` member texts,
    in clip-id order (never union-bookkeeping order, so the probe text is
    identical regardless of which side of a merge happened to become the
    union-find root)."""
    ordered_ids = sorted(member_ids)[:_BRIDGE_PROBE_MAX_MEMBERS_PER_SIDE]
    texts = [take_map[cid].text for cid in ordered_ids if cid in take_map]
    return " || ".join(texts)


def _evaluate_bridge_cohesion(
    *,
    left_members: Tuple[str, ...],
    right_members: Tuple[str, ...],
    edge: _RetryEdge,
    take_map: dict[str, CandidateTake],
    arbiter: SemanticEquivalenceArbiter | None,
    policy: SemanticEquivalenceGatePolicy,
) -> tuple[bool, dict]:
    """D-085 Section 5/6/7: does the FULL merged component -- not just the
    two touching clips -- still represent one shared audience-facing
    proposition? Fails closed (bridge rejected) on any uncertainty: no
    arbiter, arbiter declines/errors, low confidence, or a deterministic
    cross-component contradiction. Never fabricates a semantic verdict --
    `shared_proposition`/`member_support` are populated only from the SAME
    arbiter call's own `reason`/inputs, and `distinct_required_facts` only
    from the existing, already-trusted `any_pair_contradicts` primitive --
    no new semantic authority is introduced.
    """
    record: dict = {
        "left_clip_id": edge.left_id,
        "right_clip_id": edge.right_id,
        "evidence": edge.evidence,
        "triggering_confidence": round(edge.confidence, 4) if edge.evidence == "semantic" else None,
        "triggering_reason": edge.reason,
        "bridge_sensitive": True,
        "left_component_members": list(left_members),
        "right_component_members": list(right_members),
        "component_cohesion_evaluated": False,
        "shared_proposition": None,
        "member_support": [],
        "distinct_required_facts": [],
        "accepted": False,
    }

    # Deferred import: contradiction_signal transitively imports back from
    # this module (via final_sibling_grouping) -- see the D-048 FIX 1 /
    # D-083 comments elsewhere in this file for the same constraint on the
    # same kind of import.
    from .contradiction_signal import any_pair_contradicts, detect_text_contradiction

    left_texts = [take_map[cid].text for cid in left_members if cid in take_map]
    right_texts = [take_map[cid].text for cid in right_members if cid in take_map]
    # D-094.F4: the bridge question is whether the LEFT component and the
    # RIGHT component can be one family -- so only CROSS-component pairs
    # may reject it. A contradiction that already lives INSIDE one component
    # (e.g. a mid-sentence ASR fragment of a member's own negated sentence)
    # was formed by earlier, already-accepted edges; it is that family's
    # own problem (StoryValidator's contradiction invariant / the composite
    # contradiction contract catch it if both members are co-selected) and
    # must not veto a third clip's membership. Run 33969388042: the
    # hereditary family's truncated fragment "canceres son hereditarios..."
    # contradicted its own full sentence, the bridge to the restatement was
    # rejected, and the restatement was co-kept as a separate family.
    if any(
        detect_text_contradiction(left_text, right_text).has_conflict
        for left_text in left_texts for right_text in right_texts
    ):
        record["distinct_required_facts"] = ["cross_component_contradiction"]
        record["reason_rejected"] = "cross_component_contradiction"
        return False, record
    record["within_component_contradiction"] = bool(
        any_pair_contradicts(left_texts) or any_pair_contradicts(right_texts)
    )

    if arbiter is None:
        record["reason_rejected"] = "arbiter_unavailable_fail_closed"
        return False, record

    left_text = _component_probe_text(take_map, left_members)
    right_text = _component_probe_text(take_map, right_members)
    request = IdeaEquivalenceRequest(pairs=(IdeaEquivalencePair(left_text=left_text, right_text=right_text),))
    result = safe_check_idea_equivalence(arbiter, request, policy)
    decision = same_idea_by_pair_index(result).get(0)
    if decision is None:
        record["reason_rejected"] = "component_arbiter_unavailable_or_declined_fail_closed"
        return False, record

    same_retry_family, cohesion_confidence, cohesion_reason = decision
    record["component_cohesion_evaluated"] = True
    record["cohesion_confidence"] = round(cohesion_confidence, 4)
    if not same_retry_family:
        record["reason_rejected"] = "component_cohesion_declined"
        return False, record
    if cohesion_confidence < _BRIDGE_MIN_COHESION_CONFIDENCE:
        record["reason_rejected"] = "component_cohesion_below_bridge_floor"
        return False, record

    record["shared_proposition"] = cohesion_reason
    record["member_support"] = list(left_members) + list(right_members)
    record["accepted"] = True
    return True, record


def _accept_complete_pairwise_bridge(
    *,
    left_members: Tuple[str, ...],
    right_members: Tuple[str, ...],
    edge: _RetryEdge,
    edge_by_pair: dict,
    take_map: dict[str, CandidateTake],
) -> tuple[bool, dict | None]:
    """D-094.2 (policy-gated, default OFF -- see SemanticEquivalenceGatePolicy.
    accept_complete_pairwise_singleton_bridge): a SINGLETON-attaches-to-
    component bridge whose EVERY cross pair already carries its own
    accepted-candidate edge (deterministic, or a same_idea confirmation at
    >= `_BRIDGE_MIN_COHESION_CONFIDENCE` that already survived the D-083
    marker gate) is not the transitive-contamination shape D-084/D-085
    guard against -- the newcomer was confirmed against EACH existing
    member, which for a single clip IS the component-level question. It
    is accepted on that complete pairwise evidence, with the same
    deterministic cross-component contradiction safety net D-085's probe
    applies, and WITHOUT the component-level probe. Component-to-component
    merges (>= 2 members on both sides) are never routed here: D-085's own
    QA contract (always-yes pairwise arbiter must not defeat the component
    check) stays in force for them. Run 33983880111:
    the abandoned gastritis retry was confirmed 0.95 against the complete
    delivery and 0.90 against the aside, the aside 0.95 against the
    delivery, yet the concatenated "A || B" probe answered 0.2 and the
    D-020 pair was split -- the probe (a question about a synthetic
    joined text) contradicted the arbiter's own three pairwise answers.
    Returns (accepted, record) or (False, None) when the pairwise evidence
    is NOT complete -- the caller then falls through to D-085's probe,
    byte-for-byte unchanged. Bounded by construction: n*m confirmed pairs.
    """
    cross_pairs = [(l, r) for l in left_members for r in right_members]
    confirmations: list[dict] = []
    for left_id, right_id in cross_pairs:
        pair_edge = edge_by_pair.get(frozenset((left_id, right_id)))
        if pair_edge is None:
            return False, None
        if pair_edge.evidence != "deterministic" and pair_edge.confidence < _BRIDGE_MIN_COHESION_CONFIDENCE:
            return False, None
        confirmations.append({
            "left_clip_id": left_id, "right_clip_id": right_id, "evidence": pair_edge.evidence,
            "confidence": round(pair_edge.confidence, 4) if pair_edge.evidence == "semantic" else None,
            "reason": pair_edge.reason,
        })

    record: dict = {
        "left_clip_id": edge.left_id, "right_clip_id": edge.right_id,
        "evidence": edge.evidence,
        "triggering_confidence": round(edge.confidence, 4) if edge.evidence == "semantic" else None,
        "triggering_reason": edge.reason,
        "bridge_sensitive": True,
        "left_component_members": list(left_members),
        "right_component_members": list(right_members),
        "component_cohesion_evaluated": False,
        "accepted_by": "complete_pairwise_confirmation",
        "cross_pair_confirmations": confirmations,
        "shared_proposition": None,
        "member_support": list(left_members) + list(right_members),
        "distinct_required_facts": [],
        "accepted": False,
    }
    from .contradiction_signal import detect_text_contradiction  # deferred: see _evaluate_bridge_cohesion

    left_texts = [take_map[cid].text for cid in left_members if cid in take_map]
    right_texts = [take_map[cid].text for cid in right_members if cid in take_map]
    if any(
        detect_text_contradiction(left_text, right_text).has_conflict
        for left_text in left_texts for right_text in right_texts
    ):
        record["distinct_required_facts"] = ["cross_component_contradiction"]
        record["reason_rejected"] = "cross_component_contradiction"
        return False, record
    record["accepted"] = True
    return True, record


_RESTART_EVIDENCE_KINDS = frozenset({
    "same_opening_restart", "same_opening_abandoned_start", "incomplete_attempt_completed_by_retry",
    # D-100 (D-099 Gap #1): confirmed-multimodal-corroborated retry. Same
    # deterministic-evidence treatment as the lexical kinds above -- never
    # re-examined by the arbiter, subject to the same D-083 marker gate.
    "multimodal_corroborated_retry",
})


def _restart_cohesive(members: Tuple[str, ...], restart_pairs: set[frozenset]) -> bool:
    """True when EVERY pair inside `members` carries deterministic restart
    evidence -- the component is one sentence and its restarts, not a set of
    semantically-linked deliveries (D-097.A)."""
    if len(members) < 2:
        return False
    return all(
        frozenset((left_id, right_id)) in restart_pairs
        for index, left_id in enumerate(members)
        for right_id in members[index + 1:]
    )


def _accept_restart_singleton_bridge(
    *,
    left_members: Tuple[str, ...],
    right_members: Tuple[str, ...],
    edge: _RetryEdge,
    take_map: dict[str, CandidateTake],
    accepted_by: str = "deterministic_restart_evidence",
) -> tuple[bool, dict]:
    """D-097.A: a SINGLETON attaching to a component is accepted WITHOUT the
    D-085 component probe when (a) the attaching edge itself is deterministic
    RESTART evidence (`take_grouping.same_opening_restart`: the newcomer
    restarts a member's own sentence from its first words within seconds,
    sharing content beyond the opening), or (b) the attaching edge is an
    arbiter confirmation at >= `_BRIDGE_MIN_COHESION_CONFIDENCE` and the
    receiving component is RESTART-COHESIVE (every pair inside it carries
    restart evidence -- it is one sentence and its restarts, so confirming
    against one member is confirming against the component). Both are
    recording-process evidence, not the transitive semantic contamination
    D-084/D-085 guard against -- and the component probe is a question to
    the very semantic judge whose "different idea" verdict on a self-
    corrected detail isolated the clean retry in run 34008386434 (D-096
    C-1). D-085's deterministic cross-component contradiction safety net
    still applies. Component-to-component merges (>= 2 members on both
    sides), every non-restart deterministic edge (`prefix_fragment`,
    `provider_members_compatible`) and every semantic attach to a
    semantically-formed component keep D-085's probe byte-for-byte; the
    D-094.2 policy flag for SEMANTIC complete-pairwise bridges is untouched
    and still OFF."""
    record: dict = {
        "left_clip_id": edge.left_id, "right_clip_id": edge.right_id,
        "evidence": edge.evidence,
        "triggering_confidence": round(edge.confidence, 4) if edge.evidence == "semantic" else None,
        "triggering_reason": edge.reason, "bridge_sensitive": True,
        "left_component_members": list(left_members),
        "right_component_members": list(right_members),
        "component_cohesion_evaluated": False,
        "accepted_by": accepted_by,
        "shared_proposition": None,
        "member_support": list(left_members) + list(right_members),
        "distinct_required_facts": [],
        "accepted": False,
    }
    from .contradiction_signal import detect_text_contradiction  # deferred: see _evaluate_bridge_cohesion

    left_texts = [take_map[cid].text for cid in left_members if cid in take_map]
    right_texts = [take_map[cid].text for cid in right_members if cid in take_map]
    if any(
        detect_text_contradiction(left_text, right_text).has_conflict
        for left_text in left_texts for right_text in right_texts
    ):
        record["distinct_required_facts"] = ["cross_component_contradiction"]
        record["reason_rejected"] = "cross_component_contradiction"
        return False, record
    record["accepted"] = True
    return True, record


def _cross_component_blocked_pair(
    left_members: Tuple[str, ...],
    right_members: Tuple[str, ...],
    blocked_pairs: frozenset[frozenset[str]],
) -> frozenset[str] | None:
    """D-108: does ANY member of `left_members` already carry an explicit,
    already-computed non-equivalence relationship (`content_divergence_
    blocked` -- same-run evidence this function's own caller already
    produced, never a new heuristic) with ANY member of `right_members`?
    Returns the offending pair, or None. See the module comment above
    `_bridge_aware_components` for why this check runs BEFORE any
    acceptance path, not as one more path to satisfy, and for why a bare
    `arbiter_rejected_pairs` decline is deliberately NOT included here."""
    for left_id in left_members:
        for right_id in right_members:
            pair = frozenset((left_id, right_id))
            if pair in blocked_pairs:
                return pair
    return None


def _bridge_aware_components(
    group: Tuple[str, ...],
    edges: list[_RetryEdge],
    *,
    protected_ids: frozenset[str],
    take_map: dict[str, CandidateTake],
    arbiter: SemanticEquivalenceArbiter | None,
    policy: SemanticEquivalenceGatePolicy,
    edge_trace: list[dict],
    blocked_pairs: frozenset[frozenset[str]] = frozenset(),
) -> tuple[Tuple[str, ...], ...]:
    """D-085: bridge-sensitive replacement for plain union-find. Processes
    `edges` in the fixed, input-order-independent sequence `_edge_sort_key`
    defines; a non-bridge edge unions immediately (byte-identical to
    `_cohesive_components`'s own behavior for that case); a bridge edge only
    unions after `_evaluate_bridge_cohesion` accepts it. `protected_ids`
    handling is unchanged from `_cohesive_components`.

    D-108 (pimples family granularity, docs/CUTSELL_DECISIONS.md D-108):
    D-085's own bridge-cohesion probes (`_evaluate_bridge_cohesion`,
    `_accept_restart_singleton_bridge`, `_accept_complete_pairwise_bridge`)
    all ask "does the MERGED component still share one proposition?" -- none
    of them ever asks "did this run ALREADY determine two of the members
    about to be merged are NOT the same idea?". Live shape (D-107/D-108): a
    complementary beat A shares a deterministic same-opening/abandoned-start
    edge with a retry-competitor C (a thin, order-independent lexical
    coincidence -- shared opening words plus one shared content word), which
    unions A and C as an ordinary non-bridge pair BEFORE a third member B's
    own strong semantic edge to C ever runs. That later B-C edge becomes a
    bridge into the now-2-member {A, C} component, and because {A, C}
    carries a restart-evidence-kind edge, `_accept_restart_singleton_bridge`
    accepts it under `confirmed_against_restart_component` WITHOUT ever
    revisiting whether B belongs anywhere near A -- even though this SAME
    run's own within-group weak-pair check already found A and B carry an
    explicit, marker-gated content-divergence block (`content_divergence_
    blocked`) for exactly that pair. FIX: before dispatching a bridge edge
    to ANY acceptance path, check every cross pair between the two
    components against this run's own already-computed non-equivalence
    evidence (`blocked_pairs`, built by the caller from `content_divergence_
    blocked` ONLY -- reused verbatim, not a new heuristic; a bare same_
    idea=False decline is deliberately excluded, see the caller's own
    comment for why); a hit vetoes the bridge outright, whichever acceptance
    path would otherwise have granted it. An UNKNOWN relationship (a pair
    simply never evaluated, e.g. A vs C when a fixture makes that pair the
    untested one) is not in `blocked_pairs` and never triggers this veto --
    only evidence this run already produced does."""
    parent = {clip_id: clip_id for clip_id in group}
    members_of: dict[str, list[str]] = {clip_id: [clip_id] for clip_id in group}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        merged = members_of.pop(ra) + members_of.pop(rb)
        parent[ra] = rb
        members_of[rb] = merged

    # D-094.2: every accepted-candidate edge, keyed order-insensitively, so
    # a bridge can be recognised as a COMPLETE PAIRWISE CONFIRMATION (below).
    edge_by_pair: dict[frozenset, _RetryEdge] = {}
    for candidate_edge in edges:
        key = frozenset((candidate_edge.left_id, candidate_edge.right_id))
        current = edge_by_pair.get(key)
        if current is None or _edge_sort_key(candidate_edge) < _edge_sort_key(current):
            edge_by_pair[key] = candidate_edge
    # D-097.A: pairs joined by deterministic restart evidence (one sentence
    # restarted from its own opening) -- see `_accept_restart_singleton_bridge`.
    restart_pairs = {
        frozenset((candidate_edge.left_id, candidate_edge.right_id))
        for candidate_edge in edges
        if candidate_edge.evidence == "deterministic" and candidate_edge.reason in _RESTART_EVIDENCE_KINDS
    }

    for edge in sorted(edges, key=_edge_sort_key):
        if edge.left_id not in parent or edge.right_id not in parent:
            continue
        root_left, root_right = find(edge.left_id), find(edge.right_id)
        if root_left == root_right:
            continue
        left_members, right_members = members_of[root_left], members_of[root_right]
        is_bridge = len(left_members) >= 2 or len(right_members) >= 2
        if not is_bridge:
            union(edge.left_id, edge.right_id)
            edge_trace.append({
                "left_clip_id": edge.left_id, "right_clip_id": edge.right_id,
                "evidence": edge.evidence,
                "confidence": round(edge.confidence, 4) if edge.evidence == "semantic" else None,
                "reason": edge.reason, "bridge_sensitive": False, "accepted": True,
            })
            continue
        blocked_pair = _cross_component_blocked_pair(
            tuple(left_members), tuple(right_members), blocked_pairs,
        )
        if blocked_pair is not None:
            left_id, right_id = tuple(blocked_pair) if len(blocked_pair) == 2 else (edge.left_id, edge.right_id)
            edge_trace.append({
                "left_clip_id": edge.left_id, "right_clip_id": edge.right_id,
                "evidence": edge.evidence,
                "confidence": round(edge.confidence, 4) if edge.evidence == "semantic" else None,
                "reason": edge.reason, "bridge_sensitive": True,
                "left_component_members": list(left_members), "right_component_members": list(right_members),
                "component_cohesion_evaluated": False,
                "accepted": False,
                "reason_rejected": "cross_component_explicit_non_equivalence",
                "conflicting_pair": [left_id, right_id],
            })
            continue
        record = None
        if min(len(left_members), len(right_members)) == 1:
            multi_members = left_members if len(left_members) >= 2 else right_members
            restart_edge = edge.evidence == "deterministic" and edge.reason in _RESTART_EVIDENCE_KINDS
            confirmed_against_restart_component = (
                edge.evidence == "semantic"
                and edge.confidence >= _BRIDGE_MIN_COHESION_CONFIDENCE
                and _restart_cohesive(tuple(multi_members), restart_pairs)
            )
            if restart_edge or confirmed_against_restart_component:
                accepted, record = _accept_restart_singleton_bridge(
                    left_members=tuple(left_members), right_members=tuple(right_members),
                    edge=edge, take_map=take_map,
                    accepted_by=(
                        "deterministic_restart_evidence" if restart_edge
                        else "semantic_confirmation_against_restart_cohesive_component"
                    ),
                )
        if record is None and policy.accept_complete_pairwise_singleton_bridge and min(len(left_members), len(right_members)) == 1:
            accepted, record = _accept_complete_pairwise_bridge(
                left_members=tuple(left_members), right_members=tuple(right_members),
                edge=edge, edge_by_pair=edge_by_pair, take_map=take_map,
            )
        if record is None:
            accepted, record = _evaluate_bridge_cohesion(
                left_members=tuple(left_members), right_members=tuple(right_members),
                edge=edge, take_map=take_map, arbiter=arbiter, policy=policy,
            )
        edge_trace.append(record)
        if accepted:
            union(edge.left_id, edge.right_id)

    if protected_ids:
        protected_in_group = [clip_id for clip_id in group if clip_id in protected_ids]
        for anchor, other in zip(protected_in_group, protected_in_group[1:]):
            if anchor in parent and other in parent:
                union(anchor, other)

    clusters: dict[str, list[str]] = {}
    for clip_id in group:
        clusters.setdefault(find(clip_id), []).append(clip_id)
    return tuple(tuple(members) for members in clusters.values())


def split_incohesive_retry_groups(
    groups: Tuple[Tuple[str, ...], ...],
    takes: Tuple[CandidateTake, ...],
    arbiter: SemanticEquivalenceArbiter | None,
    *,
    policy: SemanticEquivalenceGatePolicy = SemanticEquivalenceGatePolicy(),
    protected_ids: frozenset[str] = frozenset(),
    prior_confirmations: Mapping[frozenset, tuple[float, str]] | None = None,
) -> tuple[Tuple[Tuple[str, ...], ...], dict]:
    """D-058 Phase 1 + D-085: require evidence of shared communicative intent
    before an already-multi-member group is trusted as one mutually-
    exclusive retry family, AND (D-085) require that a BRIDGE edge -- one
    that would connect two already-established components rather than form
    a fresh pair -- additionally prove the resulting merged component still
    represents one shared audience-facing proposition. See the module
    comments immediately above `_cohesive_components` (D-058) and above
    `_BRIDGE_MIN_COHESION_CONFIDENCE` (D-085) for the full defect and fix
    rationale. Runs once, after `reconcile_semantic_idea_equivalence`,
    before Best Take ranking.
    """
    take_map = {take.clip_id: take for take in takes}
    multi_member_groups = [group for group in groups if len(group) >= 2]
    if not multi_member_groups:
        return groups, {
            "status": "not_requested", "groups_checked": 0, "groups_split": 0,
            "weak_pair_count": 0, "checked_pair_count": 0,
            "arbiter_confirmed_pairs": [],
            "content_divergence_blocked": [], "content_divergence_blocked_count": 0,
            "prior_confirmations_reused": [], "prior_confirmations_reused_count": 0,
            "unchecked_weak_pairs": [], "unchecked_weak_pair_count": 0,
            "arbiter_rejected_pairs": [], "arbiter_rejected_pair_count": 0,
            "splits": [],
            "edge_trace": [], "bridge_evaluated_count": 0, "bridge_accepted_count": 0,
            "bridge_rejected_count": 0, "component_semantic_call_count": 0,
            "blocked_pair_veto_count": 0,
        }

    edges_by_group: dict[int, list[_RetryEdge]] = {id(group): [] for group in multi_member_groups}
    weak_pairs: list[tuple[str, str]] = []
    weak_pair_group: dict[tuple[str, str], int] = {}
    for group in multi_member_groups:
        for left_id, right_id in _within_group_candidate_pairs(group, take_map, protected_ids=protected_ids):
            left_take, right_take = take_map[left_id], take_map[right_id]
            if _provider_members_compatible(left_take, right_take):
                edges_by_group[id(group)].append(
                    _RetryEdge(left_id, right_id, "deterministic", 1.0, "provider_members_compatible")
                )
            elif _is_prefix_fragment(left_take, right_take) or _is_prefix_fragment(right_take, left_take):
                edges_by_group[id(group)].append(
                    _RetryEdge(left_id, right_id, "deterministic", 1.0, "prefix_fragment")
                )
            elif (
                (restart_kind := same_opening_restart(left_take, right_take)) is not None
                and not _within_group_arbiter_confirmation_diverges(take_map, left_id, right_id)
            ):
                # D-097.A: a sentence restarted from its own first words within
                # seconds is recording-process evidence of one retry family
                # (see take_grouping.same_opening_restart); D-083's marker
                # gate still applies exactly as it does to a semantic edge.
                edges_by_group[id(group)].append(
                    _RetryEdge(left_id, right_id, "deterministic", 1.0, restart_kind)
                )
            else:
                weak_pairs.append((left_id, right_id))
                weak_pair_group[(left_id, right_id)] = id(group)

    checked_pair_count = 0
    confirmed_pairs: list[dict] = []
    content_divergence_blocked: list[dict] = []
    prior_reused: list[dict] = []
    # D-094.F3: a weak pair the SAME run's reconcile stage already asked the
    # arbiter about (`reconcile_semantic_idea_equivalence`'s confirmed
    # merges) is existing evidence the engine already paid for -- reuse it
    # instead of re-asking, and never let it fall off the end of this pass's
    # own bounded re-ask (`policy.max_pairs_per_request`). Run 33969388042:
    # the gastritis retry pair was confirmed at 0.95 upstream, was not among
    # the 14 of 21 weak pairs re-asked here, and the group was split on
    # absence of evidence -- the abandoned retry and the complete delivery
    # were then both kept (D-020 violated silently). The D-083 divergence
    # gate applies to a reused confirmation exactly as to a fresh one.
    prior = dict(prior_confirmations or {})
    weak_pair_total = len(weak_pairs)  # reported as before: ALL weak pairs, reused or not
    remaining_weak: list[tuple[str, str]] = []
    for left_id, right_id in weak_pairs:
        hit = prior.get(frozenset((left_id, right_id)))
        if hit is None:
            remaining_weak.append((left_id, right_id))
            continue
        confidence, reason = float(hit[0]), str(hit[1])
        if _within_group_arbiter_confirmation_diverges(take_map, left_id, right_id):
            content_divergence_blocked.append({
                "left_clip_id": left_id, "right_clip_id": right_id,
                "confidence": round(confidence, 4), "reason": reason, "source": "prior_confirmation",
            })
            continue
        edges_by_group[weak_pair_group[(left_id, right_id)]].append(
            _RetryEdge(left_id, right_id, "semantic", confidence, reason)
        )
        row = {"left_clip_id": left_id, "right_clip_id": right_id,
               "confidence": round(confidence, 4), "reason": reason, "source": "prior_confirmation"}
        confirmed_pairs.append(row)
        prior_reused.append(row)
    weak_pairs = remaining_weak
    unchecked_weak_pairs: list[dict] = []
    arbiter_rejected_pairs: list[dict] = []  # D-097.A observability
    if weak_pairs and arbiter is not None:
        ranked = sorted(
            weak_pairs,
            key=lambda pair: _pair_priority_score(
                take_map[pair[0]], take_map[pair[1]],
                gap_sec=_group_gap((pair[0],), (pair[1],), take_map),
            ),
            reverse=True,
        )
        truncated = tuple(ranked[: policy.max_pairs_per_request])
        checked_pair_count = len(truncated)
        # D-094.F3 observability: pairs this bounded pass could NOT ask are
        # recorded, never silently treated as "no evidence".
        unchecked_weak_pairs = [
            {"left_clip_id": left_id, "right_clip_id": right_id}
            for left_id, right_id in ranked[policy.max_pairs_per_request:]
        ]
        request = IdeaEquivalenceRequest(pairs=tuple(
            IdeaEquivalencePair(left_text=take_map[left_id].text, right_text=take_map[right_id].text)
            for left_id, right_id in truncated
        ))
        result = safe_check_idea_equivalence(arbiter, request, policy)
        decisions = same_idea_by_pair_index(result)
        for pair_index, (left_id, right_id) in enumerate(truncated):
            decision = decisions.get(pair_index)
            if decision is None:
                continue  # fail-open: arbiter unavailable/declined -> keep separate
            same_idea, confidence, reason = decision
            if not same_idea:
                arbiter_rejected_pairs.append({
                    "left_clip_id": left_id, "right_clip_id": right_id,
                    "confidence": round(float(confidence), 4), "reason": str(reason)[:200],
                })
                continue
            if _within_group_arbiter_confirmation_diverges(take_map, left_id, right_id):
                # D-083: exactly one side carries a distinct-addition marker
                # and shows real content divergence from the other -- do not
                # trust this confirmation. See the module comment above
                # `_within_group_arbiter_confirmation_diverges` for the full
                # rationale and why this is deliberately marker-gated rather
                # than a blanket overlap floor. Retained unchanged by D-085 --
                # D-085 complements this gate, it never replaces or weakens it.
                content_divergence_blocked.append({
                    "left_clip_id": left_id, "right_clip_id": right_id,
                    "confidence": round(confidence, 4), "reason": reason,
                })
                continue
            edges_by_group[weak_pair_group[(left_id, right_id)]].append(
                _RetryEdge(left_id, right_id, "semantic", confidence, reason)
            )
            confirmed_pairs.append({
                "left_clip_id": left_id, "right_clip_id": right_id,
                "confidence": round(confidence, 4), "reason": reason,
            })

    # D-108: explicit, already-computed non-equivalence evidence this SAME
    # pass produced. Reused verbatim (never a new heuristic) as a hard veto
    # on any BRIDGE that would otherwise pull the blocked pair into one
    # retry-family component; see the module comment above
    # `_bridge_aware_components`. An unevaluated pair is absent from
    # `content_divergence_blocked` and therefore never vetoes anything on
    # its own (negative control: unknown does not mean blocked).
    #
    # Deliberately built from `content_divergence_blocked` ONLY, never
    # `arbiter_rejected_pairs`. `content_divergence_blocked` is
    # unconditionally strong: it already required the arbiter to CONFIRM
    # same_idea, an explicit distinct-addition marker on exactly one side,
    # AND a verified low shared-content floor overriding that confirmation
    # (D-048 FIX 1's own "content divergence alone decides, confidence is
    # never sole authority" rule) -- a structural signal, not a confidence
    # number. A bare `arbiter_rejected_pairs` same_idea=False verdict, by
    # contrast, is the ROUTINE, expected outcome for every topically-
    # unrelated pair inside a larger multi-member group (e.g. an unrelated
    # acne mention vs. a genuine pimples retry pair) -- proven unsafe to use
    # here by `test_regression_full_five_member_conflated_group_resolves_
    # to_three_families` (tests/test_cutsell_d083_distinct_idea_grouping_
    # safety.py): a fixed-confidence test arbiter's routine decline of an
    # unrelated pair would otherwise veto a wholly legitimate 3-member
    # retry family. A same_idea=False decline says only "not confirmed the
    # same idea," never "confirmed genuinely distinct" -- content_
    # divergence_blocked is the only signal this run produces that means
    # the latter.
    blocked_pairs: frozenset[frozenset[str]] = frozenset(
        frozenset((row["left_clip_id"], row["right_clip_id"]))
        for row in content_divergence_blocked
    )

    output_groups: list[Tuple[str, ...]] = []
    split_records: list[dict] = []
    groups_split = 0
    edge_trace: list[dict] = []
    for group in groups:
        if len(group) < 2:
            output_groups.append(group)
            continue
        components = _bridge_aware_components(
            group, edges_by_group.get(id(group), []),
            protected_ids=protected_ids, take_map=take_map,
            arbiter=arbiter, policy=policy, edge_trace=edge_trace,
            blocked_pairs=blocked_pairs,
        )
        if len(components) <= 1:
            output_groups.append(group)
            continue
        groups_split += 1
        split_records.append({
            "original_group_ids": list(group),
            "resulting_groups": [list(component) for component in components],
        })
        output_groups.extend(components)

    bridge_records = [record for record in edge_trace if record.get("bridge_sensitive")]
    bridge_accepted_count = sum(1 for record in bridge_records if record.get("accepted"))
    bridge_rejected_count = len(bridge_records) - bridge_accepted_count
    component_semantic_call_count = sum(
        1 for record in bridge_records if record.get("component_cohesion_evaluated")
    )
    # D-108: bridges vetoed purely because a cross pair already carried this
    # SAME pass's own explicit non-equivalence evidence -- never reached any
    # acceptance path at all. Counted separately from `bridge_rejected_count`
    # (which also includes ordinary `_evaluate_bridge_cohesion` rejections)
    # so a run can show WHICH guard actually stopped the merge.
    blocked_pair_veto_count = sum(
        1 for record in bridge_records
        if record.get("reason_rejected") == "cross_component_explicit_non_equivalence"
    )

    return tuple(output_groups), {
        "status": "applied" if (groups_split or checked_pair_count) else "no_incohesive_groups_found",
        "groups_checked": len(multi_member_groups),
        "groups_split": groups_split,
        "weak_pair_count": weak_pair_total,
        "checked_pair_count": checked_pair_count,
        "arbiter_confirmed_pairs": confirmed_pairs,
        "content_divergence_blocked": content_divergence_blocked,
        "content_divergence_blocked_count": len(content_divergence_blocked),
        # D-094.F3
        "prior_confirmations_reused": prior_reused,
        "prior_confirmations_reused_count": len(prior_reused),
        "unchecked_weak_pairs": unchecked_weak_pairs,
        "unchecked_weak_pair_count": len(unchecked_weak_pairs),
        "arbiter_rejected_pairs": arbiter_rejected_pairs,
        "arbiter_rejected_pair_count": len(arbiter_rejected_pairs),
        "splits": split_records,
        # D-085 bridge-aware cohesion diagnostics -- see the module comment
        # above `_BRIDGE_MIN_COHESION_CONFIDENCE` for the full contract. Each
        # `edge_trace` entry carries evidence type/confidence/reason,
        # `bridge_sensitive`, and (for bridges) `left_component_members`/
        # `right_component_members`/`component_cohesion_evaluated`/
        # `shared_proposition`/`member_support`/`distinct_required_facts`/
        # `accepted`/`reason_rejected` -- deliberately surfaced under this
        # top-level diagnostics key (unlike D-083's own `distinct_idea_
        # grouping_safety`, this is intended to be printed directly by the
        # Modal RAW workflow's diagnostic-dump script; see D-085 Section 14).
        "edge_trace": edge_trace,
        "bridge_evaluated_count": len(bridge_records),
        "bridge_accepted_count": bridge_accepted_count,
        "bridge_rejected_count": bridge_rejected_count,
        "component_semantic_call_count": component_semantic_call_count,
        "blocked_pair_veto_count": blocked_pair_veto_count,
    }


def safe_group_takes(
    provider: TakeGroupingProvider | None,
    takes: Tuple[CandidateTake, ...],
    context_text: str = "",
) -> TakeGroupingProviderResult:
    """Use semantic grouping while preserving every real candidate exactly once.

    Phase 2's semantic idea-equivalence pass (reconcile_semantic_idea_equivalence,
    below) deliberately runs OUTSIDE this function rather than being threaded
    through it: this codebase already layers several production monkeypatch
    wrappers over safe_group_takes and safe_group_takes_by_sessions (see
    final_sibling_grouping.py, session_grouping_bridge.py,
    global_session_sibling_bridge.py, local_retry_grouping.py,
    retry_group_integrity.py, hybrid_composite_best_take.py and friends), each
    hardcoding this function's current signature. Adding a new keyword here
    would silently stop propagating through every one of those wrappers in the
    real production call path -- exactly the class of regression several of
    those files' own docstrings describe fixing. pipeline.py instead calls
    reconcile_semantic_idea_equivalence directly on the final resolved groups,
    a single well-defined choke point immune to that layering.
    """
    baseline = _baseline_groups(takes)
    if provider is None or len(takes) <= 1:
        return TakeGroupingProviderResult(
            baseline,
            ProviderStatus("baseline", False, True, "lexical_fallback"),
            "baseline",
        )
    try:
        result = provider.group(takes, context_text=context_text)
        if not result.groups:
            raise ValueError("take grouping returned no groups")
        normalized_input = tuple(tuple(str(item) for item in group) for group in result.groups if group)
        repaired_groups, repaired = _repair_groups(normalized_input, takes)
        if not repaired_groups:
            raise ValueError("take grouping produced no valid candidates")
        reconciled_groups, reconciled = _reconcile_missed_retries(repaired_groups, takes)
        extended_groups, extended = _extend_adjacent_retry_groups(reconciled_groups, takes)
        final_groups, debris_absorbed = _absorb_interstitial_retry_debris(extended_groups, takes)
        reason = result.reason
        if repaired:
            reason = (reason + "; " if reason else "") + "provider_output_repaired"
        if reconciled:
            reason = (reason + "; " if reason else "") + "local_retry_reconciled"
        if extended:
            reason = (reason + "; " if reason else "") + "adjacent_retry_extended"
        if debris_absorbed:
            reason = (reason + "; " if reason else "") + "interstitial_retry_debris_absorbed"
        return TakeGroupingProviderResult(
            final_groups,
            ProviderStatus("openai", True, True, "applied"),
            reason,
        )
    except Exception as exc:
        return TakeGroupingProviderResult(
            baseline,
            ProviderStatus(
                provider=provider.__class__.__name__,
                requested=True,
                available=False,
                status="provider_error_fallback",
                reason=f"{exc.__class__.__name__}:{str(exc)[:160]}",
            ),
            "baseline_fallback",
        )
