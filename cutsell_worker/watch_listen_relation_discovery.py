"""D-161 Phase C.2 -- Watch+Listen RELATION DISCOVERY Gate.

Per docs/CUTSELL_DECISIONS.md D-148/D-154 through D-160/D-161. D-160's own
forensic proved (by code reading, not inference) that D-158's authority is
reactive-only: it only ever evaluates a pair the PRE-EXISTING semantic
candidate-pair generator/batch-cap already decided to compare (`_cross_
group_candidate_pairs` -> `_rank_candidate_pairs_with_marks` -> the 14-pair
arbiter batch), and it is never even called on a `would_merge=False` case.
A real, `SUPPORTED`-confidence Watch+Listen retry relation for a pair the
semantic path never reached (D-160's own pimples/espinillas trace) has
today's D-158 with zero opportunity to act on it either way.

This module is DISCOVERY ONLY:

    WATCH+LISTEN MAY DISCOVER.
    WATCH+LISTEN DOES NOT DECIDE.

It proposes candidate pairs the existing semantic path never generated,
using ONLY already-computed Watch+Listen Understanding evidence (D-157,
unchanged) plus the SAME deterministic proposition-evidence primitives
`take_grouping.py`'s own restart-evidence loop already uses (never a new
heuristic, never a provider call). A discovered candidate is NEVER merged
directly by this module -- every candidate is handed to `attempt_
relationship_authority.resolve_final_attempt_relation` (D-158's own
structured authority, extended with one new keyword-only parameter,
`semantic_path_evaluated=False`, fully backward compatible -- see that
module's own docstring) for the FINAL relation and merge decision. Only a
final `RETRY` relation with independently-sufficient proposition evidence
ever produces `would_merge=True`; every other relation (CORRECTION/
CONTINUATION/COMPLEMENTARY/NEW_AUDIENCE_BEAT/DISTINCT_PROPOSITION/
UNCERTAIN) is surfaced as a structured observation and NEVER merged.

## Pair-source contract (discovery vs. the existing semantic path)

Two independent candidate SOURCES now exist:

    SEMANTIC_PAIR_SOURCE   -- `_cross_group_candidate_pairs` (unchanged)
    WATCH_LISTEN_DISCOVERY -- this module (new)

A pair discovered by this module that was ALSO already resolved (merged,
restart-merged, arbiter-rejected, or D-158-conflict-blocked) by the
existing semantic path THIS run is evaluated exactly ONCE -- by whichever
mechanism reached it first (the existing semantic path always runs
first in `reconcile_semantic_idea_equivalence`; this module's own
candidates are filtered against that outcome set before evaluation,
never re-decided). `PAIR_SOURCE_BOTH` records that the pair was ALSO a
semantic-path candidate, purely for observability -- it does not change
which authority decided it.

## Proposition Firewall (D-111, reasserted)

"PROPOSITION IDENTITY PRECEDES RETRY IDENTITY." A discovered `RETRY`
relation is never sufficient alone to merge a family: `proposition_
evidence_for_pair` independently re-runs the SAME four deterministic
restart-evidence rules `reconcile_semantic_idea_equivalence`'s own
deterministic-restart loop already applies (`same_opening_restart`,
`_safe_short_prefix_retry`, `incomplete_attempt_completed_by_retry`,
`multimodal_corroborated_retry` -- all unchanged, imported directly, never
reimplemented). Only when BOTH signals independently agree (a real,
`SUPPORTED` Watch+Listen `RETRY` hypothesis AND real deterministic
restart/completion evidence) does a discovered pair ever become eligible
for `ELIGIBLE_RETRY_FAMILY`. When proposition evidence is unresolved, the
pair reports `UNCERTAIN`/`ABSTAIN_UNCERTAIN` -- exactly D-158's own
existing "never force a merge" contract, extended to the discovery path.

## Immediate-neighbor bridging (D-160's own named limitation)

D-157's own `attempt_relation_hypotheses` relate a span ONLY to its
immediate predecessor in source-timeline order. This module adds ONE
bounded, single-hop bridge: when the immediate predecessor of a span is
classified EXCLUSIVELY as a non-audience intermediary (`PRE_TAKE_SETUP`/
`POST_TAKE_RESET`/`RECORDING_PROCESS`/`FALSE_START`, never `AUDIENCE_
DELIVERY`/`CLEAN_ATTEMPT`), this module also considers that span's
predecessor's predecessor as a candidate left side -- reusing `attempt_
reconstruction._restart_evidence` (the SAME already-vetted primitive
D-157 itself calls) directly on the two `CandidateTake` texts, never a
new detector, never perception recomputation, and never more than ONE
intermediary hop -- explicitly NOT an arbitrary N-hop or all-pairs search.

## Fail-open / rollback

`watch_listen_relation_discovery_enabled()` (env `CUTSELL_WATCH_LISTEN_
RELATION_DISCOVERY_ENABLED`, default **OFF**) is a SEPARATE flag from
D-158's own `CUTSELL_WATCH_LISTEN_FAMILY_EVIDENCE_ENABLED` -- merge-veto
and discovery are different authorities (this task's own instruction).
OFF: `discover_candidate_pairs` is never called by `take_grouping_
provider.py`'s wiring; D-158's own behavior is completely unchanged. ON
with no `WatchListenUnderstanding` evidence, an invalid span, an unknown
source, or missing `CandidateTake`: every function here is a no-op --
absence is "no discovery candidate," never an error.

## Determinism

`discover_candidate_pairs` sorts sources by `source_asset_id` and spans
within a source by `(source_start, source_end, span_id)` -- the SAME sort
key `watch_listen_understanding.build_watch_listen_understanding` itself
already uses -- so output is stable for frozen inputs regardless of
dict/set iteration order, thread completion order, or caller-supplied
ordering.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Tuple
import os

from .attempt_reconstruction import _restart_evidence
from .contracts import CandidateTake
from .raw_understanding_map import (
    BEHAVIOR_AUDIENCE_DELIVERY,
    BEHAVIOR_CLEAN_ATTEMPT,
    BEHAVIOR_FALSE_START,
    BEHAVIOR_POST_TAKE_RESET,
    BEHAVIOR_PRE_TAKE_SETUP,
    BEHAVIOR_RECORDING_PROCESS,
)
from .take_grouping import (
    _safe_short_prefix_retry,
    incomplete_attempt_completed_by_retry,
    multimodal_corroborated_retry,
    same_opening_restart,
)
from .watch_listen_understanding import (
    CONFIDENCE_SUPPORTED,
    CONFIDENCE_WEAK,
    RELATION_COMPLEMENTARY,
    RELATION_CONTINUATION,
    RELATION_CORRECTION,
    RELATION_DISTINCT_PROPOSITION,
    RELATION_NEW_AUDIENCE_BEAT,
    RELATION_RETRY,
    RELATION_UNCERTAIN,
    AttemptRelationHypothesis,
    UnderstandingSpan,
)

SCHEMA_VERSION = "cutsell.watch_listen_relation_discovery.v1"

_WATCH_LISTEN_RELATION_DISCOVERY_ENV = "CUTSELL_WATCH_LISTEN_RELATION_DISCOVERY_ENABLED"

# ---------------------------------------------------------------------------
# Pair-source provenance (this task's own vocabulary).
# ---------------------------------------------------------------------------
PAIR_SOURCE_SEMANTIC = "SEMANTIC_PAIR_SOURCE"
PAIR_SOURCE_WATCH_LISTEN = "WATCH_LISTEN_DISCOVERY"
PAIR_SOURCE_BOTH = "BOTH"

# ---------------------------------------------------------------------------
# Rejection-reason vocabulary (bounded, tail-safe -- never a transcript).
# ---------------------------------------------------------------------------
REJECT_MISSING_TAKE = "missing_candidate_take"
REJECT_ALREADY_RESOLVED = "already_resolved_by_semantic_path"
REJECT_NOT_SUPPORTED = "confidence_not_supported_or_uncertain"
REJECT_PROPOSITION_UNRESOLVED = "proposition_evidence_unresolved"
REJECT_CONFLICT_FLAGGED = "meaning_safety_conflict_flagged"
REJECT_MARKED_DISTINCT_ADDITION = "marked_distinct_addition_content_divergence"

_INTERMEDIARY_BEHAVIOR_LABELS: frozenset[str] = frozenset({
    BEHAVIOR_PRE_TAKE_SETUP, BEHAVIOR_POST_TAKE_RESET, BEHAVIOR_RECORDING_PROCESS, BEHAVIOR_FALSE_START,
})
_AUDIENCE_BEHAVIOR_LABELS: frozenset[str] = frozenset({BEHAVIOR_AUDIENCE_DELIVERY, BEHAVIOR_CLEAN_ATTEMPT})


def _env_true_default_false(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def watch_listen_relation_discovery_enabled(env: Mapping[str, str] | None = None) -> bool:
    values = env if env is not None else os.environ
    return _env_true_default_false(values.get(_WATCH_LISTEN_RELATION_DISCOVERY_ENV))


@dataclass(frozen=True)
class DiscoveryCandidate:
    """One candidate pair this module proposes -- evidence, never a
    decision. `bridged=True` marks a candidate produced by the bounded
    single-hop intermediary bridge rather than a direct D-157 immediate-
    neighbor relation hypothesis."""
    left_id: str
    right_id: str
    relation: str
    confidence: str
    basis: str
    bridged: bool


def _span_behavior_labels(span: UnderstandingSpan) -> frozenset[str]:
    return frozenset(h.label for h in span.behavior_state_hypotheses)


def _is_pure_intermediary(span: UnderstandingSpan) -> bool:
    """True only when EVERY behavior hypothesis on this span is one of the
    four recognized non-audience intermediary kinds and NONE is an
    audience-relevant kind -- a real audience-delivery span is never
    bridged across, even if it also happens to carry a secondary
    intermediary-shaped hypothesis."""
    labels = _span_behavior_labels(span)
    if not labels or (labels & _AUDIENCE_BEHAVIOR_LABELS):
        return False
    return bool(labels & _INTERMEDIARY_BEHAVIOR_LABELS)


def _best_supported(
    relations: Tuple[AttemptRelationHypothesis, ...],
) -> AttemptRelationHypothesis | None:
    for relation in relations:
        if relation.confidence == CONFIDENCE_SUPPORTED and relation.relation != RELATION_UNCERTAIN:
            return relation
    return None


def _bridged_candidate(
    pp_span: UnderstandingSpan, right_span: UnderstandingSpan,
    pp_take: CandidateTake, right_take: CandidateTake,
) -> DiscoveryCandidate | None:
    """Bounded single-hop bridge (see module docstring). Reuses `attempt_
    reconstruction._restart_evidence` verbatim -- the SAME primitive
    D-157's own `_relation_for_pair` already calls -- never a new
    detector. Confidence mirrors D-157's own cadence for this same
    evidence shape: SUPPORTED only when the bridged-from take is itself
    incomplete (a genuine broken-attempt signal), WEAK otherwise."""
    if not _restart_evidence(pp_take.text, right_take.text):
        return None
    confidence = CONFIDENCE_SUPPORTED if not pp_take.complete_idea else CONFIDENCE_WEAK
    return DiscoveryCandidate(
        left_id=pp_span.span_id, right_id=right_span.span_id,
        relation=RELATION_RETRY, confidence=confidence,
        basis="lexical restart evidence bridged across a bounded non-audience intermediary span",
        bridged=True,
    )


def discover_candidate_pairs(
    understanding_spans_by_id: Mapping[str, UnderstandingSpan] | None,
    take_map: Mapping[str, CandidateTake],
) -> Tuple[DiscoveryCandidate, ...]:
    """Walks each source's own understanding-span sequence (sorted
    deterministically, mirroring D-157's own sort key) and proposes a
    `DiscoveryCandidate` for every span pair carrying a real `SUPPORTED`
    relation hypothesis (direct D-157 neighbor relation, or this module's
    own bounded single-hop bridge) -- O(n) per source, never O(n^2), never
    an arbitrary long-range comparison. Computes nothing about ASR/audio/
    visual perception; every signal is either already-computed D-157
    evidence or a direct reuse of `attempt_reconstruction._restart_
    evidence`. Fail-open: missing/empty input returns `()`, never raises."""
    if not understanding_spans_by_id:
        return ()
    spans_by_source: dict[str, list[UnderstandingSpan]] = {}
    for span in understanding_spans_by_id.values():
        spans_by_source.setdefault(span.source_asset_id, []).append(span)

    candidates: list[DiscoveryCandidate] = []
    for source_id in sorted(spans_by_source):
        spans = tuple(sorted(
            spans_by_source[source_id],
            key=lambda s: (s.source_start, s.source_end, s.span_id),
        ))
        for i in range(1, len(spans)):
            right_span = spans[i]
            left_span = spans[i - 1]
            right_take = take_map.get(right_span.span_id)
            left_take = take_map.get(left_span.span_id)

            if right_take is not None and left_take is not None:
                best = _best_supported(right_span.attempt_relation_hypotheses)
                if best is not None and best.left_span_id == left_span.span_id:
                    candidates.append(DiscoveryCandidate(
                        left_id=left_span.span_id, right_id=right_span.span_id,
                        relation=best.relation, confidence=best.confidence,
                        basis=best.basis, bridged=False,
                    ))

            if i >= 2 and right_take is not None and _is_pure_intermediary(left_span):
                pp_span = spans[i - 2]
                pp_take = take_map.get(pp_span.span_id)
                if pp_take is not None:
                    bridged = _bridged_candidate(pp_span, right_span, pp_take, right_take)
                    if bridged is not None:
                        candidates.append(bridged)

    return tuple(candidates)


def proposition_evidence_for_pair(
    left_take: CandidateTake, right_take: CandidateTake,
    *, confirmed_recording_evidence: Mapping[str, tuple] | None = None,
) -> Tuple[bool, str]:
    """The Proposition Firewall check (D-111). Reuses the SAME four
    deterministic restart-evidence rules `reconcile_semantic_idea_
    equivalence`'s own deterministic-restart loop already applies, in the
    SAME order, never reimplemented and never a provider call. Returns
    `(True, kind)` when independent proposition evidence confirms the
    pair, `(False, reason)` otherwise -- a discovered RETRY relation is
    never sufficient alone."""
    kind = same_opening_restart(left_take, right_take)
    if kind is None and _safe_short_prefix_retry(left_take, right_take):
        kind = "safe_short_prefix_retry"
    if kind is None:
        kind = incomplete_attempt_completed_by_retry(left_take, right_take)
    if kind is None and confirmed_recording_evidence:
        corroborated = multimodal_corroborated_retry(left_take, right_take, confirmed_recording_evidence)
        if corroborated is not None:
            kind = corroborated[0]
    if kind:
        return True, kind
    return False, "no independent proposition evidence for this pair"


def discovery_diagnostics(rows: Iterable[dict]) -> dict:
    """Tail-safe, counts-only CI summary (same pattern as every prior
    D-15x/D-158 compact summary) -- never dumps a transcript, basis
    string, or per-pair reason beyond the bounded per-pair trace rows
    `rows` themselves already carry."""
    rows = tuple(rows)
    label_to_key = {
        RELATION_RETRY: "watch_listen_discovery_retry_count",
        RELATION_CONTINUATION: "watch_listen_discovery_continuation_count",
        RELATION_CORRECTION: "watch_listen_discovery_correction_count",
        RELATION_COMPLEMENTARY: "watch_listen_discovery_complementary_count",
        RELATION_NEW_AUDIENCE_BEAT: "watch_listen_discovery_new_beat_count",
        RELATION_DISTINCT_PROPOSITION: "watch_listen_discovery_distinct_count",
    }
    counts = {key: 0 for key in label_to_key.values()}
    uncertain_skipped = 0
    accepted = 0
    rejected = 0
    rejection_reasons: dict[str, int] = {}
    semantic_pair_candidate_count = 0
    watch_listen_only_pair_count = 0
    both_source_pair_count = 0

    for row in rows:
        source = row.get("pair_source")
        if source == PAIR_SOURCE_SEMANTIC:
            semantic_pair_candidate_count += 1
        elif source == PAIR_SOURCE_WATCH_LISTEN:
            watch_listen_only_pair_count += 1
        elif source == PAIR_SOURCE_BOTH:
            both_source_pair_count += 1
        relation = row.get("structured_final_relation")
        if relation == RELATION_UNCERTAIN or relation is None:
            uncertain_skipped += 1
        elif relation in label_to_key:
            counts[label_to_key[relation]] += 1
        if row.get("accepted"):
            accepted += 1
        else:
            rejected += 1
            reason = row.get("rejection_reason") or "unspecified"
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    return {
        "watch_listen_discovery_evaluated_count": len(rows),
        "watch_listen_discovery_candidate_count": len(rows),
        "watch_listen_discovery_uncertain_skipped_count": uncertain_skipped,
        "semantic_pair_candidate_count": semantic_pair_candidate_count,
        "watch_listen_only_pair_count": watch_listen_only_pair_count,
        "both_source_pair_count": both_source_pair_count,
        "watch_listen_discovery_accepted_count": accepted,
        "watch_listen_discovery_rejected_count": rejected,
        "discovery_rejection_reasons": rejection_reasons,
        **counts,
    }
