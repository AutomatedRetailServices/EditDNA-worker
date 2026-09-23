"""D-172 -- Watch+Listen BestTake evidence, Zone-Usability V2 diagnostic
consumption.

Per docs/CUTSELL_CANONICAL_ENGINE_ARCHITECTURE_D098.md Section 14/13 and
docs/CUTSELL_DECISIONS.md D-163/D-164/D-167/D-170/D-172. D-163's own
BestTake diagnostic guard (`watch_listen_besttake_evidence.py`,
`evaluate_watch_listen_besttake_guard`, CLOSED, stays at ZERO diff) reads
only V1's coarse, categorical delivery-usability signal. D-164 proved that
signal saturates real candidates to a flat UNUSABLE too readily; D-167
built a refined, categorical V2 signal (severity/pattern/affected_fraction,
never a numeric score); D-170 proved V2 differentiates real candidates,
on real media, even when their coarse V1 buckets are identical -- and
found one concrete real dominance relation D-163's own guard could only
mark UNCERTAIN.

This module is the bounded adapter this task authorizes: it lets D-163's
diagnostic comparison consult V2 evidence when available, WITHOUT touching
`watch_listen_besttake_evidence.py` (D-163) itself and WITHOUT touching
`watch_listen_zone_usability_v2.py` (D-167) itself -- both remain at
literal zero diff, exactly like D-167's own "why this stays a fully
separate module" precedent for its own relationship to D-163/D-157.

## Authority principle (this task's own binding instruction)

    D-172 may improve WHAT THE GUARD KNOWS.
    It may NOT change WHAT THE ENGINE SELECTS.

`evaluate_watch_listen_besttake_guard_v2` below NEVER mutates
`selected_clip_id`/`ranked`/membership/Boundary/Pacing -- it returns a
`WatchListenBestTakeV2GuardResult` (evidence + a diagnostic status),
mirroring D-163's own `WatchListenBestTakeGuardResult` contract exactly.
`winner_before == winner_after` for every family, always, in this task.

## Design: V2 only ever ADDS discriminating power where V1 already gave up

D-163's own V1 guard (`evaluate_watch_listen_besttake_guard`, unchanged,
called first here and never re-derived) already resolves four of its five
outcomes correctly and completely: `NO_ACTION` (no evidence, or winner not
meaning-sufficient -- upstream-owned), `PRESERVE_STRUCTURED_WINNER`
(winner's own delivery usability is acceptable), `UNCERTAIN` (winner
carries a material modality conflict -- fail-open, V2 must never resolve
this either, since V2 shares the SAME `conflict_flags`/`zone_conflict`
source per D-157/D-167's own `SAME_SOURCE_DUPLICATE` audit entry), and any
outcome where V1's OWN ordinal comparison already found a dominant
meaning-sufficient alternative (`PERFORMANCE_DOMINANT_ALTERNATIVE`) --
this task never second-guesses a decision V1 already reached on its own
evidence. The ONE outcome D-164 proved is genuinely under-resolved by V1's
categorical signal is `BYPASS_POOR_USABILITY_WINNER` (winner's DELIVERY
usability is UNUSABLE, but V1's own flat OR-over-presence rule saturates
every candidate to the same rank, so no strict V1 ordinal dominance can
ever be found even when one candidate is measurably, factually less
impaired than another). ONLY in that exact V1 outcome does this module ask
V2's OWN, unmodified `zone_usability_v2_dominates` (D-167, never a second
dominance algorithm, per this task's own explicit instruction) whether the
refined per-zone/severity/pattern resolution finds a real, meaning-
sufficient, non-conflicted dominant alternative V1's coarser signal could
not see. When it does, the reported diagnostic status upgrades from
`BYPASS_POOR_USABILITY_WINNER` to `PERFORMANCE_DOMINANT_ALTERNATIVE` with
`evidence_source=V2` -- WHAT THE GUARD KNOWS changed; `action_applied`
stays `False` and `winner_after == winner_before` always -- WHAT THE
ENGINE SELECTS did not.

## Evidence source contract

Per candidate, this module classifies which evidence backs the winner's
own comparison:

    V2           -- a real `CandidateZoneUsabilityV2` (D-167) was built for
                    this candidate.
    V1_FALLBACK  -- no V2 evidence, but D-163's own V1 `WatchListenBestTake
                    Evidence` exists -- the pre-D-172 path, unchanged.
    NO_EVIDENCE  -- neither exists (fail-open to `NO_ACTION`/`UNCERTAIN`,
                    exactly like D-163's own existing fail-open contract).

`build_candidate_zone_usability_v2` never crashes: any exception while
building the fresh positioned/raw-understanding evidence (D-115/D-155,
unchanged, reused verbatim, never re-derived) is caught and treated as
"no V2 evidence for this candidate" -- V1_FALLBACK, never a pipeline
failure.

## Event-kind scope (honest, not silently widened)

The fresh `RawUnderstandingSpan` this module builds for V2 uses
`watch_listen_zone_usability_v2._DEFECT_KINDS` (the SAME event-kind union
V1's own `_usability_for_zone` and V2's own severity table both already
use) as the `event_kinds` filter into `positioned_performance_evidence.
build_positioned_performance_evidence` -- never the narrower `LOCAL_
PERFORMANCE_EVENT_KINDS` D-122's own `case_b_performance_evidence.py`
uses for its DELIVERY-only projection. This keeps this module's own
event-kind scope IDENTICAL to V1's, whatever real `TemporalEvent`s of
those kinds actually exist in `whole_video_context` for a given RAW --
never a new event kind, never a narrower or wider scope than V1's own.

## Double-counting (this task's own explicit requirement)

V2 is, by D-167's own module docstring, a re-aggregation of the SAME
underlying local-performance events D-163's V1 signal, `MediaSignals`, and
D-097/D-122 already read -- never an independent second vote. When V2
evidence is available for a candidate, it SUPERSEDES V1 for this module's
own refined zone-usability comparison dimension (the dominance search
below); V1's own evidence stays visible diagnostically (the underlying
`evaluate_watch_listen_besttake_guard` result is always computed and
carried through, never discarded) but is never independently re-summed
alongside V2 as a second confirmation.

## Meaning firewall (no exception)

A V2-cleaner-but-meaning-insufficient candidate is NEVER eligible as
`dominant_candidate_id` -- exactly mirroring D-163's own `meaning_
sufficient_ids` filter, never bypassed. When V2 dominance IS found against
such a candidate, this module records `meaning_firewall_blocked=True`
(visible diagnostic evidence a future authority could weigh) and the
reported status stays `BYPASS_POOR_USABILITY_WINNER` -- the firewall
blocks eligibility, it never silently promotes.

## CASE A/B/C ownership (D-107/D-115, reused, never re-derived)

`zone_usability_v2_dominates` (D-167, unmodified) already refuses to fire
when the winner's own `case_classification == CASE_A_BOUNDARY_ONLY`
(Boundary's own territory can never make a whole take lose) and when
either side carries a `zone_conflict` (CASE C ambiguity fails open). This
module adds no second copy of that logic -- it calls the one, real,
already-tested function.

## Feature flag

`CUTSELL_WATCH_LISTEN_ZONE_USABILITY_V2_BESTTAKE_ENABLED`, default OFF, a
SEPARATE flag from D-163's own `CUTSELL_WATCH_LISTEN_BESTTAKE_EVIDENCE_
ENABLED` -- independent rollback is needed because V1 evidence collection
must remain controllable on its own (a future RAW may want V1 evidence ON
with V2 consumption OFF, to isolate causal attribution between "collecting
Watch+Listen BestTake evidence at all" and "additionally consuming the
refined V2 signal", exactly the controlled-variable discipline this
session's RAW methodology already uses for D-164/D-170's own additive
flags). OFF: this module is never called from `pipeline.py`; behavior is
byte-identical to pre-D-172 (including D-163's own diagnostics, which stay
exactly as they were). ON (and only when D-163's OWN flag is also ON, by
construction -- see `pipeline.py`'s nesting): this module's own diagnostic
fields populate alongside D-163's existing `watch_listen_besttake_*`
family row; `selected_clip_id` remains untouched either way.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping
import os

from .contracts import CandidateTake
from .positioned_performance_evidence import build_positioned_performance_evidence
from .raw_understanding_map import build_raw_understanding_span
from .watch_listen_besttake_evidence import (
    GUARD_BYPASS_POOR_USABILITY_WINNER,
    GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE,
    WatchListenBestTakeEvidence,
    WatchListenBestTakeGuardResult,
    _existing_ladder_pick,
    evaluate_watch_listen_besttake_guard,
)
from .watch_listen_zone_usability_v2 import (
    CandidateZoneUsabilityV2,
    _DEFECT_KINDS,
    build_zone_usability_v2,
    zone_usability_v2_dominates,
)
from .whole_video_analysis import WholeVideoContext

SCHEMA_VERSION = "cutsell.watch_listen_besttake_v2_evidence.v1"

_ZONE_USABILITY_V2_BESTTAKE_ENV = "CUTSELL_WATCH_LISTEN_ZONE_USABILITY_V2_BESTTAKE_ENABLED"

# ---------------------------------------------------------------------------
# Evidence source contract (module docstring).
# ---------------------------------------------------------------------------
EVIDENCE_SOURCE_V2 = "V2"
EVIDENCE_SOURCE_V1_FALLBACK = "V1_FALLBACK"
EVIDENCE_SOURCE_NO_EVIDENCE = "NO_EVIDENCE"


def _env_true_default_false(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def zone_usability_v2_besttake_enabled(env: Mapping[str, str] | None = None) -> bool:
    values = env if env is not None else os.environ
    return _env_true_default_false(values.get(_ZONE_USABILITY_V2_BESTTAKE_ENV))


def build_candidate_zone_usability_v2(
    candidate: CandidateTake,
    whole_video_context: WholeVideoContext | None,
) -> CandidateZoneUsabilityV2 | None:
    """Fail-open builder: constructs a fresh, real `RawUnderstandingSpan`
    (D-155) from `candidate`'s own already-computed evidence (D-115's
    `PositionAwarePerformanceEvidence`, filtered to V1's OWN `_DEFECT_
    KINDS` -- module docstring's "Event-kind scope") and projects it
    through D-167's OWN, unmodified `build_zone_usability_v2`. Recomputes
    no perception -- every function called here is an existing, already-
    tested, pure projection. Returns `None` on any exception (never a
    pipeline failure) or when D-167's own builder returns `None` (no
    positioned evidence -- e.g. no aligned words)."""
    try:
        positioned = build_positioned_performance_evidence(
            candidate, whole_video_context, event_kinds=_DEFECT_KINDS,
        )
        raw_span = build_raw_understanding_span(candidate, positioned)
        return build_zone_usability_v2(candidate.clip_id, raw_span)
    except Exception:
        return None


def _evidence_source_for(
    v1_evidence: WatchListenBestTakeEvidence | None,
    v2_evidence: CandidateZoneUsabilityV2 | None,
) -> str:
    if v2_evidence is not None:
        return EVIDENCE_SOURCE_V2
    if v1_evidence is not None:
        return EVIDENCE_SOURCE_V1_FALLBACK
    return EVIDENCE_SOURCE_NO_EVIDENCE


@dataclass(frozen=True)
class WatchListenBestTakeV2GuardResult:
    """D-172's own diagnostic result -- mirrors D-163's own `WatchListen
    BestTakeGuardResult` contract exactly (no winner field beyond the
    evidence winner_id already carries; a status/reason, never a score).
    `v1_result` is always the real, unmodified D-163 verdict this result
    was derived from -- never discarded, always inspectable."""
    guard_status: str
    guard_reason: str
    winner_id: str | None
    dominant_candidate_id: str | None
    existing_ladder_agrees: bool | None
    evidence_source: str
    winner_severity: str | None
    alt_severity: str | None
    meaning_firewall_blocked: bool
    candidate_usability_summary: Mapping[str, str]
    v1_result: WatchListenBestTakeGuardResult


def evaluate_watch_listen_besttake_guard_v2(
    *,
    winner_id: str | None,
    meaning_sufficient_ids: Iterable[str],
    v1_evidence_by_id: Mapping[str, WatchListenBestTakeEvidence | None],
    v2_evidence_by_id: Mapping[str, CandidateZoneUsabilityV2 | None],
    ranked: Iterable[Mapping[str, object]] = (),
) -> WatchListenBestTakeV2GuardResult:
    """The bounded V2-consuming guard. See module docstring's "Design"
    section for the exact one-outcome-refined contract. Never mutates a
    selection; fail-open throughout."""
    meaning_sufficient_ids = frozenset(meaning_sufficient_ids)
    ranked = list(ranked)

    v1_result = evaluate_watch_listen_besttake_guard(
        winner_id=winner_id,
        meaning_sufficient_ids=meaning_sufficient_ids,
        evidence_by_id=v1_evidence_by_id,
        ranked=ranked,
    )

    winner_v1 = v1_evidence_by_id.get(winner_id) if winner_id else None
    winner_v2 = v2_evidence_by_id.get(winner_id) if winner_id else None
    winner_source = _evidence_source_for(winner_v1, winner_v2)
    winner_severity = winner_v2.delivery.zone_severity if winner_v2 is not None else None

    # Only the ONE V1 outcome D-164 proved under-resolved is ever refined
    # here -- see module docstring "Design". Every other V1 outcome
    # (including an already-found V1 dominance) is passed through
    # unchanged: V2 never second-guesses a decision V1 already reached.
    # A winner-side zone_conflict fails open exactly like D-163's own
    # `conflict_flags` check (never resolved by richer evidence either).
    if (
        v1_result.guard_status != GUARD_BYPASS_POOR_USABILITY_WINNER
        or winner_v2 is None
        or winner_v2.delivery.zone_conflict
    ):
        return WatchListenBestTakeV2GuardResult(
            guard_status=v1_result.guard_status,
            guard_reason=v1_result.guard_reason,
            winner_id=winner_id,
            dominant_candidate_id=v1_result.dominant_candidate_id,
            existing_ladder_agrees=v1_result.existing_ladder_agrees,
            evidence_source=winner_source,
            winner_severity=winner_severity,
            alt_severity=None,
            meaning_firewall_blocked=False,
            candidate_usability_summary=v1_result.candidate_usability_summary,
            v1_result=v1_result,
        )

    dominant_candidates: list[str] = []
    meaning_insufficient_dominant_found = False
    for cid, alt_v2 in v2_evidence_by_id.items():
        if cid == winner_id or alt_v2 is None:
            continue
        if alt_v2.delivery.zone_conflict:
            continue
        if not zone_usability_v2_dominates(alt_v2, winner_v2):
            continue
        if cid in meaning_sufficient_ids:
            dominant_candidates.append(cid)
        else:
            # Meaning Firewall (no exception): a real V2 dominance against
            # a meaning-insufficient candidate is reported, never eligible.
            meaning_insufficient_dominant_found = True

    if not dominant_candidates:
        return WatchListenBestTakeV2GuardResult(
            guard_status=GUARD_BYPASS_POOR_USABILITY_WINNER,
            guard_reason=(
                "v2_dominant_alternative_meaning_insufficient_blocked"
                if meaning_insufficient_dominant_found
                else v1_result.guard_reason
            ),
            winner_id=winner_id,
            dominant_candidate_id=None,
            existing_ladder_agrees=None,
            evidence_source=winner_source,
            winner_severity=winner_severity,
            alt_severity=None,
            meaning_firewall_blocked=meaning_insufficient_dominant_found,
            candidate_usability_summary=v1_result.candidate_usability_summary,
            v1_result=v1_result,
        )

    # Deterministic tie-break among dominant candidates: reuse the EXISTING
    # `ranked` score (never a new number), then clip_id -- identical
    # precedent to D-163's own V1 tie-break.
    ranked_score_by_id = {str(row.get("clip_id") or ""): float(row.get("score") or 0.0) for row in ranked}
    dominant_candidates.sort(key=lambda cid: (-ranked_score_by_id.get(cid, 0.0), cid))
    dominant_id = dominant_candidates[0]
    dominant_v2 = v2_evidence_by_id.get(dominant_id)
    alt_severity = dominant_v2.delivery.zone_severity if dominant_v2 is not None else None

    existing_pick = _existing_ladder_pick(ranked)
    return WatchListenBestTakeV2GuardResult(
        guard_status=GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE,
        guard_reason="v2_refined_dominant_alternative_found",
        winner_id=winner_id,
        dominant_candidate_id=dominant_id,
        existing_ladder_agrees=(existing_pick == dominant_id),
        evidence_source=winner_source,
        winner_severity=winner_severity,
        alt_severity=alt_severity,
        meaning_firewall_blocked=meaning_insufficient_dominant_found,
        candidate_usability_summary=v1_result.candidate_usability_summary,
        v1_result=v1_result,
    )


def watch_listen_besttake_v2_group_row(
    result: WatchListenBestTakeV2GuardResult, winner_before: str | None,
) -> dict:
    """The exact compact family-level diagnostic fields this task's own
    directive names, bounded and JSON-safe -- no transcript dump."""
    return {
        "watch_listen_besttake_evidence_source": result.evidence_source,
        "watch_listen_besttake_v2_available": result.evidence_source == EVIDENCE_SOURCE_V2,
        "watch_listen_besttake_v2_dominant_candidate": result.dominant_candidate_id,
        "watch_listen_besttake_v2_guard_status": result.guard_status,
        "watch_listen_besttake_v2_guard_reason": result.guard_reason,
        "watch_listen_besttake_v2_winner_severity": result.winner_severity,
        "watch_listen_besttake_v2_alt_severity": result.alt_severity,
        "watch_listen_besttake_v2_meaning_firewall_blocked": result.meaning_firewall_blocked,
        "winner_before": winner_before,
        "winner_after": winner_before,  # D-172: never mutated in this task
        "action_applied": False,
    }


def watch_listen_besttake_v2_diagnostics(rows: Iterable[WatchListenBestTakeV2GuardResult]) -> dict:
    """Tail-safe, counts-only CI summary -- same pattern as D-163's own
    `watch_listen_besttake_diagnostics`. Never dumps a transcript."""
    rows = tuple(rows)
    counts = {
        "v2_evaluated_count": len(rows),
        "v2_no_action_count": 0,
        "v2_dominance_count": 0,
        "v2_bypass_count": 0,
        "v2_uncertain_count": 0,
        "v2_meaning_firewall_block_count": 0,
        "v1_fallback_count": 0,
    }
    for row in rows:
        if row.guard_status == "NO_ACTION":
            counts["v2_no_action_count"] += 1
        elif row.guard_status == GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE:
            counts["v2_dominance_count"] += 1
        elif row.guard_status == GUARD_BYPASS_POOR_USABILITY_WINNER:
            counts["v2_bypass_count"] += 1
        elif row.guard_status == "UNCERTAIN":
            counts["v2_uncertain_count"] += 1
        if row.meaning_firewall_blocked:
            counts["v2_meaning_firewall_block_count"] += 1
        if row.evidence_source == EVIDENCE_SOURCE_V1_FALLBACK:
            counts["v1_fallback_count"] += 1
    return counts
