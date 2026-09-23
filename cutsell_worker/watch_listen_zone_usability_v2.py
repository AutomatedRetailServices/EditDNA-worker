"""D-167 -- Watch+Listen Zone-Usability Refinement V2.

Per docs/CUTSELL_DECISIONS.md D-164 (real-media finding) and D-165/D-166
(Language Spine, unrelated axis). This module is a bounded, additive
REFINEMENT of the FACTUAL RESOLUTION of the existing zone-usability
signal -- it grants NO new authority, mutates no selection, and does not
replace D-157's `watch_listen_understanding.py` (V1), which stays CLOSED
at ZERO diff. `watch_listen_besttake_evidence.py` (D-163) is ALSO left at
zero diff -- see "Why this stays a fully separate module" below.

## Saturation root cause (D-167's own forensic, this task)

`watch_listen_understanding._usability_for_zone` (D-157, unchanged):

    if any(e.kind in _DEFECT_KINDS for e in events):
        return USABILITY_UNUSABLE if zone == ZONE_DELIVERY else USABILITY_QUESTIONABLE

This is a single boolean OR over event PRESENCE in a zone -- it reads
NOTHING about event duration, confidence, isolation, or what fraction of
the zone a defect actually covers. `_DEFECT_KINDS` unions FIVE materially
different behavioral classes (ordinary hand/body/facial motion,
recording-process markers, false starts, abandoned-attempt confirmations,
and breaking-character) into ONE flat set with identical treatment. D-164's
real-media qualification (`docs/CUTSELL_DECISIONS.md` D-164) proved the
direct consequence: two real candidates in family `tg_31c4dc583b648824a0`
with materially different raw event counts/durations (3 events/0.2s vs.
8 events/0.533s, per D-122's own `case_b_performance_evidence.py`
projection of the SAME underlying events) were BOTH marked `UNUSABLE` on
DELIVERY by V1, because a single micro-motion event of ANY duration or
confidence already saturates the categorical rule. This is why D-163's
guard could not distinguish them: `_performance_dominates` requires a
strict ordinal difference, and V1's own signal ties everything at
UNUSABLE (rank 0) the instant any defect-kind event exists anywhere in
DELIVERY.

## Event-kind inventory and current-behavior classification (this task's
## own required audit)

| kind | source | current V1 treatment |
|---|---|---|
| `camera_disengagement_candidate` | Track C, real, D-114 | any presence -> UNUSABLE (DELIVERY) / QUESTIONABLE (ENTRY/EXIT), no duration/confidence weighting |
| `facial_expression_shift_candidate` | Track C, real, D-114 | same as above |
| `body_reset_candidate` | Track C, real, D-114 | same as above |
| `hand_motion_reset_candidate` | Track C, real, D-114 | same as above |
| `wrong_take` / `retry_setup` | D-100 deterministic-rule confirmation | same as above (bundled into `_ABANDONED_KINDS`) |
| `false_start` | deterministic rule | same as above |
| `breaking_character` | deterministic rule | same as above |
| recording-process kinds (`recording_joke`, `verbal_fumble`, `product_handling_mistake`, `accidental_laughter`, `searching_for_words`) | deterministic rule | same as above |
| `audio_silence_interval` | real, ffmpeg `silencedetect` (`audio_silence.py`) | NOT a member of `_DEFECT_KINDS` at all today -- audio evidence never drives zone usability in V1, consistent with the existing Audio Honesty finding (signal-level only, no defect classification from audio) |

## Why this stays a fully separate module (D-163/D-157 zero-diff decision)

This task's own directive lists D-163 as CLOSED under "Preserve CLOSED"
and separately instructs "Update `watch_listen_besttake_evidence.py`
ONLY AS NEEDED to expose refined severity." Because this module's own
`CandidateZoneUsabilityV2` + `zone_usability_v2_dominates` (below) fully
satisfy the directive's "improve D-163's factual comparison input"
requirement WITHOUT touching D-163's own file at all, the smallest-
footprint, lowest-regression-risk compliant reading is: the actual
"need" to modify `watch_listen_besttake_evidence.py` is zero. That file,
and `watch_listen_understanding.py` (D-157), both remain at literal zero
diff. A future, separately-authorized task may wire V2 as an alternate
input to D-163's guard; not done here.

## No new opaque score

Per this task's own explicit instruction, `zone_severity`/`zone_usability`
are CATEGORICAL, derived from a small, explicit, inspectable STRUCTURAL
lookup table over (materiality tier x pattern) -- never a weighted sum,
never a numeric master score. `affected_fraction` IS a real numeric fact
(trusted defect duration / zone duration) -- exposed for inspection and
for a future authority's own use, but this module's OWN categorical
bucketing never multiplies it by a weight; it is compared once, against
its own zone's total duration, to answer one honest structural question
("does the trusted defect evidence cover a majority of this zone's own
duration") -- not an externally-tuned threshold, and it is used only to
pick a PATTERN label, never to gate a BestTake decision directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Tuple

from .positioned_performance_evidence import PositionedEvent, ZONE_DELIVERY, ZONE_ENTRY, ZONE_EXIT
from .raw_understanding_map import (
    PROVENANCE_DETERMINISTIC_RULE,
    PROVENANCE_UNKNOWN,
    PROVENANCE_VISUAL_SIGNAL,
    RawUnderstandingSpan,
    _ABANDONED_KINDS,
    _BREAKING_CHARACTER_KINDS,
    _FALSE_START_KINDS,
    _RECORDING_PROCESS_KINDS,
    _RESET_FAMILY_KINDS,
)
from .watch_listen_besttake_evidence import (
    CASE_A_BOUNDARY_ONLY,
    CASE_B_DELIVERY_OWNED,
    CASE_C_AMBIGUOUS,
    CASE_CLEAN,
)

SCHEMA_VERSION = "cutsell.watch_listen_zone_usability_v2.v1"

# ---------------------------------------------------------------------------
# Usability vocabulary -- V1-compatible 4 states + one new intermediate
# state (IMPAIRED) this task's directive names. Redefined here (not
# imported) so watch_listen_understanding.py (D-157) stays at zero diff --
# the four shared string VALUES are intentionally identical to V1's own,
# so a future consumer that only understands the V1 vocabulary can safely
# treat IMPAIRED as UNUSABLE without any translation table.
# ---------------------------------------------------------------------------
USABILITY_USABLE = "USABLE"
USABILITY_QUESTIONABLE = "QUESTIONABLE"
USABILITY_IMPAIRED = "IMPAIRED"
USABILITY_UNUSABLE = "UNUSABLE"
USABILITY_UNKNOWN = "UNKNOWN"

# ---------------------------------------------------------------------------
# Severity vocabulary.
# ---------------------------------------------------------------------------
SEVERITY_NONE = "NONE"
SEVERITY_MILD = "MILD"
SEVERITY_MATERIAL = "MATERIAL"
SEVERITY_SEVERE = "SEVERE"
SEVERITY_UNKNOWN = "UNKNOWN"
SEVERITY_MIXED = "MIXED"

# ---------------------------------------------------------------------------
# Pattern vocabulary (isolated / repeated / sustained distinction).
# ---------------------------------------------------------------------------
PATTERN_NONE = "NONE"
PATTERN_ISOLATED = "ISOLATED"
PATTERN_REPEATED = "REPEATED"
PATTERN_SUSTAINED = "SUSTAINED"
PATTERN_UNKNOWN = "UNKNOWN"

# ---------------------------------------------------------------------------
# Confidence vocabulary -- reused verbatim from the established D-097
# per-kind floors (0.88 reset-family, 0.76 break-family;
# take_judge._RESET_KINDS/_BREAK_KINDS) rather than inventing a new
# numeric cutoff. Kinds outside those two families have no established
# floor, so ANY real confidence value (> 0.0) is SUPPORTED for them --
# this task adds no NEW numeric threshold for those kinds either.
# ---------------------------------------------------------------------------
CONFIDENCE_SUPPORTED = "SUPPORTED"
CONFIDENCE_WEAK = "WEAK"
CONFIDENCE_UNKNOWN = "UNKNOWN"

_RESET_FLOOR_KINDS = frozenset({"hand_motion_reset_candidate", "body_reset_candidate"})
_BREAK_FLOOR_KINDS = frozenset({"camera_disengagement_candidate", "facial_expression_shift_candidate"})
_RESET_FLOOR = 0.88
_BREAK_FLOOR = 0.76

# ---------------------------------------------------------------------------
# Trusted defect kinds -- IDENTICAL union to watch_listen_understanding.
# _DEFECT_KINDS (D-157, unchanged) so V2 classifies exactly the same
# events V1 already does; V2 differs only in HOW it weighs them, never in
# WHICH events count as evidence.
# ---------------------------------------------------------------------------
_DEFECT_KINDS: frozenset[str] = (
    _RESET_FAMILY_KINDS | _ABANDONED_KINDS | _RECORDING_PROCESS_KINDS
    | _FALSE_START_KINDS | _BREAKING_CHARACTER_KINDS
)

# ---------------------------------------------------------------------------
# Event-kind editorial-severity materiality tiers (this task's own
# required classification, using GENERAL behavioral meaning -- never a
# person/video-specific rule). Ordinary expressive motion (facial/body/
# hand) is deliberately the LOWEST tier -- see "Ordinary Motion Firewall"
# in `_severity_for_pattern` below.
# ---------------------------------------------------------------------------
HIGH_MATERIALITY_KINDS: frozenset[str] = _BREAKING_CHARACTER_KINDS | _ABANDONED_KINDS
MODERATE_MATERIALITY_KINDS: frozenset[str] = (
    _FALSE_START_KINDS | _RECORDING_PROCESS_KINDS | frozenset({"camera_disengagement_candidate"})
)
LOW_MATERIALITY_KINDS: frozenset[str] = frozenset({
    "facial_expression_shift_candidate", "body_reset_candidate", "hand_motion_reset_candidate",
})

# ---------------------------------------------------------------------------
# Double-counting audit (this task's own explicit requirement) -- extends
# D-163's own `DOUBLE_COUNTING_AUDIT` language to the new V2 dimensions.
# ---------------------------------------------------------------------------
DOUBLE_COUNTING_AUDIT_V2: Mapping[str, str] = {
    # Same root local-performance events already feed MediaSignals/D-097/
    # D-122's factual re-projection -- V2's duration/pattern facts are a
    # DIFFERENT aggregation (normalized-by-zone-duration, materiality-
    # tiered) of the SAME events, never an independent second measurement.
    "event_duration_total_sec": "PARTIALLY_CORRELATED",
    "affected_fraction": "PARTIALLY_CORRELATED",
    "zone_severity": "PARTIALLY_CORRELATED",
    "zone_usability": "PARTIALLY_CORRELATED",
    # Pattern (isolated/repeated/sustained) is a genuinely NEW aggregation
    # axis -- no existing consumer (MediaSignals, D-097, D-122, V1) counts
    # distinct-occurrence-vs-continuous-duration at all.
    "pattern": "INDEPENDENT",
    "dominant_event_kinds": "SAME_SOURCE_DUPLICATE",  # a direct relabeling of the same event.kind values
    "zone_conflict": "SAME_SOURCE_DUPLICATE",  # reuses D-157's own span-level conflict_flags verbatim
}


@dataclass(frozen=True)
class ZoneUsabilityResult:
    """One zone's (ENTRY/DELIVERY/EXIT) refined usability evidence. No
    winner field -- evidence only, exactly like D-163's own contract."""
    zone: str
    zone_usability: str
    zone_severity: str
    event_count: int
    event_duration_total_sec: float
    zone_duration_sec: float | None
    affected_fraction: float | None
    isolated_event: bool
    repeated_defect: bool
    sustained_defect: bool
    dominant_event_kinds: Tuple[str, ...]
    confidence: str
    provenance: str
    zone_conflict: bool


@dataclass(frozen=True)
class CandidateZoneUsabilityV2:
    """The full per-candidate V2 record -- three `ZoneUsabilityResult`s
    plus a Boundary-firewalled overall usability and the CASE A/B/C
    classification (reused from D-163's own vocabulary, never re-derived
    with new semantics)."""
    candidate_id: str
    entry: ZoneUsabilityResult
    delivery: ZoneUsabilityResult
    exit: ZoneUsabilityResult
    overall_usability: str
    case_classification: str


def _materiality_tier(kind: str) -> int:
    """2 = HIGH, 1 = MODERATE, 0 = LOW/unclassified. Ordinal, never a
    weight multiplied into a score -- used only to index the structural
    lookup table below."""
    if kind in HIGH_MATERIALITY_KINDS:
        return 2
    if kind in MODERATE_MATERIALITY_KINDS:
        return 1
    return 0


def _event_confidence_category(kind: str, confidence: float) -> str:
    if kind in _RESET_FLOOR_KINDS:
        return CONFIDENCE_SUPPORTED if confidence >= _RESET_FLOOR else CONFIDENCE_WEAK
    if kind in _BREAK_FLOOR_KINDS:
        return CONFIDENCE_SUPPORTED if confidence >= _BREAK_FLOOR else CONFIDENCE_WEAK
    return CONFIDENCE_SUPPORTED if confidence > 0.0 else CONFIDENCE_WEAK


# Structural (materiality_tier, pattern) -> (severity, usability) lookup --
# the ONE place this module's editorial-severity judgment lives. No
# formula, no weighted sum; every cell is an explicit, inspectable,
# individually-justifiable editorial classification (module docstring's
# "Ordinary Motion Firewall": LOW + ISOLATED never produces ANY severity).
_SEVERITY_TABLE: Mapping[Tuple[int, str], Tuple[str, str]] = {
    (0, PATTERN_ISOLATED):  (SEVERITY_NONE,     USABILITY_USABLE),
    (0, PATTERN_REPEATED):  (SEVERITY_MILD,     USABILITY_QUESTIONABLE),
    (0, PATTERN_SUSTAINED): (SEVERITY_MATERIAL, USABILITY_IMPAIRED),
    (1, PATTERN_ISOLATED):  (SEVERITY_MILD,     USABILITY_QUESTIONABLE),
    (1, PATTERN_REPEATED):  (SEVERITY_MATERIAL, USABILITY_IMPAIRED),
    (1, PATTERN_SUSTAINED): (SEVERITY_SEVERE,   USABILITY_UNUSABLE),
    (2, PATTERN_ISOLATED):  (SEVERITY_MATERIAL, USABILITY_IMPAIRED),
    (2, PATTERN_REPEATED):  (SEVERITY_SEVERE,   USABILITY_UNUSABLE),
    (2, PATTERN_SUSTAINED): (SEVERITY_SEVERE,   USABILITY_UNUSABLE),
}

# The literal majority-of-its-own-zone split point ("does the trusted
# defect evidence cover half or more of the time it is measured within")
# -- a self-referential structural definition of "sustained," not an
# externally-tuned BestTake cutoff. No authority reads this constant
# directly; it only selects a PATTERN label (module docstring).
_SUSTAINED_FRACTION = 0.5


def _pattern_for(event_count: int, affected_fraction: float | None) -> str:
    if event_count == 0:
        return PATTERN_NONE
    if affected_fraction is not None and affected_fraction >= _SUSTAINED_FRACTION:
        return PATTERN_SUSTAINED
    if event_count >= 2:
        return PATTERN_REPEATED
    return PATTERN_ISOLATED


def _zone_span_duration(zone: str, span: RawUnderstandingSpan) -> float | None:
    delivery = span.positioned_evidence.delivery_span
    if not delivery.available:
        return None
    d_start = float(delivery.start)  # type: ignore[arg-type]
    d_end = float(delivery.end)  # type: ignore[arg-type]
    if zone == ZONE_DELIVERY:
        return max(0.0, d_end - d_start)
    if zone == ZONE_ENTRY:
        return max(0.0, d_start - span.source_start)
    if zone == ZONE_EXIT:
        return max(0.0, span.source_end - d_end)
    return None


def _zone_conflict_for(span: RawUnderstandingSpan, zone: str) -> bool:
    """Conservative, honest reuse of D-157's own span-level `conflict_flags`
    (real, already-computed) -- applied per-zone using the SAME zone each
    flag's own basis already names (module docstring's `DOUBLE_COUNTING_
    AUDIT_V2` entry: SAME_SOURCE_DUPLICATE, never a new conflict detector)."""
    flags = span.conflict_flags
    if not flags:
        return False
    if zone == ZONE_DELIVERY and "MEANING_COMPLETE_VS_ABANDONED_ATTEMPT_EVIDENCE" in flags:
        return True
    if zone == ZONE_EXIT and "EXIT_RESET_VS_MEANING_COMPLETE" in flags:
        return True
    return False


def _build_zone_result(span: RawUnderstandingSpan, zone: str) -> ZoneUsabilityResult:
    zone_duration = _zone_span_duration(zone, span)
    conflict = _zone_conflict_for(span, zone)

    if zone_duration is None:
        return ZoneUsabilityResult(
            zone=zone, zone_usability=USABILITY_UNKNOWN, zone_severity=SEVERITY_UNKNOWN,
            event_count=0, event_duration_total_sec=0.0, zone_duration_sec=None,
            affected_fraction=None, isolated_event=False, repeated_defect=False,
            sustained_defect=False, dominant_event_kinds=(), confidence=CONFIDENCE_UNKNOWN,
            provenance=PROVENANCE_UNKNOWN, zone_conflict=conflict,
        )

    zone_events: Tuple[PositionedEvent, ...] = tuple(
        e for e in span.positioned_evidence.positioned_events
        if e.zone == zone and e.kind in _DEFECT_KINDS
    )
    event_count = len(zone_events)
    # Honest event duration: real interval end-start, never fabricated for
    # a point-like event (a zero-width event contributes exactly 0.0, per
    # this task's own "do not fabricate duration" instruction). Overlapping
    # events are summed without merging -- a documented, conservative
    # (over- rather than under-estimating impairment) simplification.
    event_duration_total = round(sum(max(0.0, e.end - e.start) for e in zone_events), 3)
    affected_fraction = (
        round(event_duration_total / zone_duration, 4) if zone_duration > 0 else None
    )

    if event_count == 0:
        severity, usability = SEVERITY_NONE, USABILITY_USABLE
        pattern = PATTERN_NONE
        dominant_kinds: Tuple[str, ...] = ()
        confidence = CONFIDENCE_UNKNOWN
        provenance = PROVENANCE_UNKNOWN
    else:
        pattern = _pattern_for(event_count, affected_fraction)
        tier = max(_materiality_tier(e.kind) for e in zone_events)
        severity, usability = _SEVERITY_TABLE[(tier, pattern)]
        # Dominant kinds: the kind(s) at the tier that decided this cell,
        # sorted for determinism -- never a raw dump of every event.
        dominant_kinds = tuple(sorted({e.kind for e in zone_events if _materiality_tier(e.kind) == tier}))
        confidences = [_event_confidence_category(e.kind, e.confidence) for e in zone_events]
        confidence = CONFIDENCE_SUPPORTED if CONFIDENCE_SUPPORTED in confidences else CONFIDENCE_WEAK
        real_kinds_visual = any(e.kind in (HIGH_MATERIALITY_KINDS | MODERATE_MATERIALITY_KINDS | LOW_MATERIALITY_KINDS) for e in zone_events)
        provenance = PROVENANCE_VISUAL_SIGNAL if real_kinds_visual else PROVENANCE_DETERMINISTIC_RULE

    if conflict:
        severity = SEVERITY_MIXED

    return ZoneUsabilityResult(
        zone=zone, zone_usability=usability, zone_severity=severity,
        event_count=event_count, event_duration_total_sec=event_duration_total,
        zone_duration_sec=round(zone_duration, 3), affected_fraction=affected_fraction,
        isolated_event=(pattern == PATTERN_ISOLATED), repeated_defect=(pattern == PATTERN_REPEATED),
        sustained_defect=(pattern == PATTERN_SUSTAINED), dominant_event_kinds=dominant_kinds,
        confidence=confidence, provenance=provenance, zone_conflict=conflict,
    )


_OVERALL_RANK = {
    USABILITY_UNUSABLE: 0, USABILITY_IMPAIRED: 1, USABILITY_UNKNOWN: 2,
    USABILITY_QUESTIONABLE: 2, USABILITY_USABLE: 3,
}


def _overall_usability_v2(entry: ZoneUsabilityResult, delivery: ZoneUsabilityResult, exit_: ZoneUsabilityResult) -> str:
    """Boundary Firewall (binding): ENTRY/EXIT severity, however high,
    NEVER by itself produces a whole-take rejection -- their own
    contribution to `overall_usability` is capped at QUESTIONABLE. Only
    DELIVERY may propagate IMPAIRED/UNUSABLE to the overall result,
    mirroring D-157's own `_overall_usability` doctrine exactly (CASE A
    ownership, restated for the 5-state vocabulary)."""
    if delivery.zone_usability == USABILITY_UNUSABLE:
        return USABILITY_UNUSABLE
    if delivery.zone_usability == USABILITY_IMPAIRED:
        return USABILITY_IMPAIRED
    if delivery.zone_usability == USABILITY_UNKNOWN:
        return USABILITY_UNKNOWN
    entry_capped = entry.zone_usability in (USABILITY_QUESTIONABLE, USABILITY_IMPAIRED, USABILITY_UNUSABLE)
    exit_capped = exit_.zone_usability in (USABILITY_QUESTIONABLE, USABILITY_IMPAIRED, USABILITY_UNUSABLE)
    if entry_capped or exit_capped:
        return USABILITY_QUESTIONABLE
    return USABILITY_USABLE


def _case_for_v2(entry: ZoneUsabilityResult, delivery: ZoneUsabilityResult, exit_: ZoneUsabilityResult) -> str:
    """CASE A/B/C classification (D-107/D-115 doctrine), reusing D-163's
    own constants verbatim -- never re-derived with new semantics. IMPAIRED
    is treated identically to UNUSABLE for case purposes (both mean
    "DELIVERY-owned defect evidence exists")."""
    if delivery.zone_usability in (USABILITY_UNUSABLE, USABILITY_IMPAIRED):
        return CASE_B_DELIVERY_OWNED
    if delivery.zone_usability == USABILITY_UNKNOWN:
        return CASE_C_AMBIGUOUS
    if entry.zone_usability in (USABILITY_QUESTIONABLE, USABILITY_IMPAIRED, USABILITY_UNUSABLE) or \
            exit_.zone_usability in (USABILITY_QUESTIONABLE, USABILITY_IMPAIRED, USABILITY_UNUSABLE):
        return CASE_A_BOUNDARY_ONLY
    return CASE_CLEAN


def build_zone_usability_v2(candidate_id: str, span: RawUnderstandingSpan | None) -> CandidateZoneUsabilityV2 | None:
    """Pure projection from an already-computed `RawUnderstandingSpan`
    (D-155, unchanged). Returns `None` when no span exists -- fail-open,
    identical contract to D-163's own `build_watch_listen_besttake_
    evidence`."""
    if span is None:
        return None
    entry = _build_zone_result(span, ZONE_ENTRY)
    delivery = _build_zone_result(span, ZONE_DELIVERY)
    exit_ = _build_zone_result(span, ZONE_EXIT)
    return CandidateZoneUsabilityV2(
        candidate_id=candidate_id, entry=entry, delivery=delivery, exit=exit_,
        overall_usability=_overall_usability_v2(entry, delivery, exit_),
        case_classification=_case_for_v2(entry, delivery, exit_),
    )


# ---------------------------------------------------------------------------
# Performance dominance V2 -- an ordinal comparison over the refined 5-state
# usability scale, mirroring D-163's own `_performance_dominates` shape
# exactly (no numeric threshold, partial-order only). NOT wired to any
# authority; provided so a future, separately-authorized task can compare
# it against D-163's V1-based comparison without re-deriving the contract.
# ---------------------------------------------------------------------------
_USABILITY_RANK_V2 = {
    USABILITY_UNUSABLE: 0, USABILITY_IMPAIRED: 1, USABILITY_UNKNOWN: 2,
    USABILITY_QUESTIONABLE: 2, USABILITY_USABLE: 3,
}


def _no_worse_v2(a: str, b: str) -> bool:
    return _USABILITY_RANK_V2.get(b, 2) >= _USABILITY_RANK_V2.get(a, 2)


def zone_usability_v2_dominates(alternative: CandidateZoneUsabilityV2, winner: CandidateZoneUsabilityV2) -> bool:
    if winner.delivery.zone_conflict or alternative.delivery.zone_conflict:
        return False
    if winner.case_classification == CASE_A_BOUNDARY_ONLY:
        return False
    if not (
        _no_worse_v2(winner.entry.zone_usability, alternative.entry.zone_usability)
        and _no_worse_v2(winner.delivery.zone_usability, alternative.delivery.zone_usability)
        and _no_worse_v2(winner.exit.zone_usability, alternative.exit.zone_usability)
    ):
        return False
    return _USABILITY_RANK_V2.get(alternative.delivery.zone_usability, 2) > _USABILITY_RANK_V2.get(winner.delivery.zone_usability, 2)


# ---------------------------------------------------------------------------
# Diagnostics (tail-safe, counts-only). No transcript. No family/clip ids
# in the top-level aggregate counts.
# ---------------------------------------------------------------------------
def candidate_zone_usability_v2_row(result: CandidateZoneUsabilityV2) -> dict:
    """Per-candidate diagnostic row -- the exact field names this task's
    directive requires."""
    return {
        "watch_listen_zone_usability_v2": result.overall_usability,
        "watch_listen_entry_severity": result.entry.zone_severity,
        "watch_listen_delivery_severity": result.delivery.zone_severity,
        "watch_listen_exit_severity": result.exit.zone_severity,
        "watch_listen_delivery_event_count": result.delivery.event_count,
        "watch_listen_delivery_event_duration_sec": result.delivery.event_duration_total_sec,
        "watch_listen_delivery_duration_sec": result.delivery.zone_duration_sec,
        "watch_listen_delivery_affected_fraction": result.delivery.affected_fraction,
        "watch_listen_delivery_pattern": (
            PATTERN_SUSTAINED if result.delivery.sustained_defect else
            PATTERN_REPEATED if result.delivery.repeated_defect else
            PATTERN_ISOLATED if result.delivery.isolated_event else PATTERN_NONE
        ),
        "watch_listen_dominant_event_kinds": list(result.delivery.dominant_event_kinds),
        "watch_listen_zone_conflict": (result.entry.zone_conflict or result.delivery.zone_conflict or result.exit.zone_conflict),
    }


def zone_usability_v2_diagnostics(results: Iterable[CandidateZoneUsabilityV2]) -> dict:
    """Tail-safe run-level summary -- same pattern as D-119/D-125/D-152/
    D-155/D-157/D-163's own compact summaries. No family/clip ids here
    (per-candidate rows carry `candidate_id`; this aggregate never does)."""
    results = tuple(results)
    counts = {
        "usable_count": 0, "questionable_count": 0, "impaired_count": 0,
        "unusable_count": 0, "unknown_count": 0,
        "isolated_defect_count": 0, "repeated_defect_count": 0, "sustained_defect_count": 0,
    }
    key_for_usability = {
        USABILITY_USABLE: "usable_count", USABILITY_QUESTIONABLE: "questionable_count",
        USABILITY_IMPAIRED: "impaired_count", USABILITY_UNUSABLE: "unusable_count",
        USABILITY_UNKNOWN: "unknown_count",
    }
    for result in results:
        counts[key_for_usability.get(result.overall_usability, "unknown_count")] += 1
        if result.delivery.isolated_event:
            counts["isolated_defect_count"] += 1
        if result.delivery.repeated_defect:
            counts["repeated_defect_count"] += 1
        if result.delivery.sustained_defect:
            counts["sustained_defect_count"] += 1
    return {"schema_version": SCHEMA_VERSION, "candidate_count": len(results), **counts}
