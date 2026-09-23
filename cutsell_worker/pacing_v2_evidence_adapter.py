"""D-217 -- PACING V2 REAL EVIDENCE-SOURCE WIRING. OFFLINE / DIAGNOSTIC ONLY.

Closes D-216's own named gap: at today's live Pacing seam, `decide_
transition` (D-215) never receives a relationship hint, Prosodic edge
evidence, or a candidate lead/tail/overlap timing window, so every live
result collapses to `HARD_CUT`/`TIGHT_CUT`/`KEEP_PAUSE`/`UNKNOWN`. This
module supplies exactly those three optional evidence sources FROM
ALREADY-COMPUTED, ALREADY-SERIALIZED real pipeline objects -- no ASR
rerun, no P1/P2/Prosodic recompute, no new provider call, no new timing
constant. It is a pure EVIDENCE ADAPTER: `pacing_v2_live_diagnostics_
integration.py` (D-216, CLOSED, untouched by this module) remains the
one diagnostics builder and `pacing_transition_decision.decide_
transition` (D-215, CLOSED, untouched) remains the one decision engine.
This module invents no new transition classifier, no new story logic, no
new pacing score -- it only maps pair -> evidence.

## Part 1 -- relationship hint (CONTINUATION/CORRECTION/RETRY/UNKNOWN)

Source: `draft.diagnostics["editorial_moment_sequence"]["moments"]` --
the ALREADY-SERIALIZED, flattened, per-source-ordered list D-198/D-199
already attach when the SEPARATE `CUTSELL_EDITORIAL_MOMENT_SEQUENCE_
DIAGNOSTICS_ENABLED` flag is on (this module never turns that flag on
itself, and never rebuilds the underlying `EditorialMomentUnderstanding`
Python objects -- those are local to `pipeline.py::build_flow_b_draft`
and do not survive to this later, post-Boundary seam; only their JSON-
safe diagnostics projection does, so that projection -- not a live
object -- is this module's one input). Each row already carries
`relation_to_predecessor` (D-198's own "SINGLE SOURCE OF TRUTH" --
D-197's grouper's own already-resolved value, never re-derived) and
`attempt_ids` (D-194's own stable-id reference list).

Mapping contract: a final selected `DraftClip` is matched to AT MOST ONE
moment row via `clip.attempt_id in row["attempt_ids"]` (never by time
proximity). A `left`/`right` pair's hint is reused ONLY when: (a) both
clips resolve to EXACTLY ONE moment row (zero or multiple matches is
UNRESOLVED, never guessed), (b) both rows share the same `source_asset_
id` as the clips themselves, and (c) `right`'s row is literally the very
next row after `left`'s row IN P1'S OWN LOCAL-SEQUENCE ORDER (index
adjacency in the per-source row list, never nearness in time) -- if
anything upstream (Boundary, BestTake, discard) removed a moment between
them, that break is honestly reported, never silently bridged. Only
`relation_to_predecessor in (RETRY, CORRECTION, CONTINUATION)` maps to a
`pacing_transition_decision` hint; `COMPLEMENTARY`/`NEW_AUDIENCE_BEAT`/
`DISTINCT_PROPOSITION`/`UNCERTAIN`/`None` all map to `None` (no
restriction -- the same "no relation asserted" default D-215 already
applies to a plain `None` hint).

## Part 2 -- Prosodic edge evidence

Source: `draft.diagnostics["take_judge_groups"][*]["prosodic_pipeline_
candidate_evidence"]` -- the ALREADY-COMPUTED, per-candidate D-187
diagnostics dict D-189/D-190 attach to each finalist-arbitration family
row, when the SEPARATE `CUTSELL_PROSODIC_FINALIST_ARBITER_DIAGNOSTICS_
ENABLED` flag (and the bounded finalist arbiter itself) are on AND that
family was D-184-eligible (2-3 candidates, a decisive terminal state).
This module never calls `analyze_prosodic_delivery` and never decodes
audio -- it only looks up, by `clip_id`, a dict D-187/D-188/D-189 already
built. Coverage is HONESTLY PARTIAL by construction: only candidates
whose family underwent finalist arbitration (and whose clip_id survived
into the final selected sequence, e.g. the winner) ever have an entry;
every other selected clip's Prosodic status is `UNAVAILABLE`, reported as
such, never invented.

## Prosodic edge adapter

`_ProsodicEdgeEvidence` carries ONLY the two transition-local safe fields
D-215's own `_prosody_supports_overlap` reads (`restart_or_interruption_
state`, `vocal_continuity_state`), duck-typed to satisfy `decide_
transition`'s `left_prosody`/`right_prosody` parameter (which only ever
reads those two attributes via `getattr`, per `pacing_transition_
decision.py`'s own docstring) -- copied verbatim from the already-
computed `prosodic_delivery_diagnostics()` dict, never recomputed, never
a BestTake ranking signal, never an emotion inference.

## Part 3 -- candidate timing windows (no magic durations)

Pure word-timing geometry over the two already-selected, already-
Boundary-finalized `DraftClip`s themselves -- no external evidence
object needed, no audio decode, no source re-probe:

    available_right_silent_head = max(0, min(w.start for w in right.words) - right.start)
    available_left_silent_tail  = max(0, right.end available -- see below)

i.e. the room, ALREADY INSIDE each clip's own selected `[start, end]`
span, between the visual cut point and the nearest real spoken word.
`None` (never `0.0`) when a side has no word timing at all -- the
WORD-TIMING FIREWALL this task requires: missing timing is UNKNOWN, never
asserted safe. Both derived bounds are, by construction, `<= (clip.end -
clip.start)` -- they can never reach outside the clip's own already-
Boundary-approved span, so the SOURCE AVAILABILITY firewall (never
extract outside source bounds) and the BOUNDARY FIREWALL (never touch
`.start`/`.end`) are both structural, not runtime checks. No `J_CUT_MS`/
`L_CUT_MS`/`OVERLAP_MS`/`IDEAL_GAP_MS` or any other duration constant is
introduced -- the ENTIRE available window (not a fraction of it, not a
fixed cap) is always offered as the candidate; `pacing_transition_
decision.decide_transition` (D-215, unmodified) remains the ONE authority
that re-validates word/meaning/double-speech safety on top of it and
decides whether to actually use it.

The MEANING FIREWALL this task also names is not re-implemented here --
`decide_transition` already reuses `semantic_claims.classify_claim`
(D-038) verbatim on whatever window it is offered; duplicating that
check here would be the "new transition classifier" this task's own
"NO DUPLICATE SEMANTIC ENGINE" rule forbids.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

from .contracts import DraftClip
from .dialogue_pacing_transition import J_CUT, L_CUT, MICRO_AUDIO_OVERLAP
from .language_proposition_relation import (
    RELATION_CONTINUATION,
    RELATION_CORRECTION,
    RELATION_RETRY,
)
from .pacing_transition_decision import (
    RELATIONSHIP_CONTINUATION,
    RELATIONSHIP_CORRECTION,
    RELATIONSHIP_RETRY,
)
from .pacing_v2_live_diagnostics_integration import build_pacing_v2_live_diagnostics

SCHEMA_VERSION = "cutsell.pacing_v2_evidence_adapter.v1"

# --- relationship-hint mapping/source vocabulary ----------------------------
RELATIONSHIP_SOURCE_P1_LOCAL_SEQUENCE = "P1_LOCAL_SEQUENCE_MOMENT_RELATION"
RELATIONSHIP_SOURCE_NONE = "NONE"

RELATIONSHIP_MAPPING_MATCHED = "MATCHED"
RELATIONSHIP_MAPPING_NO_UNDERSTANDING = "NO_UNDERSTANDING_SUPPLIED"
RELATIONSHIP_MAPPING_CLIP_NOT_IDENTIFIED = "CLIP_NOT_IDENTIFIED"
RELATIONSHIP_MAPPING_AMBIGUOUS_IDENTITY = "AMBIGUOUS_IDENTITY_MATCH"
RELATIONSHIP_MAPPING_NOT_ADJACENT_IN_P1 = "NOT_ADJACENT_IN_P1_LOCAL_SEQUENCE"
RELATIONSHIP_MAPPING_CROSS_SOURCE = "CROSS_SOURCE_PAIR"
RELATIONSHIP_MAPPING_NO_RELATION_RESOLVED = "NO_RELATION_RESOLVED"

_RELATION_TO_HINT = {
    RELATION_RETRY: RELATIONSHIP_RETRY,
    RELATION_CORRECTION: RELATIONSHIP_CORRECTION,
    RELATION_CONTINUATION: RELATIONSHIP_CONTINUATION,
}

# --- Prosodic mapping vocabulary --------------------------------------------
PROSODIC_SOURCE_TAKE_JUDGE_GROUPS = "TAKE_JUDGE_GROUPS_FINALIST_EVIDENCE"
PROSODIC_SOURCE_NONE = "NONE"
PROSODIC_STATUS_AVAILABLE = "AVAILABLE"
PROSODIC_STATUS_UNAVAILABLE = "UNAVAILABLE"
PROSODIC_PAIR_BOTH_AVAILABLE = "BOTH_AVAILABLE"
PROSODIC_PAIR_PARTIAL = "PARTIAL"
PROSODIC_PAIR_UNAVAILABLE = "UNAVAILABLE"

# --- candidate-timing vocabulary --------------------------------------------
TIMING_SOURCE_WORD_GEOMETRY = "WORD_TIMING_GEOMETRY_WITHIN_SELECTED_CLIP_BOUNDS"
CANDIDATE_TIMING_AVAILABLE = "AVAILABLE"
CANDIDATE_TIMING_NONE_SAFE = "NO_SAFE_WINDOW"
CANDIDATE_TIMING_UNKNOWN = "UNKNOWN_MISSING_WORD_TIMING"


class _ProsodicEdgeEvidence:
    """Duck-typed shim exposing ONLY the two fields `pacing_transition_
    decision._prosody_supports_overlap` reads via `getattr` -- copied
    verbatim from an already-computed `prosodic_delivery_diagnostics()`
    dict, never recomputed. Never a BestTake score, never a winner rank,
    never an emotion inference."""

    __slots__ = ("restart_or_interruption_state", "vocal_continuity_state")

    def __init__(self, restart_or_interruption_state: Optional[str], vocal_continuity_state: Optional[str]):
        self.restart_or_interruption_state = restart_or_interruption_state
        self.vocal_continuity_state = vocal_continuity_state


def _pair_key(left_id: str, right_id: str) -> Tuple[str, str]:
    return (left_id, right_id)


# ---------------------------------------------------------------------------
# Part 1 -- relationship hint
# ---------------------------------------------------------------------------

def _moments_by_source(editorial_moment_sequence_diagnostics: Optional[Mapping]) -> dict:
    moments = (editorial_moment_sequence_diagnostics or {}).get("moments") or ()
    by_source: dict = {}
    for row in moments:
        by_source.setdefault(row.get("source_asset_id"), []).append(row)
    return by_source


def _find_moment_index(clip: DraftClip, moment_rows: Sequence[Mapping]) -> Tuple[Optional[int], bool]:
    """Returns (index_or_None, ambiguous). `ambiguous=True` means more
    than one moment row claimed the same `attempt_id` -- reported
    distinctly from "no identity at all"/"zero matches", but both
    collapse to an unresolved (`None`) index -- never guessed."""
    attempt_id = getattr(clip, "attempt_id", None)
    if not attempt_id:
        return None, False
    matches = [i for i, row in enumerate(moment_rows) if attempt_id in (row.get("attempt_ids") or ())]
    if len(matches) == 1:
        return matches[0], False
    return None, len(matches) > 1


def relationship_hint_for_pair(
    left: DraftClip, right: DraftClip, moments_by_source: Mapping,
) -> dict:
    """Returns {"relationship_hint", "relationship_source",
    "relationship_mapping_status"}. `relationship_hint` is `None` (no
    restriction) unless a real, unambiguous, P1-local-sequence-adjacent
    RETRY/CORRECTION/CONTINUATION relation was found."""
    if left.source_asset_id != right.source_asset_id:
        return {
            "relationship_hint": None, "relationship_source": RELATIONSHIP_SOURCE_NONE,
            "relationship_mapping_status": RELATIONSHIP_MAPPING_CROSS_SOURCE,
        }
    rows = moments_by_source.get(left.source_asset_id)
    if not rows:
        return {
            "relationship_hint": None, "relationship_source": RELATIONSHIP_SOURCE_NONE,
            "relationship_mapping_status": RELATIONSHIP_MAPPING_NO_UNDERSTANDING,
        }
    left_index, left_ambiguous = _find_moment_index(left, rows)
    right_index, right_ambiguous = _find_moment_index(right, rows)
    if left_index is None or right_index is None:
        status = (
            RELATIONSHIP_MAPPING_AMBIGUOUS_IDENTITY if (left_ambiguous or right_ambiguous)
            else RELATIONSHIP_MAPPING_CLIP_NOT_IDENTIFIED
        )
        return {
            "relationship_hint": None, "relationship_source": RELATIONSHIP_SOURCE_NONE,
            "relationship_mapping_status": status,
        }
    if right_index != left_index + 1:
        return {
            "relationship_hint": None, "relationship_source": RELATIONSHIP_SOURCE_NONE,
            "relationship_mapping_status": RELATIONSHIP_MAPPING_NOT_ADJACENT_IN_P1,
        }
    raw_relation = rows[right_index].get("relation_to_predecessor")
    hint = _RELATION_TO_HINT.get(raw_relation)
    status = RELATIONSHIP_MAPPING_MATCHED if raw_relation is not None else RELATIONSHIP_MAPPING_NO_RELATION_RESOLVED
    return {
        "relationship_hint": hint,
        "relationship_source": RELATIONSHIP_SOURCE_P1_LOCAL_SEQUENCE,
        "relationship_mapping_status": status,
    }


# ---------------------------------------------------------------------------
# Part 2 -- Prosodic edge evidence
# ---------------------------------------------------------------------------

def _prosodic_diagnostics_by_clip_id(take_judge_groups: Sequence[Mapping]) -> dict:
    by_clip_id: dict = {}
    for row in take_judge_groups or ():
        candidate_evidence = row.get("prosodic_pipeline_candidate_evidence") or {}
        for clip_id, diag in candidate_evidence.items():
            by_clip_id.setdefault(clip_id, diag)
    return by_clip_id


def prosodic_evidence_for_clip(clip: DraftClip, prosodic_diagnostics_by_clip_id: Mapping) -> dict:
    """Returns {"evidence": _ProsodicEdgeEvidence|None, "status": AVAILABLE|
    UNAVAILABLE, "source": ...}."""
    diag = prosodic_diagnostics_by_clip_id.get(clip.clip_id)
    if diag is None:
        return {"evidence": None, "status": PROSODIC_STATUS_UNAVAILABLE, "source": PROSODIC_SOURCE_NONE}
    evidence = _ProsodicEdgeEvidence(
        restart_or_interruption_state=diag.get("prosodic_restart_state"),
        vocal_continuity_state=diag.get("prosodic_continuity_state"),
    )
    return {"evidence": evidence, "status": PROSODIC_STATUS_AVAILABLE, "source": PROSODIC_SOURCE_TAKE_JUDGE_GROUPS}


# ---------------------------------------------------------------------------
# Part 3 -- candidate timing windows
# ---------------------------------------------------------------------------

def available_silent_head_sec(clip: DraftClip) -> Optional[float]:
    """Room, INSIDE `clip`'s own already-selected span, between `clip.
    start` (the visual cut point) and its earliest real spoken word.
    `None` (UNKNOWN, never `0.0`) when there is no word timing at all."""
    if not clip.words:
        return None
    first_word_start = min(float(w.start) for w in clip.words)
    return max(0.0, first_word_start - float(clip.start))


def available_silent_tail_sec(clip: DraftClip) -> Optional[float]:
    """Room, INSIDE `clip`'s own already-selected span, between its
    latest real spoken word and `clip.end`. `None` (UNKNOWN, never
    `0.0`) when there is no word timing at all."""
    if not clip.words:
        return None
    last_word_end = max(float(w.end) for w in clip.words)
    return max(0.0, float(clip.end) - last_word_end)


def candidate_timing_for_pair(left: DraftClip, right: DraftClip) -> dict:
    """Returns the full timing-diagnostics row for one adjacent pair:
    `available_left_silent_tail`, `available_right_silent_head`, the
    resulting `lead`/`tail`/`micro_overlap` candidates (identical to the
    available windows themselves -- no fraction, no cap, no invented
    duration), and `candidate_timing_status`."""
    left_tail = available_silent_tail_sec(left)
    right_head = available_silent_head_sec(right)
    lead = right_head
    tail = left_tail
    if lead is not None and tail is not None and lead > 0 and tail > 0:
        micro_overlap = min(lead, tail)
    else:
        micro_overlap = None
    if left_tail is None and right_head is None:
        status = CANDIDATE_TIMING_UNKNOWN
    elif (left_tail or 0.0) <= 0.0 and (right_head or 0.0) <= 0.0:
        status = CANDIDATE_TIMING_NONE_SAFE
    else:
        status = CANDIDATE_TIMING_AVAILABLE
    return {
        "available_left_silent_tail": left_tail,
        "available_right_silent_head": right_head,
        "candidate_audio_lead": lead,
        "candidate_audio_tail": tail,
        "candidate_micro_overlap": micro_overlap,
        "candidate_timing_status": status,
        "candidate_timing_source": TIMING_SOURCE_WORD_GEOMETRY,
    }


# ---------------------------------------------------------------------------
# Orchestration -- builds the three evidence dicts D-216's own (unmodified)
# `build_pacing_v2_live_diagnostics` already accepts, plus the extra per-
# pair mapping/coverage diagnostics this task requires.
# ---------------------------------------------------------------------------

def build_pacing_v2_real_evidence(
    selected: Sequence[DraftClip],
    *,
    editorial_moment_sequence_diagnostics: Optional[Mapping] = None,
    take_judge_groups: Sequence[Mapping] = (),
) -> dict:
    """Pure evidence derivation, no decision. Returns:
    `relationship_hint_by_pair`, `prosody_by_clip_id` (ready to hand
    straight to `pacing_v2_live_diagnostics_integration.build_pacing_v2_
    live_diagnostics`'s own existing optional kwargs, unmodified), plus
    `candidate_timing_by_pair` and a bounded `pair_evidence_diagnostics`
    list (one row per adjacent pair, index-aligned with that function's
    own `transitions` rows) carrying the mapping-status/coverage fields
    this task's own DIAGNOSTICS section requires."""
    clips = tuple(selected)
    moments_by_source = _moments_by_source(editorial_moment_sequence_diagnostics)
    prosodic_by_clip_id = _prosodic_diagnostics_by_clip_id(take_judge_groups)

    relationship_hint_by_pair: dict = {}
    candidate_timing_by_pair: dict = {}
    prosody_by_clip_id: dict = {}
    pair_rows: list = []

    for index in range(len(clips) - 1):
        left, right = clips[index], clips[index + 1]
        key = _pair_key(left.clip_id, right.clip_id)

        relation = relationship_hint_for_pair(left, right, moments_by_source)
        relationship_hint_by_pair[key] = relation["relationship_hint"]

        left_prosody = prosodic_evidence_for_clip(left, prosodic_by_clip_id)
        right_prosody = prosodic_evidence_for_clip(right, prosodic_by_clip_id)
        if left_prosody["evidence"] is not None:
            prosody_by_clip_id[left.clip_id] = left_prosody["evidence"]
        if right_prosody["evidence"] is not None:
            prosody_by_clip_id[right.clip_id] = right_prosody["evidence"]
        if left_prosody["status"] == PROSODIC_STATUS_AVAILABLE and right_prosody["status"] == PROSODIC_STATUS_AVAILABLE:
            pair_prosodic_status = PROSODIC_PAIR_BOTH_AVAILABLE
        elif left_prosody["status"] == PROSODIC_STATUS_AVAILABLE or right_prosody["status"] == PROSODIC_STATUS_AVAILABLE:
            pair_prosodic_status = PROSODIC_PAIR_PARTIAL
        else:
            pair_prosodic_status = PROSODIC_PAIR_UNAVAILABLE

        timing = candidate_timing_for_pair(left, right)
        candidate_timing_by_pair[key] = {
            "lead": timing["candidate_audio_lead"], "tail": timing["candidate_audio_tail"],
        }

        pair_rows.append({
            "left_clip_id": left.clip_id, "right_clip_id": right.clip_id,
            "relationship_hint": relation["relationship_hint"],
            "relationship_source": relation["relationship_source"],
            "relationship_mapping_status": relation["relationship_mapping_status"],
            "left_prosodic_status": left_prosody["status"],
            "right_prosodic_status": right_prosody["status"],
            "prosodic_mapping_status": pair_prosodic_status,
            "prosodic_source": PROSODIC_SOURCE_TAKE_JUDGE_GROUPS,
            **timing,
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "relationship_hint_by_pair": relationship_hint_by_pair,
        "prosody_by_clip_id": prosody_by_clip_id,
        "candidate_timing_by_pair": candidate_timing_by_pair,
        "pair_evidence_diagnostics": pair_rows,
    }


# ---------------------------------------------------------------------------
# Orchestration entry point -- the ONE new live call site this task adds.
# Composes this module's own evidence derivation with D-216's own
# (unmodified, CLOSED) `build_pacing_v2_live_diagnostics`, then merges the
# per-pair evidence-coverage fields into its already-returned rows and adds
# this task's own required run-summary counts. `build_pacing_v2_live_
# diagnostics` itself is called exactly as D-216 already calls it -- same
# function, same signature, same decision engine underneath (D-215's
# `decide_transition`, still untouched) -- only the three optional evidence
# kwargs it already accepted now carry REAL values instead of always `None`.
# ---------------------------------------------------------------------------

_ADVANCED_MODES = (J_CUT, L_CUT, MICRO_AUDIO_OVERLAP)


def build_pacing_v2_live_diagnostics_with_real_evidence(
    selected: Sequence[DraftClip],
    *,
    dialogue_overlap_enabled: bool,
    boundary_diagnostics: Optional[Mapping] = None,
    live_transition_modes: Sequence[str] = (),
    editorial_moment_sequence_diagnostics: Optional[Mapping] = None,
    take_judge_groups: Sequence[Mapping] = (),
) -> dict:
    """The one D-217 live entry point. Derives real relationship-hint/
    Prosodic/candidate-timing evidence (this module, above) then calls
    D-216's own unmodified `build_pacing_v2_live_diagnostics` with it --
    never a second decision path. Returns that same dict shape, with each
    `transitions[i]` row additionally carrying this task's own evidence-
    coverage diagnostics (`relationship_source`, `relationship_mapping_
    status`, `prosodic_mapping_status`, `available_left_silent_tail`,
    `available_right_silent_head`, `candidate_timing_status`,
    `candidate_micro_overlap`) merged in by pair index, plus this task's
    own extra run-summary counts."""
    evidence = build_pacing_v2_real_evidence(
        selected,
        editorial_moment_sequence_diagnostics=editorial_moment_sequence_diagnostics,
        take_judge_groups=take_judge_groups,
    )
    result = build_pacing_v2_live_diagnostics(
        selected,
        dialogue_overlap_enabled=dialogue_overlap_enabled,
        boundary_diagnostics=boundary_diagnostics,
        live_transition_modes=live_transition_modes,
        prosody_by_clip_id=evidence["prosody_by_clip_id"],
        relationship_hint_by_pair=evidence["relationship_hint_by_pair"],
        candidate_timing_by_pair=evidence["candidate_timing_by_pair"],
    )
    pair_rows = evidence["pair_evidence_diagnostics"]
    transitions = list(result.get("transitions") or ())
    merged_transitions = []
    for i, row in enumerate(transitions):
        merged = dict(row)
        if i < len(pair_rows):
            extra = pair_rows[i]
            merged.update({
                "relationship_source": extra["relationship_source"],
                "relationship_mapping_status": extra["relationship_mapping_status"],
                "left_prosodic_status": extra["left_prosodic_status"],
                "right_prosodic_status": extra["right_prosodic_status"],
                "prosodic_mapping_status": extra["prosodic_mapping_status"],
                "available_left_silent_tail": extra["available_left_silent_tail"],
                "available_right_silent_head": extra["available_right_silent_head"],
                "candidate_micro_overlap": extra["candidate_micro_overlap"],
                "candidate_timing_status": extra["candidate_timing_status"],
                "candidate_timing_source": extra["candidate_timing_source"],
            })
        merged_transitions.append(merged)

    j_cut_eligible = sum(1 for row in merged_transitions if row.get("selected_mode") == J_CUT)
    l_cut_eligible = sum(1 for row in merged_transitions if row.get("selected_mode") == L_CUT)
    micro_overlap_eligible = sum(1 for row in merged_transitions if row.get("selected_mode") == MICRO_AUDIO_OVERLAP)
    candidate_j_lead_available = sum(
        1 for row in merged_transitions if (row.get("candidate_audio_lead") or 0) > 0
    )
    candidate_l_tail_available = sum(
        1 for row in merged_transitions if (row.get("candidate_audio_tail") or 0) > 0
    )
    candidate_micro_overlap_available = sum(
        1 for row in merged_transitions if (row.get("candidate_micro_overlap") or 0) > 0
    )
    candidate_timing_unavailable = sum(
        1 for row in merged_transitions if row.get("candidate_timing_status") == CANDIDATE_TIMING_UNKNOWN
    )
    relationship_hint_available = sum(1 for row in merged_transitions if row.get("relationship_hint"))
    relationship_hint_unknown = len(merged_transitions) - relationship_hint_available
    prosodic_left_available = sum(1 for row in merged_transitions if row.get("left_prosodic_status") == PROSODIC_STATUS_AVAILABLE)
    prosodic_right_available = sum(1 for row in merged_transitions if row.get("right_prosodic_status") == PROSODIC_STATUS_AVAILABLE)
    prosodic_pair_available = sum(1 for row in merged_transitions if row.get("prosodic_mapping_status") == PROSODIC_PAIR_BOTH_AVAILABLE)
    prosodic_pair_partial = sum(1 for row in merged_transitions if row.get("prosodic_mapping_status") == PROSODIC_PAIR_PARTIAL)
    prosodic_pair_unavailable = sum(1 for row in merged_transitions if row.get("prosodic_mapping_status") == PROSODIC_PAIR_UNAVAILABLE)

    merged_result = dict(result)
    merged_result["transitions"] = merged_transitions
    merged_result["evidence_schema_version"] = SCHEMA_VERSION
    merged_result["run_summary"] = {
        **(result.get("run_summary") or {}),
        "relationship_hint_available_count": relationship_hint_available,
        "relationship_hint_unknown_count": relationship_hint_unknown,
        "prosodic_left_available_count": prosodic_left_available,
        "prosodic_right_available_count": prosodic_right_available,
        "prosodic_pair_available_count": prosodic_pair_available,
        "prosodic_pair_partial_count": prosodic_pair_partial,
        "prosodic_pair_unavailable_count": prosodic_pair_unavailable,
        "candidate_j_lead_available_count": candidate_j_lead_available,
        "candidate_l_tail_available_count": candidate_l_tail_available,
        "candidate_micro_overlap_available_count": candidate_micro_overlap_available,
        "candidate_timing_unavailable_count": candidate_timing_unavailable,
        "advanced_mode_eligible_count": j_cut_eligible + l_cut_eligible + micro_overlap_eligible,
        "j_cut_eligible_count": j_cut_eligible,
        "l_cut_eligible_count": l_cut_eligible,
        "micro_overlap_eligible_count": micro_overlap_eligible,
    }
    return merged_result
