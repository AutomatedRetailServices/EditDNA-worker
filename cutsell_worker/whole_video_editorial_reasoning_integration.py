"""D-203: P2 Whole-Video Editorial Reasoning -- Phase B, CANONICAL EVIDENCE
LIVE DIAGNOSTIC INTEGRATION.

See ``docs/CUTSELL_DECISIONS.md`` D-202 (Phase A: typed foundation +
deterministic builders) and D-203 (this module) for full context. This
module is the ONE adapter layer between D-202's pure builders
(``whole_video_editorial_reasoning.py``, unchanged, zero modification) and
the REAL, already-computed per-source pipeline objects this task is
authorized to consume:

    EditorialMomentUnderstanding (D-195/D-197/D-198/D-200.3)  -- PRIMARY
        source of P1 moments/local_groups/sequence_hypotheses
    LiveLanguageSpineEvidence (D-199)                          -- PRIMARY
        source of canonical PropositionCandidate/RelationEvidence

## NO RECOMPUTE (this task's own instruction, enforced structurally)

This module imports NOTHING from ``asr.py``, ``language_spine.py``,
``language_utterance_attempt.py``'s own builders, ``language_proposition_
relation.py``'s own builders (``build_proposition_candidates``/``build_
relation_evidence``), ``watch_listen_understanding.py``'s own builders,
``local_performance.py``, ``prosodic_audio_v2.py``, ``whole_video_
openai.py``, or ``whole_video_analysis.py``. It calls no provider, decodes
no media, and re-derives no signal any upstream stage already computed --
it reads ONLY the already-built ``EditorialMomentUnderstanding``/
``LiveLanguageSpineEvidence`` objects a caller (``pipeline.py``) hands it,
exactly the same "read already-computed evidence, invent nothing" contract
``editorial_moment_sequence_integration.py`` (D-195) and ``language_spine_
live_integration.py`` (D-199) themselves already established.

## Mechanical pipeline position (D-201's own finding, restated and honored)

D-201 proved `apply_composite_resolution` (Family/CompositeResolver) runs
BEFORE P1's own diagnostic call site in ``pipeline.py``'s ``build_flow_b_
draft``. This module changes NOTHING about that order: it is called from
the SAME post-Family diagnostic block P1 (D-195) and the live Language
Spine (D-199) already occupy, as a THIRD diagnostic side-channel alongside
them -- never before Family, never inside Family/BestTake/Freeze
themselves. Any future re-ordering (P2 informing Family/Realization
construction) is a SEPARATE, not-yet-authorized integration gate.

## Flag dependency (explicit, not hidden -- this task's own instruction)

This module's own construction requires REAL P1 evidence, which only
exists when ``editorial_moment_sequence_diagnostics_enabled()`` (D-195's
flag) is also on. This is a REAL, load-bearing dependency, not invented
coupling -- it is reported HONESTLY via ``capability_status``/``missing_
evidence`` (``P1_EDITORIAL_MOMENT_SEQUENCE_DIAGNOSTICS_DISABLED``) rather
than silently auto-enabling that flag. The same honesty applies to the
live Language Spine flag (``live_language_spine_diagnostics_enabled()``,
D-199): when it is off, this module still builds regions from P1 moments/
local groups alone (proposition-free), reporting ``CAPABILITY_PARTIAL``
with ``LIVE_LANGUAGE_SPINE_DIAGNOSTICS_DISABLED`` rather than refusing to
run at all.

## No authority (restated, binding)

``build_whole_video_editorial_reasoning`` returns a read-only
``WholeVideoEditorialReasoningResult`` -- diagnostics/hypotheses only. It
never touches ``take_grouping``/``composite_resolver``/``realization_
resolver``/``bounded_finalist_arbiter``/``bounded_finalist_authority``/
``boundary_engine_pass``/``dialogue_pacing_transition``/render-plan
construction, and this module itself imports nothing from any of them
(module-leaf grep tests, ``tests/test_cutsell_d203_whole_video_editorial_
reasoning_integration.py``). **P2 HYPOTHESES DO NOT ALTER THE EDIT.**
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping, Sequence, Tuple

from .editorial_moment_sequence_integration import EditorialMomentUnderstanding
from .language_proposition_relation import PropositionCandidate
from .language_spine_live_integration import LiveLanguageSpineEvidence
from .whole_video_editorial_reasoning import (
    CAPABILITY_AVAILABLE,
    CAPABILITY_NOT_EVALUABLE,
    CAPABILITY_PARTIAL,
    WholeVideoEditorialUnderstanding,
    build_whole_video_editorial_understanding,
    whole_video_editorial_region_diagnostics,
    whole_video_editorial_understanding_run_summary,
    whole_video_proposition_realization_diagnostics,
    whole_video_supersession_diagnostics,
)

SCHEMA_VERSION = "cutsell.whole_video_editorial_reasoning_integration.v1"

_DIAGNOSTICS_ENV = "CUTSELL_WHOLE_VIDEO_EDITORIAL_REASONING_DIAGNOSTICS_ENABLED"

# ---------------------------------------------------------------------------
# Missing-evidence vocabulary (explicit, never silent auto-enable).
# ---------------------------------------------------------------------------
MISSING_P1_DIAGNOSTICS_DISABLED = "P1_EDITORIAL_MOMENT_SEQUENCE_DIAGNOSTICS_DISABLED"
MISSING_NO_P1_EVIDENCE = "NO_P1_EVIDENCE"
MISSING_LIVE_LANGUAGE_SPINE_DISABLED = "LIVE_LANGUAGE_SPINE_DIAGNOSTICS_DISABLED"


def _env_true_default_false(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def whole_video_editorial_reasoning_diagnostics_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Default OFF. When OFF, nothing in this module is ever called by
    ``pipeline.py`` -- zero P2 compute, selection/family/BestTake/D-191/
    Boundary/Pacing/render output stays byte-identical. There is no
    authority flag anywhere in this module."""
    values = env if env is not None else os.environ
    return _env_true_default_false(values.get(_DIAGNOSTICS_ENV))


@dataclass(frozen=True)
class WholeVideoEditorialReasoningResult:
    """Read-only Phase-B result. ``understanding`` is ``None`` only when
    P2 could not be evaluated at all (P1 diagnostics off, or zero P1
    evidence supplied) -- never a half-built object standing in for a
    real one."""
    understanding: WholeVideoEditorialUnderstanding | None
    capability_status: str
    missing_evidence: Tuple[str, ...]
    p2_source_count: int
    p2_missing_p1_source_count: int
    p2_missing_language_source_count: int


def _empty_result(*, missing_evidence: Tuple[str, ...], p2_missing_p1_source_count: int = 0) -> WholeVideoEditorialReasoningResult:
    return WholeVideoEditorialReasoningResult(
        understanding=None, capability_status=CAPABILITY_NOT_EVALUABLE, missing_evidence=missing_evidence,
        p2_source_count=0, p2_missing_p1_source_count=p2_missing_p1_source_count, p2_missing_language_source_count=0,
    )


def build_whole_video_editorial_reasoning(
    *,
    editorial_moment_understandings: Sequence[EditorialMomentUnderstanding] = (),
    live_language_spine_by_source: Mapping[str, LiveLanguageSpineEvidence] | None = None,
    p1_diagnostics_enabled: bool,
    live_language_spine_diagnostics_enabled: bool,
) -> WholeVideoEditorialReasoningResult:
    """The one canonical Phase-B integration entrypoint. Pure; consumes
    ONLY already-built P1/Language objects the caller passes in -- never
    recomputes ASR/Language Spine/Watch+Listen/visual/Prosodic/any
    provider itself. ``p1_diagnostics_enabled``/``live_language_spine_
    diagnostics_enabled`` are passed explicitly by the caller (rather than
    read from the environment here) so this function stays a pure,
    deterministic transform of its own arguments -- the caller
    (``pipeline.py``) is the one place flag state is actually read,
    exactly mirroring D-195/D-199's own call-site pattern."""
    live_language_spine_by_source = live_language_spine_by_source or {}

    if not p1_diagnostics_enabled:
        return _empty_result(missing_evidence=(MISSING_P1_DIAGNOSTICS_DISABLED,))

    moments_by_source: dict[str, Tuple] = {}
    local_groups_by_source: dict[str, Tuple] = {}
    sequences_by_source: dict[str, Tuple] = {}
    missing_p1_source_count = 0
    for understanding in editorial_moment_understandings:
        if not understanding.moments and not understanding.local_groups:
            missing_p1_source_count += 1
            continue
        moments_by_source[understanding.source_asset_id] = understanding.moments
        local_groups_by_source[understanding.source_asset_id] = understanding.local_groups
        sequences_by_source[understanding.source_asset_id] = understanding.sequence_hypotheses

    if not moments_by_source:
        return _empty_result(
            missing_evidence=(MISSING_NO_P1_EVIDENCE,), p2_missing_p1_source_count=missing_p1_source_count,
        )

    missing_evidence: list[str] = []
    proposition_candidates_by_id: dict[str, PropositionCandidate] = {}
    missing_language_source_count = 0
    if not live_language_spine_diagnostics_enabled:
        missing_evidence.append(MISSING_LIVE_LANGUAGE_SPINE_DISABLED)
        missing_language_source_count = len(moments_by_source)
    else:
        for source_asset_id in moments_by_source:
            evidence = live_language_spine_by_source.get(source_asset_id)
            if evidence is None or not evidence.proposition_candidates:
                missing_language_source_count += 1
                continue
            for proposition in evidence.proposition_candidates:
                proposition_candidates_by_id[proposition.proposition_candidate_id] = proposition

    understanding = build_whole_video_editorial_understanding(
        moments_by_source=moments_by_source,
        local_groups_by_source=local_groups_by_source,
        sequences_by_source=sequences_by_source,
        proposition_candidates_by_id=proposition_candidates_by_id,
    )

    if missing_evidence or missing_p1_source_count or missing_language_source_count:
        capability_status = CAPABILITY_PARTIAL
    else:
        capability_status = understanding.capability_status

    return WholeVideoEditorialReasoningResult(
        understanding=understanding,
        capability_status=capability_status,
        missing_evidence=tuple(missing_evidence),
        p2_source_count=len(moments_by_source),
        p2_missing_p1_source_count=missing_p1_source_count,
        p2_missing_language_source_count=missing_language_source_count,
    )


# ---------------------------------------------------------------------------
# Diagnostics / run summary -- tail-safe, counts/status-only, no transcript,
# no QA reference, no master score.
# ---------------------------------------------------------------------------
def whole_video_editorial_reasoning_diagnostics(result: WholeVideoEditorialReasoningResult) -> dict:
    """Bounded, JSON-safe diagnostics -- exactly the shape this task's own
    directive specifies (top-level status/counts + regions[]/proposition_
    realization_maps[]/supersession_hypotheses[] rows, each field read
    verbatim off D-202's own frozen types via D-202's own diagnostic
    functions -- no new computation)."""
    if result.understanding is None:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "not_evaluable",
            "capability_status": result.capability_status,
            "missing_evidence": list(result.missing_evidence),
            "source_count": 0,
            "region_count": 0,
            "proposition_realization_map_count": 0,
            "supersession_hypothesis_count": 0,
            "regions": [],
            "proposition_realization_maps": [],
            "supersession_hypotheses": [],
        }
    understanding = result.understanding
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "evaluated",
        "capability_status": result.capability_status,
        "missing_evidence": list(result.missing_evidence),
        "source_count": len(understanding.source_asset_ids),
        "region_count": len(understanding.regions),
        "proposition_realization_map_count": len(understanding.proposition_realization_maps),
        "supersession_hypothesis_count": len(understanding.supersession_hypotheses),
        "regions": [whole_video_editorial_region_diagnostics(r) for r in understanding.regions],
        "proposition_realization_maps": [
            whole_video_proposition_realization_diagnostics(m) for m in understanding.proposition_realization_maps
        ],
        "supersession_hypotheses": [
            whole_video_supersession_diagnostics(h) for h in understanding.supersession_hypotheses
        ],
        "global_continuity_status": understanding.global_continuity_status,
        "unresolved_conflicts": list(understanding.unresolved_conflicts),
        "provenance": list(understanding.provenance),
    }


def whole_video_editorial_reasoning_run_summary(result: WholeVideoEditorialReasoningResult) -> dict:
    """Pure aggregator -- the exact bounded field list this task's own
    "RUN SUMMARY" section requires. No master score."""
    if result.understanding is not None:
        base = whole_video_editorial_understanding_run_summary(result.understanding)
    else:
        base = {
            "schema_version": "cutsell.whole_video_editorial_reasoning.v1",
            "whole_video_region_count": 0,
            "recording_process_region_count": 0,
            "audience_delivery_region_count": 0,
            "mixed_region_count": 0,
            "proposition_realization_map_count": 0,
            "multi_realization_proposition_count": 0,
            "supersession_hypothesis_count": 0,
            "supported_supersession_count": 0,
            "partial_supersession_count": 0,
            "no_safe_supersession_count": 0,
            "conflicted_supersession_count": 0,
            "unknown_supersession_count": 0,
            "uncovered_unique_proposition_count": 0,
            "global_conflict_count": 0,
            "global_continuity_status": "UNKNOWN",
            "capability_status": CAPABILITY_NOT_EVALUABLE,
        }
    return {
        **base,
        "capability_status": result.capability_status,
        "p2_source_count": result.p2_source_count,
        "p2_missing_p1_source_count": result.p2_missing_p1_source_count,
        "p2_missing_language_source_count": result.p2_missing_language_source_count,
    }
