"""Clean orchestration for CutSell Flow B Milestone 1."""
from __future__ import annotations

import os

from collections import Counter
from dataclasses import dataclass, replace as dataclass_replace
import hashlib
from typing import Dict, Iterable, Mapping, Sequence

from .canonical_identity import (
    build_identity_chain_diagnostics,
    mint_realization_id,
    mint_retry_family_id,
    mint_semantic_idea_id,
)
from .claim_coverage_best_take import (
    _content_overlap_coefficient,
    critical_coverage_sets,
    resolve_critical_coverage_dominance,
)
from .clean_cut import apply_clean_cut
from .clean_cut_provider import CleanCutProvider, apply_provider_judgements, safe_clean_cut_judge
from .contradiction_signal import any_pair_contradicts
from .composer import compose_selected
from .composer_provider import ComposerProvider, safe_compose_order
from .final_sibling_grouping import _content
from .semantic_atom_importance import _clause_has_any
from .semantic_claims import _BELIEF_PERCEPTION_MARKERS, _RETROSPECTIVE_RECOGNITION_MARKERS
from .contracts import (
    CandidateTake,
    DraftClip,
    DraftTimeline,
    JobState,
    ProcessingRequest,
    ProcessingResult,
    SCHEMA_VERSION,
    SemanticLabel,
    SemanticRole,
    TakeGroup,
)
from .composite_resolver import (
    apply_composite_family_stabilization,
    apply_composite_group_split,
    apply_composite_resolution,
)
from .draft_review_provider import DraftReviewProvider, safe_review_draft
from .hybrid_editorial import EditorialJudge
from .semantic_compute_planner import build_cost_contract_report
from .semantic_idea_equivalence import SemanticEquivalenceArbiter, SemanticEquivalenceGatePolicy
from .session_boundaries import safe_group_takes_by_sessions
from .strategy import choose_strategy
from .take_grouping import _natural_tokens, _restart_content
from .take_grouping_provider import (
    TakeGroupingProvider,
    reconcile_semantic_idea_equivalence,
    split_incohesive_retry_groups,
)
# D-158: live-wiring the ONLY call site of `reconcile_semantic_idea_
# equivalence` to the real, already-computed D-157 Watch+Listen
# Understanding V1 objects (`flow_b.py` builds them; this module just
# threads them through). Safe as a plain top-level import -- unlike
# `take_grouping_provider.py`, nothing in this module's own dependency
# chain (`watch_listen_understanding -> attempt_reconstruction ->
# session_boundaries -> take_grouping_provider`) ever imports `pipeline.py`
# back at module load time (every existing `pipeline` reference from that
# chain's modules is a lazy `from . import pipeline` inside a function
# body) -- verified via `python3 -c "import cutsell_worker.pipeline"`.
from .attempt_relationship_authority import (
    build_understanding_span_index,
    watch_listen_family_evidence_enabled,
)
# D-161: same top-level-safe reasoning as the D-158 import above --
# `watch_listen_relation_discovery.py` sits in the SAME dependency chain
# (`attempt_reconstruction -> session_boundaries -> take_grouping_
# provider`), which never imports `pipeline.py` back at module load time.
from .watch_listen_besttake_evidence import (
    build_watch_listen_besttake_evidence,
    evaluate_watch_listen_besttake_guard,
    watch_listen_besttake_diagnostics,
    watch_listen_besttake_evidence_enabled,
    watch_listen_besttake_group_row,
)
# D-172: Zone-Usability V2 diagnostic consumption. A SEPARATE, default-OFF
# flag from D-163's own `watch_listen_besttake_evidence_enabled` above --
# see watch_listen_besttake_v2_evidence.py's own module docstring
# ("Feature flag") for why independent rollback is needed. Diagnostic-only:
# never mutates `selected_clip_id`/`ranked`/membership/Boundary/Pacing.
from .watch_listen_besttake_v2_evidence import (
    build_candidate_zone_usability_v2,
    evaluate_watch_listen_besttake_guard_v2,
    watch_listen_besttake_v2_diagnostics,
    watch_listen_besttake_v2_group_row,
    zone_usability_v2_besttake_enabled,
)
# D-174 (docs/CUTSELL_DECISIONS.md D-174): Watch+Listen BestTake Guard
# Authority Phase 1 (pure eligibility decision only -- see watch_listen_
# besttake_guard_authority.py's own module docstring for the full
# two-phase design). A SEPARATE, independently-rollbackable flag from
# D-163/D-172's own diagnostic-only flags. Nested inside D-172's own V2
# block by construction: this authority never runs on V1-only evidence.
from .watch_listen_besttake_guard_authority import (
    evaluate_watch_listen_besttake_guard_authority,
    watch_listen_besttake_guard_authority_diagnostics,
    watch_listen_besttake_guard_authority_enabled,
    watch_listen_besttake_guard_authority_row,
)
from .watch_listen_relation_discovery import watch_listen_relation_discovery_enabled
from .watch_listen_understanding import WatchListenUnderstanding
# D-184 (docs/CUTSELL_DECISIONS.md D-184): Bounded Finalist Arbiter --
# OFFLINE / DIAGNOSTIC ONLY. A SEPARATE, independently-rollbackable flag
# from D-163/D-172/D-174's own diagnostic-only flags. Consulted ONLY when
# D-183's own `TerminalBestTakeConfidence` already found the terminal
# comparison non-decisive; never mutates `selected_clip_id`/`ranked`/
# membership/Boundary/Pacing/Renderer -- see bounded_finalist_arbiter.py's
# own module docstring for the full contract.
from .bounded_finalist_arbiter import (
    FinalistArbiterInput,
    bounded_finalist_arbiter_diagnostics,
    bounded_finalist_arbiter_enabled,
    bounded_finalist_arbiter_run_summary,
    evaluate_bounded_finalist_arbiter,
)
from .take_judge import FRAGMENT_PENALTY_MARKERS, apply_delivery_cleanliness_evidence
from .take_judge_provider import TakeJudgeProvider, safe_rank_takes
from .case_b_performance_evidence import (
    build_case_b_performance_evidence,
    case_b_performance_evidence_diagnostics,
)
from .multimodal_besttake_fallback import detect_class_b_trigger, fallback_trigger_diagnostics
from .semantic_authority_observability import (
    AUTHORITY_ALLOWED,
    family_authority_diagnostics,
    semantic_authority_gate_diagnostics,
)
from .temporal_editing import refine_takes_with_temporal_context
from .whole_video_analysis import WholeVideoContext, confirmed_recording_behavior_events


def _group_id(project_id: str, key: str) -> str:
    return "tg_" + hashlib.sha256(f"{project_id}|{key}".encode()).hexdigest()[:18]


def _env_flag_enabled(name: str) -> bool:
    """D-094.2: '1'/'true'/'yes'/'on' (case-insensitive) enables; anything else,
    including unset, is OFF."""
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


# D-122 (BestTake CASE B evidence infrastructure, advisory/diagnostics only
# -- docs/CUTSELL_DECISIONS.md D-122, docs/CUTSELL_BESTTAKE_CASE_B_FORENSIC_
# D121.md): a small, side-effect-free re-derivation of `_semantic_best_
# take`'s OWN single-decisive-winner lookup (same `winner_confidence`
# default, same label/confidence test), used ONLY to expose the
# "semantic_fast_path_candidate" advisory counterfactual (D-122's
# "ADVISORY COUNTERFACTUAL" requirement) without changing that function's
# tested return signature or control flow.
def _single_semantic_winner_candidate(
    members: tuple[CandidateTake, ...],
    semantic_decisions: dict[str, tuple[str, float]],
    winner_confidence: float = 0.85,
) -> str | None:
    winners = [
        member.clip_id for member in members
        if semantic_decisions.get(member.clip_id, ("", 0.0))[0] == "winner"
        and semantic_decisions.get(member.clip_id, ("", 0.0))[1] >= winner_confidence
    ]
    return winners[0] if len(winners) == 1 else None


# D-122: `_semantic_best_take`'s own reason vocabulary, classified into the
# coarse winner_path buckets this task asks for. This is a read-only
# classification of an already-computed reason string -- it changes
# nothing about which reason is produced. "single_semantic_winner" is the
# ONLY reason string that reflects Hybrid/Gemini's decisive-label fast
# path (D-121 Section 5): DeliveryScorer's `ranked` is never consulted on
# that path. "delivery_tie_break_among_survivors" and "local_fallback" are
# the only two reasons whose final pick is `local_selected_clip_id`/
# `rank_by_id` (DeliveryScorer's own top-ranked survivor) -- everything
# else is a meaning/safety-driven resolution
# (critical_coverage_dominance, unresolved_unique_fact_asymmetry,
# unresolved_contradiction, single_member_no_contest, single_bts_unusable,
# no_usable_realization) that never reads a delivery/performance score.
# `DETERMINISTIC_OVERRIDE` is never assigned here -- it is only ever known
# once `deterministic_best_take_authority.py` runs, AFTER this diagnostics
# row already exists; see that module's own additive annotation step.
_WINNER_PATH_SEMANTIC_FAST_PATH = "SEMANTIC_FAST_PATH"
_WINNER_PATH_DELIVERYSCORE_PATH = "DELIVERYSCORE_PATH"
_WINNER_PATH_OTHER_EXISTING_PATH = "OTHER_EXISTING_PATH"
_DELIVERYSCORE_DRIVEN_REASONS = frozenset({
    "delivery_tie_break_among_survivors", "local_fallback",
})


def _winner_path_from_reason(reason: str) -> tuple[str, bool]:
    """Return (winner_path, performance_consulted_before_winner) for one
    `semantic_best_take_reason` value, per the classification above."""
    if reason == "single_semantic_winner":
        return _WINNER_PATH_SEMANTIC_FAST_PATH, False
    if reason in _DELIVERYSCORE_DRIVEN_REASONS:
        return _WINNER_PATH_DELIVERYSCORE_PATH, True
    return _WINNER_PATH_OTHER_EXISTING_PATH, False


# D-123 (docs/CUTSELL_DECISIONS.md D-123; bounded per docs/CUTSELL_
# BESTTAKE_CASE_B_FORENSIC_D121.md): a GATE ON EARLY EXIT for the
# `single_semantic_winner` fast path, not a new winner authority. D-121/
# D-122 proved that fast path never consults DeliveryScorer or D-115/D-122
# performance evidence at all. D-123 authorizes exactly one thing: when a
# real, evidenced performance conflict exists AND the DeliveryScorer-
# preferred alternative already passes the SAME meaning-sufficiency
# signals the general ladder below already uses, `_semantic_best_take`
# declines the fast-path shortcut and falls through to that unmodified
# ladder -- the ladder itself (steps 1-9) picks the actual winner exactly
# as it always has. No new score, weight, or threshold is introduced.
def _meaning_sufficient_member_ids(
    members: tuple[CandidateTake, ...],
    semantic_delete_recommended: dict[str, bool] | None,
) -> set[str]:
    """Which members already pass the EXISTING meaning-sufficiency signals
    `_semantic_best_take`'s own general ladder applies (D-081 semantic_
    delete_recommended, attempt completeness, D-103 required-condition-
    realization) -- reused verbatim as a raw per-candidate check, never a
    new classifier of any kind."""
    by_id = {member.clip_id: member for member in members}
    ids = list(by_id)
    delete_recommended_ids = {cid for cid in ids if (semantic_delete_recommended or {}).get(cid, False)}
    incomplete_ids = {cid for cid in ids if by_id[cid].complete_idea is False}
    required_missing_ids = _members_missing_required_condition_realization(ids, by_id)
    insufficient = delete_recommended_ids | incomplete_ids | required_missing_ids
    return {cid for cid in ids if cid not in insufficient}


# D-180 (docs/CUTSELL_DECISIONS.md D-180; post D-179 forensic): condition
# 4 of `_case_b_fast_path_conflict` originally compared RAW `delivery_
# event_count` alone -- D-179 proved a raw count is EVIDENCE, not
# MATERIALITY, and that a provider-backed perception count (`whole_video_
# openai.OpenAIWholeVideoProvider`) with no proven run-to-run stability
# can, via that raw comparison alone, flip a commercially material
# BestTake winner (8-vs-3 events, real Video00 gynecologist family,
# D-178B). This introduces NO new threshold family: it reuses `CaseBEvent.
# d097_would_be_counted` -- ALREADY computed by `case_b_performance_
# evidence.build_case_b_performance_evidence` for every delivery event,
# from D-097's OWN existing, already-tested interior-window geometry
# (`_CLEANLINESS_EDGE_MARGIN_SEC`) and confidence floors (0.88 RESET_
# KINDS / 0.76 BREAK_KINDS) -- as the materiality qualifier. D-167's V2
# severity vocabulary was considered and NOT used here: it is only
# computed when `CUTSELL_WATCH_LISTEN_ZONE_USABILITY_V2_BESTTAKE_ENABLED`
# is set, whereas this gate runs unconditionally on every family -- using
# it would make condition 4's behavior depend on an unrelated feature
# flag, a dependency problem this task's own directive asks to avoid.
# `d097_would_be_counted` has no "missing/unknown" state of its own: it is
# a pure, synchronous function of data already required to reach this
# point (the same `CaseBEvent` objects `winner_count`/`alt_count` are
# already computed from) -- so this refinement introduces no new failure
# mode requiring a "preserve legacy behavior" fallback.
def _material_delivery_event_count(evidence: object) -> int:
    """Count of `evidence.delivery_events` D-097's OWN existing geometry/
    confidence-floor doctrine would itself count as a real performance
    defect (`d097_would_be_counted`) -- never a new confidence/duration/
    density cutoff of this function's own invention."""
    events = getattr(evidence, "delivery_events", ()) or ()
    return sum(1 for event in events if getattr(event, "d097_would_be_counted", False))


def _case_b_fast_path_conflict(
    preferred_id: str,
    local_selected_clip_id: str,
    meaning_sufficient_ids: set[str],
    case_b_evidence_by_id: Mapping[str, object] | None,
) -> dict | None:
    """Return a factual conflict-basis dict iff ALL of the CORE RULE
    conditions hold, else None (the fast path is preserved). Condition 4
    (D-180, post D-179 forensic) is now COUNT DIFFERENCE + MATERIAL
    PERFORMANCE DIFFERENCE, not count difference alone: the raw `delivery_
    event_count` asymmetry is preserved as the entry precondition (D-123's
    original "is there a factual asymmetry at all" question), and a
    SEPARATE, already-existing D-097 materiality classification
    (`d097_would_be_counted`, see `_material_delivery_event_count` above)
    must ALSO favor the alternative before the fast path may be bypassed
    -- a raw count asymmetry driven only by events D-097's own doctrine
    would not itself count (low confidence, or geometrically outside its
    own interior window -- including an ambiguous straddle wider than the
    whole measured span, D-177's own CASE-C shape, which fails D-097's
    interior check and so never contributes here either) no longer
    bypasses on its own. Absent evidence, a tie, or evidence that favors
    the semantic winner all still return None, per this task's original
    "no guessed cutoff" / "if tied or ambiguous, preserve fast path"
    rule -- unchanged from pre-D-180 behavior for every OTHER condition."""
    if not case_b_evidence_by_id:
        return None
    if local_selected_clip_id == preferred_id:
        # Condition 3 fails: DeliveryScorer already agrees with the
        # semantic winner -- nothing to bypass for.
        return None
    if local_selected_clip_id not in meaning_sufficient_ids:
        # Condition 2 fails: the alternative DeliveryScorer prefers is
        # itself meaning-insufficient -- the fast path stands.
        return None
    winner_evidence = case_b_evidence_by_id.get(preferred_id)
    alt_evidence = case_b_evidence_by_id.get(local_selected_clip_id)
    if winner_evidence is None or alt_evidence is None:
        return None
    winner_count = winner_evidence.delivery_event_count
    alt_count = alt_evidence.delivery_event_count
    if not (winner_count > alt_count):
        # Condition 4a fails: no clear RAW factual asymmetry favoring the
        # alternative -- tied, absent, or contradicting evidence never
        # bypasses (ENTRY/EXIT-only differences are naturally 0 vs 0 here,
        # since case_b_evidence_by_id only ever contains DELIVERY-zone
        # events -- D-116's territory is never eligible).
        return None
    # D-180 Condition 4b: the raw asymmetry above must ALSO be corroborated
    # by D-097's own existing materiality doctrine, never a bare count.
    winner_material = _material_delivery_event_count(winner_evidence)
    alt_material = _material_delivery_event_count(alt_evidence)
    if not (winner_material > alt_material):
        return None
    return {
        "semantic_fast_path_candidate": preferred_id,
        "deliveryscore_top_candidate": local_selected_clip_id,
        "semantic_fast_path_candidate_delivery_event_count": winner_count,
        "deliveryscore_top_candidate_delivery_event_count": alt_count,
        "semantic_fast_path_candidate_count_by_kind": dict(winner_evidence.count_by_kind),
        "deliveryscore_top_candidate_count_by_kind": dict(alt_evidence.count_by_kind),
        "semantic_fast_path_candidate_duration_by_kind": dict(winner_evidence.duration_by_kind),
        "deliveryscore_top_candidate_duration_by_kind": dict(alt_evidence.duration_by_kind),
        # D-180: the materiality corroboration this exact conflict fired on.
        "semantic_fast_path_candidate_material_event_count": winner_material,
        "deliveryscore_top_candidate_material_event_count": alt_material,
    }


# D-180: pure, additive, observability-only companion to `_case_b_fast_
# path_conflict` -- recomputes the SAME condition-4 evaluation for
# diagnostics regardless of outcome (mirrors this module's own existing
# "before"/"after" counterfactual precedent for D-123 itself). Never
# called from inside `_semantic_best_take`'s own decision ladder; never
# influences `selected_clip_id`/`ranked`/membership/Boundary/Pacing.
def _case_b_condition4_diagnostics(
    preferred_id: str | None,
    local_selected_clip_id: str,
    meaning_sufficient_ids: set[str],
    case_b_evidence_by_id: Mapping[str, object] | None,
) -> dict:
    diag = {
        "case_b_count_difference_present": None,
        "case_b_materiality_evidence_available": False,
        "case_b_materiality_state": "NOT_EVALUATED",
        "case_b_materiality_source": "d097_would_be_counted",
        "case_b_condition4_actionable": False,
        "case_b_condition4_reason": "not_evaluated",
    }
    if preferred_id is None or not case_b_evidence_by_id:
        diag["case_b_condition4_reason"] = "no_semantic_fast_path_candidate_or_no_evidence"
        return diag
    if local_selected_clip_id == preferred_id:
        diag["case_b_condition4_reason"] = "deliveryscore_already_agrees_with_semantic_winner"
        return diag
    if local_selected_clip_id not in meaning_sufficient_ids:
        diag["case_b_condition4_reason"] = "alternative_meaning_insufficient"
        return diag
    winner_evidence = case_b_evidence_by_id.get(preferred_id)
    alt_evidence = case_b_evidence_by_id.get(local_selected_clip_id)
    if winner_evidence is None or alt_evidence is None:
        diag["case_b_condition4_reason"] = "evidence_missing_for_one_or_both_candidates"
        return diag
    diag["case_b_materiality_evidence_available"] = True
    winner_count = winner_evidence.delivery_event_count
    alt_count = alt_evidence.delivery_event_count
    count_difference_present = bool(winner_count > alt_count)
    diag["case_b_count_difference_present"] = count_difference_present
    if not count_difference_present:
        diag["case_b_materiality_state"] = "NO_COUNT_DIFFERENCE"
        diag["case_b_condition4_reason"] = "no_raw_count_asymmetry"
        return diag
    winner_material = _material_delivery_event_count(winner_evidence)
    alt_material = _material_delivery_event_count(alt_evidence)
    if winner_material > alt_material:
        diag["case_b_materiality_state"] = "MATERIAL"
        diag["case_b_condition4_actionable"] = True
        diag["case_b_condition4_reason"] = "material_count_difference_confirmed"
    else:
        diag["case_b_materiality_state"] = "NOT_MATERIAL"
        diag["case_b_condition4_reason"] = "raw_count_difference_not_materially_corroborated"
    return diag


# D-183 (docs/CUTSELL_DECISIONS.md D-183; post D-182 forensic): terminal
# BestTake confidence/decisiveness CLASSIFICATION ONLY -- no finalist
# arbiter, no winner mutation, no score-weight change. D-182 proved Steps
# 6-9 of `_semantic_best_take` (the `critical_coverage_dominance`/
# asymmetry/contradiction checks above them, and the final
# `max(tie_break_pool, key=lambda cid: rank_by_id[cid])` itself) carry NO
# concept of confidence, margin, or abstention: once 2+ candidates reach
# a comparative step, EXACTLY one winner is always forced, with no
# representation of "this comparison was not actually decisive."
#
# `TerminalBestTakeConfidence` and `_terminal_besttake_confidence` are a
# PURE, ADDITIVE observability layer over that same terminal span (Steps
# 3-9 -- every genuinely COMPARATIVE step; Steps 1/2/2.5 are exclusion-
# only safety filters, not confidence questions). They are wired into
# `_semantic_best_take` via an optional `terminal_confidence_out: dict |
# None` keyword-only side-channel (default `None`): every existing caller
# that omits it is byte-identical to pre-D-183 behavior, and even when
# supplied, the classification is recorded ALONGSIDE the existing return
# value, never used to pick, veto, or alter `selected_clip_id`/
# `preferred_id`/`reason`.
#
# CORE PRINCIPLE (this task's own words): a deterministic score
# difference is not automatically an editorially decisive difference.
# Steps 3/4's own `resolve_critical_coverage_dominance` and Step 5's own
# asymmetry/contradiction checks ALREADY run before Steps 6-9 -- by
# construction, reaching the raw-score `max()` comparison at all means
# every prior structured-evidence check already found NO dominance. So
# within Steps 6-9's own 2+-survivor branch, RAW SCORE ALONE (however
# large the numeric gap) is classified NON_DECISIVE by this module's own
# design -- DECISIVE is reserved for cases already settled by structured
# evidence (dominance found earlier; a single survivor after subset-
# exclusion; or an explicitly supplied, unanimous external comparator via
# `structured_signals`, e.g. a future D-163/D-172 wiring -- optional,
# never fabricated, never required, never double-counted against
# `rank_by_id` itself).
#
# NO NEW MAGIC MARGIN: TIED is decided by EXACT equality of the two
# scores as already rounded by `take_judge.score_take`/`rank_takes`/
# `apply_delivery_cleanliness_evidence` (`round(x, 4)`, an existing,
# already-applied precision -- never a newly invented epsilon or percent
# threshold). NON_DECISIVE vs DECISIVE is never decided by comparing the
# margin to any numeric cutoff at all -- see the paragraph above.
#
# REUSED VOCABULARY: "DECISIVE"/"NON_DECISIVE" are the SAME string
# literals `semantic_authority_observability.semantic_authority_gate_
# diagnostics` (D-146/D-150) already uses for its own (upstream, label-
# level) decisiveness question -- reused verbatim here for the SAME
# general concept at a different layer, never a duplicate ontology.
_TERMINAL_CONFIDENCE_DECISIVE = "DECISIVE"
_TERMINAL_CONFIDENCE_DECISIVE_BY_ELIMINATION = "DECISIVE_BY_ELIMINATION"
_TERMINAL_CONFIDENCE_NON_DECISIVE = "NON_DECISIVE"
_TERMINAL_CONFIDENCE_TIED = "TIED"
_TERMINAL_CONFIDENCE_CONFLICTED = "CONFLICTED"
_TERMINAL_CONFIDENCE_UNKNOWN = "UNKNOWN"

# Provenances naming WHY a state was reached -- "structured_dominance"
# (Steps 3/4's own dominance, or Step 5's own asymmetry/contradiction
# finding) and "structured_signals" (an explicitly supplied external
# comparator) are the only two that may ever justify DECISIVE/CONFLICTED
# without a raw-score comparison; "raw_score_only"/"raw_score_equal"
# never produce DECISIVE.
_TERMINAL_CONFIDENCE_PROVENANCE_STRUCTURED_DOMINANCE = "structured_dominance"
_TERMINAL_CONFIDENCE_PROVENANCE_STRUCTURED_SIGNALS = "structured_signals"
_TERMINAL_CONFIDENCE_PROVENANCE_RAW_SCORE_ONLY = "raw_score_only"
_TERMINAL_CONFIDENCE_PROVENANCE_RAW_SCORE_EQUAL = "raw_score_equal"
_TERMINAL_CONFIDENCE_PROVENANCE_SINGLE_SURVIVOR = "single_survivor"
_TERMINAL_CONFIDENCE_PROVENANCE_NO_SCORE = "no_score_available"

_TERMINAL_CONFIDENCE_STRUCTURED_PROVENANCES = frozenset({
    _TERMINAL_CONFIDENCE_PROVENANCE_STRUCTURED_DOMINANCE,
    _TERMINAL_CONFIDENCE_PROVENANCE_STRUCTURED_SIGNALS,
})


@dataclass(frozen=True)
class TerminalBestTakeConfidence:
    """The terminal BestTake decision's own confidence, additive to (never
    a replacement for) `_semantic_best_take`'s existing return value. Every
    field is a fact about the comparison itself -- never a new winner."""
    candidate_ids: tuple[str, ...]
    ranked_candidate_ids: tuple[str, ...]
    top_candidate_id: str | None
    runner_up_candidate_id: str | None
    top_score: float | None
    runner_up_score: float | None
    score_margin: float | None
    confidence_state: str
    reason: str
    provenance: str


def _terminal_confidence(
    state: str,
    reason: str,
    provenance: str,
    *,
    candidate_ids: Sequence[str] = (),
    ranked_candidate_ids: Sequence[str] = (),
    top_id: str | None = None,
    runner_up_id: str | None = None,
    top_score: float | None = None,
    runner_up_score: float | None = None,
) -> TerminalBestTakeConfidence:
    margin = (
        round(top_score - runner_up_score, 4)
        if top_score is not None and runner_up_score is not None
        else None
    )
    return TerminalBestTakeConfidence(
        candidate_ids=tuple(candidate_ids),
        ranked_candidate_ids=tuple(ranked_candidate_ids),
        top_candidate_id=top_id,
        runner_up_candidate_id=runner_up_id,
        top_score=top_score,
        runner_up_score=runner_up_score,
        score_margin=margin,
        confidence_state=state,
        reason=reason,
        provenance=provenance,
    )


def _terminal_besttake_confidence(
    candidate_ids: Sequence[str],
    rank_by_id: Mapping[str, float],
    *,
    structured_signals: Mapping[str, str | None] | None = None,
) -> TerminalBestTakeConfidence:
    """Classify the terminal 2+-survivor raw-score comparison ONLY --
    callers that already found structured dominance, a single survivor,
    or an unresolved asymmetry/contradiction never reach this function at
    all (they build their own `TerminalBestTakeConfidence` directly via
    `_terminal_confidence`, at the exact point that evidence was found).

    `structured_signals` (optional, additive, NEVER fabricated when
    absent): a mapping of an external comparator's name (e.g. a future
    "watch_listen_besttake_v2") to the candidate id it prefers among
    `candidate_ids` (or `None` for no preference from that source). No
    live caller supplies this today (D-163/D-172 are diagnostic-only and
    off by default, and are not part of `_semantic_best_take`'s own
    decision inputs) -- the parameter exists so a future, explicitly-
    authorized wiring can supply it without a second confidence ontology.
    Passing the SAME underlying evidence under two different source names
    is a caller error this function cannot detect; see
    `test_no_double_counting_source_independence` for the audit this
    task's own scope requires instead (confirming this function itself
    never re-derives or re-reads any per-signal take-level field)."""
    candidate_ids = tuple(candidate_ids)
    scored = [cid for cid in candidate_ids if cid in rank_by_id]
    if not scored:
        return _terminal_confidence(
            _TERMINAL_CONFIDENCE_UNKNOWN,
            "no_score_available_for_any_candidate",
            _TERMINAL_CONFIDENCE_PROVENANCE_NO_SCORE,
            candidate_ids=candidate_ids,
        )
    # Deterministic, candidate-order-independent ranking: the SAME sort
    # key `rank_takes` itself already uses (`(-score, clip_id)`) -- never
    # a new tie-break convention.
    ranked_ids = tuple(sorted(scored, key=lambda cid: (-rank_by_id[cid], cid)))
    if len(ranked_ids) == 1:
        only = ranked_ids[0]
        return _terminal_confidence(
            _TERMINAL_CONFIDENCE_DECISIVE_BY_ELIMINATION,
            "single_survivor_no_comparison_needed",
            _TERMINAL_CONFIDENCE_PROVENANCE_SINGLE_SURVIVOR,
            candidate_ids=candidate_ids, ranked_candidate_ids=ranked_ids,
            top_id=only, top_score=rank_by_id[only],
        )
    top_id, runner_up_id = ranked_ids[0], ranked_ids[1]
    top_score, runner_up_score = rank_by_id[top_id], rank_by_id[runner_up_id]

    if structured_signals:
        preferences = {
            cid for cid in (structured_signals or {}).values()
            if cid in (top_id, runner_up_id)
        }
        if preferences == {top_id}:
            return _terminal_confidence(
                _TERMINAL_CONFIDENCE_DECISIVE,
                "structured_comparators_unanimously_agree_with_top_ranked_candidate",
                _TERMINAL_CONFIDENCE_PROVENANCE_STRUCTURED_SIGNALS,
                candidate_ids=candidate_ids, ranked_candidate_ids=ranked_ids,
                top_id=top_id, runner_up_id=runner_up_id,
                top_score=top_score, runner_up_score=runner_up_score,
            )
        if len(preferences) >= 2:
            return _terminal_confidence(
                _TERMINAL_CONFIDENCE_CONFLICTED,
                "structured_comparators_disagree_on_preferred_candidate",
                _TERMINAL_CONFIDENCE_PROVENANCE_STRUCTURED_SIGNALS,
                candidate_ids=candidate_ids, ranked_candidate_ids=ranked_ids,
                top_id=top_id, runner_up_id=runner_up_id,
                top_score=top_score, runner_up_score=runner_up_score,
            )
        # A single structured comparator naming the RUNNER-UP (not the top
        # score) is itself a disagreement with the raw score -- CONFLICTED,
        # never silently overridden and never ignored.
        if preferences == {runner_up_id}:
            return _terminal_confidence(
                _TERMINAL_CONFIDENCE_CONFLICTED,
                "structured_comparator_disagrees_with_raw_score_ranking",
                _TERMINAL_CONFIDENCE_PROVENANCE_STRUCTURED_SIGNALS,
                candidate_ids=candidate_ids, ranked_candidate_ids=ranked_ids,
                top_id=top_id, runner_up_id=runner_up_id,
                top_score=top_score, runner_up_score=runner_up_score,
            )
        # Signals present but naming neither finalist -- no opinion here.

    if top_score == runner_up_score:
        return _terminal_confidence(
            _TERMINAL_CONFIDENCE_TIED,
            "terminal_score_exact_tie",
            _TERMINAL_CONFIDENCE_PROVENANCE_RAW_SCORE_EQUAL,
            candidate_ids=candidate_ids, ranked_candidate_ids=ranked_ids,
            top_id=top_id, runner_up_id=runner_up_id,
            top_score=top_score, runner_up_score=runner_up_score,
        )
    return _terminal_confidence(
        _TERMINAL_CONFIDENCE_NON_DECISIVE,
        "raw_score_difference_without_structured_dominance",
        _TERMINAL_CONFIDENCE_PROVENANCE_RAW_SCORE_ONLY,
        candidate_ids=candidate_ids, ranked_candidate_ids=ranked_ids,
        top_id=top_id, runner_up_id=runner_up_id,
        top_score=top_score, runner_up_score=runner_up_score,
    )


def terminal_besttake_confidence_diagnostics(confidence: TerminalBestTakeConfidence) -> dict:
    """JSON-safe, bounded per-family diagnostics row -- no transcript
    dump, no QA-reference info. Field names match this task's own
    directive verbatim."""
    return {
        "terminal_besttake_confidence_state": confidence.confidence_state,
        "terminal_besttake_confidence_reason": confidence.reason,
        "terminal_besttake_candidate_count": len(confidence.candidate_ids),
        "terminal_besttake_top_candidate_id": confidence.top_candidate_id,
        "terminal_besttake_runner_up_candidate_id": confidence.runner_up_candidate_id,
        "terminal_besttake_top_score": confidence.top_score,
        "terminal_besttake_runner_up_score": confidence.runner_up_score,
        "terminal_besttake_score_margin": confidence.score_margin,
        "terminal_besttake_structured_dominance_present": (
            confidence.provenance in _TERMINAL_CONFIDENCE_STRUCTURED_PROVENANCES
            and confidence.confidence_state == _TERMINAL_CONFIDENCE_DECISIVE
        ),
        "terminal_besttake_conflict_present": confidence.confidence_state == _TERMINAL_CONFIDENCE_CONFLICTED,
        "terminal_besttake_decisive": confidence.confidence_state in (
            _TERMINAL_CONFIDENCE_DECISIVE, _TERMINAL_CONFIDENCE_DECISIVE_BY_ELIMINATION,
        ),
    }


# D-183 run-level tail-safe summary -- a pure aggregator over already-
# computed per-family diagnostics rows (e.g. `take_judge_groups`), never
# a recomputation of any family's own confidence. Mirrors D-152/D-181's
# own workflow-side aggregation pattern, kept here as a plain, directly
# testable function since this task does not authorize a workflow change.
def terminal_besttake_confidence_run_summary(rows: Iterable[Mapping]) -> dict:
    counts = {
        "terminal_besttake_evaluated_count": 0,
        "terminal_besttake_decisive_count": 0,
        "terminal_besttake_non_decisive_count": 0,
        "terminal_besttake_tied_count": 0,
        "terminal_besttake_conflicted_count": 0,
        "terminal_besttake_unknown_count": 0,
    }
    for row in rows:
        state = row.get("terminal_besttake_confidence_state") if isinstance(row, Mapping) else None
        if state is None:
            continue
        counts["terminal_besttake_evaluated_count"] += 1
        if state in (_TERMINAL_CONFIDENCE_DECISIVE, _TERMINAL_CONFIDENCE_DECISIVE_BY_ELIMINATION):
            counts["terminal_besttake_decisive_count"] += 1
        elif state == _TERMINAL_CONFIDENCE_NON_DECISIVE:
            counts["terminal_besttake_non_decisive_count"] += 1
        elif state == _TERMINAL_CONFIDENCE_TIED:
            counts["terminal_besttake_tied_count"] += 1
        elif state == _TERMINAL_CONFIDENCE_CONFLICTED:
            counts["terminal_besttake_conflicted_count"] += 1
        elif state == _TERMINAL_CONFIDENCE_UNKNOWN:
            counts["terminal_besttake_unknown_count"] += 1
    return counts


def _draft_clip(take: CandidateTake, *, role: SemanticRole, group_id: str | None, selected: bool) -> DraftClip:
    # D-050A: `group_id` here is already the FINAL, post-semantic-
    # equivalence take-group id (pipeline.py is its one minting owner --
    # see canonical_identity.py's ID OWNERSHIP table). semantic_idea_id
    # and retry_family_id are minted from it, additively; take_group_id
    # itself is left completely unchanged for every existing consumer.
    semantic_idea_id = mint_semantic_idea_id(group_id) if group_id else None
    retry_family_id = mint_retry_family_id(group_id) if group_id else None
    return DraftClip(
        clip_id=take.clip_id,
        source_asset_id=take.source_asset_id,
        source_order=take.source_order,
        start=take.start,
        end=take.end,
        text=take.text,
        caption_text=take.text,
        words=take.words,
        semantic_role=role,
        take_group_id=group_id,
        selected=selected,
        # Carry the take's local face/pose/motion evidence through to the
        # draft so downstream Selection authorities (Hybrid or Unified) can
        # actually see it instead of it being silently dropped at this
        # conversion. See local_performance.py / MediaSignals.
        signals=take.signals,
        # D-050A: carried unchanged from the CandidateTake this clip was
        # built from -- never recomputed here.
        realization_id=take.realization_id,
        semantic_idea_id=semantic_idea_id,
        retry_family_id=retry_family_id,
        # D-050C1.6: carried unchanged from the CandidateTake -- see
        # DraftClip.complete_idea's own docstring.
        complete_idea=take.complete_idea,
        # D-076: carried unchanged from the CandidateTake -- see
        # DraftClip.source_span_id/.attempt_id's own docstring.
        source_span_id=take.source_span_id,
        attempt_id=take.attempt_id,
    )


def family_scoped_semantic_decisions(
    members: tuple[CandidateTake, ...],
    semantic_decisions: dict[str, tuple[str, float]],
    window_rows: Iterable[Mapping] | None,
) -> tuple[dict[str, tuple[str, float]], dict | None]:
    """D-094.3 (F8): a hybrid "winner"/"alternate" label is COMPARATIVE -- it
    answers "best among the candidates this window saw". The per-clip merge
    across windows (`hybrid_session_cleanup`'s best-priority-per-clip) keeps
    a clip's strongest label from ANY window, so a take judged "winner" in
    a window that never saw its better sibling keeps that "winner" even when
    the one window that saw the WHOLE family ranked it "alternate". Run
    33983880111: the pimples monolith was "winner" 0.96 in a window without
    the later delivery, "alternate" 0.88 in the window holding all three
    takes (where the later delivery was "winner" 0.95); the merge produced
    two "winners", the ladder fell to DeliveryScorer and the monolith won.

    When at least one window contains EVERY member of this retry family,
    those windows' labels are the family-level answer and replace the
    cross-window merge for these members (merged by the same priority rule
    across the complete windows only). Otherwise the global merge is used
    unchanged. Returns (decisions, source_info); source_info is None when no
    family-complete window exists. Windows are the per-chunk rows of
    `hybrid_cleanup.diagnostics` (member_ids + decisions); rows without
    member_ids (hook diagnostics) are ignored."""
    from .hybrid_session_cleanup import _decision_priority

    family = [member.clip_id for member in members]
    if not window_rows or len(family) < 2:
        return dict(semantic_decisions), None
    family_set = set(family)
    complete = [
        row for row in window_rows
        if isinstance(row, Mapping) and family_set <= set(row.get("member_ids") or ())
    ]
    if not complete:
        return dict(semantic_decisions), None
    merged: dict[str, tuple[str, float]] = {}
    for row in complete:
        for decision in row.get("decisions") or ():
            clip_id = decision.get("clip_id")
            if clip_id not in family_set:
                continue
            candidate = (str(decision.get("label") or ""), float(decision.get("confidence") or 0.0))
            current = merged.get(clip_id)
            if current is None or _decision_priority(*candidate) > _decision_priority(*current):
                merged[clip_id] = candidate
    scoped = dict(semantic_decisions)
    for clip_id in family:
        if clip_id in merged:
            scoped[clip_id] = merged[clip_id]
    return scoped, {
        "family_complete_window_chunk_indices": [row.get("chunk_index") for row in complete],
        "family_window_labels": {cid: list(merged[cid]) for cid in family if cid in merged},
        "global_merge_labels": {cid: list(semantic_decisions.get(cid, ("", 0.0))) for cid in family},
    }


# D-097.8 (R10): the same 0.85 floor the Resolver applies to a hybrid
# `failed` label (realization_resolver._SEMANTIC_FAILED_THRESHOLD) and the
# winner-label floor `_semantic_best_take` already requires.
_BTS_SINGLETON_UNUSABLE_CONFIDENCE = 0.85


# D-103 (P0 follow-up, papillary-diagnosis closure): D-102's own real-media
# qualification proved condition (d) below cannot catch every required-
# meaning loss -- the D-101 papillary realization carries ZERO claims of
# any type under `semantic_claims.py`'s classifier (its own sentences are
# generic ACTION_EVENT/SUPPORTING), so `critical_coverage_sets` has
# nothing to compare and the fast path stayed unprotected. Investigation
# (see D-103 decision entry) confirmed no ALREADY-PRODUCTION-AUTHORITATIVE
# representation captures this class of loss either: `semantic_ledger.py`
# and `realization_resolver.py`'s `RequirementGroup`/`build_requirement_
# groups` are both explicitly SHADOW-ONLY ("never consulted by ... today's
# engine" -- their own module docstrings) and cannot be bridged into a live
# veto without a separate authority-cutover directive, out of this bounded
# task's scope.
#
# The minimum general representation added here is narrow and local to
# this ONE veto -- it does not touch `semantic_claims.classify_claim`,
# ClaimCoverage, StoryValidator, or D-063 dominance, and never reclassifies
# anything CRITICAL globally. It reuses the SAME general (non-Video00)
# marker vocabulary `semantic_claims.py` already uses for its own
# `CONTRASTIVE_HINDSIGHT_NEGATION` claim role (`_BELIEF_PERCEPTION_
# MARKERS`/`_RETROSPECTIVE_RECOGNITION_MARKERS`) -- a belief/perception
# verb about one's own state, paired with an explicit retrospective-
# recognition marker ("looking back", "now that I ...", "in hindsight") --
# but recognizes it as a REQUIRED before/after realization about a
# condition regardless of whether a negation marker also happens to be
# present (the existing classifier only grants that pattern CRITICAL
# status when it IS negated; the identical propositional shape phrased
# positively is invisible to it today). This is a general linguistic
# pattern, not a Video00 fact/phrase: it fires on the SHAPE of the
# sentence, never on specific transcript wording.
#
# The veto only fires when a sibling matches this pattern AND the
# proposed winner's own text does not already substantially overlap with
# that sibling's content (`_content_overlap_coefficient`, the same
# Szymkiewicz-Simpson overlap primitive `claim_coverage_best_take.py`'s
# own D-065/D-066 hindsight-alignment machinery already uses) -- so a
# winner that already expresses the same realization (a paraphrase) is
# never forced open, and two candidates that both happen to carry this
# pattern about genuinely different subjects are still correctly
# distinguished by the overlap check, not merged or composited.
_REQUIRED_REALIZATION_OVERLAP_FLOOR = 0.40


def _is_retrospective_condition_realization(text: str) -> bool:
    """A general (non-Video00) linguistic pattern: a belief/perception verb
    about one's own state paired with an explicit retrospective-
    recognition marker -- the same general vocabulary `semantic_claims.py`
    already uses for `CONTRASTIVE_HINDSIGHT_NEGATION`, but without
    requiring the negation marker that classifier currently insists on.
    See the D-103 module comment above `_single_winner_safety_veto` for
    the full rationale."""
    return _clause_has_any(text, _BELIEF_PERCEPTION_MARKERS) and _clause_has_any(
        text, _RETROSPECTIVE_RECOGNITION_MARKERS
    )


def _members_missing_required_condition_realization(
    member_ids: list[str], by_id: dict[str, CandidateTake]
) -> set[str]:
    """Every member id whose own text fails to preserve ANOTHER member's
    required before/after condition realization (`_is_retrospective_
    condition_realization`). Shared by `_single_winner_safety_veto` (which
    only needs to know whether the LABEL's own pick is missing one) and
    the general ladder's own exclusion step below (D-103): a pure veto
    that merely forces fallthrough is not enough here, because the
    dominance step a few lines down is driven by the SAME CRITICAL-claim
    classifier that is blind to this realization -- a sibling with zero
    CRITICAL claims never registers as "dominant" over a winner that
    happens to hold an unrelated CRITICAL claim of its own, so without an
    explicit exclusion the ladder would silently re-derive the exact
    label it was just told not to trust. This exclusion is therefore
    computed and applied the SAME way D-081's `semantic_delete_
    recommended` exclusion already is -- soft, `_exclude_unless_all`,
    fail-open toward WHEN-UNCERTAIN-KEEP -- never a second, divergent
    resolution path."""
    missing: set[str] = set()
    for cid in member_ids:
        text = str(by_id[cid].text or "")
        for other_id in member_ids:
            if other_id == cid:
                continue
            other_text = str(by_id[other_id].text or "")
            if not _is_retrospective_condition_realization(other_text):
                continue
            overlap = _content_overlap_coefficient(
                frozenset(_content(other_text)), frozenset(_content(text))
            )
            if overlap < _REQUIRED_REALIZATION_OVERLAP_FLOOR:
                missing.add(cid)
                break
    return missing


# D-101 ROOT CAUSE #1 (P0 forensic, hereditary-cancer/papillary-diagnosis
# cluster): `single_semantic_winner` used to trust a lone Hybrid/Gemini
# "winner" label unconditionally, with NONE of the safety checks the
# `len(winners) != 1` branch below already has -- on real Video00 footage
# this let a per-window classification error discard the realization
# carrying unique required meaning while keeping an unrelated one. This is
# a SAFETY VETO only, not a return to maximum semantic coverage: it never
# manufactures a composite and never restores a losing realization merely
# because it carries extra SUPPORTING/low-value content -- it only refuses
# to trust the label when the label's own pick would (a) itself carry a
# D-081 semantic-delete-recommended flag, (b) itself be an EXPLICITLY
# incomplete attempt (`complete_idea is False` -- WHEN-UNCERTAIN-KEEP:
# unset/unknown is never a veto trigger), (c) factually contradict another
# member (`contradiction_signal.any_pair_contradicts`, the same safety
# gate used everywhere else in this function), (d) fail to cover a
# CRITICAL claim (`claim_coverage_best_take.critical_coverage_sets`) that
# another member uniquely covers, or (e, D-103) fail to preserve a sibling's
# required before/after condition realization (see above) that the
# CRITICAL-claim classifier alone does not recognize. No new heuristic
# beyond (e)'s own narrow, local pattern: every other check below is the
# SAME deterministic function the multi-candidate branch already calls.
# Vetoed cases fall through to that same branch (treated exactly like a
# non-decisive label set), never a bespoke resolution path.
def _single_winner_safety_veto(
    preferred_id: str,
    members: tuple[CandidateTake, ...],
    semantic_delete_recommended: dict[str, bool] | None,
) -> str | None:
    """Return a veto reason, or None when the single "winner" label is safe
    to trust as-is (the common case -- byte-identical to pre-D-101
    behavior whenever nothing below fires)."""
    by_id = {member.clip_id: member for member in members}
    member_ids = list(by_id)
    if (semantic_delete_recommended or {}).get(preferred_id, False):
        return "winner_carries_delete_recommended_evidence"
    if by_id[preferred_id].complete_idea is False:
        return "winner_is_explicitly_incomplete"
    texts = [str(by_id[cid].text or "") for cid in member_ids]
    if any_pair_contradicts(texts):
        return "members_contradict"
    members_pairs = [(cid, by_id[cid]) for cid in member_ids]
    coverage = critical_coverage_sets(members_pairs, member_ids)
    if coverage:
        preferred_coverage = coverage.get(preferred_id, frozenset())
        for cid, covered in coverage.items():
            if cid == preferred_id:
                continue
            if not covered.issubset(preferred_coverage):
                return "winner_missing_unique_critical_claim"
    if preferred_id in _members_missing_required_condition_realization(member_ids, by_id):
        return "winner_missing_required_condition_realization"
    return None


# D-101 ROOT CAUSE #2 (P0 forensic, same cluster): once the semantic label
# stops being decisive, `delivery_tie_break_among_survivors` used to pick
# among survivors by raw DeliveryScorer rank alone, with no awareness that
# one survivor could be a strict, literal content subset of another --
# on real footage this let an incomplete short prefix of a fuller passage
# outscore (and so discard) the complete passage that contains and
# completes it. Deliberately NOT "longer text always wins": the match
# requires the shorter candidate's ENTIRE natural-token sequence to occur
# verbatim, contiguously, inside the longer one (never a bag-of-words/
# lexical-similarity test), so an unrelated pair sharing only a topic or
# opening (same_opening_restart's own territory), two independently
# COMPLETE statements, or a complementary pair never qualifies -- and a
# genuine factual disagreement between the two (`any_pair_contradicts`,
# the same D-063-family safety gate used throughout this function) always
# suppresses the protection, so a contradicting fuller candidate is never
# treated as automatically safe either.
_SUBSET_MINIMUM_SHORT_CONTENT_TOKENS = 3
_SUBSET_MINIMUM_EXTRA_CONTENT_TOKENS = 2


def _is_incomplete_content_subset(short: CandidateTake, long: CandidateTake) -> bool:
    """True when `short`'s full natural-token sequence is a contiguous
    subsequence of `long`'s, with `long` carrying at least a minimum of
    genuinely additional content beyond the matched span. See the D-101
    Root Cause #2 module comment above for the full rationale/bounds."""
    short_tokens = _natural_tokens(str(short.text or ""))
    long_tokens = _natural_tokens(str(long.text or ""))
    if len(short_tokens) >= len(long_tokens):
        return False
    short_content = _restart_content(short_tokens)
    if len(short_content) < _SUBSET_MINIMUM_SHORT_CONTENT_TOKENS:
        return False
    window = len(short_tokens)
    match_start = None
    for start in range(len(long_tokens) - window + 1):
        if long_tokens[start:start + window] == short_tokens:
            match_start = start
            break
    if match_start is None:
        return False
    extra_tokens = long_tokens[:match_start] + long_tokens[match_start + window:]
    extra_content = _restart_content(extra_tokens)
    return len(extra_content) >= _SUBSET_MINIMUM_EXTRA_CONTENT_TOKENS


def _exclude_incomplete_subset_losers(
    candidate_ids: list[str],
    by_id: dict[str, CandidateTake],
) -> list[str]:
    """Drop any candidate that is an incomplete content subset of ANOTHER
    candidate in `candidate_ids`, unless the pair factually contradicts
    (in which case the protection is suppressed for that pair -- the
    fuller candidate is never assumed safe just because it is fuller).
    Fails open: never excludes every candidate."""
    excluded: set[str] = set()
    for short_id in candidate_ids:
        for long_id in candidate_ids:
            if short_id == long_id:
                continue
            if not _is_incomplete_content_subset(by_id[short_id], by_id[long_id]):
                continue
            if any_pair_contradicts([str(by_id[short_id].text or ""), str(by_id[long_id].text or "")]):
                continue
            excluded.add(short_id)
    survivors = [cid for cid in candidate_ids if cid not in excluded]
    return survivors if survivors else candidate_ids


def _semantic_best_take(
    members: tuple[CandidateTake, ...],
    semantic_decisions: dict[str, tuple[str, float]],
    local_selected_clip_id: str,
    ranked: tuple = (),
    *,
    winner_confidence: float = 0.85,
    semantic_delete_recommended: dict[str, bool] | None = None,
    deterministic_unusable: dict[str, bool] | None = None,
    case_b_evidence_by_id: Mapping[str, object] | None = None,
    semantic_comparative_authority: str | None = None,
    terminal_confidence_out: dict | None = None,
) -> tuple[str | None, str | None, str]:
    """Honor one clear semantic winner only inside an already-proven retry group.

    D-183 (docs/CUTSELL_DECISIONS.md D-183; post D-182 forensic):
    `terminal_confidence_out` is optional and purely additive -- omitted
    or `None` (every existing caller before D-183), this function is
    byte-identical to pre-D-183 behavior in every respect, including
    return value. When a caller supplies a fresh `dict`, this function
    populates `terminal_confidence_out["terminal_besttake_confidence"]`
    with a `TerminalBestTakeConfidence` describing how DECISIVE the
    comparative span (Steps 3-9 -- dominance, asymmetry/contradiction,
    and the final raw-score tie-break) actually was, at the EXACT point
    each outcome is reached -- never a second pass, never a re-derivation
    that could drift from the real decision. It NEVER reads from or
    writes to `selected_clip_id`/`preferred_id`/the returned reason.

    D-150 (Phase B; docs/CUTSELL_DECISIONS.md D-150): `semantic_
    comparative_authority` is optional and additive -- omitted or `None`
    (every existing caller before D-150, plus this function's own D-123
    counterfactual call above), this function is byte-identical to
    pre-D-150 behavior. When the caller (pipeline.py's per-family loop)
    passes an explicit status from `semantic_authority_observability.
    resolve_semantic_comparative_authority` -- `"ABSTAIN_CONFLICT"` or
    `"ABSTAIN_INCOMPLETE_CONTEXT"` -- the `single_semantic_winner` early
    exit below is skipped entirely, exactly like an existing D-101 `_
    single_winner_safety_veto` hit: the fast path never runs (no
    confidence check, no case_b gate, nothing), and control falls straight
    through to the SAME general ladder immediately below, completely
    unmodified. This never selects a winner itself, never touches
    `semantic_delete_recommended`/`deterministic_unusable`/the ladder's own
    steps, and never calls a provider -- it only decides whether the
    comparative label ABOVE this point may be trusted at all.

    D-123 (docs/CUTSELL_DECISIONS.md D-123; bounded per docs/CUTSELL_
    BESTTAKE_CASE_B_FORENSIC_D121.md): `case_b_evidence_by_id` is optional
    and additive -- omitted or `None`, this function is byte-identical to
    pre-D-123 behavior for every existing caller. When provided (mapping
    clip_id -> a `CaseBPerformanceEvidence`, D-122), it can ONLY gate the
    `single_semantic_winner` early exit below (see `_case_b_fast_path_
    conflict`'s own docstring for the exact four conditions) -- it never
    selects a winner itself. A gated-out fast path falls through to the
    SAME general ladder immediately below, completely unmodified.

    D-097.B (all-failed family): when EVERY member carries D-081 semantic
    delete-recommended evidence, Best Take no longer elects a survivor by
    delivery tie-break ("least bad wins"). It looks for a USABLE member --
    one without deterministic unusability evidence (`deterministic_unusable`:
    a ranker fragment penalty relative to a sibling, or local-performance
    corroboration of failure). A usable member competes normally (the label
    conflict is recorded by the caller, never silently trusted); if NONE is
    usable the family yields NO winner (`(None, None, "no_usable_realization")`)
    and the caller marks the candidate incomplete for review rather than
    presenting a clean complete story. Labels alone never delete: a family
    whose members lack deterministic unusability evidence keeps today's
    behaviour byte-for-byte.

    Hybrid session cleanup sees the full message and may recognize which delivery is the
    intended final take. The local Watch+Listen ranker still establishes the fallback,
    but a unique medium-high semantic winner may override a tiny local score difference.
    This can never create a group: it only chooses among members the deterministic
    grouping stage already proved to be competing retries.

    D-082: when semantic labels are NOT decisive (zero or 2+ "winner" labels
    -- the exact D-080 sonography shape, where labels degraded from a
    decisive {"failed", "winner"} to a non-actionable {"keep", "keep"}
    across two live runs on byte-identical DeliveryScorer scores), this no
    longer falls straight to `local_selected_clip_id` (the raw,
    completeness-blind DeliveryScorer rank -- D-080's proven root cause).
    It instead consults, in order, deterministic evidence this codebase
    already computes elsewhere -- no new authority, no weighted scoring:

      1. D-081 `semantic_delete_recommended` evidence (soft: a candidate
         carrying it is excluded from consideration unless that would
         eliminate every candidate -- same "WHEN UNCERTAIN, KEEP" fail-open
         rule as every other check below, and never a second irreversible
         delete authority -- D-081/D-082 Section 12);
      2. attempt completeness (`CandidateTake.complete_idea`, the exact
         signal `take_judge.score_take` already weights for delivery
         scoring -- soft, same fail-open rule: only an EXPLICIT False
         excludes, never an unset/unknown value);
      3. D-063/D-065/D-066 CRITICAL_COVERAGE_DOMINANCE, reused verbatim via
         `claim_coverage_best_take.resolve_critical_coverage_dominance` --
         never reimplemented, including its own safety gates (never prefers
         a proven-incomplete candidate, never overrides a real
         contradiction);
      4. only once dominance finds no single winner AND the survivors'
         CRITICAL-claim coverage sets are genuinely IDENTICAL (a true tie)
         does the local DeliveryScorer ranking get to decide among them --
         its proper role once content is effectively tied (D-082 Section
         8/10). A genuinely disjoint/asymmetric coverage split (distinct
         unique facts, neither a superset of the other) is left exactly as
         it was -- `local_selected_clip_id`, this function's own pre-D-082
         safe default -- rather than forcing delivery to pick a side
         (D-082 Section 7: "remain unresolved / preserve safe behavior").

    Any step that finds nothing decisive falls open to the next one; the
    final fallback is always `local_selected_clip_id`, never worse than
    today's behavior for a genuinely unresolved family.

    D-101 (P0 forensic, hereditary-cancer/papillary-diagnosis cluster):
    two additional, purely additive safety checks close the two proven
    root causes found there -- see `_single_winner_safety_veto`'s and
    `_exclude_incomplete_subset_losers`'s own module comments immediately
    above this function for the full rationale. Neither changes this
    function's contract for the common, safe case.
    """
    winners = []
    for member in members:
        label, confidence = semantic_decisions.get(member.clip_id, ("", 0.0))
        if label == "winner" and confidence >= winner_confidence:
            winners.append((member.clip_id, confidence))
    # D-150 (Phase B): ABSTAIN_CONFLICT/ABSTAIN_INCOMPLETE_CONTEXT veto the
    # fast path exactly like a D-101 `_single_winner_safety_veto` hit --
    # `None` (every pre-D-150 caller) and `"AUTHORITATIVE"` are the only
    # values that ever let this run, so this line is a no-op for every
    # existing caller.
    semantic_authority_blocks_fast_path = semantic_comparative_authority not in (None, AUTHORITY_ALLOWED)
    if len(winners) == 1 and not semantic_authority_blocks_fast_path:
        preferred_id, _ = winners[0]
        veto_reason = _single_winner_safety_veto(preferred_id, members, semantic_delete_recommended)
        if veto_reason is None:
            case_b_conflict = None
            if case_b_evidence_by_id:
                meaning_sufficient_ids = _meaning_sufficient_member_ids(members, semantic_delete_recommended)
                case_b_conflict = _case_b_fast_path_conflict(
                    preferred_id, local_selected_clip_id, meaning_sufficient_ids, case_b_evidence_by_id,
                )
            if case_b_conflict is None:
                if terminal_confidence_out is not None:
                    # D-183: a single, confidence-floor-passing, safety-
                    # veto-clear, case-b-uncontested semantic label is
                    # ALREADY the most decisive structured evidence this
                    # ladder ever produces -- DECISIVE, never re-derived
                    # from `ranked`/`rank_by_id` (which this path never
                    # even consults).
                    terminal_confidence_out["terminal_besttake_confidence"] = _terminal_confidence(
                        _TERMINAL_CONFIDENCE_DECISIVE, "single_semantic_winner",
                        _TERMINAL_CONFIDENCE_PROVENANCE_STRUCTURED_DOMINANCE,
                        candidate_ids=tuple(member.clip_id for member in members),
                        ranked_candidate_ids=(preferred_id,), top_id=preferred_id,
                    )
                if preferred_id == local_selected_clip_id:
                    return local_selected_clip_id, preferred_id, "single_semantic_winner"
                return preferred_id, preferred_id, "single_semantic_winner"
            # D-123: a real, evidenced performance conflict gates this
            # early exit -- fall through to the SAME general ladder below,
            # exactly as though this family had zero/multiple "winner"
            # labels. CASE B never picks a winner here; the ladder does,
            # exactly as it always has.
        # D-101 Root Cause #1: the label is vetoed -- fall through to the
        # general resolution ladder below exactly as though this family
        # had zero/multiple "winner" labels (never a bespoke path).

    member_ids = [member.clip_id for member in members]
    by_id = {member.clip_id: member for member in members}
    if len(member_ids) < 2:
        # D-097.8 (R10): a lone `bts` realization (not audience content at
        # all, so no idea can vanish with it -- unlike D-097.B's lone
        # `failed` delivery, which stays kept) with the Hybrid label at or
        # above the usable floor AND deterministic local corroboration is
        # the D-081 `corroborated_bts_delete` basis whose application was
        # deferred to "the authoritative resolution boundary" -- this is
        # that boundary, and for a singleton nobody else ever reached it:
        # RAW 34043967265 ended the video on "No, no, no, no, no." (bts
        # 0.95, dense_physical_reset:5, visual_fumble:0.85, kept fail-open).
        # Lost BY DECISION (recorded as no_usable_realization), never a
        # label-only drop: without the deterministic evidence it is kept.
        only_id = member_ids[0]
        label, confidence = semantic_decisions.get(only_id, ("", 0.0))
        if (
            label == "bts"
            and confidence >= _BTS_SINGLETON_UNUSABLE_CONFIDENCE
            and (deterministic_unusable or {}).get(only_id, False)
        ):
            return None, None, "single_bts_unusable"
        return local_selected_clip_id, None, "single_member_no_contest"

    def _exclude_unless_all(ids: list[str], excluded: set[str]) -> list[str]:
        survivors = [cid for cid in ids if cid not in excluded]
        return survivors if survivors else list(ids)

    # Step 1: D-081 semantic-delete-recommended evidence.
    delete_recommended_ids = {
        cid for cid in member_ids if (semantic_delete_recommended or {}).get(cid, False)
    }
    if delete_recommended_ids == set(member_ids):
        # D-097.B: every member is semantically failed -- consult objective
        # usability instead of falling open to a tie-break among failures.
        usable = [cid for cid in member_ids if not (deterministic_unusable or {}).get(cid, False)]
        if not usable:
            if terminal_confidence_out is not None:
                # D-183: no valid finalist at all -- reuse existing
                # "no_usable_realization" semantics rather than inventing
                # a new no-candidate vocabulary.
                terminal_confidence_out["terminal_besttake_confidence"] = _terminal_confidence(
                    _TERMINAL_CONFIDENCE_UNKNOWN, "no_usable_realization", _TERMINAL_CONFIDENCE_PROVENANCE_NO_SCORE,
                    candidate_ids=member_ids,
                )
            return None, None, "no_usable_realization"
        survivors = usable
    else:
        survivors = _exclude_unless_all(member_ids, delete_recommended_ids)

    # Step 2: attempt completeness.
    incomplete_ids = {cid for cid in survivors if by_id[cid].complete_idea is False}
    survivors = _exclude_unless_all(survivors, incomplete_ids)

    # Step 2.5 (D-103): required-condition-realization safety. Excluded
    # HERE, not merely vetoed at the fast path, because Steps 3/4's own
    # CRITICAL_COVERAGE_DOMINANCE is driven by the SAME claim classifier
    # that is blind to this realization -- a survivor missing it would
    # otherwise still win dominance purely by holding an unrelated
    # CRITICAL claim of its own. See `_members_missing_required_condition_
    # realization`'s own docstring for the full rationale.
    required_realization_missing_ids = _members_missing_required_condition_realization(survivors, by_id)
    survivors = _exclude_unless_all(survivors, required_realization_missing_ids)

    if len(survivors) >= 2:
        members_pairs = [(cid, by_id[cid]) for cid in member_ids]

        # Step 3/4: D-063/D-065/D-066 CRITICAL_COVERAGE_DOMINANCE, reused.
        dominant_id, _hindsight_rows = resolve_critical_coverage_dominance(members_pairs, survivors)
        if dominant_id is not None:
            if terminal_confidence_out is not None:
                # D-183: Steps 3/4's own dominance IS structured evidence
                # that already, genuinely settles the comparison -- DECISIVE,
                # never re-derived from `rank_by_id`.
                terminal_confidence_out["terminal_besttake_confidence"] = _terminal_confidence(
                    _TERMINAL_CONFIDENCE_DECISIVE, "critical_coverage_dominance",
                    _TERMINAL_CONFIDENCE_PROVENANCE_STRUCTURED_DOMINANCE,
                    candidate_ids=survivors, ranked_candidate_ids=(dominant_id,), top_id=dominant_id,
                )
            if dominant_id == local_selected_clip_id:
                return local_selected_clip_id, dominant_id, "critical_coverage_dominance"
            return dominant_id, dominant_id, "critical_coverage_dominance"

        # Step 5: unique-required-fact safety -- delivery may only settle a
        # GENUINE tie, never an asymmetric/disjoint split. A defensive
        # second check reuses `any_pair_contradicts` (the SAME safety gate
        # `_critical_coverage_dominant_candidate` itself already applies
        # internally) directly on the surviving texts: two candidates that
        # factually contradict each other must never be handed to delivery
        # merely because their CRITICAL-claim coverage sets happened to
        # look identical (e.g. a negated and non-negated claim minted under
        # colliding canonical ids).
        coverage = critical_coverage_sets(members_pairs, survivors)
        if coverage:
            coverage_values = list(coverage.values())
            if any(value != coverage_values[0] for value in coverage_values[1:]):
                if terminal_confidence_out is not None:
                    # D-183: an asymmetric/disjoint CRITICAL-claim split is
                    # structured evidence that genuinely DISAGREES on which
                    # survivor should win -- CONFLICTED, per this task's
                    # own semantics ("structured evidence sources
                    # materially disagree").
                    terminal_confidence_out["terminal_besttake_confidence"] = _terminal_confidence(
                        _TERMINAL_CONFIDENCE_CONFLICTED, "unresolved_unique_fact_asymmetry",
                        _TERMINAL_CONFIDENCE_PROVENANCE_STRUCTURED_DOMINANCE,
                        candidate_ids=survivors,
                    )
                return local_selected_clip_id, None, "unresolved_unique_fact_asymmetry"
        if any_pair_contradicts([str(by_id[cid].text or "") for cid in survivors]):
            if terminal_confidence_out is not None:
                terminal_confidence_out["terminal_besttake_confidence"] = _terminal_confidence(
                    _TERMINAL_CONFIDENCE_CONFLICTED, "unresolved_contradiction",
                    _TERMINAL_CONFIDENCE_PROVENANCE_STRUCTURED_DOMINANCE,
                    candidate_ids=survivors,
                )
            return local_selected_clip_id, None, "unresolved_contradiction"

    # Steps 6-9: delivery score / richness tie-break among the surviving,
    # safe candidate set -- delivery's proper role once content is
    # effectively tied.
    rank_by_id = {row.clip_id: row.score for row in ranked}
    survivor_ranked = [cid for cid in survivors if cid in rank_by_id]
    if survivor_ranked:
        # D-101 Root Cause #2: raw delivery score alone may not settle a
        # tie in favor of a candidate that is an incomplete, literal
        # content subset of another survivor -- see
        # `_exclude_incomplete_subset_losers`'s own module comment above.
        tie_break_pool = _exclude_incomplete_subset_losers(survivor_ranked, by_id)
        best = max(tie_break_pool, key=lambda cid: rank_by_id[cid])
        if terminal_confidence_out is not None:
            # D-183: the ACTUAL terminal comparison this decision was made
            # on -- `tie_break_pool`/`rank_by_id`, the exact same values
            # `max()` itself just used, never re-derived or duplicated.
            terminal_confidence_out["terminal_besttake_confidence"] = _terminal_besttake_confidence(
                tie_break_pool, rank_by_id,
            )
        if best == local_selected_clip_id:
            return local_selected_clip_id, None, "delivery_tie_break_among_survivors"
        return best, best, "delivery_tie_break_among_survivors"

    if terminal_confidence_out is not None:
        terminal_confidence_out["terminal_besttake_confidence"] = _terminal_confidence(
            _TERMINAL_CONFIDENCE_UNKNOWN, "no_score_available_for_any_survivor",
            _TERMINAL_CONFIDENCE_PROVENANCE_NO_SCORE, candidate_ids=survivors,
        )
    return local_selected_clip_id, None, "local_fallback"


def build_flow_b_draft(
    request: ProcessingRequest,
    takes: Iterable[CandidateTake],
    semantic_labels: Iterable[SemanticLabel] = (),
    take_judge_provider: TakeJudgeProvider | None = None,
    clean_cut_provider: CleanCutProvider | None = None,
    composer_provider: ComposerProvider | None = None,
    take_grouping_provider: TakeGroupingProvider | None = None,
    draft_review_provider: DraftReviewProvider | None = None,
    editorial_judge: EditorialJudge | None = None,
    whole_video_context: WholeVideoContext | None = None,
    temporal_trim_diagnostics: Iterable[dict] = (),
    attempt_reconstruction_diagnostics: dict | None = None,
    performance_confirmation_diagnostics: Iterable[dict] = (),
    semantic_equivalence_arbiter: SemanticEquivalenceArbiter | None = None,
    boundary_owner: str = "pre_freeze",
    watch_listen_understandings: Iterable[WatchListenUnderstanding] = (),
) -> ProcessingResult:
    """Build an editable draft after understanding the complete source context.

    ``boundary_owner`` (D-097.C/E) is read only by the physical draft
    wrappers installed around this function (edge-only boundary, interior gap
    trim): ``"post_freeze"`` tells them the universal Clean Cut path owns
    those operations in its post-Freeze BoundaryEngine pass, so they skip
    here instead of acting on a pre-authority candidate set. The legacy
    ``process_local_sources`` callers keep the default.
    """
    del boundary_owner  # consumed by the wrappers; never influences the draft itself
    take_tuple = tuple(takes)
    temporal_trim_diagnostics = tuple(temporal_trim_diagnostics)
    performance_confirmation_diagnostics = tuple(performance_confirmation_diagnostics)
    attempt_reconstruction_diagnostics = dict(attempt_reconstruction_diagnostics or {})
    label_map: dict[str, SemanticLabel] = {label.clip_id: label for label in semantic_labels}
    context_text = whole_video_context.compact_text() if whole_video_context is not None else ""

    # D-050D1: mint `realization_id` here -- on the COMPLETE candidate pool
    # (AttemptReconstructor output plus any `preserved_subspan_candidates`,
    # already merged into `takes` by flow_b.py before this function is
    # ever called), before ANY editorial stage (clean_cut, provider
    # judgements, hybrid/composite resolution) can keep, discard, or
    # transform a candidate. This used to run only over the survivors of
    # all three of those stages (see D-050C3/D-050D's own audit), which
    # meant anything they removed never received a canonical identity at
    # all -- the exact mechanism behind every orphan realization traced in
    # that audit. `apply_clean_cut`/`apply_provider_judgements` are pure
    # partitions of the SAME CandidateTake objects (never rebuild them),
    # and `apply_composite_resolution`'s own kept/deleted split is the
    # same shape -- so minting here, once, is carried forward unchanged
    # into every branch, kept or discarded, by construction. Still the
    # single canonical owner (`mint_realization_id`, canonical_identity.py)
    # -- no second minting implementation anywhere else in this function.
    take_tuple = tuple(
        take if take.realization_id else dataclass_replace(
            take, realization_id=mint_realization_id(take.source_asset_id, take.attempt_id, take.text),
        )
        for take in take_tuple
    )

    # Pass 1: deterministic/local cleanup remains the backbone and removes obvious
    # recording garbage before optional semantic reasoning spends anything.
    kept, deterministic_discarded, decisions = apply_clean_cut(take_tuple, whole_video_context)
    clean_judged = safe_clean_cut_judge(clean_cut_provider, kept)
    kept, provider_discarded, clean_judge_diagnostics = apply_provider_judgements(kept, clean_judged)
    discarded = tuple(deterministic_discarded) + tuple(provider_discarded)

    for item in clean_judge_diagnostics:
        if not item.get("applied_mixed_trim"):
            continue
        parent_id = str(item.get("clip_id") or "")
        parent_label = label_map.get(parent_id)
        if parent_label is None:
            continue
        child_ids = [item.get("kept_clip_id"), *(item.get("discarded_clip_ids") or [])]
        for child_id in child_ids:
            if child_id:
                label_map[str(child_id)] = SemanticLabel(
                    str(child_id), parent_label.role, parent_label.confidence, parent_label.reason
                )

    # Pass 2: batch semantic intent by bounded creator mini-session. This catches BTS,
    # self-review and failed attempts with context while avoiding one paid call for every
    # singleton/retry group. Semantic winner/alternate evidence is retained for Pass 3.
    #
    # CompositeResolver (composite_resolver.py, see D-023): the single, directly-
    # callable authority for delivery restoration/rescue/composite marking. Owns
    # what used to be 14 separately-monkeypatched hybrid_* authorities layered
    # onto this one call, in the same order, same algorithms -- now one explicit
    # composition instead of an implicit import-time chain.
    hybrid_cleanup, composite_split_ids = apply_composite_resolution(
        kept,
        whole_video_context,
        editorial_judge,
    )
    kept = hybrid_cleanup.kept
    discarded = (*discarded, *hybrid_cleanup.deleted)
    hybrid_semantic_decisions = {
        clip_id: (label, float(confidence))
        for clip_id, label, confidence in hybrid_cleanup.semantic_decisions
    }
    # D-082 Section 12: surface D-081's semantic_delete_recommended evidence
    # (recorded per-window inside hybrid_cleanup.diagnostics, never
    # destructive on its own) so _semantic_best_take's non-decisive-label
    # fallback can treat it as soft negative evidence. OR-across windows:
    # if any window flagged the candidate, that evidence is never silently
    # dropped, matching D-081's own "never discard the evidence" posture.
    hybrid_semantic_delete_recommended: dict[str, bool] = {}
    for diagnostic in hybrid_cleanup.diagnostics:
        for decision in diagnostic.get("decisions") or ():
            clip_id = decision.get("clip_id")
            if not clip_id:
                continue
            if decision.get("semantic_delete_recommended"):
                hybrid_semantic_delete_recommended[clip_id] = True
            else:
                hybrid_semantic_delete_recommended.setdefault(clip_id, False)

    # D-097.B: D-081's local-performance corroboration per candidate (the
    # `local_failure_corroborated` flag hybrid_session_cleanup records on
    # every window decision) -- deterministic unusability evidence, never a
    # label.
    hybrid_local_failure_corroborated: dict[str, bool] = {}
    for diagnostic in hybrid_cleanup.diagnostics:
        for decision in diagnostic.get("decisions") or ():
            clip_id = decision.get("clip_id")
            if clip_id and decision.get("local_failure_corroborated"):
                hybrid_local_failure_corroborated[clip_id] = True

    # D-050D1: `realization_id` is minted once, above, before Pass 1 even
    # starts -- every member of `kept` here already carries it (see the
    # single minting pass at the top of this function). No second minting
    # pass; no reminting.

    # Pass 3: deterministic retry grouping + Best Take runs after semantic garbage is
    # removed. The local ranker remains the fallback. If Hybrid already identified one
    # clear winner among members of the same proven retry group, that editorial winner
    # takes precedence over a marginal local score difference.
    take_by_id = {take.clip_id: take for take in kept}
    grouping = safe_group_takes_by_sessions(
        take_grouping_provider,
        kept,
        whole_video_context,
        context_text=context_text,
    )
    # CompositeResolver's composite-marked pairs (see above) are forced into
    # singleton groups here so BestTakeResolver's one-winner competition
    # cannot re-collapse an intended composite delivery. Direct call, no
    # ContextVar, no monkeypatch of safe_group_takes_by_sessions.
    grouping = apply_composite_group_split(grouping, kept, composite_split_ids)

    # Phase 2 of the architecture rebalance: a narrow, gated semantic-
    # equivalence arbiter may confirm that two groups the lexical layer left
    # separate are recording attempts of the same intended idea, merging
    # them into one retry contest BEFORE the completeness/performance
    # ranking (safe_rank_takes) and deterministic Best Take run below. This
    # runs here, directly on safe_group_takes_by_sessions's resolved output,
    # rather than being threaded as a parameter through that call -- see
    # take_grouping_provider.safe_group_takes's docstring for why: this
    # function is already wrapped by several production monkeypatch layers
    # that hardcode its current signature, and this is the one choke point
    # every one of those layers' output must pass through regardless.
    # D-025: composite_split_ids are protected here too, not just at the
    # grouping-split step above -- otherwise this call's own, separate
    # arbiter invocation can re-merge an accepted composite's pieces (or
    # merge one into an unrelated group), silently discarding a decision
    # CompositeResolver already made. See reconcile_semantic_idea_
    # equivalence's own docstring for the exact RAW that exposed this.
    # D-100 (D-099 Gap #1): `whole_video_context` is already a live local
    # variable at this exact call site (used one call earlier for session
    # partitioning's `context_text`) -- D-099 traced that it was never
    # threaded any further into the deterministic restart-evidence merge
    # loop below, so confirmed multimodal recording-behavior evidence
    # (`performance_confirmation.py`'s `wrong_take`/`retry_setup` events)
    # never reached the retry-family authority. This is the minimal bridge:
    # a narrow, plain-tuple extraction (never the full context object) so
    # `reconcile_semantic_idea_equivalence` can let that evidence corroborate
    # a weaker lexical link than its existing rules require -- optional and
    # purely additive; see `take_grouping.multimodal_corroborated_retry`.
    confirmed_recording_evidence = confirmed_recording_behavior_events(whole_video_context)
    # D-158: real Watch+Listen evidence only ever reaches the authority when
    # the capability flag is ON (default OFF -- pre-D-158 behavior);
    # `reconcile_semantic_idea_equivalence` itself also gates on the same
    # flag, so this is belt-and-suspenders, not a second flag definition --
    # skipping the index build entirely when OFF/empty keeps the OFF path
    # byte-identical work, not just byte-identical output.
    # D-161: the Relation Discovery Gate is a SEPARATE capability flag from
    # D-158's merge-veto flag (different authorities -- see D-161 decision
    # entry) and it also needs this same span index to build discovery
    # candidates from. Build the index when EITHER flag is ON so that
    # discovery works even when the family-evidence flag stays OFF; both
    # flags OFF (the shared default) still skips the build entirely.
    # D-163: the BestTake evidence guard is a THIRD, separate authority
    # (docs/CUTSELL_DECISIONS.md D-163) that reuses this SAME span index
    # (per-member entry/delivery/exit usability, behavior hypotheses,
    # conflict flags) -- never a fourth recomputation.
    watch_listen_spans_by_id = (
        build_understanding_span_index(watch_listen_understandings)
        if (
            watch_listen_family_evidence_enabled()
            or watch_listen_relation_discovery_enabled()
            or watch_listen_besttake_evidence_enabled()
        )
        and watch_listen_understandings
        else None
    )
    semantic_equivalence_groups, semantic_equivalence_diagnostics = reconcile_semantic_idea_equivalence(
        grouping.groups, kept, semantic_equivalence_arbiter,
        protected_ids=composite_split_ids,
        confirmed_recording_evidence=confirmed_recording_evidence,
        watch_listen_spans_by_id=watch_listen_spans_by_id,
    )

    # D-058 Phase 1: one final cohesion-validation pass -- see
    # take_grouping_provider.split_incohesive_retry_groups's own module
    # comment for the full defect/fix rationale (docs/CUTSELL_DECISIONS.md
    # D-057/D-058). Runs after every merging step above, on whatever groups
    # they produced, and before Best Take ranking below ever treats a group
    # as one mutually-exclusive contest. Same `protected_ids` contract as
    # the arbiter merge immediately above -- an accepted composite's pieces
    # are never re-examined here either.
    # D-094.F3: hand the reconcile stage's own confirmed merges to the
    # cohesion pass as prior evidence (same run, same arbiter, same pair
    # texts) so a confirmed retry pair can never be split merely because it
    # fell outside this pass's bounded re-ask.
    prior_confirmations = {
        frozenset((str(row.get("left_clip_id") or ""), str(row.get("right_clip_id") or ""))):
        (float(row.get("confidence") or 0.0), str(row.get("reason") or ""))
        for row in (semantic_equivalence_diagnostics.get("merges") or ())
        if isinstance(row, dict) and row.get("left_clip_id") and row.get("right_clip_id")
    }
    semantic_equivalence_groups, cohesion_diagnostics = split_incohesive_retry_groups(
        semantic_equivalence_groups, kept, semantic_equivalence_arbiter,
        protected_ids=composite_split_ids,
        prior_confirmations=prior_confirmations,
        # D-094.2: runtime-config only (default OFF); see the policy field.
        policy=SemanticEquivalenceGatePolicy(
            accept_complete_pairwise_singleton_bridge=_env_flag_enabled(
                "CUTSELL_BRIDGE_COMPLETE_PAIRWISE_SINGLETON"
            ),
        ),
    )
    cohesion_diagnostics = {
        **cohesion_diagnostics,
        "accept_complete_pairwise_singleton_bridge": _env_flag_enabled(
            "CUTSELL_BRIDGE_COMPLETE_PAIRWISE_SINGLETON"
        ),
    }
    group_members = [tuple(take_by_id[clip_id] for clip_id in ids) for ids in semantic_equivalence_groups]

    groups = []
    clip_to_group: Dict[str, str] = {}
    judge_statuses = Counter()
    judge_reasons = Counter()
    alternate_group_count = 0
    semantic_best_take_override_count = 0
    judge_group_diagnostics = []
    watch_listen_besttake_results: list = []
    watch_listen_besttake_v2_results: list = []
    watch_listen_besttake_guard_authority_results: list = []
    bounded_finalist_arbiter_diagnostics_rows: list = []
    no_usable_realization_ids: set[str] = set()
    events_by_source: dict[str, tuple] = {}
    if whole_video_context is not None:
        for source in whole_video_context.sources:
            events_by_source[source.source_asset_id] = tuple(source.events)

    for members in group_members:
        if not members:
            continue
        if len(members) >= 2:
            alternate_group_count += 1
        judged = safe_rank_takes(members, take_judge_provider)
        # D-097 (PO adjustment §2): measured cleanliness evidence (accidental
        # interior dead air, multimodal resets) adjusts the family ranking
        # BEFORE any winner is read off it; markers land in `ranked[].reason`.
        source_events = events_by_source.get(members[0].source_asset_id, ())
        ranked, cleanliness_rows = apply_delivery_cleanliness_evidence(judged.ranked, members, source_events)
        judge_statuses[judged.status.status] += 1
        if judged.status.reason:
            judge_reasons[judged.status.reason] += 1
        local_selected_clip_id = ranked[0].clip_id
        # D-094.3 (F8): prefer the labels of a window that judged the WHOLE
        # family together over the per-clip cross-window merge.
        family_semantic_decisions, semantic_label_source = family_scoped_semantic_decisions(
            members, hybrid_semantic_decisions, hybrid_cleanup.diagnostics,
        )
        # D-146 (Phase A, observability only -- docs/CUTSELL_DECISIONS.md
        # D-146): a pure, additive projection of the SAME hybrid_cleanup.
        # diagnostics window rows and the SAME semantic_label_source computed
        # immediately above. Read by nothing above this line and nothing
        # below it that makes a decision -- see semantic_authority_
        # observability.py's own module docstring for the non-circularity
        # proof and today's honest authority description.
        family_semantic_authority_observability = family_authority_diagnostics(
            [member.clip_id for member in members],
            hybrid_cleanup.diagnostics,
            semantic_label_source,
        )
        # D-097.B: deterministic unusability evidence per member -- a ranker
        # fragment penalty relative to a sibling, or D-081 local-performance
        # corroboration. Labels are never part of this map.
        ranked_reason_by_id = {row.clip_id: str(row.reason or "") for row in ranked}
        deterministic_unusable = {
            member.clip_id: bool(
                any(marker in ranked_reason_by_id.get(member.clip_id, "") for marker in FRAGMENT_PENALTY_MARKERS)
                or hybrid_local_failure_corroborated.get(member.clip_id, False)
            )
            for member in members
        }
        # D-123 (docs/CUTSELL_DECISIONS.md D-123): D-122's CASE B evidence
        # objects (raw dataclasses, not yet JSON-projected) are built HERE,
        # before the decision, so `_semantic_best_take` can use them to gate
        # (never replace) its own `single_semantic_winner` early exit.
        case_b_evidence_objects = {
            member.clip_id: build_case_b_performance_evidence(member, whole_video_context)
            for member in members
        }
        # D-123 counterfactual: the decision `_semantic_best_take` would
        # make WITHOUT CASE B evidence -- byte-identical to pre-D-123/D-122
        # behavior (omits `case_b_evidence_by_id`) -- computed for
        # observability only (`winner_path_before`), never used as the
        # actual decision below.
        _before_selected_clip_id, _before_preferred_clip_id, before_semantic_best_take_reason = _semantic_best_take(
            members,
            family_semantic_decisions,
            local_selected_clip_id,
            ranked,
            semantic_delete_recommended=hybrid_semantic_delete_recommended,
            deterministic_unusable=deterministic_unusable,
        )
        # D-150 (Phase B; docs/CUTSELL_DECISIONS.md D-150): the ONE narrow
        # authority gate over `_semantic_best_take`'s `single_semantic_
        # winner` fast path. Consumes `family_semantic_authority_
        # observability`'s OWN already-computed `family_complete_context`/
        # `complete_context_conflict` fields (single source of truth --
        # never recomputed here) to decide whether the family-scoped
        # comparative label above may be trusted at all. Computed for
        # EVERY group (including singletons -- `resolve_semantic_
        # comparative_authority` returns AUTHORITATIVE for `len(members) <
        # 2`, so this is a no-op there) so the diagnostics below are always
        # populated, exactly like `family_semantic_authority_observability`
        # itself already is.
        semantic_authority_gate = semantic_authority_gate_diagnostics(
            [member.clip_id for member in members],
            family_semantic_decisions,
            family_semantic_authority_observability,
        )
        # D-183 (docs/CUTSELL_DECISIONS.md D-183): a fresh dict the REAL
        # decision call below populates additively at the exact point its
        # own comparative outcome (dominance / asymmetry / contradiction /
        # raw-score tie-break) is reached -- never a second invocation,
        # never a re-derivation that could drift from the actual decision.
        _terminal_confidence_out: dict = {}
        selected_clip_id, semantic_preferred_clip_id, semantic_best_take_reason = _semantic_best_take(
            members,
            family_semantic_decisions,
            local_selected_clip_id,
            ranked,
            semantic_delete_recommended=hybrid_semantic_delete_recommended,
            deterministic_unusable=deterministic_unusable,
            case_b_evidence_by_id=case_b_evidence_objects,
            semantic_comparative_authority=semantic_authority_gate["semantic_authority_gate_status"],
            terminal_confidence_out=_terminal_confidence_out,
        )
        _terminal_besttake_confidence_result = _terminal_confidence_out.get("terminal_besttake_confidence")
        no_usable_realization = selected_clip_id is None
        all_delete_recommended = len(members) >= 2 and all(
            hybrid_semantic_delete_recommended.get(member.clip_id, False) for member in members
        )
        if no_usable_realization:
            selected_clip_id = ""
            no_usable_realization_ids.update(member.clip_id for member in members)
        if semantic_preferred_clip_id and selected_clip_id != local_selected_clip_id:
            semantic_best_take_override_count += 1
        membership_key = "semantic:" + hashlib.sha256(
            "|".join(sorted(member.clip_id for member in members)).encode()
        ).hexdigest()[:16]
        gid = _group_id(request.project_id, membership_key)
        groups.append(TakeGroup(
            group_id=gid,
            semantic_key=membership_key,
            candidate_ids=tuple(member.clip_id for member in members),
            ranked=ranked,
            selected_clip_id=selected_clip_id,
        ))
        # D-097.8 (R10): a singleton dropped as no-usable (a corroborated
        # `bts` lone take) is recorded exactly like a dropped family so the
        # StoryValidator classifies it LOST_IN_NO_USABLE_REALIZATION_FAMILY
        # (a decision, never an accidental loss) -- ordinary singletons are
        # still not contests and stay out of these rows.
        if len(members) >= 2 or no_usable_realization:
            # D-122 (advisory/diagnostics only -- see docs/CUTSELL_DECISIONS.md
            # D-122): expose D-121's confirmed single_semantic_winner bypass
            # and D-115's DELIVERY-zone performance evidence per competitor.
            # `winner_path`/`performance_consulted_before_winner` reflect the
            # ACTUAL final `semantic_best_take_reason` (D-123-aware, i.e.
            # after any bypass); `deterministic_best_take_authority.py`
            # additively upgrades both to DETERMINISTIC_OVERRIDE later, once
            # that is known.
            case_b_winner_path, case_b_performance_consulted = _winner_path_from_reason(semantic_best_take_reason)
            case_b_semantic_fast_path_candidate = _single_semantic_winner_candidate(members, family_semantic_decisions)
            case_b_evidence_by_id = {
                clip_id: case_b_performance_evidence_diagnostics(evidence)
                for clip_id, evidence in case_b_evidence_objects.items()
            }
            # D-123 (docs/CUTSELL_DECISIONS.md D-123): the gate is a pure,
            # deterministic function of already-computed inputs -- recomputed
            # here (never threaded through `_semantic_best_take`'s return
            # value, matching D-122's own `_single_semantic_winner_candidate`
            # precedent) so diagnostics can show WHY the bypass did or did
            # not fire, using the EXACT same function `_semantic_best_take`
            # itself called.
            winner_path_before, performance_consulted_before = _winner_path_from_reason(before_semantic_best_take_reason)
            winner_path_after, performance_consulted_after = case_b_winner_path, case_b_performance_consulted
            meaning_sufficient_ids = _meaning_sufficient_member_ids(members, hybrid_semantic_delete_recommended)
            case_b_conflict_basis = None
            if case_b_semantic_fast_path_candidate is not None:
                case_b_conflict_basis = _case_b_fast_path_conflict(
                    case_b_semantic_fast_path_candidate, local_selected_clip_id,
                    meaning_sufficient_ids, case_b_evidence_objects,
                )
            case_b_conflict_present = case_b_conflict_basis is not None
            # D-180 (docs/CUTSELL_DECISIONS.md D-180): pure, additive
            # observability companion -- recomputes the SAME condition-4
            # evaluation regardless of outcome so diagnostics can show WHY
            # a bypass did or did not fire, mirroring D-123's own before/
            # after precedent above. Never consulted by `_semantic_best_
            # take`'s own ladder; never changes `case_b_conflict_basis`/
            # `case_b_conflict_present`/`selected_clip_id`/membership.
            case_b_condition4_diagnostics = _case_b_condition4_diagnostics(
                case_b_semantic_fast_path_candidate, local_selected_clip_id,
                meaning_sufficient_ids, case_b_evidence_objects,
            )
            semantic_fast_path_bypassed = bool(
                before_semantic_best_take_reason == "single_semantic_winner"
                and semantic_best_take_reason != "single_semantic_winner"
            )
            bypass_reason = "case_b_performance_conflict" if semantic_fast_path_bypassed else None
            # D-128 (docs/CUTSELL_DECISIONS.md D-128; docs/CUTSELL_
            # MULTIMODAL_FALLBACK_ARBITER_FORENSIC_D127.md): Phase 1
            # SHADOW-ONLY Class B detection. `detect_class_b_trigger` is a
            # pure, read-only classifier of fields already computed above
            # -- it NEVER calls a provider, NEVER changes `selected_
            # clip_id`/`ranked`/`winner_path`/membership/Boundary. Reuses
            # `_single_winner_safety_veto` (unchanged, D-101/D-103) as the
            # SAME deterministic safety check the general ladder already
            # applies -- never a new semantic classifier.
            fallback_safety_excluded_ids = {
                member.clip_id for member in members
                if member.clip_id != case_b_semantic_fast_path_candidate
                and _single_winner_safety_veto(
                    member.clip_id, members, hybrid_semantic_delete_recommended,
                ) is not None
            }
            fallback_trigger_decision = detect_class_b_trigger(
                family_id=gid,
                member_ids=tuple(member.clip_id for member in members),
                semantic_winner=case_b_semantic_fast_path_candidate,
                deliveryscore_winner=local_selected_clip_id,
                meaning_sufficient_ids=meaning_sufficient_ids,
                case_b_evidence_by_id=case_b_evidence_objects,
                safety_excluded_ids=fallback_safety_excluded_ids,
            )
            # D-163 Phase D (docs/CUTSELL_DECISIONS.md D-163): Watch+Listen
            # PERFORMANCE/USABILITY evidence, diagnostic-only in this task
            # (see watch_listen_besttake_evidence.py's own module docstring
            # for the full "why diagnostic, not action" contract). Default
            # OFF (CUTSELL_WATCH_LISTEN_BESTTAKE_EVIDENCE_ENABLED) -- a total
            # no-op (byte-identical selected_clip_id/ranked/membership) when
            # off or when no Watch+Listen span exists for a member.
            watch_listen_besttake_row: dict = {}
            if watch_listen_besttake_evidence_enabled() and watch_listen_spans_by_id:
                _wlbt_evidence_by_id = {
                    member.clip_id: build_watch_listen_besttake_evidence(
                        member.clip_id,
                        watch_listen_spans_by_id.get(member.clip_id),
                        meaning_sufficient=member.clip_id in meaning_sufficient_ids,
                    )
                    for member in members
                }
                if any(ev is not None for ev in _wlbt_evidence_by_id.values()):
                    _wlbt_result = evaluate_watch_listen_besttake_guard(
                        winner_id=selected_clip_id or None,
                        meaning_sufficient_ids=meaning_sufficient_ids,
                        evidence_by_id=_wlbt_evidence_by_id,
                        ranked=[{"clip_id": item.clip_id, "score": item.score} for item in ranked],
                    )
                    watch_listen_besttake_row = watch_listen_besttake_group_row(_wlbt_result, selected_clip_id)
                    watch_listen_besttake_results.append(_wlbt_result)
                    # D-172 (docs/CUTSELL_DECISIONS.md D-172): Zone-Usability
                    # V2 diagnostic consumption. Nested inside D-163's own
                    # evidence-collection block by construction -- V2
                    # consumption is never evaluated when D-163's own base
                    # evidence flag is off (see watch_listen_besttake_v2_
                    # evidence.py's own "Feature flag" section). Recomputes
                    # no perception: `build_candidate_zone_usability_v2`
                    # wraps D-115/D-155/D-167's own already-tested, pure
                    # projections. Diagnostic-only -- never mutates
                    # `selected_clip_id`/`ranked`/membership/Boundary/Pacing;
                    # `winner_after == winner_before` always in this task.
                    if zone_usability_v2_besttake_enabled():
                        _wlbt_v2_evidence_by_id = {
                            member.clip_id: build_candidate_zone_usability_v2(member, whole_video_context)
                            for member in members
                        }
                        _wlbt_v2_result = evaluate_watch_listen_besttake_guard_v2(
                            winner_id=selected_clip_id or None,
                            meaning_sufficient_ids=meaning_sufficient_ids,
                            v1_evidence_by_id=_wlbt_evidence_by_id,
                            v2_evidence_by_id=_wlbt_v2_evidence_by_id,
                            ranked=[{"clip_id": item.clip_id, "score": item.score} for item in ranked],
                        )
                        watch_listen_besttake_row.update(
                            watch_listen_besttake_v2_group_row(_wlbt_v2_result, selected_clip_id)
                        )
                        watch_listen_besttake_v2_results.append(_wlbt_v2_result)
                        # D-174 (docs/CUTSELL_DECISIONS.md D-174): Watch+
                        # Listen BestTake Guard Authority, Phase 1 (pure
                        # eligibility decision -- see watch_listen_besttake_
                        # guard_authority.py's own module docstring). Nested
                        # inside D-172's own V2 block: this authority never
                        # considers V1-only evidence. Diagnostic-only here --
                        # the real winner mutation (Phase 2) happens later,
                        # in universal_clean_cut.py, only if this evaluation
                        # marks GUARD_REJECT_CURRENT_WINNER and only when its
                        # own separate flag is ON.
                        if watch_listen_besttake_guard_authority_enabled():
                            _wlbt_authority_result = evaluate_watch_listen_besttake_guard_authority(
                                winner_id=selected_clip_id or None,
                                meaning_sufficient_ids=meaning_sufficient_ids,
                                member_count=len(members),
                                v2_result=_wlbt_v2_result,
                                v2_evidence_by_id=_wlbt_v2_evidence_by_id,
                                case_b_conflict_present=case_b_conflict_present,
                            )
                            watch_listen_besttake_row.update(
                                watch_listen_besttake_guard_authority_row(_wlbt_authority_result)
                            )
                            watch_listen_besttake_guard_authority_results.append(_wlbt_authority_result)
            # D-184 (docs/CUTSELL_DECISIONS.md D-184): Bounded Finalist
            # Arbiter -- OFFLINE / DIAGNOSTIC ONLY. Independently flag-gated
            # (never depends on D-163/D-172/D-174's own flags being on).
            # Consumes D-183's OWN already-computed `_terminal_besttake_
            # confidence_result` (never re-derived) plus a fresh D-172 V2
            # projection per member (built here so this diagnostic never
            # depends on D-172's own flag) and P0 language/proposition
            # evidence from each member's own text. Zero effect on
            # `selected_clip_id`/`ranked`/membership/Boundary/Pacing/
            # Renderer -- `action_applied` is always `False`.
            bounded_finalist_arbiter_row: dict = {}
            if bounded_finalist_arbiter_enabled():
                _arbiter_terminal_state = (
                    _terminal_besttake_confidence_result.confidence_state
                    if _terminal_besttake_confidence_result is not None else None
                )
                _arbiter_result = evaluate_bounded_finalist_arbiter(FinalistArbiterInput(
                    family_id=gid,
                    candidate_ids=tuple(member.clip_id for member in members),
                    meaning_sufficient_candidate_ids=tuple(sorted(meaning_sufficient_ids)),
                    terminal_confidence_state=_arbiter_terminal_state,
                    terminal_scores={item.clip_id: item.score for item in ranked},
                    candidate_texts={member.clip_id: member.text for member in members},
                    v2_evidence_by_id={
                        member.clip_id: build_candidate_zone_usability_v2(member, whole_video_context)
                        for member in members
                    },
                    performance_evidence_by_id=case_b_evidence_objects,
                ))
                bounded_finalist_arbiter_row = bounded_finalist_arbiter_diagnostics(_arbiter_result)
                bounded_finalist_arbiter_diagnostics_rows.append(bounded_finalist_arbiter_row)
            judge_group_diagnostics.append({
                "group_id": gid,
                "selected_clip_id": selected_clip_id,
                "local_selected_clip_id": local_selected_clip_id,
                "semantic_preferred_clip_id": semantic_preferred_clip_id,
                "semantic_override_applied": selected_clip_id != local_selected_clip_id,
                "semantic_best_take_reason": semantic_best_take_reason,
                # D-094.3 (F8): the labels the decision was actually made on
                # (family-window labels when a family-complete window exists).
                "semantic_label_source": semantic_label_source,
                # D-146 (Phase A, observability only): family_complete_context
                # (true/false/unknown), omitted_candidate_ids, partial_window_
                # conflict, provider_authority_applied (today's ACTUAL behavior,
                # never a Phase-B aspiration), provider_config (temperature/
                # prompt_version honestly reported UNKNOWN -- neither field
                # exists in the current EditorialJudgeResult contract), and
                # family_scoped_source_info (the same semantic_label_source
                # above, re-exposed under this module's own field name for a
                # self-contained report). Zero effect on selected_clip_id,
                # ranked, membership, grouping, or Boundary.
                "semantic_authority_observability": family_semantic_authority_observability,
                # D-150 (Phase B; docs/CUTSELL_DECISIONS.md D-150): the
                # gate's own decision -- semantic_authority_gate_status is
                # ALSO what was actually passed as semantic_comparative_
                # authority to the real _semantic_best_take call above, so
                # this is not merely descriptive, it is the exact input
                # that produced selected_clip_id/semantic_best_take_reason.
                **semantic_authority_gate,
                "semantic_candidates": [
                    {
                        "clip_id": member.clip_id,
                        "label": family_semantic_decisions.get(member.clip_id, ("", 0.0))[0],
                        "confidence": family_semantic_decisions.get(member.clip_id, ("", 0.0))[1],
                    }
                    for member in members
                ],
                "execution_status": judged.status.status,
                "execution_reason": judged.status.reason,
                "ranked": [
                    {"clip_id": item.clip_id, "score": item.score, "reason": item.reason}
                    for item in ranked
                ],
                # D-097: cleanliness evidence rows and the all-failed outcome.
                "delivery_cleanliness": cleanliness_rows,
                "no_usable_realization": no_usable_realization,
                # D-097.9 (R11): WHY the family has no usable realization --
                # "no_usable_realization" (D-097.B: every delivery of an
                # intended idea failed; the story is incomplete) or
                # "single_bts_unusable" (D-097.8 R10: a corroborated lone
                # `bts` take -- recording-process material, no audience idea
                # vanishes with it, the story stays complete). RAW 34045158712
                # dropped "¡Vamos!" correctly and then refused delivery
                # because the two were indistinguishable downstream.
                "no_usable_realization_basis": (semantic_best_take_reason if no_usable_realization else None),
                "all_members_delete_recommended": all_delete_recommended,
                "label_conflict_routed": bool(all_delete_recommended and not no_usable_realization),
                "member_usability": {
                    member.clip_id: {
                        "delete_recommended": bool(hybrid_semantic_delete_recommended.get(member.clip_id, False)),
                        "deterministic_unusable": deterministic_unusable.get(member.clip_id, False),
                        "local_failure_corroborated": hybrid_local_failure_corroborated.get(member.clip_id, False),
                        "ranker_reason": ranked_reason_by_id.get(member.clip_id, ""),
                    }
                    for member in members
                },
                # D-122: BestTake CASE B evidence infrastructure -- advisory
                # only, never consulted above this line, never changes
                # `selected_clip_id`/`ranked`/membership/grouping/Boundary.
                "winner_path": case_b_winner_path,
                "performance_consulted_before_winner": case_b_performance_consulted,
                "deliveryscore_top_candidate": local_selected_clip_id,
                "semantic_fast_path_candidate": case_b_semantic_fast_path_candidate,
                "case_b_evidence": case_b_evidence_by_id,
                # D-123 (docs/CUTSELL_DECISIONS.md D-123): the bounded
                # fast-path gate's own observability -- "before" reflects the
                # semantic-only counterfactual (as if D-123 did not exist),
                # "after" reflects what `_semantic_best_take` actually
                # returned once the gate was consulted. A bypass NEVER picks
                # a winner from CASE B evidence directly; it only prevents
                # the single_semantic_winner early return so the EXISTING
                # DeliveryScorer/deterministic path decides instead.
                "winner_path_before": winner_path_before,
                "winner_path_after": winner_path_after,
                "semantic_fast_path_bypassed": semantic_fast_path_bypassed,
                "bypass_reason": bypass_reason,
                "case_b_conflict_present": case_b_conflict_present,
                "case_b_conflict_basis": case_b_conflict_basis,
                # D-180: condition-4 materiality-stabilization observability
                # (case_b_count_difference_present, case_b_materiality_
                # evidence_available, case_b_materiality_state, case_b_
                # materiality_source, case_b_condition4_actionable, case_b_
                # condition4_reason) -- no transcript dump, no QA reference
                # info, advisory only.
                **case_b_condition4_diagnostics,
                # D-183 (docs/CUTSELL_DECISIONS.md D-183; post D-182
                # forensic): terminal BestTake confidence/decisiveness
                # CLASSIFICATION ONLY -- no finalist arbiter, no winner
                # mutation. `None` only when this exact family never
                # reached a comparative step at all (a true single-member
                # family, out of this classification's own scope per its
                # own docstring); every genuine >=2-member contest above
                # is always populated.
                **(
                    terminal_besttake_confidence_diagnostics(_terminal_besttake_confidence_result)
                    if _terminal_besttake_confidence_result is not None
                    else {
                        "terminal_besttake_confidence_state": None,
                        "terminal_besttake_confidence_reason": "not_a_contest_single_member_family",
                        "terminal_besttake_candidate_count": len(members),
                        "terminal_besttake_top_candidate_id": None,
                        "terminal_besttake_runner_up_candidate_id": None,
                        "terminal_besttake_top_score": None,
                        "terminal_besttake_runner_up_score": None,
                        "terminal_besttake_score_margin": None,
                        "terminal_besttake_structured_dominance_present": False,
                        "terminal_besttake_conflict_present": False,
                        "terminal_besttake_decisive": False,
                    }
                ),
                "meaning_sufficient_candidates": sorted(meaning_sufficient_ids),
                "final_winner": selected_clip_id,
                # D-128 (docs/CUTSELL_DECISIONS.md D-128): Phase 1
                # SHADOW-ONLY Class B fallback trigger observability --
                # never consulted above this line, never changes
                # selected_clip_id/ranked/membership/grouping/Boundary.
                **fallback_trigger_diagnostics(fallback_trigger_decision),
                # D-163: Watch+Listen BestTake evidence guard -- diagnostic
                # only in this task; empty dict when the flag is off or no
                # Watch+Listen evidence exists for this family.
                **watch_listen_besttake_row,
                # D-184: Bounded Finalist Arbiter -- diagnostic only; empty
                # dict when CUTSELL_BOUNDED_FINALIST_ARBITER_ENABLED is off.
                **bounded_finalist_arbiter_row,
            })
        for member in members:
            clip_to_group[member.clip_id] = gid

    # Pass 4: only after retry families have been judged and a logical winner has been
    # chosen do we touch physical edit boundaries. Keep the logical clip IDs stable so
    # semantic labels, retry-group membership and the already-made Best Take decision
    # cannot be invalidated by a later timestamp adjustment.
    kept, post_best_take_trim_diagnostics = refine_takes_with_temporal_context(
        kept,
        whole_video_context,
        preserve_clip_id=True,
    )
    temporal_trim_diagnostics = (
        *temporal_trim_diagnostics,
        *post_best_take_trim_diagnostics,
    )

    surviving_labels = tuple(label_map[take.clip_id] for take in kept if take.clip_id in label_map)
    strategy = choose_strategy(surviving_labels, kept)
    natural_selected = compose_selected(kept, groups, surviving_labels)

    composition = safe_compose_order(
        composer_provider,
        natural_selected,
        surviving_labels,
        strategy,
        context_text=context_text,
    )
    selected_map = {take.clip_id: take for take in natural_selected}
    composed_takes = tuple(selected_map[clip_id] for clip_id in composition.ordered_clip_ids)

    review = safe_review_draft(
        draft_review_provider,
        composed_takes,
        surviving_labels,
        strategy,
        context_text=context_text,
    )
    composed_map = {take.clip_id: take for take in composed_takes}
    selected_takes = tuple(composed_map[clip_id] for clip_id in review.ordered_clip_ids)
    selected_ids = {take.clip_id for take in selected_takes}

    initially_removed_ids = set(composed_map) - selected_ids
    removed_group_ids = {
        clip_to_group[clip_id]
        for clip_id in initially_removed_ids
        if clip_id in clip_to_group
    }
    review_removed_ids = {
        take.clip_id
        for take in kept
        if take.clip_id in initially_removed_ids or clip_to_group.get(take.clip_id) in removed_group_ids
    }
    review_removed = tuple(take for take in kept if take.clip_id in review_removed_ids)

    selected = tuple(
        _draft_clip(
            take,
            role=label_map.get(take.clip_id, SemanticLabel(take.clip_id, SemanticRole.OTHER, 0.0)).role,
            group_id=clip_to_group.get(take.clip_id),
            selected=True,
        )
        for take in selected_takes
    )
    # D-097.B: members of a family with no usable realization are DISCARDED
    # (never parked as alternates where a later pass could re-select them);
    # the drop is recorded below and surfaced as an incomplete-story review
    # signal downstream, never as a clean complete story.
    no_usable_removed = tuple(take for take in kept if take.clip_id in no_usable_realization_ids)
    alternates = tuple(
        _draft_clip(
            take,
            role=label_map.get(take.clip_id, SemanticLabel(take.clip_id, SemanticRole.OTHER, 0.0)).role,
            group_id=clip_to_group.get(take.clip_id),
            selected=False,
        )
        for take in kept
        if take.clip_id not in selected_ids
        and take.clip_id not in review_removed_ids
        and take.clip_id not in no_usable_realization_ids
    )
    discarded_clips = tuple(
        _draft_clip(
            take,
            role=label_map.get(take.clip_id, SemanticLabel(take.clip_id, SemanticRole.OTHER, 0.0)).role,
            # D-050C3 Section 5: this used to hardcode group_id=None
            # unconditionally, unlike the selected/alternates buckets above
            # which correctly consult clip_to_group. `discarded` (pre-
            # grouping clean_cut/hybrid-cleanup rejects) never had a group
            # to begin with, so clip_to_group.get() is a no-op there -- but
            # `review_removed` (post-grouping draft_review rejects, see
            # removed_group_ids above) DID go through grouping and have a
            # real group_id, which the hardcoded None silently discarded,
            # stripping semantic_idea_id/retry_family_id (see _draft_clip)
            # from every such realization regardless of whether grouping
            # found a real retry family for it. Fixed to the same lookup
            # selected/alternates already use -- no clip-id hardcoding,
            # general to any discard path that reaches this constructor.
            group_id=clip_to_group.get(take.clip_id),
            selected=False,
        )
        for take in (*discarded, *review_removed, *no_usable_removed)
    )

    whole_video_diag = {
        "status": whole_video_context.status.__dict__ if whole_video_context is not None else None,
        "dominant_edit_mode": whole_video_context.dominant_edit_mode if whole_video_context is not None else "natural",
        "sources": [
            {
                "source_asset_id": source.source_asset_id,
                "summary": source.summary,
                "dominant_style": source.dominant_style,
                "creator_intent": source.creator_intent,
                "edit_mode": source.edit_mode,
                "sales_intent": source.sales_intent,
                "main_topic": source.main_topic,
                "product_or_subject": source.product_or_subject,
                "story_logic": source.story_logic,
                "events": [event.__dict__ for event in source.events],
            }
            for source in (whole_video_context.sources if whole_video_context is not None else ())
        ],
    }

    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION,
        project_id=request.project_id,
        strategy=strategy,
        selected=selected,
        alternates=alternates,
        discarded=discarded_clips,
        diagnostics={
            "whole_video_context": whole_video_diag,
            "attempt_reconstruction": attempt_reconstruction_diagnostics,
            "performance_confirmation": list(performance_confirmation_diagnostics)[:300],
            "temporal_performance_trims": list(temporal_trim_diagnostics)[:300],
            "clean_cut_decisions": [decision.__dict__ for decision in decisions],
            "clean_cut_judge_status": clean_judged.status.__dict__,
            "clean_cut_judge": list(clean_judge_diagnostics)[:100],
            "clean_cut_judge_deleted_count": sum(1 for item in clean_judge_diagnostics if item.get("applied_delete")),
            "clean_cut_judge_mixed_count": sum(1 for item in clean_judge_diagnostics if item["action"] == "mixed"),
            "clean_cut_judge_mixed_trimmed_count": sum(1 for item in clean_judge_diagnostics if item.get("applied_mixed_trim")),
            "hybrid_editorial_requested_chunk_count": hybrid_cleanup.requested_chunk_count,
            "hybrid_editorial_available_chunk_count": hybrid_cleanup.available_chunk_count,
            "hybrid_editorial_deleted_count": len(hybrid_cleanup.deleted),
            "hybrid_editorial_semantic_decision_count": len(hybrid_cleanup.semantic_decisions),
            "hybrid_editorial_chunks": list(hybrid_cleanup.diagnostics)[:100],
            # D-094.F2: explicit, counted starvation evidence (runs 33960713625 /
            # 33969388042: 2 of 6 planned P1 retry-equivalence windows refused
            # by the $0.0075 per-edit ledger, silently fail-open before this).
            "hybrid_editorial_budget_exhausted_chunk_count": sum(
                1 for row in hybrid_cleanup.diagnostics if row.get("budget_exhausted")
            ),
            "hybrid_editorial_budget_exhausted_chunk_indices": [
                row.get("chunk_index") for row in hybrid_cleanup.diagnostics if row.get("budget_exhausted")
            ],
            "hybrid_editorial_budget_exhausted_estimated_shortfall_usd": round(sum(
                float(row.get("estimated_cost_usd") or 0.0)
                for row in hybrid_cleanup.diagnostics if row.get("budget_exhausted")
            ), 6),
            "hybrid_editorial_budget_exhausted_member_ids": sorted({
                cid for row in hybrid_cleanup.diagnostics if row.get("budget_exhausted")
                for cid in (row.get("member_ids") or ())
            }),
            # D-052 Part B: present only when CUTSELL_SEMANTIC_COMPUTE_PLANNER
            # was enabled for this run -- None otherwise (today's default).
            "semantic_compute_plan": (
                build_cost_contract_report(hybrid_cleanup.semantic_compute_plan)
                if hybrid_cleanup.semantic_compute_plan is not None else None
            ),
            "hybrid_semantic_best_take_override_count": semantic_best_take_override_count,
            "take_group_count": len(groups),
            "alternate_group_count": alternate_group_count,
            "take_grouping_status": grouping.status.__dict__,
            "take_grouping_reason": grouping.reason,
            "take_group_members": [list(group) for group in semantic_equivalence_groups][:100],
            "semantic_idea_equivalence": semantic_equivalence_diagnostics,
            "distinct_idea_grouping_safety": cohesion_diagnostics,
            "source_count": len(request.sources),
            "take_judge_status_counts": dict(judge_statuses),
            "take_judge_fallback_reasons": dict(judge_reasons),
            "take_judge_groups": judge_group_diagnostics[:50],
            # D-163 (docs/CUTSELL_DECISIONS.md D-163): tail-safe, counts-only
            # summary of the Watch+Listen BestTake evidence guard -- same
            # pattern as D-158/D-161's own compact summaries. {"status":
            # "disabled"} when the flag is off; the per-family fields
            # already live on each take_judge_groups row above.
            "watch_listen_besttake_evidence": (
                {"status": "disabled"} if not watch_listen_besttake_evidence_enabled()
                else (
                    {"status": "no_families_evaluated"} if not watch_listen_besttake_results
                    else {"status": "evaluated", **watch_listen_besttake_diagnostics(watch_listen_besttake_results)}
                )
            ),
            # D-172 (docs/CUTSELL_DECISIONS.md D-172): Zone-Usability V2
            # diagnostic-consumption tail-safe summary -- same pattern as
            # D-158/D-161/D-163's own compact summaries. {"status":
            # "disabled"} when the (separate, default-OFF) V2 flag is off;
            # the per-family fields already live on each take_judge_groups
            # row above via watch_listen_besttake_v2_group_row.
            "watch_listen_besttake_v2": (
                {"status": "disabled"} if not zone_usability_v2_besttake_enabled()
                else (
                    {"status": "no_families_evaluated"} if not watch_listen_besttake_v2_results
                    else {"status": "evaluated", **watch_listen_besttake_v2_diagnostics(watch_listen_besttake_v2_results)}
                )
            ),
            # D-174 (docs/CUTSELL_DECISIONS.md D-174): Watch+Listen BestTake
            # Guard Authority Phase-1 tail-safe summary -- same pattern as
            # D-158/D-161/D-163/D-172's own compact summaries. {"status":
            # "disabled"} when the (separate, default-OFF) authority flag is
            # off; the per-family Phase-1 fields already live on each
            # take_judge_groups row above via watch_listen_besttake_guard_
            # authority_row. Phase-2 outcome fields on that same row (patched
            # in place by universal_clean_cut.py's apply_watch_listen_
            # besttake_guard_authority, only when a real GUARD_REJECT_
            # CURRENT_WINNER fired and the ladder resolved a replacement)
            # are NOT reflected in this pipeline.py-stage summary -- this
            # summary is Phase 1 only, computed before Phase 2 ever runs.
            "watch_listen_besttake_guard_authority": (
                {"status": "disabled"} if not watch_listen_besttake_guard_authority_enabled()
                else (
                    {"status": "no_families_evaluated"} if not watch_listen_besttake_guard_authority_results
                    else {
                        "status": "evaluated",
                        **watch_listen_besttake_guard_authority_diagnostics(
                            watch_listen_besttake_guard_authority_results
                        ),
                    }
                )
            ),
            # D-184 (docs/CUTSELL_DECISIONS.md D-184): Bounded Finalist
            # Arbiter tail-safe summary -- same pattern as D-158/D-161/
            # D-163/D-172/D-174's own compact summaries. {"status":
            # "disabled"} when the (separate, default-OFF) arbiter flag is
            # off; the per-family fields already live on each take_judge_
            # groups row above via bounded_finalist_arbiter_diagnostics.
            "bounded_finalist_arbiter": (
                {"status": "disabled"} if not bounded_finalist_arbiter_enabled()
                else (
                    {"status": "no_families_evaluated"} if not bounded_finalist_arbiter_diagnostics_rows
                    else {
                        "status": "evaluated",
                        **bounded_finalist_arbiter_run_summary(bounded_finalist_arbiter_diagnostics_rows),
                    }
                )
            ),
            "composer_status": composition.status.__dict__,
            "composer_reason": composition.reason,
            "composer_order": list(composition.ordered_clip_ids),
            "draft_review_status": review.status.__dict__,
            "draft_review_postable": review.postable,
            "draft_review_issues": list(review.issues),
            "draft_review_reason": review.reason,
            "draft_review_order": list(review.ordered_clip_ids),
            "draft_review_removed_ids": [take.clip_id for take in review_removed],
            "draft_review_removed_group_ids": sorted(removed_group_ids),
            # D-097.B: families Best Take refused to force a winner for.
            "no_usable_realization_removed_ids": [take.clip_id for take in no_usable_removed],
            "no_usable_realization_family_count": sum(
                1 for row in judge_group_diagnostics if row.get("no_usable_realization")
            ),
            # D-097.9 (R11): the split the delivery gate reads.
            "no_usable_realization_bts_singleton_count": sum(
                1 for row in judge_group_diagnostics
                if row.get("no_usable_realization") and row.get("no_usable_realization_basis") == "single_bts_unusable"
            ),
        },
    )

    # CompositeResolver step 16 (composite_resolver.py, D-023): the one
    # genuinely downstream extension, operating on the built draft rather
    # than raw takes -- repairs a concise discarded delivery + later winner
    # that jointly cover a redundant selected monolith better than the
    # monolith alone. Called explicitly here instead of via a monkeypatch
    # on this function.
    draft = apply_composite_family_stabilization(draft)

    # D-050A observability (Section 7): the full identity chain for every
    # selected clip, in one place, computed AFTER the last stage in this
    # function that can change `selected` membership -- read-only, never
    # fed back into any decision here or downstream.
    draft = dataclass_replace(
        draft,
        diagnostics={
            **(draft.diagnostics or {}),
            "canonical_identity_chain": build_identity_chain_diagnostics(draft),
        },
    )

    if alternate_group_count == 0:
        judge_stage = "not_applicable_no_alternates"
    elif judge_statuses.get("applied"):
        judge_stage = "provider_complete"
    elif judge_statuses.get("provider_error_fallback"):
        judge_stage = "degraded_fallback"
    else:
        judge_stage = "baseline_complete"

    if clean_cut_provider is None:
        clean_cut_stage = "context_aware_deterministic_complete"
    elif clean_judged.status.status == "applied":
        clean_cut_stage = "provider_complete"
    elif clean_judged.status.status == "provider_error":
        clean_cut_stage = "degraded_fail_open"
    else:
        clean_cut_stage = clean_judged.status.status

    # D-097.2: a window the per-edit dollar ledger refused (D-094.F2's
    # counted starvation) is a PARTIAL semantic pass, never "complete" --
    # the families inside it go unlabeled and get decided by tie-breaks,
    # so the stage status must say so where every consumer (RAW prints,
    # CLEAN RAW gate, active-path identity) reads it.
    hybrid_budget_refused_count = sum(1 for row in hybrid_cleanup.diagnostics if row.get("budget_exhausted"))
    if editorial_judge is None:
        hybrid_stage = "disabled_local_only"
    elif (
        hybrid_cleanup.requested_chunk_count
        and hybrid_cleanup.available_chunk_count == hybrid_cleanup.requested_chunk_count
        and not hybrid_budget_refused_count
    ):
        hybrid_stage = "provider_complete"
    elif hybrid_cleanup.requested_chunk_count and hybrid_cleanup.available_chunk_count:
        hybrid_stage = (
            f"provider_partial:{hybrid_cleanup.available_chunk_count}/{hybrid_cleanup.requested_chunk_count}"
            + (f":budget_refused={hybrid_budget_refused_count}" if hybrid_budget_refused_count else "")
        )
    elif hybrid_cleanup.requested_chunk_count:
        hybrid_stage = "degraded_fail_open"
    else:
        hybrid_stage = "confidence_gate_local"

    if composer_provider is None or len(natural_selected) <= 1:
        composer_stage = "natural_order"
    elif composition.status.status == "applied":
        composer_stage = "provider_complete"
    else:
        composer_stage = "degraded_natural_order_fallback"

    if take_grouping_provider is None or len(kept) <= 1:
        grouping_stage = "baseline_complete"
    elif grouping.status.status == "applied":
        grouping_stage = "provider_complete"
    else:
        grouping_stage = "degraded_baseline_fallback"

    if draft_review_provider is None or len(composed_takes) <= 1:
        review_stage = "not_requested"
    elif review.status.status == "applied":
        review_stage = "postable" if review.postable else "needs_attention"
    else:
        review_stage = "degraded_fallback"

    temporal_applied_count = sum(
        1 for item in temporal_trim_diagnostics if item.get("applied")
    )

    return ProcessingResult(
        schema_version=SCHEMA_VERSION,
        project_id=request.project_id,
        state=JobState.DRAFT_READY,
        draft=draft,
        stage_status={
            "whole_video_context": whole_video_context.status.status if whole_video_context is not None else "not_requested",
            "edit_mode": whole_video_context.dominant_edit_mode if whole_video_context is not None else "natural",
            "attempt_reconstruction": "applied" if attempt_reconstruction_diagnostics else "not_requested",
            "temporal_performance": "applied" if temporal_applied_count else "no_edge_trim_needed",
            "clean_cut": clean_cut_stage,
            "hybrid_editorial": hybrid_stage,
            "take_grouping": grouping_stage,
            "take_judge": judge_stage,
            "semantic": "provided" if label_map else "not_provided",
            "composer": composer_stage,
            "draft_review": review_stage,
        },
    )