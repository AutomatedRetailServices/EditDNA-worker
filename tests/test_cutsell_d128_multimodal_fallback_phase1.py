"""D-128 Phase 1 -- Multimodal BestTake fallback: Class B trigger + shadow-only
arbiter contract.

docs/CUTSELL_DECISIONS.md D-128, docs/CUTSELL_MULTIMODAL_FALLBACK_ARBITER_
FORENSIC_D127.md. This suite proves the ONE authorized Phase 1 deliverable:
SHADOW-ONLY observability for exactly one general trigger class (Class B --
semantic winner and DeliveryScorer's own top pick AGREE, but a different
meaning-sufficient member's D-122 CASE B evidence is factually cleaner). It
never:

- calls a real multimodal provider (no network reference exists anywhere in
  `multimodal_besttake_fallback.py`/`multimodal_besttake_arbiter.py`/
  `multimodal_besttake_eval.py`, and the live pipeline never imports the
  arbiter module at all -- only the pure trigger classifier);
- changes `selected_clip_id`/`ranked`/`winner_path`/membership/Boundary/
  render-plan behavior (the new `judge_group_diagnostics` fields are an
  ADDITIVE observability projection, appended after every existing D-122/
  D-123 field, never consulted above that line);
- invents a score, weight, or threshold (`_factual_dominance` is a pure
  partial-order comparison over already-computed D-122 CASE B aggregates);
- widens D-123's own disagreement territory (`SEMANTIC_DELIVERYSCORE_
  DISAGREE` always excludes Class B, matching D-127 Section 4's scope
  boundary).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cutsell_worker.case_b_performance_evidence import CaseBPerformanceEvidence
from cutsell_worker.multimodal_besttake_arbiter import (
    ABSTAINED,
    BEST_TAKE,
    ERROR,
    INVALID_RESPONSE,
    MultimodalBestTakeGatePolicy,
    MultimodalBestTakeRequest,
    MultimodalBestTakeResponse,
    MultimodalFinalistInput,
    NOT_INVOKED,
    NullMultimodalBestTakeArbiter,
    UNCERTAIN,
    VALID_OUTCOMES,
    WOULD_INVOKE_SHADOW,
    build_multimodal_besttake_request,
    safe_arbitrate,
    should_request_multimodal_arbitration,
    validate_multimodal_besttake_response,
)
from cutsell_worker.multimodal_besttake_eval import eval_cases, run_offline_eval
from cutsell_worker.multimodal_besttake_fallback import (
    CASE_B_NOT_DOMINATED,
    CLASS_B,
    CLASS_B_ELIGIBLE,
    NO_CASE_B_EVIDENCE,
    NO_DELIVERYSCORE_WINNER,
    NO_MEANING_SUFFICIENT_ALTERNATIVE,
    NO_SEMANTIC_WINNER,
    NOT_MULTI_MEMBER,
    SAFETY_EXCLUSION,
    SEMANTIC_DELIVERYSCORE_DISAGREE,
    detect_class_b_trigger,
    fallback_trigger_diagnostics,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _evidence(candidate_id: str, count_by_kind: dict, duration_by_kind: dict) -> CaseBPerformanceEvidence:
    total_count = sum(count_by_kind.values())
    total_duration = round(sum(duration_by_kind.values()), 3)
    return CaseBPerformanceEvidence(
        candidate_id=candidate_id,
        source_asset_id="src",
        delivery_available=True,
        delivery_start=0.0,
        delivery_end=10.0,
        delivery_span_duration=10.0,
        delivery_events=(),
        delivery_event_count=total_count,
        delivery_event_duration_total=total_duration,
        count_by_kind=dict(count_by_kind),
        duration_by_kind=dict(duration_by_kind),
        event_density=(total_duration / 10.0) if total_duration else 0.0,
    )


def _finalist(candidate_id: str, evidence: CaseBPerformanceEvidence) -> MultimodalFinalistInput:
    return MultimodalFinalistInput(
        candidate_id=candidate_id,
        source_asset_id="src",
        source_start=0.0,
        source_end=10.0,
        transcript_text="some transcript",
        meaning_sufficient=True,
        case_b_evidence=evidence,
        semantic_label="pimples",
        semantic_confidence=0.9,
        deliveryscore_summary=1.0,
    )


# --------------------------------------------------------------------------
# 1. Class B positive trigger (pimples-shaped fixture)
# --------------------------------------------------------------------------

def test_class_b_positive_trigger_pimples_shaped():
    """D-126's own real shape: agreed winner A carries MORE CASE B evidence
    than meaning-sufficient alternative B -- ELIGIBLE, and B is the (only)
    dominating alternative."""
    winner_evidence = _evidence("A", {"hand_motion_reset_candidate": 9}, {"hand_motion_reset_candidate": 4.0})
    alt_evidence = _evidence("B", {"hand_motion_reset_candidate": 7}, {"hand_motion_reset_candidate": 2.0})
    decision = detect_class_b_trigger(
        family_id="tg_pimples", member_ids=("A", "B"),
        semantic_winner="A", deliveryscore_winner="A",
        meaning_sufficient_ids={"A", "B"},
        case_b_evidence_by_id={"A": winner_evidence, "B": alt_evidence},
        safety_excluded_ids=set(),
    )
    assert decision.eligible is True
    assert decision.trigger_class == CLASS_B
    assert decision.reason == CLASS_B_ELIGIBLE
    assert decision.structured_winner == "A"
    assert decision.alternative_candidates == ("B",)


# --------------------------------------------------------------------------
# 2 / 13. Semantic/DeliveryScorer disagreement excluded (D-123's territory)
# --------------------------------------------------------------------------

def test_semantic_deliveryscore_disagreement_excluded():
    a = _evidence("A", {"k": 7}, {"k": 0.4})
    b = _evidence("B", {"k": 1}, {"k": 0.1})
    decision = detect_class_b_trigger(
        family_id="tg_disagree", member_ids=("A", "B"),
        semantic_winner="A", deliveryscore_winner="B",
        meaning_sufficient_ids={"A", "B"},
        case_b_evidence_by_id={"A": a, "B": b},
        safety_excluded_ids=set(),
    )
    assert decision.eligible is False
    assert decision.reason == SEMANTIC_DELIVERYSCORE_DISAGREE


# --------------------------------------------------------------------------
# 3. Meaning-insufficient alternative excluded
# --------------------------------------------------------------------------

def test_meaning_insufficient_alternative_excluded():
    a = _evidence("A", {"k": 9}, {"k": 4.0})
    b = _evidence("B", {"k": 1}, {"k": 0.1})
    decision = detect_class_b_trigger(
        family_id="tg_insufficient", member_ids=("A", "B"),
        semantic_winner="A", deliveryscore_winner="A",
        meaning_sufficient_ids={"A"},  # B is NOT meaning-sufficient
        case_b_evidence_by_id={"A": a, "B": b},
        safety_excluded_ids=set(),
    )
    assert decision.eligible is False
    assert decision.reason == NO_MEANING_SUFFICIENT_ALTERNATIVE


# --------------------------------------------------------------------------
# 4. Tied CASE B evidence excluded
# --------------------------------------------------------------------------

def test_tied_case_b_evidence_excluded():
    a = _evidence("A", {"k": 3}, {"k": 0.3})
    b = _evidence("B", {"k": 3}, {"k": 0.3})
    decision = detect_class_b_trigger(
        family_id="tg_tied", member_ids=("A", "B"),
        semantic_winner="A", deliveryscore_winner="A",
        meaning_sufficient_ids={"A", "B"},
        case_b_evidence_by_id={"A": a, "B": b},
        safety_excluded_ids=set(),
    )
    assert decision.eligible is False
    assert decision.reason == CASE_B_NOT_DOMINATED


# --------------------------------------------------------------------------
# 5. Partial-order conflict excluded (better on one dimension, worse on
#    another -- never a guess)
# --------------------------------------------------------------------------

def test_partial_order_conflict_excluded():
    a = _evidence("A", {"hand_motion_reset_candidate": 3, "facial_expression_shift_candidate": 1},
                  {"hand_motion_reset_candidate": 0.2, "facial_expression_shift_candidate": 0.02})
    b = _evidence("B", {"hand_motion_reset_candidate": 1, "facial_expression_shift_candidate": 3},
                  {"hand_motion_reset_candidate": 0.07, "facial_expression_shift_candidate": 0.3})
    decision = detect_class_b_trigger(
        family_id="tg_partial_order", member_ids=("A", "B"),
        semantic_winner="A", deliveryscore_winner="A",
        meaning_sufficient_ids={"A", "B"},
        case_b_evidence_by_id={"A": a, "B": b},
        safety_excluded_ids=set(),
    )
    assert decision.eligible is False
    assert decision.reason == CASE_B_NOT_DOMINATED


# --------------------------------------------------------------------------
# 6. Missing CASE B evidence excluded -- winner side, then alternative side
# --------------------------------------------------------------------------

def test_missing_case_b_evidence_for_winner_excluded():
    b = _evidence("B", {"k": 1}, {"k": 0.1})
    decision = detect_class_b_trigger(
        family_id="tg_no_winner_evidence", member_ids=("A", "B"),
        semantic_winner="A", deliveryscore_winner="A",
        meaning_sufficient_ids={"A", "B"},
        case_b_evidence_by_id={"B": b},  # no entry for A
        safety_excluded_ids=set(),
    )
    assert decision.eligible is False
    assert decision.reason == NO_CASE_B_EVIDENCE


def test_missing_case_b_evidence_for_alternative_excluded():
    a = _evidence("A", {"k": 9}, {"k": 4.0})
    decision = detect_class_b_trigger(
        family_id="tg_no_alt_evidence", member_ids=("A", "B"),
        semantic_winner="A", deliveryscore_winner="A",
        meaning_sufficient_ids={"A", "B"},
        case_b_evidence_by_id={"A": a},  # no entry for B
        safety_excluded_ids=set(),
    )
    assert decision.eligible is False
    assert decision.reason == NO_CASE_B_EVIDENCE


# --------------------------------------------------------------------------
# 7. Single-member family excluded
# --------------------------------------------------------------------------

def test_single_member_family_excluded():
    a = _evidence("A", {}, {})
    decision = detect_class_b_trigger(
        family_id="tg_singleton", member_ids=("A",),
        semantic_winner="A", deliveryscore_winner="A",
        meaning_sufficient_ids={"A"},
        case_b_evidence_by_id={"A": a},
        safety_excluded_ids=set(),
    )
    assert decision.eligible is False
    assert decision.reason == NOT_MULTI_MEMBER


# --------------------------------------------------------------------------
# 8. ENTRY/EXIT-only difference excluded -- D-115's DELIVERY-zone filter
#    already keeps such events out of CASE B evidence entirely, so both
#    candidates present as zero/zero here; no safe dominance is possible.
# --------------------------------------------------------------------------

def test_entry_exit_only_difference_excluded():
    a = _evidence("A", {}, {})
    b = _evidence("B", {}, {})
    decision = detect_class_b_trigger(
        family_id="tg_entry_exit", member_ids=("A", "B"),
        semantic_winner="A", deliveryscore_winner="A",
        meaning_sufficient_ids={"A", "B"},
        case_b_evidence_by_id={"A": a, "B": b},
        safety_excluded_ids=set(),
    )
    assert decision.eligible is False
    assert decision.reason == CASE_B_NOT_DOMINATED


# --------------------------------------------------------------------------
# 9. Safety exclusion -- a factually cleaner alternative excluded by an
#    existing deterministic safety veto (reused verbatim, never
#    re-implemented in this module)
# --------------------------------------------------------------------------

def test_safety_excluded_alternative_never_fires():
    a = _evidence("A", {"k": 5}, {"k": 0.33})
    b = _evidence("B", {"k": 1}, {"k": 0.07})
    decision = detect_class_b_trigger(
        family_id="tg_safety", member_ids=("A", "B"),
        semantic_winner="A", deliveryscore_winner="A",
        meaning_sufficient_ids={"A", "B"},
        case_b_evidence_by_id={"A": a, "B": b},
        safety_excluded_ids={"B"},
    )
    assert decision.eligible is False
    assert decision.reason == SAFETY_EXCLUSION


# --------------------------------------------------------------------------
# 10 / 11. No decisive semantic winner / no DeliveryScorer winner excluded
# --------------------------------------------------------------------------

def test_no_semantic_winner_excluded():
    a = _evidence("A", {"k": 2}, {"k": 0.1})
    b = _evidence("B", {"k": 1}, {"k": 0.05})
    decision = detect_class_b_trigger(
        family_id="tg_no_semantic", member_ids=("A", "B"),
        semantic_winner=None, deliveryscore_winner="A",
        meaning_sufficient_ids={"A", "B"},
        case_b_evidence_by_id={"A": a, "B": b},
        safety_excluded_ids=set(),
    )
    assert decision.eligible is False
    assert decision.reason == NO_SEMANTIC_WINNER


def test_no_deliveryscore_winner_excluded():
    a = _evidence("A", {"k": 2}, {"k": 0.1})
    b = _evidence("B", {"k": 1}, {"k": 0.05})
    decision = detect_class_b_trigger(
        family_id="tg_no_deliveryscore", member_ids=("A", "B"),
        semantic_winner="A", deliveryscore_winner=None,
        meaning_sufficient_ids={"A", "B"},
        case_b_evidence_by_id={"A": a, "B": b},
        safety_excluded_ids=set(),
    )
    assert decision.eligible is False
    assert decision.reason == NO_DELIVERYSCORE_WINNER


# --------------------------------------------------------------------------
# 12. Trivial agreement with no CASE B evidence at all (grouping-absent
#     shape) -- must never fabricate a conflict.
# --------------------------------------------------------------------------

def test_trivial_agreement_no_case_b_evidence_excluded():
    decision = detect_class_b_trigger(
        family_id="tg_trivial", member_ids=("A", "B"),
        semantic_winner="A", deliveryscore_winner="A",
        meaning_sufficient_ids={"A", "B"},
        case_b_evidence_by_id={},
        safety_excluded_ids=set(),
    )
    assert decision.eligible is False
    assert decision.reason == NO_CASE_B_EVIDENCE


# --------------------------------------------------------------------------
# 14. Request assembly includes ONLY legitimate finalists named by an
#     ELIGIBLE decision -- never a candidate outside it.
# --------------------------------------------------------------------------

def test_request_has_only_legitimate_finalists():
    winner_evidence = _evidence("A", {"k": 9}, {"k": 4.0})
    alt_evidence = _evidence("B", {"k": 7}, {"k": 2.0})
    decision = detect_class_b_trigger(
        family_id="tg_request", member_ids=("A", "B", "C"),
        semantic_winner="A", deliveryscore_winner="A",
        meaning_sufficient_ids={"A", "B"},  # C is not even meaning-sufficient
        case_b_evidence_by_id={"A": winner_evidence, "B": alt_evidence},
        safety_excluded_ids=set(),
    )
    assert decision.eligible is True
    finalists_by_id = {
        "A": _finalist("A", winner_evidence),
        "B": _finalist("B", alt_evidence),
        # A finalist for C is deliberately NOT supplied -- it must never
        # leak into the request even if a caller mistakenly built one.
        "C": _finalist("C", _evidence("C", {"k": 100}, {"k": 50.0})),
    }
    request = build_multimodal_besttake_request(decision, finalists_by_id)
    assert request is not None
    candidate_ids = {f.candidate_id for f in request.finalists}
    assert candidate_ids == {"A", "B"}
    assert "C" not in candidate_ids


def test_request_returns_none_when_trigger_not_eligible():
    a = _evidence("A", {"k": 3}, {"k": 0.3})
    b = _evidence("B", {"k": 3}, {"k": 0.3})
    decision = detect_class_b_trigger(
        family_id="tg_not_eligible", member_ids=("A", "B"),
        semantic_winner="A", deliveryscore_winner="A",
        meaning_sufficient_ids={"A", "B"},
        case_b_evidence_by_id={"A": a, "B": b},
        safety_excluded_ids=set(),
    )
    assert decision.eligible is False
    request = build_multimodal_besttake_request(decision, {"A": _finalist("A", a), "B": _finalist("B", b)})
    assert request is None


def test_request_returns_none_when_fewer_than_two_finalists_resolvable():
    winner_evidence = _evidence("A", {"k": 9}, {"k": 4.0})
    alt_evidence = _evidence("B", {"k": 7}, {"k": 2.0})
    decision = detect_class_b_trigger(
        family_id="tg_underpopulated", member_ids=("A", "B"),
        semantic_winner="A", deliveryscore_winner="A",
        meaning_sufficient_ids={"A", "B"},
        case_b_evidence_by_id={"A": winner_evidence, "B": alt_evidence},
        safety_excluded_ids=set(),
    )
    assert decision.eligible is True
    # Only A's finalist input is available -- must refuse to build a request.
    request = build_multimodal_besttake_request(decision, {"A": _finalist("A", winner_evidence)})
    assert request is None


# --------------------------------------------------------------------------
# 15. Request never includes full RAW / raw media bytes -- references only,
#     defaulted to None/() in Phase 1.
# --------------------------------------------------------------------------

def test_finalist_input_never_carries_raw_media_by_default():
    finalist = _finalist("A", _evidence("A", {"k": 1}, {"k": 0.1}))
    assert finalist.video_span_reference is None
    assert finalist.audio_span_reference is None
    assert finalist.sampled_frame_references == ()
    # Only bounded, already-computed fields -- no raw byte payload field
    # exists on the dataclass at all.
    field_names = {f for f in MultimodalFinalistInput.__dataclass_fields__}
    assert "raw_media_bytes" not in field_names
    assert "frame_bytes" not in field_names


# --------------------------------------------------------------------------
# 16. BEST_TAKE response must reference a supplied candidate id
# --------------------------------------------------------------------------

def test_best_take_response_must_reference_supplied_candidate():
    finalists = (
        _finalist("A", _evidence("A", {"k": 9}, {"k": 4.0})),
        _finalist("B", _evidence("B", {"k": 7}, {"k": 2.0})),
    )
    request = MultimodalBestTakeRequest(family_id="tg_x", proposition_context="", finalists=finalists)
    bad_response = MultimodalBestTakeResponse(
        family_id="tg_x", outcome=BEST_TAKE, best_take_candidate_id="Z",
        confidence=0.9, reason="bad", provider="none", model="none",
        requested=True, available=True,
    )
    with pytest.raises(ValueError):
        validate_multimodal_besttake_response(request, bad_response)

    good_response = MultimodalBestTakeResponse(
        family_id="tg_x", outcome=BEST_TAKE, best_take_candidate_id="A",
        confidence=0.9, reason="ok", provider="none", model="none",
        requested=True, available=True,
    )
    validated = validate_multimodal_besttake_response(request, good_response)
    assert validated.best_take_candidate_id == "A"


def test_invalid_confidence_rejected():
    finalists = (_finalist("A", _evidence("A", {"k": 1}, {"k": 0.1})),
                 _finalist("B", _evidence("B", {"k": 1}, {"k": 0.1})))
    request = MultimodalBestTakeRequest(family_id="tg_x", proposition_context="", finalists=finalists)
    response = MultimodalBestTakeResponse(
        family_id="tg_x", outcome=UNCERTAIN, best_take_candidate_id=None,
        confidence=1.5, reason="bad", provider="none", model="none",
        requested=True, available=True,
    )
    with pytest.raises(ValueError):
        validate_multimodal_besttake_response(request, response)


# --------------------------------------------------------------------------
# 17. UNCERTAIN is a valid outcome
# --------------------------------------------------------------------------

def test_uncertain_is_valid_outcome():
    assert UNCERTAIN in VALID_OUTCOMES
    finalists = (_finalist("A", _evidence("A", {"k": 1}, {"k": 0.1})),
                 _finalist("B", _evidence("B", {"k": 1}, {"k": 0.1})))
    request = MultimodalBestTakeRequest(family_id="tg_x", proposition_context="", finalists=finalists)
    response = NullMultimodalBestTakeArbiter().arbitrate(request)
    validated = validate_multimodal_besttake_response(request, response)
    assert validated.outcome == UNCERTAIN
    assert validated.available is False


# --------------------------------------------------------------------------
# 18. Invalid response rejected safely (safe_arbitrate never raises)
# --------------------------------------------------------------------------

class _InvalidOutcomeArbiter:
    def arbitrate(self, request: MultimodalBestTakeRequest) -> MultimodalBestTakeResponse:
        return MultimodalBestTakeResponse(
            family_id=request.family_id, outcome="NOT_A_REAL_OUTCOME",
            best_take_candidate_id=None, confidence=0.5, reason="broken",
            provider="fake", model="fake", requested=True, available=True,
        )


def test_invalid_response_rejected_safely_via_safe_arbitrate():
    finalists = (_finalist("A", _evidence("A", {"k": 1}, {"k": 0.1})),
                 _finalist("B", _evidence("B", {"k": 1}, {"k": 0.1})))
    request = MultimodalBestTakeRequest(family_id="tg_x", proposition_context="", finalists=finalists)
    outcome, response = safe_arbitrate(_InvalidOutcomeArbiter(), request)
    # D-136 Phase 2 (docs/CUTSELL_DECISIONS.md D-136): safe_arbitrate now
    # classifies a response-validation failure as the more specific
    # INVALID_RESPONSE (previously the generic ERROR, which Phase 1 had no
    # way to distinguish since no provider call existed yet to raise
    # anything more specific) -- this is the exact "extend it minimally
    # for concrete failure mapping" this task's own directive required.
    assert outcome == INVALID_RESPONSE
    assert response is None


# --------------------------------------------------------------------------
# 19. No provider call anywhere in Phase 1 -- structural + behavioral proof
# --------------------------------------------------------------------------

def test_no_provider_call_with_null_arbiter():
    finalists = (_finalist("A", _evidence("A", {"k": 1}, {"k": 0.1})),
                 _finalist("B", _evidence("B", {"k": 1}, {"k": 0.1})))
    request = MultimodalBestTakeRequest(family_id="tg_x", proposition_context="", finalists=finalists)
    outcome, response = safe_arbitrate(NullMultimodalBestTakeArbiter(), request)
    # NullMultimodalBestTakeArbiter always returns available=False -- the
    # `safe_arbitrate` fail-open wrapper reports this as NOT_INVOKED (never
    # a real ABSTAINED verdict, since no provider was ever consulted).
    assert outcome == NOT_INVOKED
    assert response.provider == "none"
    assert response.requested is False


def test_no_provider_call_when_arbiter_is_none():
    finalists = (_finalist("A", _evidence("A", {"k": 1}, {"k": 0.1})),
                 _finalist("B", _evidence("B", {"k": 1}, {"k": 0.1})))
    request = MultimodalBestTakeRequest(family_id="tg_x", proposition_context="", finalists=finalists)
    outcome, response = safe_arbitrate(None, request)
    assert outcome == NOT_INVOKED
    assert response is None


def test_gate_policy_rejects_over_the_finalist_ceiling():
    finalists = tuple(
        _finalist(cid, _evidence(cid, {"k": 1}, {"k": 0.1})) for cid in ("A", "B", "C", "D")
    )
    request = MultimodalBestTakeRequest(family_id="tg_x", proposition_context="", finalists=finalists)
    assert should_request_multimodal_arbitration(request, MultimodalBestTakeGatePolicy(max_finalists_per_request=3)) is False


def test_no_network_reference_in_shadow_modules():
    """Structural proof: none of the three D-128 Phase 1 modules reference
    any network/provider-call primitive -- there is nothing to call."""
    forbidden = ("requests.", "httpx.", "urlopen", "openai.", "socket.", "aiohttp")
    for name in (
        "multimodal_besttake_fallback.py",
        "multimodal_besttake_arbiter.py",
        "multimodal_besttake_eval.py",
    ):
        source = (REPO_ROOT / "cutsell_worker" / name).read_text()
        for token in forbidden:
            assert token not in source, f"{name} unexpectedly references {token!r}"


def test_pipeline_never_imports_the_arbiter_module():
    """`pipeline.py` imports ONLY the pure trigger classifier
    (`detect_class_b_trigger`/`fallback_trigger_diagnostics`) -- never
    `multimodal_besttake_arbiter` (the request/response/provider contract),
    matching this task's explicit shadow-only wiring scope."""
    source = (REPO_ROOT / "cutsell_worker" / "pipeline.py").read_text()
    assert "from .multimodal_besttake_fallback import" in source
    assert "multimodal_besttake_arbiter" not in source


def test_boundary_and_render_modules_never_reference_fallback():
    """Boundary/render modules are untouched by D-128 -- physical timing and
    rendering never consult Class B fallback observability."""
    for name in ("boundary_engine_pass.py", "render_plan.py", "live_render_qc.py"):
        source = (REPO_ROOT / "cutsell_worker" / name).read_text()
        assert "multimodal_besttake" not in source


# --------------------------------------------------------------------------
# 20 / 22. score_take/rank_takes/winner/membership/Boundary/render-plan
#     unchanged -- proven structurally: the new diagnostics are appended
#     strictly AFTER `final_winner` in `judge_group_diagnostics`, and no
#     shadow field name collides with an existing D-122/D-123 key.
# --------------------------------------------------------------------------

def test_shadow_diagnostic_keys_never_collide_with_existing_keys():
    existing_d122_d123_keys = {
        "group_id", "selected_clip_id", "local_selected_clip_id",
        "semantic_preferred_clip_id", "semantic_override_applied",
        "semantic_best_take_reason", "semantic_label_source",
        "semantic_candidates", "execution_status", "execution_reason",
        "ranked", "delivery_cleanliness", "no_usable_realization",
        "no_usable_realization_basis", "all_members_delete_recommended",
        "label_conflict_routed", "member_usability", "winner_path",
        "performance_consulted_before_winner", "deliveryscore_top_candidate",
        "semantic_fast_path_candidate", "case_b_evidence", "winner_path_before",
        "winner_path_after", "semantic_fast_path_bypassed", "bypass_reason",
        "case_b_conflict_present", "case_b_conflict_basis",
        "meaning_sufficient_candidates", "final_winner",
    }
    winner_evidence = _evidence("A", {"k": 9}, {"k": 4.0})
    alt_evidence = _evidence("B", {"k": 7}, {"k": 2.0})
    decision = detect_class_b_trigger(
        family_id="tg_keys", member_ids=("A", "B"),
        semantic_winner="A", deliveryscore_winner="A",
        meaning_sufficient_ids={"A", "B"},
        case_b_evidence_by_id={"A": winner_evidence, "B": alt_evidence},
        safety_excluded_ids=set(),
    )
    shadow_keys = set(fallback_trigger_diagnostics(decision).keys())
    assert shadow_keys.isdisjoint(existing_d122_d123_keys)
    assert all(k.startswith("fallback_") for k in shadow_keys)


# --------------------------------------------------------------------------
# 21. Diagnostics are JSON-safe and bounded (never raw delivery_events)
# --------------------------------------------------------------------------

def test_diagnostics_json_safe_and_bounded():
    winner_evidence = _evidence("A", {"k": 9}, {"k": 4.0})
    alt_evidence = _evidence("B", {"k": 7}, {"k": 2.0})
    decision = detect_class_b_trigger(
        family_id="tg_json", member_ids=("A", "B"),
        semantic_winner="A", deliveryscore_winner="A",
        meaning_sufficient_ids={"A", "B"},
        case_b_evidence_by_id={"A": winner_evidence, "B": alt_evidence},
        safety_excluded_ids=set(),
    )
    diagnostics = fallback_trigger_diagnostics(decision)
    encoded = json.dumps(diagnostics)  # must not raise
    assert "delivery_events" not in encoded
    assert diagnostics["fallback_shadow_eligible"] is True
    assert diagnostics["fallback_provider_invoked"] is False
    assert diagnostics["fallback_winner_changed"] is False


def test_diagnostics_non_eligible_case_has_empty_candidate_fields():
    a = _evidence("A", {"k": 3}, {"k": 0.3})
    b = _evidence("B", {"k": 3}, {"k": 0.3})
    decision = detect_class_b_trigger(
        family_id="tg_non_eligible", member_ids=("A", "B"),
        semantic_winner="A", deliveryscore_winner="A",
        meaning_sufficient_ids={"A", "B"},
        case_b_evidence_by_id={"A": a, "B": b},
        safety_excluded_ids=set(),
    )
    diagnostics = fallback_trigger_diagnostics(decision)
    assert diagnostics["fallback_shadow_eligible"] is False
    assert diagnostics["fallback_structured_winner"] is None
    assert diagnostics["fallback_candidate_ids"] == []
    assert diagnostics["fallback_case_b_dominance"] == {}


# --------------------------------------------------------------------------
# 23. Offline eval harness: every D-127 Section 20 case passes
# --------------------------------------------------------------------------

def test_offline_eval_harness_all_cases_pass():
    rows = run_offline_eval()
    assert len(rows) == len(eval_cases()) >= 9
    failures = [row for row in rows if not row["passed"]]
    assert not failures, f"offline eval regressions: {failures}"


def test_offline_eval_harness_includes_the_pimples_shaped_positive():
    rows = run_offline_eval()
    pimples_row = next(row for row in rows if row["name"] == "pimples_shaped_positive")
    assert pimples_row["expected_eligible"] is True
    assert pimples_row["actual_eligible"] is True
    assert pimples_row["actual_reason"] == CLASS_B_ELIGIBLE


# --------------------------------------------------------------------------
# 24 / 25. Output vocabulary is bounded to the KEEP/DISCARD-only doctrine --
#     no KEEP_BOTH_COMPLEMENTARY/COMPOSITE/REWRITE outcome exists.
# --------------------------------------------------------------------------

def test_output_vocabulary_excludes_illegal_outcomes():
    illegal = {"KEEP_BOTH_COMPLEMENTARY", "NEW_COMPOSITE", "NEW_CANDIDATE", "REWRITE", "MERGE_SPEECH"}
    assert VALID_OUTCOMES.isdisjoint(illegal)
    assert VALID_OUTCOMES == {BEST_TAKE, "EQUIVALENT", "GOOD_TAKE_TRIM_ENTRY", "GOOD_TAKE_TRIM_EXIT", UNCERTAIN}


def test_gate_policy_finalist_ceiling_matches_task_scope():
    # "2-3 legitimate finalists" is this task's own definitional scope, not
    # an invented calibration number.
    assert MultimodalBestTakeGatePolicy().max_finalists_per_request == 3
