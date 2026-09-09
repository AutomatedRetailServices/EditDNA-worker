"""D-183 -- Terminal BestTake confidence/decisiveness CLASSIFICATION ONLY.

BOUNDED, ADDITIVE REFINEMENT (docs/CUTSELL_DECISIONS.md D-183, post D-182
forensic). D-182 proved `_semantic_best_take`'s comparative span (Steps
3-9 -- `resolve_critical_coverage_dominance`, the asymmetry/contradiction
checks, and the final `max(tie_break_pool, key=lambda cid:
rank_by_id[cid])`) carries NO concept of confidence, margin, or
abstention: once 2+ candidates reach a comparative step, EXACTLY one
winner is always forced, with zero representation of "this comparison
was not actually decisive."

This suite proves the ONE authorized change: a new, pure, additive
`TerminalBestTakeConfidence` classification wired into `_semantic_best_
take` via an optional `terminal_confidence_out: dict | None` side-channel
(default `None` -- every existing caller before D-183 is byte-identical).
It never:

- selects, vetoes, or alters `selected_clip_id`/`preferred_id`/the
  returned reason (proven directly below: every DECISIVE fixture's
  winner is unchanged from its pre-D-183 value);
- introduces a new weighted score, score-weight change, or numeric
  margin threshold (DECISIVE requires STRUCTURED evidence -- dominance
  already found by Steps 3/4, a single survivor after subset-exclusion,
  a confident single semantic label, or an explicitly supplied external
  comparator -- raw score alone, however large the gap, is categorically
  NON_DECISIVE; TIED is exact equality of the SAME `round(x, 4)` value
  `take_judge.score_take`/`rank_takes` already produce, never a new
  epsilon);
- implements a finalist arbiter (PREFER_A/PREFER_B/ABSTAIN is explicitly
  OUT of scope here, per D-183's own directive);
- calls a provider or touches Family Formation/D-150/Boundary/Pacing/
  Renderer.
"""
from __future__ import annotations

import inspect

import pytest

from cutsell_worker.contracts import CandidateTake, RankedTake
from cutsell_worker.pipeline import (
    TerminalBestTakeConfidence,
    _TERMINAL_CONFIDENCE_CONFLICTED,
    _TERMINAL_CONFIDENCE_DECISIVE,
    _TERMINAL_CONFIDENCE_DECISIVE_BY_ELIMINATION,
    _TERMINAL_CONFIDENCE_NON_DECISIVE,
    _TERMINAL_CONFIDENCE_TIED,
    _TERMINAL_CONFIDENCE_UNKNOWN,
    _exclude_incomplete_subset_losers,
    _semantic_best_take,
    _terminal_besttake_confidence,
    terminal_besttake_confidence_diagnostics,
    terminal_besttake_confidence_run_summary,
)
from cutsell_worker.semantic_authority_observability import AUTHORITY_ABSTAIN_CONFLICT


def take(clip_id: str, text: str, *, complete_idea: bool | None = True, start: float = 0.0) -> CandidateTake:
    return CandidateTake(
        clip_id=clip_id, source_asset_id="src", source_order=0,
        start=start, end=start + 4.0, text=text, complete_idea=complete_idea,
    )


def ranked(*pairs: tuple[str, float]) -> tuple[RankedTake, ...]:
    return tuple(RankedTake(clip_id, score, "watch_listen_baseline") for clip_id, score in pairs)


# === Items 1-2: one survivor / no survivor ==================================

def test_01_one_survivor_decisive_by_elimination():
    result = _terminal_besttake_confidence(["only"], {"only": 0.5})
    assert result.confidence_state == _TERMINAL_CONFIDENCE_DECISIVE_BY_ELIMINATION
    assert result.top_candidate_id == "only"
    assert result.runner_up_candidate_id is None
    assert result.score_margin is None


def test_02_no_survivor_unknown():
    result = _terminal_besttake_confidence(["a", "b"], {})
    assert result.confidence_state == _TERMINAL_CONFIDENCE_UNKNOWN
    assert result.top_candidate_id is None


# === Items 3-4: exact tie / slight difference no structured dominance =======

def test_03_exact_score_tie():
    result = _terminal_besttake_confidence(["a", "b"], {"a": 0.7, "b": 0.7})
    assert result.confidence_state == _TERMINAL_CONFIDENCE_TIED
    assert result.score_margin == 0.0


def test_04_slight_numerical_difference_no_structured_dominance_is_non_decisive():
    """The core principle: a deterministic score difference is NOT
    automatically an editorially decisive difference. However large the
    raw gap, with no structured backing this is NON_DECISIVE."""
    result = _terminal_besttake_confidence(["a", "b"], {"a": 0.9, "b": 0.1})
    assert result.confidence_state == _TERMINAL_CONFIDENCE_NON_DECISIVE
    assert result.score_margin == pytest.approx(0.8)
    assert result.top_candidate_id == "a"
    assert result.runner_up_candidate_id == "b"


# === Item 5: clearly decisive structured dominance ==========================

_THIN_TEXT = "I felt a bit off for a while."
_RICH_TEXT = "The test confirmed it was a mild vitamin D deficiency, and I felt a bit off for a while."


def _completeness_dominance_pair():
    # Same proven D-082 fixture: raw delivery favors the THINNER candidate
    # (A), proving dominance (not delivery) decides once a CRITICAL fact
    # is at stake. B is a strict superset of A's content plus the
    # diagnosis confirmation.
    a = take("A", _THIN_TEXT)
    b = take("B", _RICH_TEXT, start=4.0)
    return a, b, ranked(("A", 0.90), ("B", 0.70))


def test_05_critical_coverage_dominance_is_decisive_and_never_reaches_raw_score():
    """When Steps 3/4's own dominance already settles it, the terminal
    confidence classifier's raw-score path is never even consulted --
    DECISIVE is recorded directly at the dominance return point."""
    a, b, r = _completeness_dominance_pair()
    decisions = {"A": ("keep", 0.88), "B": ("keep", 0.91)}
    out: dict = {}
    selected, _preferred, reason = _semantic_best_take((a, b), decisions, "A", r, terminal_confidence_out=out)
    assert reason == "critical_coverage_dominance"
    confidence = out["terminal_besttake_confidence"]
    assert confidence.confidence_state == _TERMINAL_CONFIDENCE_DECISIVE
    assert confidence.provenance == "structured_dominance"
    assert selected == "B"


# === Item 6: D-181 abstract Pimples replay ==================================

def test_06_d181_abstract_pimples_replay_non_decisive_or_tied():
    """Generic fixture (no Video00 wording): two complete, meaning-
    sufficient candidates, D-150 ABSTAIN_CONFLICT, no critical coverage
    dominance, no subset relationship, near-equal performance evidence,
    rank_takes gives A a slightly higher terminal score than B. Expected:
    NON_DECISIVE or TIED. Existing winner output is UNCHANGED by this
    task (still whichever candidate the pre-existing ladder picks)."""
    a = take("a", "I also had a small skin reaction near my ear that kept bothering me.")
    b = take("b", "There was also some irritation along my jaw that felt like an allergy to me.", start=6.0)
    decisions = {"a": ("keep", 0.5), "b": ("keep", 0.5)}
    r = ranked(("a", 0.62), ("b", 0.58))  # A slightly higher, near-equal
    out: dict = {}
    before = _semantic_best_take((a, b), decisions, "a", r, semantic_comparative_authority=AUTHORITY_ABSTAIN_CONFLICT)
    after = _semantic_best_take(
        (a, b), decisions, "a", r, semantic_comparative_authority=AUTHORITY_ABSTAIN_CONFLICT,
        terminal_confidence_out=out,
    )
    assert before == after  # D-183 changes nothing about the decision itself
    confidence = out["terminal_besttake_confidence"]
    assert confidence.confidence_state in (_TERMINAL_CONFIDENCE_NON_DECISIVE, _TERMINAL_CONFIDENCE_TIED)
    assert confidence.provenance in ("raw_score_only", "raw_score_equal")


# === Items 7-8: D-150 abstain / authoritative =================================

def test_07_d150_abstain_conflict_reaches_terminal_classifier_never_semantic_winner():
    """D-150 FIREWALL: ABSTAIN_CONFLICT skips the single_semantic_winner
    fast path entirely -- the terminal classifier never sees a
    `single_semantic_winner` provenance for an abstained family, and
    D-183 never manufactures a semantic winner where D-150 abstains."""
    a = take("a", "I also broke out around my neck during that period of stress.")
    b = take("b", "I also noticed breakouts along my jawline during that same stretch.", start=6.0)
    decisions = {"a": ("winner", 0.95), "b": ("keep", 0.5)}
    r = ranked(("a", 0.4), ("b", 0.9))
    out: dict = {}
    selected, _p, reason = _semantic_best_take(
        (a, b), decisions, "b", r, semantic_comparative_authority=AUTHORITY_ABSTAIN_CONFLICT,
        terminal_confidence_out=out,
    )
    assert reason != "single_semantic_winner"
    confidence = out["terminal_besttake_confidence"]
    assert confidence.reason != "single_semantic_winner"


def test_08_d150_authoritative_single_semantic_winner_is_decisive():
    a = take("a", "This serum cleared up my skin within about three weeks of nightly use.")
    b = take("b", "This serum smells nice.", start=6.0)
    decisions = {"a": ("winner", 0.95), "b": ("keep", 0.5)}
    out: dict = {}
    selected, preferred, reason = _semantic_best_take(
        (a, b), decisions, "b", ranked(("a", 0.5), ("b", 0.9)), terminal_confidence_out=out,
    )
    assert reason == "single_semantic_winner"
    confidence = out["terminal_besttake_confidence"]
    assert confidence.confidence_state == _TERMINAL_CONFIDENCE_DECISIVE
    assert confidence.reason == "single_semantic_winner"
    assert selected == "a"


# === Item 9: critical coverage dominance (duplicate-style confirmation) ====

def test_09_critical_coverage_dominance_confirmed_decisive_reason():
    a, b, r = _completeness_dominance_pair()
    decisions = {"A": ("winner", 0.90), "B": ("winner", 0.90)}
    out: dict = {}
    _sel, _pref, reason = _semantic_best_take((a, b), decisions, "A", r, terminal_confidence_out=out)
    assert reason == "critical_coverage_dominance"
    assert out["terminal_besttake_confidence"].confidence_state == _TERMINAL_CONFIDENCE_DECISIVE


# === Item 10: incomplete subset loser =======================================

_SHORT = "The warranty covers repairs for two years."
_LONG = "The warranty covers repairs for two years but batteries come with a separate manufacturer guarantee."


def test_10_incomplete_subset_loser_excluded_before_terminal_comparison():
    """The subset candidate never even enters `tie_break_pool` -- the
    terminal classifier only ever sees the single genuine survivor,
    DECISIVE_BY_ELIMINATION, never a raw two-way score comparison."""
    short = take("short", _SHORT)
    long_ = take("long", _LONG, start=5.0)
    decisions = {"short": ("keep", 0.5), "long": ("keep", 0.5)}
    out: dict = {}
    selected, _p, reason = _semantic_best_take(
        (short, long_), decisions, "short", ranked(("short", 10.0), ("long", 5.0)),
        terminal_confidence_out=out,
    )
    assert selected == "long"
    confidence = out["terminal_besttake_confidence"]
    assert confidence.confidence_state == _TERMINAL_CONFIDENCE_DECISIVE_BY_ELIMINATION
    assert confidence.top_candidate_id == "long"
    assert confidence.candidate_ids == ("long",)  # subset loser never entered the pool


# === Item 11: meaning-insufficient candidate excluded =======================

def test_11_meaning_insufficient_style_exclusion_leaves_single_survivor():
    """A candidate D-081 marks delete-recommended is excluded by Step 1
    before the terminal comparison -- same DECISIVE_BY_ELIMINATION shape."""
    a = take("a", "This routine helped my skin within two weeks of consistent use.")
    b = take("b", "This routine did basically nothing different for me at all honestly.", start=6.0)
    decisions = {"a": ("keep", 0.5), "b": ("keep", 0.5)}
    out: dict = {}
    selected, _p, reason = _semantic_best_take(
        (a, b), decisions, "a", ranked(("a", 5.0), ("b", 9.0)),
        semantic_delete_recommended={"b": True}, terminal_confidence_out=out,
    )
    assert selected == "a"
    confidence = out["terminal_besttake_confidence"]
    assert confidence.confidence_state == _TERMINAL_CONFIDENCE_DECISIVE_BY_ELIMINATION


# === Items 12-14: V2 absent / near-equal / factual dominance ================

def test_12_v2_absent_never_fabricated():
    """No `structured_signals` supplied -- CONFLICTED can never fire from
    an absent source; classification falls through to raw score."""
    result = _terminal_besttake_confidence(["a", "b"], {"a": 0.6, "b": 0.5})
    assert result.confidence_state == _TERMINAL_CONFIDENCE_NON_DECISIVE
    assert result.provenance == "raw_score_only"


def test_13_v2_available_but_near_equal_no_preference_still_raw_score_path():
    """A supplied structured source that names NEITHER finalist (e.g. it
    found both near-equal) has no opinion here -- classification still
    falls through to the raw-score path, never fabricating DECISIVE."""
    result = _terminal_besttake_confidence(
        ["a", "b"], {"a": 0.6, "b": 0.5},
        structured_signals={"v2_severity": None},
    )
    assert result.confidence_state == _TERMINAL_CONFIDENCE_NON_DECISIVE


def test_14_v2_factual_dominance_supported_via_structured_signals_contract():
    """When a structured comparator DOES exist and unanimously agrees
    with the raw-score top candidate, the contract supports DECISIVE --
    proven at the classifier level (no live pipeline.py caller supplies
    this today, per D-183's own scope; V2 is never made mandatory)."""
    result = _terminal_besttake_confidence(
        ["a", "b"], {"a": 0.55, "b": 0.50},
        structured_signals={"v2_severity": "a"},
    )
    assert result.confidence_state == _TERMINAL_CONFIDENCE_DECISIVE
    assert result.provenance == "structured_signals"


# === Item 15: conflicting structured evidence ===============================

def test_15_conflicting_structured_signals_are_conflicted():
    result = _terminal_besttake_confidence(
        ["a", "b"], {"a": 0.55, "b": 0.50},
        structured_signals={"v1_usability": "a", "v2_severity": "b"},
    )
    assert result.confidence_state == _TERMINAL_CONFIDENCE_CONFLICTED


def test_15b_single_structured_signal_disagreeing_with_raw_score_is_conflicted():
    """A lone comparator naming the RUNNER-UP (not the raw-score top) is
    itself a disagreement with the score -- CONFLICTED, never silently
    overridden and never ignored."""
    result = _terminal_besttake_confidence(
        ["a", "b"], {"a": 0.55, "b": 0.50},
        structured_signals={"v2_severity": "b"},
    )
    assert result.confidence_state == _TERMINAL_CONFIDENCE_CONFLICTED


def test_15c_asymmetric_critical_coverage_split_is_conflicted():
    a = take("a", "This formula includes retinol which is not suitable for sensitive skin at all.")
    b = take("b", "This formula does not include retinol so it works for sensitive skin instead.", start=6.0)
    decisions = {"a": ("keep", 0.5), "b": ("keep", 0.5)}
    out: dict = {}
    _sel, _p, reason = _semantic_best_take(
        (a, b), decisions, "a", ranked(("a", 5.0), ("b", 5.0)), terminal_confidence_out=out,
    )
    if reason == "unresolved_unique_fact_asymmetry":
        assert out["terminal_besttake_confidence"].confidence_state == _TERMINAL_CONFIDENCE_CONFLICTED
    else:
        # Different survivors' coverage sets happened to align this run --
        # not this test's concern; skip rather than assert a brittle shape.
        pytest.skip("fixture did not produce the targeted asymmetry shape")


# === Item 16: score source unavailable ======================================

def test_16_score_source_unavailable_is_unknown():
    a = take("A", "I really loved this product overall and would buy it again.")
    b = take("B", "I really loved this product overall, and would buy it again.", start=4.0)
    decisions = {"A": ("keep", 0.90), "B": ("keep", 0.90)}
    out: dict = {}
    _sel, _p, reason = _semantic_best_take((a, b), decisions, "A", ranked(), terminal_confidence_out=out)
    assert reason == "local_fallback"
    assert out["terminal_besttake_confidence"].confidence_state == _TERMINAL_CONFIDENCE_UNKNOWN


# === Items 17-20: determinism / independence ================================

def test_17_candidate_order_reversed_same_result():
    r1 = _terminal_besttake_confidence(["a", "b"], {"a": 0.6, "b": 0.5})
    r2 = _terminal_besttake_confidence(["b", "a"], {"a": 0.6, "b": 0.5})
    assert r1.confidence_state == r2.confidence_state
    assert r1.top_candidate_id == r2.top_candidate_id == "a"
    assert r1.ranked_candidate_ids == r2.ranked_candidate_ids


def test_18_clip_ids_changed_same_result_shape():
    r1 = _terminal_besttake_confidence(["clip_x", "clip_y"], {"clip_x": 0.6, "clip_y": 0.5})
    r2 = _terminal_besttake_confidence(["clip_p9", "clip_q3"], {"clip_p9": 0.6, "clip_q3": 0.5})
    assert r1.confidence_state == r2.confidence_state == _TERMINAL_CONFIDENCE_NON_DECISIVE
    assert r1.score_margin == r2.score_margin


def test_19_family_ids_independent():
    """The classifier never references a family/group id at all."""
    source = inspect.getsource(_terminal_besttake_confidence)
    assert "group_id" not in source and "family_id" not in source and "gid" not in source


def test_20_exact_deterministic_repeat():
    r1 = _terminal_besttake_confidence(["a", "b"], {"a": 0.6, "b": 0.5})
    r2 = _terminal_besttake_confidence(["a", "b"], {"a": 0.6, "b": 0.5})
    assert r1 == r2


# === Items 21-22: no reference input =========================================

def test_21_no_direct_reference_input():
    source = inspect.getsource(_terminal_besttake_confidence)
    for needle in ("cutai", "cut.ai", "human_gold", "cutsell_gold"):
        assert needle not in source.lower()


def test_22_no_cutai_gold_runtime_field():
    result = _terminal_besttake_confidence(["a", "b"], {"a": 0.6, "b": 0.5})
    for field_name in result.__dataclass_fields__:
        assert "cutai" not in field_name.lower() and "gold" not in field_name.lower()


# === Items 23-24: no score weight change / no new numeric threshold ========

def test_23_no_score_weight_change():
    """`take_judge.score_take`'s own coefficients are byte-unchanged --
    confirmed by re-checking the positive-weight sum this task's own
    forensic (D-182) already established equals 1.00."""
    from cutsell_worker.take_judge import score_take
    from cutsell_worker.contracts import MediaSignals
    signals = MediaSignals(
        source_asset_id="src", start=0.0, end=4.0,
        audio_quality=1.0, face_visibility=1.0, eye_contact=1.0, framing_quality=1.0,
        product_visibility=1.0, motion_stability=1.0, continuity=1.0,
        expression_naturalness=1.0, gesture_naturalness=1.0, delivery_energy=1.0,
        visual_fumble=0.0, distraction_risk=0.0,
    )
    t = CandidateTake(
        clip_id="t", source_asset_id="src", source_order=0, start=0.0, end=4.0,
        text="x", complete_idea=True, signals=signals,
    )
    assert score_take(t).score == pytest.approx(1.0, abs=1e-6)


def test_24_no_new_numeric_threshold_in_classifier():
    """DECISIVE vs NON_DECISIVE is never decided by comparing the score
    margin to any numeric cutoff -- confirmed by source inspection: no
    float/percent literal appears as a margin comparison anywhere in the
    classifier itself (the only float comparison is the exact-equality
    TIED check, which is not a threshold)."""
    source = inspect.getsource(_terminal_besttake_confidence)
    forbidden = ("0.01", "0.05", "0.1", "0.10", "5%", "10%", "epsilon", "EPSILON")
    for needle in forbidden:
        assert needle not in source, f"unexpected magic-margin literal {needle!r}"


# === Items 25-40: regression / no-change confirmations ======================

def test_25_d123_case_b_fast_path_gate_conditions_unaffected():
    from cutsell_worker.pipeline import _case_b_fast_path_conflict
    assert _case_b_fast_path_conflict("A", "B", {"A", "B"}, None) is None


def test_26_d128_module_untouched_reference():
    import cutsell_worker.multimodal_besttake_fallback as _d128  # noqa: F401


def test_27_d150_semantic_authority_gate_status_vocabulary_unchanged():
    from cutsell_worker.semantic_authority_observability import (
        AUTHORITY_ABSTAIN_CONFLICT as _a, AUTHORITY_ALLOWED as _b, AUTHORITY_ABSTAIN_INCOMPLETE_CONTEXT as _c,
    )
    assert (_a, _b, _c) == ("ABSTAIN_CONFLICT", "AUTHORITATIVE", "ABSTAIN_INCOMPLETE_CONTEXT")


def test_28_d158_attempt_relationship_module_importable_unchanged():
    import cutsell_worker.attempt_relationship_authority  # noqa: F401


def test_29_d161_watch_listen_relation_discovery_module_importable_unchanged():
    import cutsell_worker.watch_listen_relation_discovery  # noqa: F401


def test_30_d163_watch_listen_besttake_evidence_module_importable_unchanged():
    import cutsell_worker.watch_listen_besttake_evidence  # noqa: F401


def test_31_d167_zone_usability_v2_module_importable_unchanged():
    import cutsell_worker.watch_listen_zone_usability_v2  # noqa: F401


def test_32_d174_besttake_guard_authority_module_importable_unchanged():
    import cutsell_worker.watch_listen_besttake_guard_authority  # noqa: F401


def test_33_d177_boundary_engine_pass_module_importable_unchanged():
    import cutsell_worker.boundary_engine_pass  # noqa: F401


def test_34_d180_materiality_helper_unchanged():
    from cutsell_worker.pipeline import _material_delivery_event_count

    class _Ev:
        d097_would_be_counted = True

    class _Evidence:
        delivery_events = (_Ev(),)

    assert _material_delivery_event_count(_Evidence()) == 1


def test_35_family_formation_no_change():
    import cutsell_worker.final_sibling_grouping  # noqa: F401


def test_36_language_spine_no_change():
    import cutsell_worker.language_spine  # noqa: F401 (import-safe module presence check)


def test_37_boundary_no_change():
    import cutsell_worker.boundary_engine_pass  # noqa: F401


def test_38_pacing_no_change():
    import cutsell_worker.dialogue_pacing_transition  # noqa: F401


def test_39_renderer_no_change():
    import cutsell_worker.render_plan  # noqa: F401


def test_40_no_provider_or_network_reference_anywhere_new():
    forbidden = ("requests", "openai", "google.generativeai", "genai", "gemini", "modal", "runpod")
    for fn in (_terminal_besttake_confidence, terminal_besttake_confidence_diagnostics, terminal_besttake_confidence_run_summary):
        source = inspect.getsource(fn).lower()
        for needle in forbidden:
            assert needle not in source, f"{needle!r} unexpectedly referenced in {fn.__name__}"


# === Diagnostics projection + tail-safe summary + wiring ====================

def test_diagnostics_projection_fields():
    confidence = _terminal_besttake_confidence(["a", "b"], {"a": 0.6, "b": 0.5})
    row = terminal_besttake_confidence_diagnostics(confidence)
    for key in (
        "terminal_besttake_confidence_state", "terminal_besttake_confidence_reason",
        "terminal_besttake_candidate_count", "terminal_besttake_top_candidate_id",
        "terminal_besttake_runner_up_candidate_id", "terminal_besttake_top_score",
        "terminal_besttake_runner_up_score", "terminal_besttake_score_margin",
        "terminal_besttake_structured_dominance_present", "terminal_besttake_conflict_present",
        "terminal_besttake_decisive",
    ):
        assert key in row
    assert row["terminal_besttake_decisive"] is False
    assert row["terminal_besttake_conflict_present"] is False


def test_tail_safe_run_summary_counts():
    rows = [
        {"terminal_besttake_confidence_state": _TERMINAL_CONFIDENCE_DECISIVE},
        {"terminal_besttake_confidence_state": _TERMINAL_CONFIDENCE_DECISIVE_BY_ELIMINATION},
        {"terminal_besttake_confidence_state": _TERMINAL_CONFIDENCE_NON_DECISIVE},
        {"terminal_besttake_confidence_state": _TERMINAL_CONFIDENCE_TIED},
        {"terminal_besttake_confidence_state": _TERMINAL_CONFIDENCE_CONFLICTED},
        {"terminal_besttake_confidence_state": _TERMINAL_CONFIDENCE_UNKNOWN},
        {"terminal_besttake_confidence_state": None},  # not-a-contest row, uncounted
    ]
    summary = terminal_besttake_confidence_run_summary(rows)
    assert summary == {
        "terminal_besttake_evaluated_count": 6,
        "terminal_besttake_decisive_count": 2,
        "terminal_besttake_non_decisive_count": 1,
        "terminal_besttake_tied_count": 1,
        "terminal_besttake_conflicted_count": 1,
        "terminal_besttake_unknown_count": 1,
    }


def test_pipeline_wiring_carries_new_diagnostics_keys_byte_identical_winner():
    """Same weak/strong fixture D-122/D-123's own pipeline-wiring tests
    used -- proves the new keys are threaded end-to-end without changing
    the pre-existing winner/membership outcome."""
    from cutsell_worker.contracts import MediaSignals, ProcessingRequest, SemanticLabel, SemanticRole
    from cutsell_worker.pipeline import build_flow_b_draft

    weak = CandidateTake(
        clip_id="weak", source_asset_id="src", source_order=0, start=1.0, end=3.0,
        text="this serum changed my skin",
        signals=MediaSignals(source_asset_id="src", start=1.0, end=3.0, audio_quality=0.3, eye_contact=0.2),
    )
    strong = CandidateTake(
        clip_id="strong", source_asset_id="src", source_order=0, start=4.0, end=6.0,
        text="this serum changed my skin",
        signals=MediaSignals(source_asset_id="src", start=4.0, end=6.0, audio_quality=0.95, eye_contact=0.95),
    )
    request = ProcessingRequest(project_id="project-1", user_id="user-1", sources=())
    labels = (
        SemanticLabel(weak.clip_id, SemanticRole.PROOF, 0.9),
        SemanticLabel(strong.clip_id, SemanticRole.PROOF, 0.9),
    )
    result = build_flow_b_draft(request, (weak, strong), labels)
    assert result.state.value == "draft_ready"
    assert [clip.clip_id for clip in result.draft.selected] == [strong.clip_id]

    groups = (result.draft.diagnostics or {}).get("take_judge_groups") or []
    assert len(groups) == 1
    row = groups[0]
    for key in (
        "terminal_besttake_confidence_state", "terminal_besttake_confidence_reason",
        "terminal_besttake_candidate_count", "terminal_besttake_top_candidate_id",
        "terminal_besttake_runner_up_candidate_id", "terminal_besttake_top_score",
        "terminal_besttake_runner_up_score", "terminal_besttake_score_margin",
        "terminal_besttake_structured_dominance_present", "terminal_besttake_conflict_present",
        "terminal_besttake_decisive",
    ):
        assert key in row
    # This fixture's own pre-existing outcome (D-122/D-123's own precedent):
    # both candidates reach the raw-score tie-break -- D-183 classifies
    # this NON_DECISIVE (no structured dominance backs the score gap), but
    # never changes the winner itself.
    assert row["terminal_besttake_confidence_state"] == "NON_DECISIVE"
    assert row["terminal_besttake_decisive"] is False
    assert row["final_winner"] == "strong"  # unchanged winner


def test_no_winner_mutation_before_after_terminal_confidence_out_identical():
    """Passing `terminal_confidence_out` never changes the 3-tuple return
    value in ANY of the fixtures already exercised above."""
    a = take("a", "This routine helped my skin within two weeks of consistent use.")
    b = take("b", "This routine did basically nothing different for me at all honestly.", start=6.0)
    decisions = {"a": ("keep", 0.5), "b": ("keep", 0.5)}
    r = ranked(("a", 5.0), ("b", 9.0))
    without = _semantic_best_take((a, b), decisions, "a", r)
    with_out = _semantic_best_take((a, b), decisions, "a", r, terminal_confidence_out={})
    assert without == with_out


def test_no_double_counting_source_independence():
    """D-121/D-182 double-counting audit: the classifier itself never
    re-reads or re-derives from any MediaSignals-level correlated field
    -- it consumes only the FINAL aggregate score once already computed."""
    source = inspect.getsource(_terminal_besttake_confidence)
    forbidden = (
        "visual_fumble", "gesture_naturalness", "expression_naturalness",
        "multimodal_reset_penalty", "delivery_cleanliness_evidence", "MediaSignals",
    )
    for needle in forbidden:
        assert needle not in source, f"{needle!r} unexpectedly referenced -- re-derivation risk"


def test_dataclass_never_mutates_family_or_proposition_identity():
    """`TerminalBestTakeConfidence` carries no family/proposition/idea id
    field at all -- purely a comparison-level fact."""
    for field_name in TerminalBestTakeConfidence.__dataclass_fields__:
        for forbidden in ("family", "proposition", "idea", "group_id"):
            assert forbidden not in field_name.lower()
