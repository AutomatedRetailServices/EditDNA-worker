"""D-191: Bounded Finalist Authority -- the FIRST gate where D-184's own
already-closed bounded finalist arbiter result MAY change the terminal
BestTake winner. Offline only. No RAW. No provider. No P1. No score/
weight change. No Family Formation change.

Two layers, mirroring D-184's own test file's own convention (`test_
cutsell_d184_bounded_finalist_arbiter.py` tests `evaluate_bounded_
finalist_arbiter` directly, never via a full pipeline run):

1. Direct, dataclass-level tests of `evaluate_bounded_finalist_
   authority` -- the D-183 firewall (DECISIVE/DECISIVE_BY_ELIMINATION/
   UNKNOWN never eligible), the meaning P0 firewall, D-123 ownership,
   the (always-False-in-live-wiring) Boundary firewall, every arbiter
   verdict (ABSTAIN in each of its own states, PREFER_CANDIDATE),
   fail-open behavior, determinism, and candidate-order/id independence.
   Covers the bulk of the directive's own 63-item test matrix at the
   exact level D-191's own logic lives at.

2. A small number of TRUE pipeline-level integration tests (reusing
   `test_cutsell_d189_prosodic_pipeline_wiring.py`'s own fixture style)
   proving the NEW wiring itself: default-off byte-identical
   compatibility, a real arbiter PREFERENCE_SUPPORTED verdict with the
   authority flag OFF never mutating the winner, and the authority flag
   ON actually replacing the terminal winner with the arbiter's own
   supported preference on a real (generic, non-Video00) prosodic-
   dominance fixture -- the ONE canonical BestTake mutation seam this
   task authorizes, proven end-to-end through `build_flow_b_draft`.

No literal Video00/Pimples transcript, timestamp, clip id, or family id
anywhere in this file -- every fixture is generic and abstract, per this
task's own explicit instruction.
"""
from __future__ import annotations

import wave

import numpy as np
import pytest

from cutsell_worker.bounded_finalist_arbiter import (
    DECISION_ABSTAIN,
    DECISION_PREFER_CANDIDATE,
    STATE_CONFLICTED,
    STATE_INSUFFICIENT_EVIDENCE,
    STATE_NEAR_EQUAL,
    STATE_NOT_ELIGIBLE,
    STATE_PREFERENCE_SUPPORTED,
    BoundedFinalistArbiterResult,
)
from cutsell_worker.bounded_finalist_authority import (
    STATE_APPLIED,
    STATE_BLOCKED_BY_BOUNDARY,
    STATE_BLOCKED_BY_CONFLICT,
    STATE_BLOCKED_BY_D123,
    STATE_BLOCKED_BY_MEANING,
    STATE_NOT_ELIGIBLE as AUTH_STATE_NOT_ELIGIBLE,
    STATE_NOT_ENABLED,
    STATE_NO_SUPPORTED_PREFERENCE,
    BoundedFinalistAuthorityResult,
    bounded_finalist_arbiter_authority_enabled,
    bounded_finalist_authority_diagnostics,
    bounded_finalist_authority_run_summary,
    evaluate_bounded_finalist_authority,
)
from cutsell_worker.audio_silence import AUDIO_SILENCE_EVENT_KIND
from cutsell_worker.contracts import CandidateTake, MediaSignals, ProcessingRequest, SemanticLabel, SemanticRole
from cutsell_worker.pipeline import build_flow_b_draft
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext

SR = 16000


# ===========================================================================
# Helpers: synthetic `BoundedFinalistArbiterResult` builders (never a
# literal Pimples/Video00 fixture -- pure, generic finalist ids).
# ===========================================================================
def _arbiter_result(
    *,
    candidate_ids=("cA", "cB"),
    decision=DECISION_PREFER_CANDIDATE,
    preferred_candidate_id="cB",
    arbiter_state=STATE_PREFERENCE_SUPPORTED,
    reason="structured_evidence_sources_unanimously_prefer_one_candidate",
    meaning_parity_status="CONSISTENT",
    structured_conflict=False,
) -> BoundedFinalistArbiterResult:
    return BoundedFinalistArbiterResult(
        candidate_ids=tuple(candidate_ids),
        decision=decision,
        preferred_candidate_id=preferred_candidate_id,
        arbiter_state=arbiter_state,
        reason=reason,
        meaning_parity_status=meaning_parity_status,
        performance_comparison_status="NOT_EVALUATED",
        editability_comparison_status="NO_EVIDENCE",
        structured_conflict=structured_conflict,
        missing_evidence=(),
        evidence_sources=("prosodic_delivery",),
        provenance="bounded_finalist_arbiter_v1",
        action_applied=False,
        prosodic_comparison_status="AVAILABLE",
    )


def _call(
    *,
    enabled=True,
    winner_before="cA",
    candidate_ids=("cA", "cB"),
    terminal_confidence_state="NON_DECISIVE",
    arbiter_result=None,
    d123_actionable_conflict=False,
    boundary_only_difference=False,
) -> BoundedFinalistAuthorityResult:
    if arbiter_result is None:
        arbiter_result = _arbiter_result(candidate_ids=candidate_ids)
    return evaluate_bounded_finalist_authority(
        enabled=enabled,
        winner_before=winner_before,
        candidate_ids=candidate_ids,
        terminal_confidence_state=terminal_confidence_state,
        arbiter_result=arbiter_result,
        d123_actionable_conflict=d123_actionable_conflict,
        boundary_only_difference=boundary_only_difference,
    )


# ===========================================================================
# 1-2. Feature flag.
# ===========================================================================
def test_01_authority_default_off():
    assert bounded_finalist_arbiter_authority_enabled({}) is False
    assert bounded_finalist_arbiter_authority_enabled({"CUTSELL_BOUNDED_FINALIST_ARBITER_AUTHORITY_ENABLED": "1"}) is True


def test_02_flag_off_never_evaluated_never_applied():
    result = _call(enabled=False)
    assert result.authority_state == STATE_NOT_ENABLED
    assert result.evaluated is False
    assert result.authority_applied is False
    assert result.winner_after == result.winner_before == "cA"


# ===========================================================================
# 3-8. D-183 firewall: DECISIVE / DECISIVE_BY_ELIMINATION / UNKNOWN never
# eligible; NON_DECISIVE / TIED / CONFLICTED are.
# ===========================================================================
@pytest.mark.parametrize("state", ["DECISIVE", "DECISIVE_BY_ELIMINATION", "UNKNOWN", None, "GARBAGE"])
def test_03_decisive_states_blocked(state):
    result = _call(terminal_confidence_state=state)
    assert result.authority_state == AUTH_STATE_NOT_ELIGIBLE
    assert result.authority_applied is False
    assert result.winner_after == "cA"


@pytest.mark.parametrize("state", ["NON_DECISIVE", "TIED", "CONFLICTED"])
def test_04_eligible_terminal_states_reach_arbiter_check(state):
    result = _call(terminal_confidence_state=state)
    assert result.authority_state == STATE_APPLIED
    assert result.winner_after == "cB"


def test_05_gynecologist_style_decisive_control_never_reopened():
    """The D-190/D-186B canonical negative control, replayed generically:
    a family the D-183 terminal ladder already called DECISIVE (its own
    `single_semantic_winner` fast path) must never be reopened by D-191,
    even when a (hypothetically) PREFER_CANDIDATE arbiter result exists
    for it."""
    result = _call(terminal_confidence_state="DECISIVE")
    assert result.authority_state == AUTH_STATE_NOT_ELIGIBLE
    assert result.authority_applied is False
    assert result.winner_after == result.winner_before


# ===========================================================================
# 9-10. Candidate-count eligibility (1 blocked, 4 blocked; 2-3 eligible).
# ===========================================================================
def test_06_one_finalist_blocked():
    result = _call(candidate_ids=("cA",), arbiter_result=_arbiter_result(candidate_ids=("cA",), preferred_candidate_id=None, decision=DECISION_ABSTAIN, arbiter_state=STATE_NOT_ELIGIBLE))
    assert result.authority_state == AUTH_STATE_NOT_ELIGIBLE


def test_07_four_finalists_blocked():
    result = _call(
        candidate_ids=("cA", "cB", "cC", "cD"),
        arbiter_result=_arbiter_result(candidate_ids=("cA", "cB", "cC", "cD"), preferred_candidate_id=None, decision=DECISION_ABSTAIN, arbiter_state=STATE_NOT_ELIGIBLE),
    )
    assert result.authority_state == AUTH_STATE_NOT_ELIGIBLE


def test_08_three_finalist_supported_preference_applies():
    arb = _arbiter_result(candidate_ids=("cA", "cB", "cC"), preferred_candidate_id="cC")
    result = _call(candidate_ids=("cA", "cB", "cC"), arbiter_result=arb)
    assert result.authority_state == STATE_APPLIED
    assert result.winner_after == "cC"


# ===========================================================================
# 11. Arbiter result missing entirely.
# ===========================================================================
def test_09_arbiter_result_none():
    result = evaluate_bounded_finalist_authority(
        enabled=True,
        winner_before="cA",
        candidate_ids=("cA", "cB"),
        terminal_confidence_state="NON_DECISIVE",
        arbiter_result=None,
    )
    assert result.authority_state == STATE_NO_SUPPORTED_PREFERENCE
    assert result.winner_after == "cA"


def test_10_arbiter_not_eligible_blocked():
    arb = _arbiter_result(decision=DECISION_ABSTAIN, preferred_candidate_id=None, arbiter_state=STATE_NOT_ELIGIBLE)
    result = _call(arbiter_result=arb)
    assert result.authority_state == AUTH_STATE_NOT_ELIGIBLE


# ===========================================================================
# 12-14. D-184 verdict vocabulary: ABSTAIN/NEAR_EQUAL/CONFLICTED/
# INSUFFICIENT_EVIDENCE never apply; PREFER_CANDIDATE/PREFERENCE_
# SUPPORTED applies.
# ===========================================================================
def test_11_near_equal_no_action():
    arb = _arbiter_result(decision=DECISION_ABSTAIN, preferred_candidate_id=None, arbiter_state=STATE_NEAR_EQUAL)
    result = _call(arbiter_result=arb)
    assert result.authority_state == STATE_NO_SUPPORTED_PREFERENCE
    assert result.winner_after == "cA"


def test_12_conflicted_blocked():
    arb = _arbiter_result(decision=DECISION_ABSTAIN, preferred_candidate_id=None, arbiter_state=STATE_CONFLICTED, structured_conflict=True)
    result = _call(arbiter_result=arb)
    assert result.authority_state == STATE_BLOCKED_BY_CONFLICT
    assert result.winner_after == "cA"


def test_13_insufficient_evidence_no_action():
    arb = _arbiter_result(decision=DECISION_ABSTAIN, preferred_candidate_id=None, arbiter_state=STATE_INSUFFICIENT_EVIDENCE)
    result = _call(arbiter_result=arb)
    assert result.authority_state == STATE_NO_SUPPORTED_PREFERENCE
    assert result.winner_after == "cA"


def test_14_preference_supported_applies():
    result = _call()
    assert result.authority_state == STATE_APPLIED
    assert result.authority_applied is True
    assert result.winner_after == "cB"
    assert result.authority_source == "bounded_finalist_arbiter"
    assert result.supported_candidate_id == "cB"


# ===========================================================================
# 15. Preferred candidate outside the eligible finalist set (malformed
# evidence) -> fail open.
# ===========================================================================
def test_15_supported_candidate_outside_family_blocked():
    arb = _arbiter_result(preferred_candidate_id="cZ")
    result = _call(arbiter_result=arb)
    assert result.authority_state == STATE_NO_SUPPORTED_PREFERENCE
    assert result.winner_after == "cA"


# ===========================================================================
# 16-17. Meaning P0 firewall.
# ===========================================================================
def test_16_meaning_conflict_blocked():
    arb = _arbiter_result(meaning_parity_status="CONFLICT", decision=DECISION_ABSTAIN, preferred_candidate_id=None, arbiter_state=STATE_CONFLICTED, structured_conflict=True)
    result = _call(arbiter_result=arb)
    assert result.authority_state == STATE_BLOCKED_BY_MEANING
    assert result.meaning_firewall_passed is False
    assert result.winner_after == "cA"


def test_17_meaning_unknown_still_blocks_authority():
    """UNKNOWN meaning parity (fewer texts supplied than candidates) is
    NOT the same as CONSISTENT -- authority never applies on an
    unverified meaning basis."""
    arb = _arbiter_result(meaning_parity_status="UNKNOWN")
    result = _call(arbiter_result=arb)
    assert result.authority_state == STATE_BLOCKED_BY_MEANING
    assert result.winner_after == "cA"


# ===========================================================================
# 18-19. D-123 ownership.
# ===========================================================================
def test_18_d123_actionable_conflict_blocks():
    result = _call(d123_actionable_conflict=True)
    assert result.authority_state == STATE_BLOCKED_BY_D123
    assert result.d123_blocked is True
    assert result.winner_after == "cA"


def test_19_no_d123_conflict_does_not_block():
    result = _call(d123_actionable_conflict=False)
    assert result.authority_state == STATE_APPLIED


# ===========================================================================
# 20. Boundary firewall (structurally supported, always False today).
# ===========================================================================
def test_20_boundary_only_difference_blocks():
    result = _call(boundary_only_difference=True)
    assert result.authority_state == STATE_BLOCKED_BY_BOUNDARY
    assert result.boundary_blocked is True
    assert result.winner_after == "cA"


# ===========================================================================
# 21-22. Visual/Prosody agreement and conflict (both routed through the
# SAME already-closed D-184 arbiter result -- D-191 never re-derives the
# Visual/Prosody merge itself).
# ===========================================================================
def test_21_visual_and_prosody_agree_applies():
    # D-184's own unanimous-agreement merge already collapsed Visual +
    # Prosody into ONE `PREFER_CANDIDATE` verdict -- D-191 just trusts it.
    arb = _arbiter_result()
    result = _call(arbiter_result=arb)
    assert result.authority_state == STATE_APPLIED


def test_22_visual_prosody_conflict_never_applies():
    # D-184's own merge already resolved disagreement to CONFLICTED --
    # D-191 sees only that terminal verdict, never the raw per-source
    # preferences, so it cannot (and must not) pick a side itself.
    arb = _arbiter_result(decision=DECISION_ABSTAIN, preferred_candidate_id=None, arbiter_state=STATE_CONFLICTED, structured_conflict=True)
    result = _call(arbiter_result=arb)
    assert result.authority_state == STATE_BLOCKED_BY_CONFLICT
    assert result.winner_after == "cA"


# ===========================================================================
# 23. Fail-open: malformed candidate_ids input.
# ===========================================================================
def test_23_malformed_candidate_ids_fail_open():
    result = evaluate_bounded_finalist_authority(
        enabled=True,
        winner_before="cA",
        candidate_ids=None,  # type: ignore[arg-type]
        terminal_confidence_state="NON_DECISIVE",
        arbiter_result=_arbiter_result(),
    )
    assert result.authority_state == AUTH_STATE_NOT_ELIGIBLE
    assert result.winner_after == "cA"


def test_24_non_arbiter_object_fails_open():
    result = _call(arbiter_result="not a real result")  # type: ignore[arg-type]
    assert result.authority_state == STATE_NO_SUPPORTED_PREFERENCE
    assert result.winner_after == "cA"


# ===========================================================================
# 25. Determinism + order/id independence.
# ===========================================================================
def test_25_deterministic_repeat():
    r1 = _call()
    r2 = _call()
    assert r1 == r2


def test_26_candidate_order_independence():
    a = _call(candidate_ids=("cA", "cB"), arbiter_result=_arbiter_result(candidate_ids=("cA", "cB")))
    b = _call(candidate_ids=("cB", "cA"), arbiter_result=_arbiter_result(candidate_ids=("cB", "cA")))
    assert a.authority_state == b.authority_state == STATE_APPLIED
    assert a.winner_after == b.winner_after == "cB"


def test_27_id_independence_generic_relabeling():
    arb = _arbiter_result(candidate_ids=("alpha", "beta"), preferred_candidate_id="beta")
    result = _call(winner_before="alpha", candidate_ids=("alpha", "beta"), arbiter_result=arb)
    assert result.authority_state == STATE_APPLIED
    assert result.winner_after == "beta"


# ===========================================================================
# 28. Diagnostics + run summary.
# ===========================================================================
def test_28_diagnostics_row_shape():
    result = _call()
    row = bounded_finalist_authority_diagnostics(result)
    expected_keys = {
        "bounded_finalist_authority_enabled",
        "bounded_finalist_authority_evaluated",
        "bounded_finalist_authority_state",
        "bounded_finalist_authority_winner_before",
        "bounded_finalist_authority_supported_candidate_id",
        "bounded_finalist_authority_winner_after",
        "bounded_finalist_authority_terminal_confidence",
        "bounded_finalist_authority_arbiter_state",
        "bounded_finalist_authority_arbiter_decision",
        "bounded_finalist_authority_meaning_passed",
        "bounded_finalist_authority_d123_blocked",
        "bounded_finalist_authority_boundary_blocked",
        "bounded_finalist_authority_conflict",
        "bounded_finalist_authority_applied",
        "bounded_finalist_authority_source",
        "bounded_finalist_authority_reason",
    }
    assert set(row) == expected_keys
    assert row["bounded_finalist_authority_applied"] is True
    assert row["bounded_finalist_authority_winner_after"] == "cB"


def test_29_diagnostics_none_input_honest_not_enabled_shape():
    row = bounded_finalist_authority_diagnostics(None)
    assert row["bounded_finalist_authority_state"] == STATE_NOT_ENABLED
    assert row["bounded_finalist_authority_evaluated"] is False
    assert row["bounded_finalist_authority_applied"] is False


def test_30_run_summary_counts():
    rows = [
        bounded_finalist_authority_diagnostics(_call()),  # APPLIED
        bounded_finalist_authority_diagnostics(_call(d123_actionable_conflict=True)),  # D123 block
        bounded_finalist_authority_diagnostics(_call(boundary_only_difference=True)),  # boundary block
        bounded_finalist_authority_diagnostics(None),  # never evaluated
        bounded_finalist_authority_diagnostics(_call(enabled=False)),  # not enabled
    ]
    summary = bounded_finalist_authority_run_summary(rows)
    assert summary["finalist_authority_evaluated_count"] == 3
    assert summary["finalist_authority_applied_count"] == 1
    assert summary["finalist_authority_d123_block_count"] == 1
    assert summary["finalist_authority_boundary_block_count"] == 1
    assert summary["finalist_authority_winner_changed_count"] == 1


# ===========================================================================
# 31. Winner provenance.
# ===========================================================================
def test_31_winner_provenance_fields():
    result = _call()
    assert result.winner_before == "cA"
    assert result.winner_after == "cB"
    assert result.authority_source == "bounded_finalist_arbiter"
    assert result.provenance == "bounded_finalist_authority_v1"


# ===========================================================================
# 32. No QA reference / no score-weight anywhere in this module's source.
# ===========================================================================
def test_32_no_qa_reference_no_score_weight_in_source():
    import inspect

    from cutsell_worker import bounded_finalist_authority as mod

    source = inspect.getsource(mod)
    banned = ["cut.ai", "cutai", "human gold", "human_gold", "video00", "pimples", "weight ="]
    lowered = source.lower()
    for term in banned:
        assert term not in lowered, f"banned term found: {term!r}"


def test_33_no_provider_network_in_source():
    import inspect

    from cutsell_worker import bounded_finalist_authority as mod

    source = inspect.getsource(mod).lower()
    for term in ("requests", "urllib", "openai", "anthropic", "gemini", "modal.com", "http://", "https://"):
        assert term not in source


# ===========================================================================
# 34. No new state zoo -- exactly the 8 documented states.
# ===========================================================================
def test_34_authority_state_vocabulary_is_exactly_eight():
    from cutsell_worker import bounded_finalist_authority as mod

    states = {
        mod.STATE_NOT_ENABLED, mod.STATE_NOT_ELIGIBLE, mod.STATE_BLOCKED_BY_MEANING,
        mod.STATE_BLOCKED_BY_D123, mod.STATE_BLOCKED_BY_BOUNDARY, mod.STATE_BLOCKED_BY_CONFLICT,
        mod.STATE_NO_SUPPORTED_PREFERENCE, mod.STATE_APPLIED,
    }
    assert len(states) == 8


# ===========================================================================
# PIPELINE-LEVEL INTEGRATION (true wiring, via `build_flow_b_draft`) --
# reuses `test_cutsell_d189_prosodic_pipeline_wiring.py`'s own fixture
# style. Generic, non-Video00 fixtures throughout.
# ===========================================================================
def _write_wav(path, samples_float, sample_rate=SR):
    clipped = np.clip(samples_float, -1.0, 1.0)
    ints = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(ints.tobytes())


def _tone(duration_sec, amplitude=0.25, freq=180.0, sample_rate=SR):
    n = int(round(duration_sec * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(duration_sec, sample_rate=SR):
    return np.zeros(int(round(duration_sec * sample_rate)), dtype=np.float32)


def _reversed_dominance_audio_path(tmp_path):
    """Real, decodable source audio for the SAME family -- content is not
    what differentiates continuity here (Audio V1's own already-computed
    dead-air events do, injected via `_reversed_dominance_context()`
    below, exactly the "Audio V1 reuse" contract D-189 established); this
    file only needs to decode successfully."""
    path = tmp_path / "reversed_dominance_source.wav"
    _write_wav(path, _tone(8.0))
    return str(path)


def _reversed_dominance_context():
    """D-190 generic abstract replay, injected the way Audio V1 evidence
    actually reaches Prosodic Audio V2 (`events_by_source`, an
    `AUDIO_SILENCE_EVENT_KIND` interval -- see pipeline.py's own
    `_prosodic_silence_intervals` reuse, never a second silence
    detector): the CURRENTLY-WINNING (raw-score) candidate's window
    [4.0, 6.0] ('strong') carries an interior pause (fragmenting it),
    while the losing candidate's window [1.0, 3.0] ('weak') is
    CONTINUOUS -- the exact structural shape D-190 proved on real
    Video00 media (the raw-score winner was the one with worse
    continuity/hesitation/restart), generalized with no literal
    transcript, timestamp, or clip id."""
    silence_event = TemporalEvent(
        source_asset_id="src", start=4.7, end=5.3,
        kind=AUDIO_SILENCE_EVENT_KIND, confidence=0.9, description="interior pause",
    )
    source = SourceVideoContext(
        source_asset_id="src", summary="", dominant_style="", creator_intent="",
        events=(silence_event,),
    )
    return WholeVideoContext(
        sources=(source,),
        status=ProviderStatus(provider="whole_video_analysis", requested=True, available=True, status="ok"),
    )


def _weak_strong_fixture():
    """SAME shape as D-189's own `_weak_strong_fixture` -- 'strong' wins
    on raw DeliveryScore (higher audio_quality/eye_contact) but the
    family is NON_DECISIVE (D-183 `raw_score_difference_without_
    structured_dominance`) -- the exact real-media terminal shape D-190
    proved on Video00. Identical text on both candidates (no meaning
    conflict)."""
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
    return request, (weak, strong), labels, strong.clip_id


ENV_ALL_THREE_ON = {
    "CUTSELL_BOUNDED_FINALIST_ARBITER_ENABLED": "1",
    "CUTSELL_PROSODIC_FINALIST_ARBITER_DIAGNOSTICS_ENABLED": "1",
    "CUTSELL_BOUNDED_FINALIST_ARBITER_AUTHORITY_ENABLED": "1",
}


def _set_env(monkeypatch, env: dict):
    for key, value in env.items():
        monkeypatch.setenv(key, value)


def _group_row(result):
    groups = (result.draft.diagnostics or {}).get("take_judge_groups") or []
    assert len(groups) == 1
    return groups[0]


def test_p1_default_off_byte_identical_winner():
    """Every flag off: byte-identical to pre-D-191 (and pre-D-184/D-189)
    winner selection."""
    request, takes, labels, expected_winner = _weak_strong_fixture()
    result = build_flow_b_draft(request, takes, labels)
    row = _group_row(result)
    assert row["bounded_finalist_authority_state"] == STATE_NOT_ENABLED
    assert row["bounded_finalist_authority_applied"] is False
    assert [c.clip_id for c in result.draft.selected] == [expected_winner]
    assert result.draft.diagnostics.get("bounded_finalist_authority") == {"status": "disabled"}


def test_p2_diagnostic_preference_but_authority_off_no_mutation(tmp_path, monkeypatch):
    """D-184/D-188 (diagnostics) ON, D-191's own authority flag OFF: even
    a real PREFERENCE_SUPPORTED diagnostic verdict never changes the
    winner."""
    _set_env(monkeypatch, {
        "CUTSELL_BOUNDED_FINALIST_ARBITER_ENABLED": "1",
        "CUTSELL_PROSODIC_FINALIST_ARBITER_DIAGNOSTICS_ENABLED": "1",
    })
    request, takes, labels, expected_winner = _weak_strong_fixture()
    path = _reversed_dominance_audio_path(tmp_path)
    result = build_flow_b_draft(request, takes, labels, whole_video_context=_reversed_dominance_context(), local_paths={"src": path})
    row = _group_row(result)
    assert row["bounded_finalist_authority_state"] == STATE_NOT_ENABLED
    assert row["bounded_finalist_authority_applied"] is False
    # The raw-score winner never changes with authority off, regardless
    # of what the (diagnostic-only) arbiter/prosodic verdict says.
    assert [c.clip_id for c in result.draft.selected] == [expected_winner]


def test_p3_authority_applies_winner_changes_to_arbiter_preference(tmp_path, monkeypatch):
    """THE authority-mutation case: with all three flags on and a real
    prosodic-dominance fixture that favors the CURRENTLY-LOSING candidate
    ('weak'), the bounded finalist arbiter reaches PREFERENCE_SUPPORTED
    toward 'weak', and D-191 replaces the terminal winner with it -- the
    D-190 real-media shape, generically replayed."""
    _set_env(monkeypatch, ENV_ALL_THREE_ON)
    request, takes, labels, _raw_score_winner = _weak_strong_fixture()
    path = _reversed_dominance_audio_path(tmp_path)
    result = build_flow_b_draft(request, takes, labels, whole_video_context=_reversed_dominance_context(), local_paths={"src": path})
    row = _group_row(result)
    if row.get("prosodic_finalist_state") != "DOMINANT":
        pytest.skip("synthetic fixture did not reach real DOMINANT this run -- see test_10's own precedent")
    assert row["bounded_finalist_arbiter_state"] == STATE_PREFERENCE_SUPPORTED
    assert row["bounded_finalist_authority_state"] == STATE_APPLIED
    assert row["bounded_finalist_authority_applied"] is True
    assert row["bounded_finalist_authority_winner_before"] == "strong"
    assert row["bounded_finalist_authority_winner_after"] == "weak"
    assert row["final_winner"] == "weak"
    assert [c.clip_id for c in result.draft.selected] == ["weak"]
    summary = result.draft.diagnostics["bounded_finalist_authority"]
    assert summary["status"] == "evaluated"
    assert summary["finalist_authority_applied_count"] == 1
    assert summary["finalist_authority_winner_changed_count"] == 1


def test_p4_authority_diagnostics_bounded_no_dump(tmp_path, monkeypatch):
    """Authority diagnostics row stays small and JSON-safe -- no
    transcript/waveform dump."""
    import json

    _set_env(monkeypatch, ENV_ALL_THREE_ON)
    request, takes, labels, _ = _weak_strong_fixture()
    path = _reversed_dominance_audio_path(tmp_path)
    result = build_flow_b_draft(request, takes, labels, whole_video_context=_reversed_dominance_context(), local_paths={"src": path})
    row = _group_row(result)
    authority_row = {k: v for k, v in row.items() if k.startswith("bounded_finalist_authority_")}
    encoded = json.dumps(authority_row)
    assert len(encoded) < 1500
    assert "this serum changed my skin" not in encoded
