"""D-155 -- independent audit of D-154's integration.

D-154 gave `PerceptualReview` an explicit 4-state `watch_listen_status`
(BLOCKED / HUMAN_REVIEW_REQUIRED / SYSTEM_PASS / HUMAN_APPROVED), but never
wired it into the CLEAN RAW gate: `compute_clean_raw_metrics()` never
extracted the field from the persisted `result["perceptual_watch_listen"]`
artifact, and `evaluate_clean_raw_gate()` only ever inspected the old
`perceptual_status` (PASS/FAIL/UNCERTAIN from `overall_status()`). A
reviewed candidate genuinely awaiting a human's eyes
(`watch_listen_status=HUMAN_REVIEW_REQUIRED`, `perceptual_status=
UNCERTAIN`) with everything else clean therefore reached `GATE_PASS` --
reproduced exactly here and proven fixed alongside the other three states.
"""
from benchmarks.clean_raw_gate import (
    GATE_FAIL,
    GATE_INCOMPLETE,
    GATE_PASS,
    build_gate_report,
    compute_clean_raw_metrics,
    evaluate_clean_raw_gate,
)
from cutsell_worker import perceptual_watch_listen as pwl


def _clean_result(perceptual):
    return {
        "stage_status": {"story_completeness": "complete", "no_usable_realization_family_count": 0},
        "diagnostics": {},
        "live_render_qc": {"status": "PASS", "deliverable": True, "delivery_status": "DELIVERABLE",
                           "attempts": [{"status": "PASS", "findings": []}]},
        "perceptual_watch_listen": perceptual,
    }


def _clean_ladder():
    return {"physical_regions": [{"level": "LEVEL_3", "kind": "consensus_keep", "duration_sec": 5.0}],
            "summary": {"cutai_parity": {}}}


# ---------------------------------------------------------------------------
# The exact reported reproduction: watch_listen_status=HUMAN_REVIEW_REQUIRED,
# perceptual_status=UNCERTAIN, everything else clean -- must never be PASS.
# ---------------------------------------------------------------------------

def test_human_review_required_never_produces_clean_raw_pass():
    result = _clean_result({
        "status": "UNCERTAIN",
        "watch_listen_status": pwl.WATCH_LISTEN_HUMAN_REVIEW_REQUIRED,
        "gate_mode": pwl.GATE_MODE_STATE_MACHINE_V1,
        "capability_status_counts": {}, "routing": {},
    })
    metrics = compute_clean_raw_metrics(result, _clean_ladder())
    assert metrics["perceptual_watch_listen_status"] == pwl.WATCH_LISTEN_HUMAN_REVIEW_REQUIRED
    gate = evaluate_clean_raw_gate(metrics)
    assert gate["status"] != GATE_PASS
    assert gate["status"] == GATE_INCOMPLETE
    assert any("HUMAN_REVIEW_REQUIRED" in item for item in gate["missing_evidence"])
    # The candidate MP4 is a kept deliverable -- HUMAN_REVIEW_REQUIRED is a
    # review-pending/incomplete state, never a confirmed technical failure.
    assert gate["blocking"] == []


def test_human_review_required_never_produces_clean_raw_pass_via_full_report():
    report = build_gate_report(_clean_result({
        "status": "UNCERTAIN",
        "watch_listen_status": pwl.WATCH_LISTEN_HUMAN_REVIEW_REQUIRED,
        "gate_mode": pwl.GATE_MODE_STATE_MACHINE_V1,
        "capability_status_counts": {}, "routing": {},
    }), _clean_ladder())
    assert report["gate"]["status"] != GATE_PASS


# ---------------------------------------------------------------------------
# BLOCKED (EVALUATED_FAIL/ERROR on a capability) -> GATE_FAIL.
# ---------------------------------------------------------------------------

def test_blocked_fails_the_gate():
    result = _clean_result({
        "status": "FAIL",
        "watch_listen_status": pwl.WATCH_LISTEN_BLOCKED,
        "gate_mode": pwl.GATE_MODE_STATE_MACHINE_V1,
        "capability_status_counts": {}, "routing": {},
    })
    gate = evaluate_clean_raw_gate(compute_clean_raw_metrics(result, _clean_ladder()))
    assert gate["status"] == GATE_FAIL
    assert "perceptual:BLOCKED" in gate["blocking"]


# ---------------------------------------------------------------------------
# SYSTEM_PASS -> clears the perceptual portion of the gate.
# ---------------------------------------------------------------------------

def test_system_pass_clears_the_perceptual_portion_of_the_gate():
    result = _clean_result({
        "status": "PASS",
        "watch_listen_status": pwl.WATCH_LISTEN_SYSTEM_PASS,
        "gate_mode": pwl.GATE_MODE_STATE_MACHINE_V1,
        "capability_status_counts": {}, "routing": {},
    })
    gate = evaluate_clean_raw_gate(compute_clean_raw_metrics(result, _clean_ladder()))
    assert gate["status"] == GATE_PASS
    assert gate["blocking"] == [] and gate["missing_evidence"] == []


# ---------------------------------------------------------------------------
# HUMAN_APPROVED -> explicit human approval is recorded and clears the gate.
# ---------------------------------------------------------------------------

def test_human_approved_is_recorded_and_clears_the_gate():
    result = _clean_result({
        "status": "UNCERTAIN",
        "watch_listen_status": pwl.WATCH_LISTEN_HUMAN_APPROVED,
        "gate_mode": pwl.GATE_MODE_STATE_MACHINE_V1,
        "capability_status_counts": {}, "routing": {},
    })
    metrics = compute_clean_raw_metrics(result, _clean_ladder())
    assert metrics["perceptual_human_approved"] is True
    gate = evaluate_clean_raw_gate(metrics)
    assert gate["status"] == GATE_PASS
    assert gate["blocking"] == [] and gate["missing_evidence"] == []


def test_perceptual_human_approved_metric_is_false_for_the_other_three_states():
    for status in (pwl.WATCH_LISTEN_BLOCKED, pwl.WATCH_LISTEN_HUMAN_REVIEW_REQUIRED, pwl.WATCH_LISTEN_SYSTEM_PASS):
        result = _clean_result({"status": "x", "watch_listen_status": status})
        assert compute_clean_raw_metrics(result, _clean_ladder())["perceptual_human_approved"] is False


# ---------------------------------------------------------------------------
# Backward compatibility: a result.json produced before D-154 has no
# `watch_listen_status` key at all -- the gate must fall back to the legacy
# `perceptual_status` PASS/FAIL/UNCERTAIN check rather than treat the field
# as an unrecognized/blocking value.
# ---------------------------------------------------------------------------

def test_missing_watch_listen_status_falls_back_to_legacy_perceptual_status():
    fail_result = _clean_result({"status": "FAIL"})
    gate = evaluate_clean_raw_gate(compute_clean_raw_metrics(fail_result, _clean_ladder()))
    assert gate["status"] == GATE_FAIL and "perceptual:FAIL" in gate["blocking"]

    absent_result = _clean_result({})
    gate = evaluate_clean_raw_gate(compute_clean_raw_metrics(absent_result, _clean_ladder()))
    assert gate["status"] == GATE_INCOMPLETE and "perceptual_watch_listen" in gate["missing_evidence"]


def test_unrecognized_watch_listen_status_never_passes_silently():
    result = _clean_result({"status": "UNCERTAIN", "watch_listen_status": "SOME_FUTURE_STATE"})
    gate = evaluate_clean_raw_gate(compute_clean_raw_metrics(result, _clean_ladder()))
    assert gate["status"] != GATE_PASS
    assert any("unrecognized_status" in item for item in gate["missing_evidence"])
