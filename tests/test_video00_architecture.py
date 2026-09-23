import json

from benchmarks.validate_video00_architecture import validate


def write_json(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _valid_v1_result(**overrides):
    result = {
        "stage_status": {
            "semantic": "clean_cut_core_v1_idea_first",
            "selection_phase_authority": "clean_cut_core_v1_idea_first_keep_discard",
            "unified_selection_reasoner": "disabled_clean_cut_core_v1",
            "freeze_blocked_pending_coherence_review": False,
            "human_boundary_polish": "source_evidenced_multimodal_v5_boundary_only_complete",
        },
        "diagnostics": {
            "canonical_edit_plan": {
                "plan_id": "plan_abc123",
                "plan_version": 1,
                "semantic_hash": "hash_abc123",
                "validation_state": "frozen_ready",
            },
            "final_edit_reviewer": {"status": "PASS", "findings": [], "warnings": []},
            "repair_loop": {"status": "PASS", "attempt_count": 0, "attempts": []},
            "final_story_coherence_validation": {"freeze_blocked": False},
            "selection_boundary_contract": {
                "status": "verified",
                "plan_id": "plan_abc123",
                "plan_version": 1,
                "plan_semantic_hash": "hash_abc123",
            },
        },
        "live_render_qc": {"status": "PASS", "render_attempt_count": 1, "attempts": []},
        "alternate_count": 0,
        "hybrid_group_diagnostic_count": 3,
        "hybrid_requested_group_count": 5,
        "hybrid_available_group_count": 4,
    }
    result.update(overrides)
    return result


# ---------------------------------------------------------------------------
# 9. The new architecture verifier passes a valid Clean Cut Core V1 run
# ---------------------------------------------------------------------------

def test_verifier_passes_a_valid_clean_cut_core_v1_run(tmp_path):
    ok, report = validate(write_json(tmp_path, "result.json", _valid_v1_result()))

    assert ok is True
    assert report["architecture_verified"] is True
    assert report["failed_check_count"] == 0


# ---------------------------------------------------------------------------
# 10. Old whole-video reasoner being absent is expected, not an error
# ---------------------------------------------------------------------------

def test_absent_whole_video_reasoner_is_expected_not_an_error(tmp_path):
    result = _valid_v1_result()
    # Exactly the shape a real RAW's result JSON has: the key is missing
    # entirely (never set) rather than present-and-false.
    assert "unified_selection_reasoner" not in result["diagnostics"]

    ok, report = validate(write_json(tmp_path, "result.json", result))

    assert ok is True
    names = {c["name"]: c["ok"] for c in report["checks"]}
    assert names["whole_video_unified_selection_reasoner_absent_as_expected"] is True


# ---------------------------------------------------------------------------
# 11. SWAP being absent is expected, not an error
# ---------------------------------------------------------------------------

def test_swap_absent_is_expected_not_an_error(tmp_path):
    result = _valid_v1_result(alternate_count=0)

    ok, report = validate(write_json(tmp_path, "result.json", result))

    assert ok is True
    names = {c["name"]: c["ok"] for c in report["checks"]}
    assert names["swap_absent_as_expected"] is True


def test_a_nonzero_alternate_count_fails_the_swap_absent_check(tmp_path):
    # SWAP is out of scope for Clean Cut Core V1 (D-019) -- a parked
    # alternate would mean SWAP leaked back into the active path.
    result = _valid_v1_result(alternate_count=2)

    ok, report = validate(write_json(tmp_path, "result.json", result))

    assert ok is False
    assert "swap_absent_as_expected" in report["failed_checks"]


# ---------------------------------------------------------------------------
# 12. The verifier fails when CanonicalEditPlan/FinalEditReviewer/freeze
#     contract did not actually run
# ---------------------------------------------------------------------------

def test_verifier_fails_when_canonical_edit_plan_never_ran(tmp_path):
    result = _valid_v1_result()
    result["diagnostics"]["canonical_edit_plan"] = {}

    ok, report = validate(write_json(tmp_path, "result.json", result))

    assert ok is False
    assert "canonical_edit_plan_created" in report["failed_checks"]


def test_verifier_fails_when_final_edit_reviewer_never_ran(tmp_path):
    result = _valid_v1_result()
    result["diagnostics"]["final_edit_reviewer"] = {}

    ok, report = validate(write_json(tmp_path, "result.json", result))

    assert ok is False
    assert "final_edit_reviewer_executed" in report["failed_checks"]


def test_verifier_fails_when_freeze_contract_never_verified(tmp_path):
    result = _valid_v1_result()
    result["diagnostics"]["selection_boundary_contract"] = {"status": "frozen"}  # never verified

    ok, report = validate(write_json(tmp_path, "result.json", result))

    assert ok is False
    assert "freeze_references_exact_validated_plan" in report["failed_checks"]


def test_verifier_fails_the_stale_legacy_shape_that_broke_raw_33409169518(tmp_path):
    # The exact stale assertion this verifier replaces would have demanded
    # selection_reasoner_status == "applied" on a real V1 run that never
    # sets it at all. Confirm a real V1-shaped result (no legacy fields
    # whatsoever) still passes cleanly under the NEW verifier.
    result = _valid_v1_result()
    assert "selection_reasoner_status" not in result

    ok, _report = validate(write_json(tmp_path, "result.json", result))

    assert ok is True


# ---------------------------------------------------------------------------
# Freeze-blocked shape: architecture ran correctly even though it never
# reached Boundary/Render for this candidate.
# ---------------------------------------------------------------------------

def test_verifier_passes_a_correctly_freeze_blocked_run(tmp_path):
    result = _valid_v1_result(
        stage_status={
            "semantic": "clean_cut_core_v1_idea_first",
            "selection_phase_authority": "clean_cut_core_v1_idea_first_keep_discard+freeze_blocked_pending_human_review",
            "unified_selection_reasoner": "disabled_clean_cut_core_v1",
            "freeze_blocked_pending_coherence_review": True,
            "human_boundary_polish": "not_applicable_freeze_blocked_by_coherence_validation",
        },
        live_render_qc={"status": "not_attempted", "reason": "freeze_blocked_no_render", "render_attempt_count": 0, "attempts": []},
    )
    result["diagnostics"]["final_story_coherence_validation"] = {"freeze_blocked": True}
    result["diagnostics"]["selection_boundary_contract"] = {
        "status": "not_frozen_freeze_blocked_by_coherence_review",
        "plan_id": "plan_abc123",
        "plan_version": 1,
        "semantic_hash": "hash_abc123",
    }

    ok, report = validate(write_json(tmp_path, "result.json", result))

    assert ok is True
    assert report["freeze_blocked"] is True


def test_verifier_fails_if_freeze_blocked_but_boundary_ran_anyway(tmp_path):
    # A real bug this verifier must catch: semantic validation failed but
    # Boundary/render ran anyway, silently producing an unsafe candidate.
    result = _valid_v1_result(
        stage_status={
            "semantic": "clean_cut_core_v1_idea_first",
            "selection_phase_authority": "clean_cut_core_v1_idea_first_keep_discard+freeze_blocked_pending_human_review",
            "unified_selection_reasoner": "disabled_clean_cut_core_v1",
            "freeze_blocked_pending_coherence_review": True,
            "human_boundary_polish": "source_evidenced_multimodal_v5_boundary_only_complete",  # ran anyway -- bug
        },
    )
    result["diagnostics"]["final_story_coherence_validation"] = {"freeze_blocked": True}

    ok, report = validate(write_json(tmp_path, "result.json", result))

    assert ok is False
    assert "semantic_failure_correctly_blocked_freeze_and_boundary" in report["failed_checks"]
