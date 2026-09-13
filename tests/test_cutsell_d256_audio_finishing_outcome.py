"""Audio Finishing OUTCOME / PRODUCT-STATE CONTRACT (D-256).

Pure-Python tests -- no ffmpeg, no real media, no RAW. `audio_finishing_
outcome.py` is a pure derivation layer over already-computed D-247/D-249/
D-251 records, so every fixture here is either a directly-constructed
`AudioFinishingMeasurement`/`AudioFinishingPlan`/`AudioFinishingExecution
Record`/`ExecutionVerificationResult`, or the real `generate_audio_
finishing_plan` fed a synthetic measurement -- matching D-249's own test
philosophy exactly (only the D-247 measurement layer needs real ffmpeg
fixtures; everything downstream of a measurement is pure).
"""
from __future__ import annotations

import inspect

from cutsell_worker import audio_finishing_outcome as outcome_mod
from cutsell_worker.audio_finishing_executor import (
    EXECUTION_STATUS_FFMPEG_FAILURE,
    EXECUTION_STATUS_NO_ACTION_NEEDED,
    EXECUTION_STATUS_PLAN_NOT_EXECUTABLE,
    EXECUTION_STATUS_SUCCESS,
    VERIFICATION_STATUS_PASS,
    VERIFICATION_STATUS_POLICY_OUT_OF_RANGE,
    VERIFICATION_STATUS_TECHNICAL_FAILURE,
    AudioFinishingExecutionRecord,
    ExecutionVerificationResult,
)
from cutsell_worker.audio_finishing_measurement import (
    CLIPPING_STATUS_NO_CLIPPING_DETECTED,
    MEASUREMENT_STATUS_COMPLETE,
    MEASUREMENT_STATUS_PARTIAL,
    AudioFinishingMeasurement,
)
from cutsell_worker.audio_finishing_outcome import (
    EXECUTION_STATE_FAILED,
    EXECUTION_STATE_NO_ACTION_NEEDED,
    EXECUTION_STATE_NOT_RUN,
    EXECUTION_STATE_SUCCEEDED,
    OUTCOME_VERSION,
    PRODUCT_STATE_ABSTAINED,
    PRODUCT_STATE_BLOCKED_SAFETY,
    PRODUCT_STATE_COMPLETE,
    PRODUCT_STATE_PARTIAL_TOO_LOUD,
    PRODUCT_STATE_PARTIAL_TOO_QUIET,
    PRODUCT_STATE_SOURCE_RESCUE_REQUIRED,
    PRODUCT_STATE_UNKNOWN,
    REFINISH_DECISION_ALREADY_FINISHED_SAME_POLICY,
    REFINISH_DECISION_FINISHED_OUTPUT_AS_NEW_SOURCE,
    REFINISH_DECISION_NEW_SOURCE,
    REFINISH_DECISION_SAME_SOURCE_NEW_POLICY_VERSION,
    SOURCE_CLASS_ABSTAINED,
    SOURCE_CLASS_BLOCKED_SAFETY,
    SOURCE_CLASS_EXTREME_OVER_LEVEL,
    SOURCE_CLASS_EXTREME_UNDER_LEVEL,
    SOURCE_CLASS_NORMAL_CORRECTABLE,
    SOURCE_CLASS_UNKNOWN,
    WARNING_ALREADY_FINISHED,
    WARNING_FINISHED_OUTPUT_AS_SOURCE,
    WARNING_FINISHING_ABSTAINED,
    WARNING_FINISHING_BLOCKED_SAFETY,
    WARNING_FINISHING_PARTIAL,
    WARNING_MEASUREMENT_INCOMPLETE,
    WARNING_PEAK_SAFETY_UNVERIFIED,
    WARNING_SOURCE_TOO_LOUD,
    WARNING_SOURCE_TOO_QUIET,
    build_audio_finishing_outcome,
    classify_execution,
    classify_source,
    compute_export_allowed,
    compute_finishing_identity,
    compute_policy_complete,
    compute_product_state,
    compute_source_rescue_required,
    compute_warnings,
    decide_refinishing,
)
from cutsell_worker.audio_finishing_policy import (
    GAIN_STATE_ABSTAIN_INSUFFICIENT_EVIDENCE,
    GAIN_STATE_BLOCKED_SILENCE,
    GAIN_STATE_CORRECTION_ALLOWED,
    GAIN_STATE_CORRECTION_LIMITED,
    GAIN_STATE_NO_CHANGE_NEEDED,
    MAX_AUTOMATIC_GAIN_CORRECTION_DB,
    PEAK_EVIDENCE_TRUE_PEAK,
    POLICY_VERSION,
    TARGET_INTEGRATED_LOUDNESS_LUFS,
    TRUE_PEAK_CEILING_DBTP,
    generate_audio_finishing_plan,
)


def _measurement(
    *,
    integrated_loudness_lufs: float | None = -14.0,
    true_peak_dbfs: float | None = -3.0,
    measurement_status: str = MEASUREMENT_STATUS_COMPLETE,
) -> AudioFinishingMeasurement:
    return AudioFinishingMeasurement(
        media_path="/synthetic/d256_fixture.wav",
        window_start_sec=None, window_end_sec=None,
        duration_sec=144.854362, sample_rate_hz=48000,
        channel_count=2, channel_layout="stereo",
        integrated_loudness_lufs=integrated_loudness_lufs, loudness_range_lu=2.0,
        true_peak_dbfs=true_peak_dbfs, sample_peak_dbfs=true_peak_dbfs,
        clipping_status=CLIPPING_STATUS_NO_CLIPPING_DETECTED, silence_result=None,
        measurement_status=measurement_status, measurement_errors=(),
        provenance={"tool": "synthetic-fixture"},
    )


def _plan_for(lufs: float, *, true_peak_dbfs: float = -3.0):
    return generate_audio_finishing_plan(_measurement(integrated_loudness_lufs=lufs, true_peak_dbfs=true_peak_dbfs))


def _execution_record(*, execution_status: str, plan) -> AudioFinishingExecutionRecord:
    return AudioFinishingExecutionRecord(
        execution_id="exec-fixture-0001",
        policy_version=plan.policy_version,
        plan_status=plan.plan_status,
        input_path="/synthetic/d256_fixture.wav",
        output_path="/synthetic/d256_fixture_out.wav" if execution_status == EXECUTION_STATUS_SUCCESS else None,
        requested_whole_video_gain_db=plan.requested_whole_video_gain_db,
        authorized_whole_video_gain_db=plan.authorized_whole_video_gain_db,
        limiter_authorized=plan.limiter_authorized,
        true_peak_ceiling_dbtp=plan.true_peak_ceiling_dbtp,
        peak_evidence_source=plan.peak_evidence_source,
        filters_applied=("volume",) if execution_status == EXECUTION_STATUS_SUCCESS else (),
        execution_status=execution_status,
        ffmpeg_return_code=0 if execution_status == EXECUTION_STATUS_SUCCESS else None,
    )


def _verification(*, status: str, after_lufs: float | None, true_peak_within_ceiling: bool | None = True) -> ExecutionVerificationResult:
    return ExecutionVerificationResult(
        measurement_status=MEASUREMENT_STATUS_COMPLETE,
        integrated_loudness_lufs=after_lufs,
        true_peak_dbfs=-5.0,
        sample_peak_dbfs=-5.5,
        peak_evidence_source=PEAK_EVIDENCE_TRUE_PEAK,
        audio_present=True,
        duration_preserved=True,
        duration_delta_sec=0.0,
        sample_rate_expected=True,
        channel_count_expected=True,
        loudness_in_target_range=(status == VERIFICATION_STATUS_PASS),
        true_peak_within_ceiling=true_peak_within_ceiling,
        verification_status=status,
    )


# ---------------------------------------------------------------------------
# 1-2: module identity / no fabricated version drift.
# ---------------------------------------------------------------------------

def test_outcome_version_is_stable_string():
    assert OUTCOME_VERSION == "audio_finishing_outcome.v1"


def test_no_new_numeric_policy_constants_reexported():
    # Stage 1 binding: this module must not re-declare/re-export any of
    # the six canonical numeric constants -- it only ever reads them off
    # an already-decided AudioFinishingPlan.
    for name in (
        "TARGET_INTEGRATED_LOUDNESS_LUFS", "LOUDNESS_TOLERANCE_LU",
        "ADJACENT_TAKE_MISMATCH_THRESHOLD_LU", "MAX_AUTOMATIC_GAIN_CORRECTION_DB",
        "TRUE_PEAK_CEILING_DBTP", "MINIMUM_RELIABLE_LOUDNESS_WINDOW_SEC",
    ):
        assert not hasattr(outcome_mod, name), f"{name} must not be declared in the outcome module"


# ---------------------------------------------------------------------------
# 3-9: classify_source -- the ±6dB-envelope-only classification.
# ---------------------------------------------------------------------------

def test_classify_source_no_change_needed_is_normal_correctable():
    plan = _plan_for(-14.0)
    assert plan.whole_video_state == GAIN_STATE_NO_CHANGE_NEEDED
    assert classify_source(plan) == SOURCE_CLASS_NORMAL_CORRECTABLE


def test_classify_source_correction_allowed_is_normal_correctable():
    plan = _plan_for(-18.0)  # 4dB, within envelope
    assert plan.whole_video_state == GAIN_STATE_CORRECTION_ALLOWED
    assert classify_source(plan) == SOURCE_CLASS_NORMAL_CORRECTABLE


def test_classify_source_extreme_under_level():
    plan = _plan_for(-32.2)  # D-254C exact replay value, +18.2dB requested
    assert plan.whole_video_state == GAIN_STATE_CORRECTION_LIMITED
    assert plan.requested_whole_video_gain_db > 0
    assert classify_source(plan) == SOURCE_CLASS_EXTREME_UNDER_LEVEL


def test_classify_source_extreme_over_level():
    plan = _plan_for(4.0, true_peak_dbfs=-8.0)  # far too loud, negative requested gain
    assert plan.whole_video_state == GAIN_STATE_CORRECTION_LIMITED
    assert plan.requested_whole_video_gain_db < 0
    assert classify_source(plan) == SOURCE_CLASS_EXTREME_OVER_LEVEL


def test_classify_source_abstained():
    plan = _plan_for(None)  # no loudness measurement at all
    assert plan.whole_video_state == GAIN_STATE_ABSTAIN_INSUFFICIENT_EVIDENCE
    assert classify_source(plan) == SOURCE_CLASS_ABSTAINED


def test_classify_source_blocked_safety():
    measurement = _measurement(integrated_loudness_lufs=-40.0)
    from cutsell_worker.post_render_watch_listen_qc import PostRenderFinding, PostRenderQCResult
    silence = PostRenderQCResult(
        status="FAIL",
        findings=(PostRenderFinding(kind="LINGERING_ACCIDENTAL_SILENCE", start=0.0, end=5.0, detail={}, routes_to="BoundaryEngine"),),
    )
    measurement = AudioFinishingMeasurement(**{**measurement.__dict__, "silence_result": silence})
    plan = generate_audio_finishing_plan(measurement)
    assert plan.whole_video_state == GAIN_STATE_BLOCKED_SILENCE
    assert classify_source(plan) == SOURCE_CLASS_BLOCKED_SAFETY


def test_classify_source_unknown_when_correction_limited_without_requested_gain():
    plan = _plan_for(-32.2)
    plan = plan.__class__(**{**plan.__dict__, "requested_whole_video_gain_db": None})
    assert classify_source(plan) == SOURCE_CLASS_UNKNOWN


# ---------------------------------------------------------------------------
# 10-13: classify_execution.
# ---------------------------------------------------------------------------

def test_classify_execution_none_is_not_run():
    assert classify_execution(None) == EXECUTION_STATE_NOT_RUN


def test_classify_execution_plan_not_executable_is_not_run_not_failed():
    plan = _plan_for(None)
    record = _execution_record(execution_status=EXECUTION_STATUS_PLAN_NOT_EXECUTABLE, plan=plan)
    assert classify_execution(record) == EXECUTION_STATE_NOT_RUN


def test_classify_execution_success():
    plan = _plan_for(-18.0)
    record = _execution_record(execution_status=EXECUTION_STATUS_SUCCESS, plan=plan)
    assert classify_execution(record) == EXECUTION_STATE_SUCCEEDED


def test_classify_execution_ffmpeg_failure_is_failed():
    plan = _plan_for(-18.0)
    record = _execution_record(execution_status=EXECUTION_STATUS_FFMPEG_FAILURE, plan=plan)
    assert classify_execution(record) == EXECUTION_STATE_FAILED


def test_classify_execution_no_action_needed():
    plan = _plan_for(-14.0)
    record = _execution_record(execution_status=EXECUTION_STATUS_NO_ACTION_NEEDED, plan=plan)
    assert classify_execution(record) == EXECUTION_STATE_NO_ACTION_NEEDED


# ---------------------------------------------------------------------------
# 14-17: compute_policy_complete, including the NO_ACTION_NEEDED special case.
# ---------------------------------------------------------------------------

def test_policy_complete_true_on_pass_verification():
    verification = _verification(status=VERIFICATION_STATUS_PASS, after_lufs=-14.0)
    assert compute_policy_complete(EXECUTION_STATE_SUCCEEDED, verification, None) is True


def test_policy_complete_false_on_out_of_range_verification():
    verification = _verification(status=VERIFICATION_STATUS_POLICY_OUT_OF_RANGE, after_lufs=-26.2)
    assert compute_policy_complete(EXECUTION_STATE_SUCCEEDED, verification, None) is False


def test_policy_complete_true_for_no_action_needed_already_compliant():
    plan = _plan_for(-14.0)
    assert compute_policy_complete(EXECUTION_STATE_NO_ACTION_NEEDED, None, plan) is True


def test_policy_complete_false_when_no_verification_and_not_no_action_needed():
    assert compute_policy_complete(EXECUTION_STATE_FAILED, None, None) is False
    assert compute_policy_complete(EXECUTION_STATE_NOT_RUN, None, None) is False


# ---------------------------------------------------------------------------
# 18-21: compute_export_allowed independence from policy_complete.
# ---------------------------------------------------------------------------

def test_export_allowed_true_even_when_policy_incomplete_but_technically_safe():
    verification = _verification(status=VERIFICATION_STATUS_POLICY_OUT_OF_RANGE, after_lufs=-26.2)
    assert compute_export_allowed(EXECUTION_STATE_SUCCEEDED, verification, "PASS") is True


def test_export_allowed_false_on_technical_qc_failure():
    verification = _verification(status=VERIFICATION_STATUS_PASS, after_lufs=-14.0)
    assert compute_export_allowed(EXECUTION_STATE_SUCCEEDED, verification, "FAIL") is False


def test_export_allowed_false_on_technical_failure_verification():
    verification = _verification(status=VERIFICATION_STATUS_TECHNICAL_FAILURE, after_lufs=None)
    assert compute_export_allowed(EXECUTION_STATE_SUCCEEDED, verification, "PASS") is False


def test_export_allowed_false_on_execution_failed():
    assert compute_export_allowed(EXECUTION_STATE_FAILED, None, "PASS") is False


def test_export_allowed_true_when_execution_not_run_no_action_needed():
    assert compute_export_allowed(EXECUTION_STATE_NO_ACTION_NEEDED, None, None) is True


# ---------------------------------------------------------------------------
# 22: source_rescue_required always False in V1.
# ---------------------------------------------------------------------------

def test_source_rescue_required_always_false():
    assert compute_source_rescue_required() is False


# ---------------------------------------------------------------------------
# 23-28: compute_product_state -- ordering and guard correctness.
# ---------------------------------------------------------------------------

def test_product_state_complete():
    assert compute_product_state(SOURCE_CLASS_NORMAL_CORRECTABLE, EXECUTION_STATE_SUCCEEDED, True) == PRODUCT_STATE_COMPLETE


def test_product_state_partial_too_quiet():
    assert compute_product_state(SOURCE_CLASS_EXTREME_UNDER_LEVEL, EXECUTION_STATE_SUCCEEDED, False) == PRODUCT_STATE_PARTIAL_TOO_QUIET


def test_product_state_partial_too_loud():
    assert compute_product_state(SOURCE_CLASS_EXTREME_OVER_LEVEL, EXECUTION_STATE_SUCCEEDED, False) == PRODUCT_STATE_PARTIAL_TOO_LOUD


def test_product_state_abstained_regardless_of_execution_state():
    assert compute_product_state(SOURCE_CLASS_ABSTAINED, EXECUTION_STATE_NOT_RUN, False) == PRODUCT_STATE_ABSTAINED


def test_product_state_blocked_safety_regardless_of_execution_state():
    assert compute_product_state(SOURCE_CLASS_BLOCKED_SAFETY, EXECUTION_STATE_NOT_RUN, False) == PRODUCT_STATE_BLOCKED_SAFETY


def test_product_state_unknown_when_extreme_source_but_dsp_never_ran():
    # A genuine defect case: an extreme source that SHOULD have been
    # actionable, but the DSP failed/never ran -- must not be mislabeled
    # as if a partial correction had actually been applied.
    assert compute_product_state(SOURCE_CLASS_EXTREME_UNDER_LEVEL, EXECUTION_STATE_FAILED, False) == PRODUCT_STATE_UNKNOWN
    assert compute_product_state(SOURCE_CLASS_EXTREME_UNDER_LEVEL, EXECUTION_STATE_NOT_RUN, False) == PRODUCT_STATE_UNKNOWN


def test_product_state_never_produces_reserved_rescue_state():
    # SOURCE_RESCUE_REQUIRED is reserved for a future pipeline; this
    # module must never assign it.
    for source_class in (SOURCE_CLASS_NORMAL_CORRECTABLE, SOURCE_CLASS_EXTREME_UNDER_LEVEL,
                         SOURCE_CLASS_EXTREME_OVER_LEVEL, SOURCE_CLASS_ABSTAINED,
                         SOURCE_CLASS_BLOCKED_SAFETY, SOURCE_CLASS_UNKNOWN):
        for execution_state in (EXECUTION_STATE_SUCCEEDED, EXECUTION_STATE_NO_ACTION_NEEDED,
                                EXECUTION_STATE_FAILED, EXECUTION_STATE_NOT_RUN):
            for policy_complete in (True, False):
                assert compute_product_state(source_class, execution_state, policy_complete) != PRODUCT_STATE_SOURCE_RESCUE_REQUIRED


# ---------------------------------------------------------------------------
# 29-33: compute_warnings.
# ---------------------------------------------------------------------------

def test_warnings_extreme_under_level():
    warnings = compute_warnings(SOURCE_CLASS_EXTREME_UNDER_LEVEL, None, None)
    assert WARNING_SOURCE_TOO_QUIET in warnings
    assert WARNING_FINISHING_PARTIAL in warnings


def test_warnings_extreme_over_level():
    warnings = compute_warnings(SOURCE_CLASS_EXTREME_OVER_LEVEL, None, None)
    assert WARNING_SOURCE_TOO_LOUD in warnings
    assert WARNING_FINISHING_PARTIAL in warnings


def test_warnings_abstained_and_blocked():
    assert compute_warnings(SOURCE_CLASS_ABSTAINED, None, None) == (WARNING_FINISHING_ABSTAINED,)
    assert compute_warnings(SOURCE_CLASS_BLOCKED_SAFETY, None, None) == (WARNING_FINISHING_BLOCKED_SAFETY,)


def test_warnings_measurement_incomplete():
    partial_measurement = _measurement(measurement_status=MEASUREMENT_STATUS_PARTIAL)
    warnings = compute_warnings(SOURCE_CLASS_NORMAL_CORRECTABLE, partial_measurement, None)
    assert WARNING_MEASUREMENT_INCOMPLETE in warnings


def test_warnings_peak_safety_unverified():
    verification = _verification(status=VERIFICATION_STATUS_PASS, after_lufs=-14.0, true_peak_within_ceiling=None)
    warnings = compute_warnings(SOURCE_CLASS_NORMAL_CORRECTABLE, None, verification)
    assert WARNING_PEAK_SAFETY_UNVERIFIED in warnings


def test_warnings_deterministic_ordering():
    w1 = compute_warnings(SOURCE_CLASS_EXTREME_UNDER_LEVEL, _measurement(measurement_status=MEASUREMENT_STATUS_PARTIAL), None)
    w2 = compute_warnings(SOURCE_CLASS_EXTREME_UNDER_LEVEL, _measurement(measurement_status=MEASUREMENT_STATUS_PARTIAL), None)
    assert w1 == w2


# ---------------------------------------------------------------------------
# 34-38: finishing identity / double-finishing firewall.
# ---------------------------------------------------------------------------

def test_finishing_identity_deterministic():
    plan = _plan_for(-32.2)
    record = _execution_record(execution_status=EXECUTION_STATUS_SUCCESS, plan=plan)
    id1 = compute_finishing_identity("sha-source-a", plan, record, "sha-output-a")
    id2 = compute_finishing_identity("sha-source-a", plan, record, "sha-output-a")
    assert id1 == id2


def test_finishing_identity_changes_with_source_content():
    plan = _plan_for(-32.2)
    record = _execution_record(execution_status=EXECUTION_STATUS_SUCCESS, plan=plan)
    id1 = compute_finishing_identity("sha-source-a", plan, record, "sha-output-a")
    id2 = compute_finishing_identity("sha-source-b", plan, record, "sha-output-a")
    assert id1 != id2


def test_finishing_identity_changes_with_policy_version():
    plan_a = _plan_for(-32.2)
    plan_b = plan_a.__class__(**{**plan_a.__dict__, "policy_version": "V2"})
    record_a = _execution_record(execution_status=EXECUTION_STATUS_SUCCESS, plan=plan_a)
    record_b = _execution_record(execution_status=EXECUTION_STATUS_SUCCESS, plan=plan_b)
    id_a = compute_finishing_identity("sha-source-a", plan_a, record_a, "sha-output-a")
    id_b = compute_finishing_identity("sha-source-a", plan_b, record_b, "sha-output-a")
    assert id_a != id_b


def test_finishing_identity_not_filename_based():
    # Same content hash, different (irrelevant) media_path on the plan's
    # own measurement reference -- identity must not change.
    measurement_a = _measurement(integrated_loudness_lufs=-32.2)
    measurement_b = AudioFinishingMeasurement(**{**measurement_a.__dict__, "media_path": "/some/other/name.wav"})
    plan_a = generate_audio_finishing_plan(measurement_a)
    plan_b = generate_audio_finishing_plan(measurement_b)
    record_a = _execution_record(execution_status=EXECUTION_STATUS_SUCCESS, plan=plan_a)
    record_b = _execution_record(execution_status=EXECUTION_STATUS_SUCCESS, plan=plan_b)
    id_a = compute_finishing_identity("sha-source-a", plan_a, record_a, "sha-output-a")
    id_b = compute_finishing_identity("sha-source-a", plan_b, record_b, "sha-output-a")
    assert id_a == id_b


def test_decide_refinishing_new_source():
    decision = decide_refinishing("sha-new", POLICY_VERSION, None, None, None, None)
    assert decision == REFINISH_DECISION_NEW_SOURCE


def test_decide_refinishing_already_finished_same_policy():
    decision = decide_refinishing(
        "sha-a", POLICY_VERSION, "fid-1", "sha-a", POLICY_VERSION, "sha-a-out",
    )
    assert decision == REFINISH_DECISION_ALREADY_FINISHED_SAME_POLICY


def test_decide_refinishing_same_source_new_policy_version():
    decision = decide_refinishing(
        "sha-a", "V2", "fid-1", "sha-a", POLICY_VERSION, "sha-a-out",
    )
    assert decision == REFINISH_DECISION_SAME_SOURCE_NEW_POLICY_VERSION


def test_decide_refinishing_finished_output_supplied_as_new_source_fail_closed():
    # The candidate's own content hash equals a PRIOR finishing's OUTPUT
    # hash -- caught even though no other record matches.
    decision = decide_refinishing(
        "sha-a-out", POLICY_VERSION, "fid-1", "sha-a", POLICY_VERSION, "sha-a-out",
    )
    assert decision == REFINISH_DECISION_FINISHED_OUTPUT_AS_NEW_SOURCE


# ---------------------------------------------------------------------------
# 39: the exact D-254C real-media replay regression fixture.
# ---------------------------------------------------------------------------

def test_d254c_exact_replay_fixture():
    """before_lufs=-32.2, requested_gain=+18.2, authorized_gain=+6.0,
    after_lufs=-26.2, final_true_peak=-5.0, technical_qc=PASS -- expected:
    EXTREME_UNDER_LEVEL / AUDIO_FINISHING_PARTIAL_SOURCE_TOO_QUIET /
    export_allowed=True / policy_complete=False (D-256 Stage 8)."""
    plan = _plan_for(-32.2)
    assert plan.whole_video_state == GAIN_STATE_CORRECTION_LIMITED
    assert round(plan.requested_whole_video_gain_db, 1) == 18.2
    assert plan.authorized_whole_video_gain_db == MAX_AUTOMATIC_GAIN_CORRECTION_DB

    record = _execution_record(execution_status=EXECUTION_STATUS_SUCCESS, plan=plan)
    verification = _verification(status=VERIFICATION_STATUS_POLICY_OUT_OF_RANGE, after_lufs=-26.2, true_peak_within_ceiling=True)

    result = build_audio_finishing_outcome(
        plan=plan,
        execution_record=record,
        verification=verification,
        technical_qc_status="PASS",
        source_sha256="sha-d254c-source",
        output_sha256="sha-d254c-output",
    )

    assert result.source_classification == SOURCE_CLASS_EXTREME_UNDER_LEVEL
    assert result.product_state == PRODUCT_STATE_PARTIAL_TOO_QUIET
    assert result.export_allowed is True
    assert result.policy_complete is False
    assert result.source_rescue_required is False
    assert result.before_integrated_loudness_lufs == -32.2
    assert result.after_integrated_loudness_lufs == -26.2
    assert result.authorized_gain_db == MAX_AUTOMATIC_GAIN_CORRECTION_DB
    assert result.outcome_version == OUTCOME_VERSION
    assert WARNING_SOURCE_TOO_QUIET in result.warnings
    assert WARNING_FINISHING_PARTIAL in result.warnings
    assert result.finishing_already_applied is False


# ---------------------------------------------------------------------------
# 40-43: normal-complete / extreme-over / abstain / blocked-safety
# end-to-end build_audio_finishing_outcome fixtures.
# ---------------------------------------------------------------------------

def test_normal_complete_fixture():
    plan = _plan_for(-14.0)
    record = _execution_record(execution_status=EXECUTION_STATUS_NO_ACTION_NEEDED, plan=plan)
    result = build_audio_finishing_outcome(plan=plan, execution_record=record, technical_qc_status="PASS")
    assert result.product_state == PRODUCT_STATE_COMPLETE
    assert result.export_allowed is True
    assert result.policy_complete is True


def test_extreme_over_level_fixture():
    plan = _plan_for(4.0, true_peak_dbfs=-8.0)
    record = _execution_record(execution_status=EXECUTION_STATUS_SUCCESS, plan=plan)
    verification = _verification(status=VERIFICATION_STATUS_POLICY_OUT_OF_RANGE, after_lufs=-2.0)
    result = build_audio_finishing_outcome(plan=plan, execution_record=record, verification=verification, technical_qc_status="PASS")
    assert result.source_classification == SOURCE_CLASS_EXTREME_OVER_LEVEL
    assert result.product_state == PRODUCT_STATE_PARTIAL_TOO_LOUD
    assert result.export_allowed is True
    assert result.policy_complete is False


def test_abstain_fixture():
    plan = _plan_for(None)
    result = build_audio_finishing_outcome(plan=plan, execution_record=None, technical_qc_status="PASS")
    assert result.source_classification == SOURCE_CLASS_ABSTAINED
    assert result.product_state == PRODUCT_STATE_ABSTAINED
    assert result.execution_status == EXECUTION_STATE_NOT_RUN
    assert result.export_allowed is True  # abstaining is safe, not a technical defect
    assert result.policy_complete is False


def test_blocked_safety_fixture():
    from cutsell_worker.post_render_watch_listen_qc import PostRenderFinding, PostRenderQCResult
    measurement = _measurement(integrated_loudness_lufs=-40.0)
    silence = PostRenderQCResult(
        status="FAIL",
        findings=(PostRenderFinding(kind="LINGERING_ACCIDENTAL_SILENCE", start=0.0, end=5.0, detail={}, routes_to="BoundaryEngine"),),
    )
    measurement = AudioFinishingMeasurement(**{**measurement.__dict__, "silence_result": silence})
    plan = generate_audio_finishing_plan(measurement)
    result = build_audio_finishing_outcome(plan=plan, execution_record=None, technical_qc_status="PASS")
    assert result.source_classification == SOURCE_CLASS_BLOCKED_SAFETY
    assert result.product_state == PRODUCT_STATE_BLOCKED_SAFETY
    assert result.execution_status == EXECUTION_STATE_NOT_RUN


# ---------------------------------------------------------------------------
# 44-45: technical QC separation proofs.
# ---------------------------------------------------------------------------

def test_technical_qc_failure_blocks_export_even_when_policy_complete():
    plan = _plan_for(-14.0)
    record = _execution_record(execution_status=EXECUTION_STATUS_NO_ACTION_NEEDED, plan=plan)
    result = build_audio_finishing_outcome(plan=plan, execution_record=record, technical_qc_status="FAIL")
    assert result.policy_complete is True
    assert result.export_allowed is False


def test_technical_qc_pass_does_not_force_policy_complete():
    plan = _plan_for(-32.2)
    record = _execution_record(execution_status=EXECUTION_STATUS_SUCCESS, plan=plan)
    verification = _verification(status=VERIFICATION_STATUS_POLICY_OUT_OF_RANGE, after_lufs=-26.2)
    result = build_audio_finishing_outcome(plan=plan, execution_record=record, verification=verification, technical_qc_status="PASS")
    assert result.export_allowed is True
    assert result.policy_complete is False


# ---------------------------------------------------------------------------
# 46-48: double-finishing firewall wired end-to-end through
# build_audio_finishing_outcome.
# ---------------------------------------------------------------------------

def test_build_outcome_flags_already_finished_same_policy():
    plan = _plan_for(-32.2)
    record = _execution_record(execution_status=EXECUTION_STATUS_SUCCESS, plan=plan)
    verification = _verification(status=VERIFICATION_STATUS_POLICY_OUT_OF_RANGE, after_lufs=-26.2)
    result = build_audio_finishing_outcome(
        plan=plan, execution_record=record, verification=verification, technical_qc_status="PASS",
        source_sha256="sha-a", output_sha256="sha-a-out",
        previous_finishing_identity="fid-1", previous_source_sha256="sha-a",
        previous_policy_version=plan.policy_version, previous_output_sha256="sha-a-out",
    )
    assert result.finishing_already_applied is True
    assert WARNING_ALREADY_FINISHED in result.warnings


def test_build_outcome_flags_finished_output_supplied_as_new_source():
    plan = _plan_for(-32.2)
    record = _execution_record(execution_status=EXECUTION_STATUS_SUCCESS, plan=plan)
    verification = _verification(status=VERIFICATION_STATUS_POLICY_OUT_OF_RANGE, after_lufs=-26.2)
    result = build_audio_finishing_outcome(
        plan=plan, execution_record=record, verification=verification, technical_qc_status="PASS",
        source_sha256="sha-a-out",  # candidate IS a prior output
        previous_finishing_identity="fid-1", previous_source_sha256="sha-a",
        previous_policy_version=plan.policy_version, previous_output_sha256="sha-a-out",
    )
    assert result.finishing_already_applied is False
    assert WARNING_FINISHED_OUTPUT_AS_SOURCE in result.warnings


def test_build_outcome_new_policy_version_is_not_flagged_as_already_finished():
    plan = _plan_for(-32.2)
    record = _execution_record(execution_status=EXECUTION_STATUS_SUCCESS, plan=plan)
    verification = _verification(status=VERIFICATION_STATUS_POLICY_OUT_OF_RANGE, after_lufs=-26.2)
    result = build_audio_finishing_outcome(
        plan=plan, execution_record=record, verification=verification, technical_qc_status="PASS",
        source_sha256="sha-a", output_sha256="sha-a-out",
        previous_finishing_identity="fid-1", previous_source_sha256="sha-a",
        previous_policy_version="V0-OLD", previous_output_sha256="sha-a-out",
    )
    assert result.finishing_already_applied is False
    assert WARNING_ALREADY_FINISHED not in result.warnings


# ---------------------------------------------------------------------------
# 49-50: no shared mutable state / no new numeric policy.
# ---------------------------------------------------------------------------

def test_no_shared_mutable_state_between_calls():
    plan = _plan_for(-32.2)
    record = _execution_record(execution_status=EXECUTION_STATUS_SUCCESS, plan=plan)
    verification = _verification(status=VERIFICATION_STATUS_POLICY_OUT_OF_RANGE, after_lufs=-26.2)
    r1 = build_audio_finishing_outcome(plan=plan, execution_record=record, verification=verification, technical_qc_status="PASS")
    r2 = build_audio_finishing_outcome(plan=plan, execution_record=record, verification=verification, technical_qc_status="PASS")
    assert r1 == r2
    assert r1.warnings is not r2.warnings or r1.warnings == r2.warnings  # tuples: value-equal, no shared mutation risk


def test_canonical_numeric_constants_unchanged():
    assert TARGET_INTEGRATED_LOUDNESS_LUFS == -14.0
    assert MAX_AUTOMATIC_GAIN_CORRECTION_DB == 6.0
    assert TRUE_PEAK_CEILING_DBTP == -1.0
    assert POLICY_VERSION == "V1"


# ---------------------------------------------------------------------------
# 51-52: structural "no DSP / no renderer/QC-authority touch" proofs.
# ---------------------------------------------------------------------------

# Actual DSP-invocation signatures, not prose mentions -- the module's own
# docstring legitimately explains (in English) that it never calls ffmpeg,
# so a bare "ffmpeg" substring check would false-positive on that sentence.
# These tokens only ever appear here if real DSP code were added.
_FORBIDDEN_DSP_TOKENS = ("loudnorm", "alimiter", "volume=", "import subprocess", "subprocess.run(", "subprocess.Popen(")


def test_outcome_module_contains_no_dsp_invocation():
    source = inspect.getsource(outcome_mod)
    for token in _FORBIDDEN_DSP_TOKENS:
        assert token not in source, f"unexpected DSP/process token {token!r} found in audio_finishing_outcome.py"


def test_outcome_module_does_not_import_renderer_or_qc_authorities():
    source = inspect.getsource(outcome_mod)
    for forbidden_import in (
        "import render", "from .render", "from .live_render_qc", "from .pacing",
        "from .human_boundary_polish", "from .selection_freeze",
    ):
        assert forbidden_import not in source


# ---------------------------------------------------------------------------
# 53-56: AudioFinishingOutcome field-list completeness (Stage 2 contract).
# ---------------------------------------------------------------------------

def test_outcome_dataclass_field_list_matches_stage2_contract():
    from dataclasses import fields
    names = {f.name for f in fields(outcome_mod.AudioFinishingOutcome)}
    expected = {
        "outcome_version", "execution_status", "policy_status", "product_state",
        "export_allowed", "policy_complete", "source_rescue_required",
        "technical_qc_status", "source_classification",
        "measurement_reference", "plan_reference", "execution_reference", "verification_reference",
        "before_integrated_loudness_lufs", "after_integrated_loudness_lufs",
        "target_loudness_lufs", "true_peak_ceiling_dbtp",
        "requested_gain_db", "authorized_gain_db",
        "reasons", "warnings", "provenance",
        "finishing_identity", "finishing_already_applied",
    }
    assert expected <= names


def test_outcome_dataclass_is_frozen():
    plan = _plan_for(-14.0)
    result = build_audio_finishing_outcome(plan=plan, execution_record=None, technical_qc_status="PASS")
    import dataclasses
    try:
        result.export_allowed = False
        assert False, "AudioFinishingOutcome must be frozen"
    except dataclasses.FrozenInstanceError:
        pass
