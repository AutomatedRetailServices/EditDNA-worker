"""Audio Finishing POLICY + PLAN GENERATION (D-249).

Pure-Python policy tests -- no ffmpeg, no real media, no RAW. Every fixture
is a directly-constructed `AudioFinishingMeasurement` (the D-247 dataclass)
with deterministic numeric fields, matching this gate's own "synthetic
policy fixtures... no real RAW" instruction: the policy layer consumes
already-computed measurements, so testing it needs no subprocess/media at
all -- only the measurement layer (D-247, already tested separately)
needs real ffmpeg fixtures.
"""
from __future__ import annotations

import pytest

from cutsell_worker.audio_finishing_measurement import (
    CLIPPING_STATUS_CLIPPING_DETECTED,
    CLIPPING_STATUS_NO_CLIPPING_DETECTED,
    CLIPPING_STATUS_UNKNOWN,
    MEASUREMENT_STATUS_COMPLETE,
    AudioFinishingMeasurement,
)
from cutsell_worker.audio_finishing_policy import (
    ACCEPTABLE_LOUDNESS_LOWER_BOUND_LUFS,
    ACCEPTABLE_LOUDNESS_UPPER_BOUND_LUFS,
    ADJACENT_TAKE_MISMATCH_THRESHOLD_LU,
    GAIN_STATE_ABSTAIN_INSUFFICIENT_EVIDENCE,
    GAIN_STATE_BLOCKED_CLIPPING,
    GAIN_STATE_BLOCKED_PEAK_RISK,
    GAIN_STATE_BLOCKED_SILENCE,
    GAIN_STATE_CORRECTION_ALLOWED,
    GAIN_STATE_CORRECTION_LIMITED,
    GAIN_STATE_NO_CHANGE_NEEDED,
    LOUDNESS_TOLERANCE_LU,
    MAX_AUTOMATIC_GAIN_CORRECTION_DB,
    MINIMUM_RELIABLE_LOUDNESS_WINDOW_SEC,
    PEAK_EVIDENCE_FALLBACK_SAMPLE_PEAK,
    PEAK_EVIDENCE_TRUE_PEAK,
    PEAK_EVIDENCE_UNAVAILABLE,
    PLAN_STATUS_ABSTAIN,
    PLAN_STATUS_BLOCKED,
    PLAN_STATUS_READY_FOR_CORRECTION,
    PLAN_STATUS_READY_NO_CHANGE,
    PLAN_STATUS_READY_WITH_LIMITER,
    TARGET_INTEGRATED_LOUDNESS_LUFS,
    TRUE_PEAK_CEILING_DBTP,
    AdjacentTakeAdjustment,
    evaluate_adjacent_take_continuity,
    evaluate_peak_safety,
    evaluate_whole_video_loudness,
    generate_audio_finishing_plan,
)
from cutsell_worker.post_render_watch_listen_qc import PostRenderFinding, PostRenderQCResult


def _measurement(
    *,
    duration_sec: float = 5.0,
    integrated_loudness_lufs: float | None = -14.0,
    true_peak_dbfs: float | None = -3.0,
    sample_peak_dbfs: float | None = -3.5,
    clipping_status: str = CLIPPING_STATUS_NO_CLIPPING_DETECTED,
    silence_result: PostRenderQCResult | None = None,
    channel_count: int | None = 2,
    channel_layout: str | None = "stereo",
    sample_rate_hz: int | None = 48000,
    measurement_status: str = MEASUREMENT_STATUS_COMPLETE,
) -> AudioFinishingMeasurement:
    return AudioFinishingMeasurement(
        media_path="/synthetic/fixture.wav",
        window_start_sec=None, window_end_sec=None,
        duration_sec=duration_sec, sample_rate_hz=sample_rate_hz,
        channel_count=channel_count, channel_layout=channel_layout,
        integrated_loudness_lufs=integrated_loudness_lufs, loudness_range_lu=2.0,
        true_peak_dbfs=true_peak_dbfs, sample_peak_dbfs=sample_peak_dbfs,
        clipping_status=clipping_status, silence_result=silence_result,
        measurement_status=measurement_status, measurement_errors=(),
        provenance={"tool": "synthetic-fixture"},
    )


def _silence_finding() -> PostRenderQCResult:
    return PostRenderQCResult(
        status="FAIL",
        findings=(PostRenderFinding(kind="LINGERING_ACCIDENTAL_SILENCE", start=0.0, end=5.0, detail={}, routes_to="BoundaryEngine"),),
    )


# ---------------------------------------------------------------------------
# Stage 18, items 1-3: canonical constants.
# ---------------------------------------------------------------------------

def test_canonical_constants_exact():
    assert TARGET_INTEGRATED_LOUDNESS_LUFS == -14.0
    assert LOUDNESS_TOLERANCE_LU == 1.0
    assert ADJACENT_TAKE_MISMATCH_THRESHOLD_LU == 2.0
    assert MAX_AUTOMATIC_GAIN_CORRECTION_DB == 6.0
    assert TRUE_PEAK_CEILING_DBTP == -1.0
    assert MINIMUM_RELIABLE_LOUDNESS_WINDOW_SEC == 1.5


def test_acceptable_lower_bound_exact():
    assert ACCEPTABLE_LOUDNESS_LOWER_BOUND_LUFS == -15.0


def test_acceptable_upper_bound_exact():
    assert ACCEPTABLE_LOUDNESS_UPPER_BOUND_LUFS == -13.0


# ---------------------------------------------------------------------------
# Whole-video: 1 (-14 exact), 2 (-13.5), 3 (-14.8), 4 (-18 quiet), 5 (-10 loud).
# ---------------------------------------------------------------------------

def test_target_exact_is_no_change():
    decision = evaluate_whole_video_loudness(_measurement(integrated_loudness_lufs=-14.0))
    assert decision.gain_state == GAIN_STATE_NO_CHANGE_NEEDED
    assert decision.requested_gain_db == 0.0
    assert decision.authorized_gain_db == 0.0


@pytest.mark.parametrize("lufs", [-13.5, -14.8])
def test_within_tolerance_is_no_change(lufs):
    decision = evaluate_whole_video_loudness(_measurement(integrated_loudness_lufs=lufs))
    assert decision.gain_state == GAIN_STATE_NO_CHANGE_NEEDED


def test_too_quiet_generates_requested_positive_gain():
    decision = evaluate_whole_video_loudness(_measurement(integrated_loudness_lufs=-18.0))
    assert decision.gain_state == GAIN_STATE_CORRECTION_ALLOWED
    assert decision.requested_gain_db == pytest.approx(4.0)
    assert decision.authorized_gain_db == pytest.approx(4.0)


def test_too_loud_generates_requested_negative_gain():
    decision = evaluate_whole_video_loudness(_measurement(integrated_loudness_lufs=-10.0))
    assert decision.gain_state == GAIN_STATE_CORRECTION_ALLOWED
    assert decision.requested_gain_db == pytest.approx(-4.0)
    assert decision.authorized_gain_db == pytest.approx(-4.0)


def test_requested_gain_preserved_separately_from_authorized_gain():
    # -18 LUFS needs +4dB (within envelope) -- requested == authorized here,
    # so use a case where they diverge: -25 LUFS needs +11dB, envelope is 6dB.
    decision = evaluate_whole_video_loudness(_measurement(integrated_loudness_lufs=-25.0, true_peak_dbfs=-20.0, sample_peak_dbfs=-20.0))
    assert decision.requested_gain_db == pytest.approx(11.0)
    assert decision.authorized_gain_db == pytest.approx(6.0)
    assert decision.requested_gain_db != decision.authorized_gain_db


def test_over_6db_does_not_silently_clamp_to_target():
    decision = evaluate_whole_video_loudness(_measurement(integrated_loudness_lufs=-25.0, true_peak_dbfs=-20.0, sample_peak_dbfs=-20.0))
    assert decision.gain_state == GAIN_STATE_CORRECTION_LIMITED
    # target would need +11dB; only +6dB authorized -> post-correction loudness
    # would be -19 LUFS, nowhere near -14 -- the plan must not pretend otherwise.
    assert decision.authorized_gain_db == 6.0
    assert decision.integrated_loudness_lufs + decision.authorized_gain_db != TARGET_INTEGRATED_LOUDNESS_LUFS


def test_over_6db_negative_also_limited():
    decision = evaluate_whole_video_loudness(_measurement(integrated_loudness_lufs=-2.0, true_peak_dbfs=-2.0, sample_peak_dbfs=-2.5))
    assert decision.gain_state == GAIN_STATE_CORRECTION_LIMITED
    assert decision.authorized_gain_db == pytest.approx(-6.0)


# ---------------------------------------------------------------------------
# Silence never receives positive gain.
# ---------------------------------------------------------------------------

def test_silence_never_receives_positive_gain():
    decision = evaluate_whole_video_loudness(_measurement(integrated_loudness_lufs=float("-inf")))
    assert decision.gain_state == GAIN_STATE_BLOCKED_SILENCE
    assert decision.authorized_gain_db == 0.0


def test_silence_result_fail_also_blocks():
    decision = evaluate_whole_video_loudness(_measurement(integrated_loudness_lufs=-30.0, silence_result=_silence_finding()))
    assert decision.gain_state == GAIN_STATE_BLOCKED_SILENCE


def test_missing_loudness_abstains():
    decision = evaluate_whole_video_loudness(_measurement(integrated_loudness_lufs=None))
    assert decision.gain_state == GAIN_STATE_ABSTAIN_INSUFFICIENT_EVIDENCE


def test_short_window_abstains():
    decision = evaluate_whole_video_loudness(_measurement(duration_sec=1.0, integrated_loudness_lufs=-20.0))
    assert decision.gain_state == GAIN_STATE_ABSTAIN_INSUFFICIENT_EVIDENCE


# ---------------------------------------------------------------------------
# Clipping proxy blocks positive gain, never declares a categorical defect.
# ---------------------------------------------------------------------------

def test_clipping_proxy_blocks_positive_gain():
    decision = evaluate_whole_video_loudness(_measurement(integrated_loudness_lufs=-20.0, clipping_status=CLIPPING_STATUS_CLIPPING_DETECTED))
    assert decision.gain_state == GAIN_STATE_BLOCKED_CLIPPING
    assert decision.authorized_gain_db == 0.0


def test_clipping_proxy_does_not_block_negative_gain():
    decision = evaluate_whole_video_loudness(_measurement(integrated_loudness_lufs=-8.0, clipping_status=CLIPPING_STATUS_CLIPPING_DETECTED))
    assert decision.gain_state != GAIN_STATE_BLOCKED_CLIPPING
    assert decision.authorized_gain_db < 0


def test_clipping_proxy_never_a_standalone_categorical_verdict_field():
    # Structural guard: the plan/decision objects never carry a bare
    # "is_clipped" boolean verdict field -- only the bounded gain_state.
    decision = evaluate_whole_video_loudness(_measurement(clipping_status=CLIPPING_STATUS_CLIPPING_DETECTED, integrated_loudness_lufs=-14.0))
    field_names = set(decision.__dataclass_fields__.keys())
    assert "is_clipped" not in field_names
    assert "clipping_verdict" not in field_names


# ---------------------------------------------------------------------------
# True peak preferred; sample peak fallback tagged; unknown peak fails closed.
# ---------------------------------------------------------------------------

def test_true_peak_preferred_when_both_present():
    result = evaluate_peak_safety(_measurement(true_peak_dbfs=-3.0, sample_peak_dbfs=-1.0), candidate_gain_db=1.0)
    assert result.peak_evidence_source == PEAK_EVIDENCE_TRUE_PEAK
    assert result.existing_peak_dbfs == -3.0


def test_sample_peak_fallback_tagged():
    result = evaluate_peak_safety(_measurement(true_peak_dbfs=None, sample_peak_dbfs=-3.0), candidate_gain_db=1.0)
    assert result.peak_evidence_source == PEAK_EVIDENCE_FALLBACK_SAMPLE_PEAK
    assert result.existing_peak_dbfs == -3.0


def test_unknown_peak_fails_closed_on_positive_gain():
    decision = evaluate_whole_video_loudness(_measurement(integrated_loudness_lufs=-20.0, true_peak_dbfs=None, sample_peak_dbfs=None))
    assert decision.gain_state == GAIN_STATE_ABSTAIN_INSUFFICIENT_EVIDENCE or decision.gain_state == GAIN_STATE_BLOCKED_PEAK_RISK


def test_unknown_peak_never_silently_assumed_safe():
    result = evaluate_peak_safety(_measurement(true_peak_dbfs=None, sample_peak_dbfs=None), candidate_gain_db=4.0)
    assert result.blocked is True
    assert result.peak_evidence_source == PEAK_EVIDENCE_UNAVAILABLE


# ---------------------------------------------------------------------------
# Limiter authorization only under peak-risk + authorized correction; never primary.
# ---------------------------------------------------------------------------

def test_limiter_authorized_when_positive_gain_would_cross_ceiling():
    decision = evaluate_whole_video_loudness(_measurement(integrated_loudness_lufs=-17.5, true_peak_dbfs=-2.5, sample_peak_dbfs=-3.0))
    assert decision.gain_state == GAIN_STATE_CORRECTION_ALLOWED
    assert decision.limiter_needed is True


def test_no_limiter_when_headroom_is_ample():
    decision = evaluate_whole_video_loudness(_measurement(integrated_loudness_lufs=-17.5, true_peak_dbfs=-20.0, sample_peak_dbfs=-20.0))
    assert decision.limiter_needed is False


def test_existing_peak_already_at_ceiling_blocks_positive_gain():
    decision = evaluate_whole_video_loudness(_measurement(integrated_loudness_lufs=-18.0, true_peak_dbfs=-0.5, sample_peak_dbfs=-0.5))
    assert decision.gain_state == GAIN_STATE_BLOCKED_PEAK_RISK
    assert decision.authorized_gain_db == 0.0


def test_limiter_never_used_for_negative_gain_or_no_change():
    decision = evaluate_whole_video_loudness(_measurement(integrated_loudness_lufs=-10.0, true_peak_dbfs=-0.9, sample_peak_dbfs=-1.0))
    assert decision.limiter_needed is False  # negative move never needs a limiter
    decision2 = evaluate_whole_video_loudness(_measurement(integrated_loudness_lufs=-14.0, true_peak_dbfs=-0.9))
    assert decision2.limiter_needed is False


# ---------------------------------------------------------------------------
# Adjacent-take continuity: <=2 LU no change, >2 LU allowed, >6 LU limited,
# short windows abstain.
# ---------------------------------------------------------------------------

def test_adjacent_delta_within_threshold_no_change():
    left = _measurement(integrated_loudness_lufs=-14.0)
    right = _measurement(integrated_loudness_lufs=-15.5)  # 1.5 LU delta
    adj = evaluate_adjacent_take_continuity(left, right, left_segment_id="a", right_segment_id="b")
    assert adj.gain_state == GAIN_STATE_NO_CHANGE_NEEDED


def test_adjacent_delta_exact_threshold_no_change():
    left = _measurement(integrated_loudness_lufs=-14.0)
    right = _measurement(integrated_loudness_lufs=-16.0)  # exactly 2.0 LU
    adj = evaluate_adjacent_take_continuity(left, right, left_segment_id="a", right_segment_id="b")
    assert adj.gain_state == GAIN_STATE_NO_CHANGE_NEEDED


def test_adjacent_delta_over_threshold_plan_generated():
    left = _measurement(integrated_loudness_lufs=-14.0)
    right = _measurement(integrated_loudness_lufs=-17.0)  # 3.0 LU delta
    adj = evaluate_adjacent_take_continuity(left, right, left_segment_id="a", right_segment_id="b")
    assert adj.gain_state == GAIN_STATE_CORRECTION_ALLOWED
    assert adj.requested_correction_db == pytest.approx(3.0)
    assert adj.authorized_correction_db == pytest.approx(3.0)
    assert adj.direction == "RAISE_RIGHT"


def test_adjacent_delta_over_envelope_bounded():
    left = _measurement(integrated_loudness_lufs=-14.0)
    right = _measurement(integrated_loudness_lufs=-22.0, true_peak_dbfs=-20.0, sample_peak_dbfs=-20.0)  # 8 LU delta
    adj = evaluate_adjacent_take_continuity(left, right, left_segment_id="a", right_segment_id="b")
    assert adj.gain_state == GAIN_STATE_CORRECTION_LIMITED
    assert adj.authorized_correction_db == pytest.approx(6.0)
    assert adj.requested_correction_db == pytest.approx(8.0)


def test_adjacent_short_left_window_abstains():
    left = _measurement(duration_sec=0.5, integrated_loudness_lufs=-14.0)
    right = _measurement(integrated_loudness_lufs=-20.0)
    adj = evaluate_adjacent_take_continuity(left, right, left_segment_id="a", right_segment_id="b")
    assert adj.gain_state == GAIN_STATE_ABSTAIN_INSUFFICIENT_EVIDENCE


def test_adjacent_short_right_window_abstains():
    left = _measurement(integrated_loudness_lufs=-14.0)
    right = _measurement(duration_sec=0.5, integrated_loudness_lufs=-20.0)
    adj = evaluate_adjacent_take_continuity(left, right, left_segment_id="a", right_segment_id="b")
    assert adj.gain_state == GAIN_STATE_ABSTAIN_INSUFFICIENT_EVIDENCE


def test_adjacent_mono_stereo_metadata_alone_does_not_change_outcome():
    left = _measurement(integrated_loudness_lufs=-14.0, channel_count=1, channel_layout="mono")
    right = _measurement(integrated_loudness_lufs=-14.0, channel_count=2, channel_layout="stereo")
    adj = evaluate_adjacent_take_continuity(left, right, left_segment_id="a", right_segment_id="b")
    assert adj.gain_state == GAIN_STATE_NO_CHANGE_NEEDED


def test_adjacent_clipping_blocks_the_raised_side():
    left = _measurement(integrated_loudness_lufs=-14.0)
    right = _measurement(integrated_loudness_lufs=-20.0, clipping_status=CLIPPING_STATUS_CLIPPING_DETECTED)
    adj = evaluate_adjacent_take_continuity(left, right, left_segment_id="a", right_segment_id="b")
    assert adj.gain_state == GAIN_STATE_BLOCKED_CLIPPING


# ---------------------------------------------------------------------------
# No per-clip independent normalization; no renderer mutation; no correction
# filters used anywhere in this module.
# ---------------------------------------------------------------------------

def test_per_clip_independent_normalization_impossible_by_construction():
    # There is no function anywhere in the policy module that normalizes a
    # single measurement to an absolute target independent of its neighbor --
    # only evaluate_whole_video_loudness (one whole-video Level-2 pass) and
    # evaluate_adjacent_take_continuity (relative to its neighbor, Level 1)
    # exist. Confirm the module defines no other gain-computing function.
    import inspect

    import cutsell_worker.audio_finishing_policy as policy_module

    module_level_functions = {
        name for name, obj in vars(policy_module).items()
        if inspect.isfunction(obj) and obj.__module__ == policy_module.__name__
    }
    assert module_level_functions == {
        "evaluate_peak_safety",
        "evaluate_whole_video_loudness",
        "evaluate_adjacent_take_continuity",
        "generate_audio_finishing_plan",
        "_is_silent",
        "_has_reliable_window",
        "_derive_plan_status",
    }


def test_no_ffmpeg_or_subprocess_import_in_policy_module():
    import ast
    import inspect

    import cutsell_worker.audio_finishing_policy as policy_module

    tree = ast.parse(inspect.getsource(policy_module))
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module.split(".")[0])
    assert "subprocess" not in imported_names
    assert not hasattr(policy_module, "subprocess")

    # Real DSP filter invocations (not prose mentioning them in the
    # docstring) never appear as actual code -- checked by scanning every
    # non-docstring, non-comment line for an ffmpeg-style filter token.
    source_lines = inspect.getsource(policy_module).splitlines()
    in_docstring = False
    for line in source_lines:
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            in_docstring = not in_docstring or stripped.count('"""') + stripped.count("'''") >= 2
            continue
        if in_docstring or stripped.startswith("#"):
            continue
        for forbidden in ("subprocess.", "loudnorm=", "alimiter=", "acompressor=", "afftdn=", "-af "):
            assert forbidden not in line, f"real code line references {forbidden!r}: {line!r}"


def test_plan_has_no_raw_ffmpeg_strings():
    plan = generate_audio_finishing_plan(_measurement(integrated_loudness_lufs=-20.0))
    for value in (plan.whole_video_state, plan.plan_status, plan.peak_evidence_source):
        assert "=" not in value and "filter" not in value.lower()


# ---------------------------------------------------------------------------
# Full plan generation: status vocabulary + natural-dynamics firewall.
# ---------------------------------------------------------------------------

def test_plan_ready_no_change():
    plan = generate_audio_finishing_plan(_measurement(integrated_loudness_lufs=-14.0))
    assert plan.plan_status == PLAN_STATUS_READY_NO_CHANGE
    assert plan.authorized_whole_video_gain_db == 0.0


def test_plan_ready_for_correction():
    plan = generate_audio_finishing_plan(_measurement(integrated_loudness_lufs=-18.0, true_peak_dbfs=-20.0, sample_peak_dbfs=-20.0))
    assert plan.plan_status == PLAN_STATUS_READY_FOR_CORRECTION
    assert plan.limiter_authorized is False


def test_plan_ready_with_limiter():
    plan = generate_audio_finishing_plan(_measurement(integrated_loudness_lufs=-17.5, true_peak_dbfs=-2.5, sample_peak_dbfs=-3.0))
    assert plan.plan_status == PLAN_STATUS_READY_WITH_LIMITER
    assert plan.limiter_authorized is True


def test_plan_blocked_on_silence():
    plan = generate_audio_finishing_plan(_measurement(integrated_loudness_lufs=float("-inf")))
    assert plan.plan_status == PLAN_STATUS_BLOCKED


def test_plan_abstain_on_missing_evidence():
    plan = generate_audio_finishing_plan(_measurement(integrated_loudness_lufs=None))
    assert plan.plan_status == PLAN_STATUS_ABSTAIN
    assert plan.abstentions


def test_plan_includes_adjacent_adjustments_and_traceable_provenance():
    left = _measurement(integrated_loudness_lufs=-14.0)
    right = _measurement(integrated_loudness_lufs=-19.0)
    plan = generate_audio_finishing_plan(
        _measurement(integrated_loudness_lufs=-14.0),
        adjacent_pairs=((left, right, "clip_1", "clip_2"),),
        provenance={"render_commit": "abc123"},
    )
    assert len(plan.adjacent_take_adjustments) == 1
    assert plan.adjacent_take_adjustments[0].left_segment_id == "clip_1"
    assert plan.provenance["render_commit"] == "abc123"
    assert plan.measurement_reference.integrated_loudness_lufs == -14.0


def test_plan_generation_accepts_no_semantic_identity_arguments():
    # Structural natural-dynamics-firewall guard (Stage 9): the function
    # signature has no speaker/take-family/recording-intent parameter to
    # even accidentally use for inference.
    import inspect

    sig = inspect.signature(generate_audio_finishing_plan)
    forbidden_params = {"speaker", "speaker_id", "take_family", "recording_intent", "same_speaker"}
    assert forbidden_params.isdisjoint(sig.parameters.keys())

    sig2 = inspect.signature(evaluate_adjacent_take_continuity)
    assert forbidden_params.isdisjoint(sig2.parameters.keys())


def test_adjacent_take_adjustment_is_frozen_dataclass():
    import dataclasses

    assert dataclasses.is_dataclass(AdjacentTakeAdjustment)
    left = _measurement(integrated_loudness_lufs=-14.0)
    right = _measurement(integrated_loudness_lufs=-14.0)
    adj = evaluate_adjacent_take_continuity(left, right)
    with pytest.raises(dataclasses.FrozenInstanceError):
        adj.gain_state = "TAMPERED"
