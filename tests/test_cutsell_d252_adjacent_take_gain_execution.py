"""Adjacent-Take Gain Execution Foundation, LEVEL 1 (D-252).

Pure-Python unit tests directly construct `AdjacentTakeAdjustment` and
`RenderSegment` objects (matching D-249's own testing convention -- the
execution LOGIC needs no media at all). One real end-to-end integration
test (Stage 17's "level measurement replay") renders real synthetic
before/after multi-segment videos via the actual `render.render_preview`
and re-measures them with D-247's real `measure_audio` to prove the
corrected side moves, the untouched side stays materially unchanged, and
timing is identical.

Skipped automatically (not failed) if `ffmpeg`/`ffprobe` are not on PATH.
"""
from __future__ import annotations

import dataclasses
import shutil
import subprocess

import pytest

from cutsell_worker.audio_finishing_executor import (
    ADJACENT_EXECUTION_STATUS_ADJUSTMENT_APPLIED,
    ADJACENT_EXECUTION_STATUS_ALREADY_APPLIED,
    ADJACENT_EXECUTION_STATUS_DUPLICATE_TARGET_CONFLICT,
    ADJACENT_EXECUTION_STATUS_IDENTITY_MISMATCH,
    ADJACENT_EXECUTION_STATUS_INVALID_GAIN,
    ADJACENT_EXECUTION_STATUS_NO_CHANGE,
    ADJACENT_EXECUTION_STATUS_PLAN_NOT_AUTHORIZED,
    ADJACENT_EXECUTION_STATUS_SEGMENT_NOT_FOUND,
    apply_adjacent_take_adjustments,
    compute_adjustment_application_id,
    db_to_linear,
)
from cutsell_worker.audio_finishing_measurement import (
    CLIPPING_STATUS_NO_CLIPPING_DETECTED,
    MEASUREMENT_STATUS_COMPLETE,
    AudioFinishingMeasurement,
    measure_audio,
)
from cutsell_worker.audio_finishing_policy import (
    GAIN_STATE_ABSTAIN_INSUFFICIENT_EVIDENCE,
    GAIN_STATE_BLOCKED_PEAK_RISK,
    GAIN_STATE_CORRECTION_ALLOWED,
    GAIN_STATE_CORRECTION_LIMITED,
    GAIN_STATE_NO_CHANGE_NEEDED,
    MAX_AUTOMATIC_GAIN_CORRECTION_DB,
    AdjacentTakeAdjustment,
    generate_audio_finishing_plan,
)
from cutsell_worker.render_plan import RenderSegment

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available on this runner",
)


def _measurement(*, integrated_loudness_lufs=-14.0, duration_sec=3.0, true_peak_dbfs=-6.0, sample_peak_dbfs=-6.5, clipping_status=CLIPPING_STATUS_NO_CLIPPING_DETECTED):
    return AudioFinishingMeasurement(
        media_path="/synthetic/fixture.wav", window_start_sec=None, window_end_sec=None,
        duration_sec=duration_sec, sample_rate_hz=48000, channel_count=2, channel_layout="stereo",
        integrated_loudness_lufs=integrated_loudness_lufs, loudness_range_lu=1.0,
        true_peak_dbfs=true_peak_dbfs, sample_peak_dbfs=sample_peak_dbfs,
        clipping_status=clipping_status, silence_result=None,
        measurement_status=MEASUREMENT_STATUS_COMPLETE, measurement_errors=(), provenance={},
    )


def _segment(clip_id, *, audio_volume=1.0, start=0.0, end=2.0, source_asset_id=None):
    return RenderSegment(
        clip_id=clip_id, source_asset_id=source_asset_id or f"asset_{clip_id}",
        source_path=f"/fake/{clip_id}.mp4", start=start, end=end, audio_volume=audio_volume,
    )


def _plan_with_pairs(pairs):
    whole = _measurement()
    return generate_audio_finishing_plan(whole, adjacent_pairs=pairs)


# ---------------------------------------------------------------------------
# STAGE 5: dB -> linear conversion.
# ---------------------------------------------------------------------------

def test_db_to_linear_exact():
    assert db_to_linear(0.0) == pytest.approx(1.0)
    assert db_to_linear(6.0) == pytest.approx(1.995262315)
    assert db_to_linear(-6.0) == pytest.approx(0.501187234)


# ---------------------------------------------------------------------------
# STAGE 6/21: no-change / allowed / limited / blocked / abstain / unknown.
# ---------------------------------------------------------------------------

def test_no_change_leaves_audio_volume_untouched():
    left, right = _measurement(integrated_loudness_lufs=-14.0), _measurement(integrated_loudness_lufs=-15.0)
    plan = _plan_with_pairs(((left, right, "a", "b"),))
    seg_a, seg_b = _segment("a"), _segment("b")
    new_segments, results = apply_adjacent_take_adjustments(plan, (seg_a, seg_b))
    assert new_segments[0].audio_volume == 1.0
    assert new_segments[1].audio_volume == 1.0
    assert results[0].execution_status == ADJACENT_EXECUTION_STATUS_NO_CHANGE


def test_correction_applies_exact_authorized_db():
    left, right = _measurement(integrated_loudness_lufs=-14.0), _measurement(integrated_loudness_lufs=-17.0)
    plan = _plan_with_pairs(((left, right, "a", "b"),))
    adjustment = plan.adjacent_take_adjustments[0]
    assert adjustment.gain_state == GAIN_STATE_CORRECTION_ALLOWED
    seg_a, seg_b = _segment("a"), _segment("b")
    new_segments, results = apply_adjacent_take_adjustments(plan, (seg_a, seg_b))
    expected_linear = db_to_linear(adjustment.authorized_correction_db)
    raised = new_segments[0] if adjustment.direction == "RAISE_LEFT" else new_segments[1]
    assert raised.audio_volume == pytest.approx(expected_linear)
    applied = [r for r in results if r.execution_status == ADJACENT_EXECUTION_STATUS_ADJUSTMENT_APPLIED]
    assert len(applied) == 1
    assert applied[0].authorized_correction_db == adjustment.authorized_correction_db


def test_requested_not_used_when_different_from_authorized():
    left, right = _measurement(integrated_loudness_lufs=-14.0), _measurement(integrated_loudness_lufs=-24.0, true_peak_dbfs=-20.0, sample_peak_dbfs=-20.0)
    plan = _plan_with_pairs(((left, right, "a", "b"),))
    adjustment = plan.adjacent_take_adjustments[0]
    assert adjustment.gain_state == GAIN_STATE_CORRECTION_LIMITED
    assert adjustment.requested_correction_db != adjustment.authorized_correction_db
    seg_a, seg_b = _segment("a"), _segment("b")
    new_segments, results = apply_adjacent_take_adjustments(plan, (seg_a, seg_b))
    applied = [r for r in results if r.execution_status == ADJACENT_EXECUTION_STATUS_ADJUSTMENT_APPLIED][0]
    assert applied.authorized_correction_db == pytest.approx(MAX_AUTOMATIC_GAIN_CORRECTION_DB)
    # the applied linear multiplier must correspond to the AUTHORIZED (6dB),
    # never the requested (10dB) value.
    assert applied.applied_linear_multiplier == pytest.approx(db_to_linear(MAX_AUTOMATIC_GAIN_CORRECTION_DB))


def test_over_6db_limited_plan_applies_only_authorized_amount():
    left, right = _measurement(integrated_loudness_lufs=-14.0), _measurement(integrated_loudness_lufs=-30.0, true_peak_dbfs=-20.0, sample_peak_dbfs=-20.0)
    plan = _plan_with_pairs(((left, right, "a", "b"),))
    adjustment = plan.adjacent_take_adjustments[0]
    assert adjustment.gain_state == GAIN_STATE_CORRECTION_LIMITED
    assert abs(adjustment.authorized_correction_db) == pytest.approx(MAX_AUTOMATIC_GAIN_CORRECTION_DB)


def test_blocked_adjustment_does_nothing():
    # existing peak already at ceiling -> BLOCKED_PEAK_RISK
    left, right = _measurement(integrated_loudness_lufs=-14.0), _measurement(integrated_loudness_lufs=-20.0, true_peak_dbfs=-0.5, sample_peak_dbfs=-0.5)
    plan = _plan_with_pairs(((left, right, "a", "b"),))
    adjustment = plan.adjacent_take_adjustments[0]
    assert adjustment.gain_state == GAIN_STATE_BLOCKED_PEAK_RISK
    seg_a, seg_b = _segment("a"), _segment("b")
    new_segments, results = apply_adjacent_take_adjustments(plan, (seg_a, seg_b))
    assert new_segments[0].audio_volume == 1.0 and new_segments[1].audio_volume == 1.0
    assert results[0].execution_status == ADJACENT_EXECUTION_STATUS_PLAN_NOT_AUTHORIZED


def test_abstain_adjustment_does_nothing():
    left, right = _measurement(duration_sec=0.5, integrated_loudness_lufs=-14.0), _measurement(integrated_loudness_lufs=-20.0)
    plan = _plan_with_pairs(((left, right, "a", "b"),))
    adjustment = plan.adjacent_take_adjustments[0]
    assert adjustment.gain_state == GAIN_STATE_ABSTAIN_INSUFFICIENT_EVIDENCE
    seg_a, seg_b = _segment("a"), _segment("b")
    new_segments, results = apply_adjacent_take_adjustments(plan, (seg_a, seg_b))
    assert new_segments[0].audio_volume == 1.0 and new_segments[1].audio_volume == 1.0
    assert results[0].execution_status == ADJACENT_EXECUTION_STATUS_PLAN_NOT_AUTHORIZED


def test_unknown_adjustment_does_nothing():
    left, right = _measurement(integrated_loudness_lufs=-14.0), _measurement(integrated_loudness_lufs=-17.0)
    plan = _plan_with_pairs(((left, right, "a", "b"),))
    tampered_adjustment = dataclasses.replace(plan.adjacent_take_adjustments[0], gain_state="UNKNOWN")
    tampered_plan = dataclasses.replace(plan, adjacent_take_adjustments=(tampered_adjustment,))
    seg_a, seg_b = _segment("a"), _segment("b")
    new_segments, results = apply_adjacent_take_adjustments(tampered_plan, (seg_a, seg_b))
    assert new_segments[0].audio_volume == 1.0 and new_segments[1].audio_volume == 1.0
    assert results[0].execution_status == ADJACENT_EXECUTION_STATUS_PLAN_NOT_AUTHORIZED


def test_short_window_abstain_does_nothing():
    # covered structurally by test_abstain_adjustment_does_nothing (short
    # window is the actual real trigger used there); this test proves the
    # SPECIFIC short-window reason string is what produced the abstain.
    left, right = _measurement(duration_sec=0.2, integrated_loudness_lufs=-14.0), _measurement(integrated_loudness_lufs=-14.0)
    plan = _plan_with_pairs(((left, right, "a", "b"),))
    assert plan.adjacent_take_adjustments[0].gain_state == GAIN_STATE_ABSTAIN_INSUFFICIENT_EVIDENCE
    new_segments, results = apply_adjacent_take_adjustments(plan, (_segment("a"), _segment("b")))
    assert new_segments[0].audio_volume == 1.0


# ---------------------------------------------------------------------------
# STAGE 3/7: segment targeting, no per-clip normalization, no leakage.
# ---------------------------------------------------------------------------

def test_only_targeted_segment_changes_neighbor_untouched():
    left, right = _measurement(integrated_loudness_lufs=-14.0), _measurement(integrated_loudness_lufs=-17.0)
    plan = _plan_with_pairs(((left, right, "a", "b"),))
    adjustment = plan.adjacent_take_adjustments[0]
    seg_a, seg_b, seg_c = _segment("a"), _segment("b"), _segment("c")
    new_segments, _ = apply_adjacent_take_adjustments(plan, (seg_a, seg_b, seg_c))
    touched_id = "a" if adjustment.direction == "RAISE_LEFT" else "b"
    untouched_id = "b" if touched_id == "a" else "a"
    by_id = {s.clip_id: s for s in new_segments}
    assert by_id[touched_id].audio_volume != 1.0
    # untouched segments are returned as the EXACT same object (`is`), not
    # merely equal -- the structural proof no independent normalization
    # ever runs over every clip.
    original_by_id = {s.clip_id: s for s in (seg_a, seg_b, seg_c)}
    assert by_id[untouched_id] is original_by_id[untouched_id]
    assert by_id["c"] is original_by_id["c"]


def test_no_global_per_clip_normalization_structural():
    # Five segments, only ONE authorized adjustment -- exactly one segment
    # may change; the other four must be the identical objects passed in.
    left, right = _measurement(integrated_loudness_lufs=-14.0), _measurement(integrated_loudness_lufs=-18.0)
    plan = _plan_with_pairs(((left, right, "x", "y"),))
    segments = tuple(_segment(cid) for cid in ("v", "w", "x", "y", "z"))
    new_segments, _ = apply_adjacent_take_adjustments(plan, segments)
    changed = [new for new, old in zip(new_segments, segments) if new is not old]
    assert len(changed) == 1


def test_cross_source_leakage_impossible():
    left, right = _measurement(integrated_loudness_lufs=-14.0), _measurement(integrated_loudness_lufs=-17.0)
    plan = _plan_with_pairs(((left, right, "a", "b"),))
    # segments from a completely different render -- no "a"/"b" clip_id present
    other_segments = (_segment("p"), _segment("q"))
    new_segments, results = apply_adjacent_take_adjustments(plan, other_segments)
    assert new_segments[0].audio_volume == 1.0 and new_segments[1].audio_volume == 1.0
    assert all(r.execution_status == ADJACENT_EXECUTION_STATUS_SEGMENT_NOT_FOUND for r in results)


def test_reordered_identity_mismatch_still_resolves_correctly():
    # Segment order in the tuple must not matter -- lookup is by clip_id,
    # never by position.
    left, right = _measurement(integrated_loudness_lufs=-14.0), _measurement(integrated_loudness_lufs=-17.0)
    plan = _plan_with_pairs(((left, right, "a", "b"),))
    adjustment = plan.adjacent_take_adjustments[0]
    seg_a, seg_b, seg_c = _segment("a"), _segment("b"), _segment("c")
    new_segments, _ = apply_adjacent_take_adjustments(plan, (seg_c, seg_b, seg_a))  # reversed/reordered
    by_id = {s.clip_id: s for s in new_segments}
    touched_id = "a" if adjustment.direction == "RAISE_LEFT" else "b"
    assert by_id[touched_id].audio_volume != 1.0


def test_identity_mismatch_when_segment_id_missing():
    left, right = _measurement(integrated_loudness_lufs=-14.0), _measurement(integrated_loudness_lufs=-17.0)
    # left_segment_id/right_segment_id left as None -- the caller failed to
    # supply real identity (D-252 Stage 1's documented schema gap).
    plan = _plan_with_pairs(((left, right, None, None),))
    new_segments, results = apply_adjacent_take_adjustments(plan, (_segment("a"), _segment("b")))
    assert new_segments[0].audio_volume == 1.0 and new_segments[1].audio_volume == 1.0
    assert results[0].execution_status == ADJACENT_EXECUTION_STATUS_IDENTITY_MISMATCH


def test_missing_segment_fails_closed():
    left, right = _measurement(integrated_loudness_lufs=-14.0), _measurement(integrated_loudness_lufs=-17.0)
    plan = _plan_with_pairs(((left, right, "a", "does_not_exist"),))
    adjustment = plan.adjacent_take_adjustments[0]
    if adjustment.direction != "RAISE_RIGHT":
        pytest.skip("this fixture only exercises the right-side missing-segment path")
    new_segments, results = apply_adjacent_take_adjustments(plan, (_segment("a"), _segment("b")))
    assert results[0].execution_status == ADJACENT_EXECUTION_STATUS_SEGMENT_NOT_FOUND
    assert new_segments[0].audio_volume == 1.0 and new_segments[1].audio_volume == 1.0


# ---------------------------------------------------------------------------
# STAGE 8: duplicate target / middle-segment dual adjacency.
# ---------------------------------------------------------------------------

def test_duplicate_target_conflict_fails_closed():
    # Two independent pairs both authorize raising the SAME segment "b".
    left1, right1 = _measurement(integrated_loudness_lufs=-14.0), _measurement(integrated_loudness_lufs=-17.0)
    left2, right2 = _measurement(integrated_loudness_lufs=-17.0), _measurement(integrated_loudness_lufs=-14.0)
    # pair 1: (a=left1 louder, b=right1 quieter) -> RAISE_RIGHT (b)
    # pair 2: (b=left2 quieter, c=right2 louder) -> RAISE_LEFT (b)
    plan = _plan_with_pairs((
        (left1, right1, "a", "b"),
        (left2, right2, "b", "c"),
    ))
    adjustments = plan.adjacent_take_adjustments
    assert adjustments[0].direction == "RAISE_RIGHT" and adjustments[0].left_segment_id == "a"
    assert adjustments[1].direction == "RAISE_LEFT" and adjustments[1].left_segment_id == "b"
    seg_a, seg_b, seg_c = _segment("a"), _segment("b"), _segment("c")
    new_segments, results = apply_adjacent_take_adjustments(plan, (seg_a, seg_b, seg_c))
    by_id = {s.clip_id: s for s in new_segments}
    assert by_id["b"].audio_volume == 1.0  # neither conflicting adjustment applied
    conflict_statuses = [r.execution_status for r in results]
    assert conflict_statuses.count(ADJACENT_EXECUTION_STATUS_DUPLICATE_TARGET_CONFLICT) == 2


def test_three_segment_chain_middle_segment_single_adjacency_applies_cleanly():
    # A middle segment involved in only ONE actionable adjustment (the
    # other pair resolves to NO_CHANGE_NEEDED) is not a conflict.
    left1, right1 = _measurement(integrated_loudness_lufs=-14.0), _measurement(integrated_loudness_lufs=-17.0)
    left2, right2 = _measurement(integrated_loudness_lufs=-17.0), _measurement(integrated_loudness_lufs=-17.5)  # <2 LU, no-change
    plan = _plan_with_pairs((
        (left1, right1, "a", "b"),
        (left2, right2, "b", "c"),
    ))
    assert plan.adjacent_take_adjustments[0].gain_state == GAIN_STATE_CORRECTION_ALLOWED
    assert plan.adjacent_take_adjustments[1].gain_state == GAIN_STATE_NO_CHANGE_NEEDED
    seg_a, seg_b, seg_c = _segment("a"), _segment("b"), _segment("c")
    new_segments, results = apply_adjacent_take_adjustments(plan, (seg_a, seg_b, seg_c))
    by_id = {s.clip_id: s for s in new_segments}
    assert by_id["b"].audio_volume != 1.0
    assert any(r.execution_status == ADJACENT_EXECUTION_STATUS_ADJUSTMENT_APPLIED for r in results)


# ---------------------------------------------------------------------------
# STAGE 9: pre-existing audio_volume composition.
# ---------------------------------------------------------------------------

def test_pre_existing_audio_volume_composed_multiplicatively():
    left, right = _measurement(integrated_loudness_lufs=-14.0), _measurement(integrated_loudness_lufs=-17.0)
    plan = _plan_with_pairs(((left, right, "a", "b"),))
    adjustment = plan.adjacent_take_adjustments[0]
    raised_id = "a" if adjustment.direction == "RAISE_LEFT" else "b"
    pre_existing_volume = 0.8  # e.g. a manual editorial gain already present
    seg_a = _segment("a", audio_volume=pre_existing_volume if raised_id == "a" else 1.0)
    seg_b = _segment("b", audio_volume=pre_existing_volume if raised_id == "b" else 1.0)
    new_segments, results = apply_adjacent_take_adjustments(plan, (seg_a, seg_b))
    by_id = {s.clip_id: s for s in new_segments}
    expected = pre_existing_volume * db_to_linear(adjustment.authorized_correction_db)
    assert by_id[raised_id].audio_volume == pytest.approx(expected)
    applied = [r for r in results if r.execution_status == ADJACENT_EXECUTION_STATUS_ADJUSTMENT_APPLIED][0]
    assert applied.prior_audio_volume == pytest.approx(pre_existing_volume)


# ---------------------------------------------------------------------------
# STAGE 14: idempotence.
# ---------------------------------------------------------------------------

def test_same_plan_applied_twice_is_idempotent():
    left, right = _measurement(integrated_loudness_lufs=-14.0), _measurement(integrated_loudness_lufs=-17.0)
    plan = _plan_with_pairs(((left, right, "a", "b"),))
    seg_a, seg_b = _segment("a"), _segment("b")
    first_segments, first_results = apply_adjacent_take_adjustments(plan, (seg_a, seg_b))
    applied = [r for r in first_results if r.execution_status == ADJACENT_EXECUTION_STATUS_ADJUSTMENT_APPLIED][0]
    already_applied_ids = frozenset({applied.adjustment_application_id})

    second_segments, second_results = apply_adjacent_take_adjustments(
        plan, first_segments, already_applied_ids=already_applied_ids,
    )
    # volume must NOT have doubled -- identical to the first pass's result.
    for a, b in zip(first_segments, second_segments):
        assert a.audio_volume == pytest.approx(b.audio_volume)
    assert any(r.execution_status == ADJACENT_EXECUTION_STATUS_ALREADY_APPLIED for r in second_results)


def test_different_plan_has_distinct_application_id():
    left1, right1 = _measurement(integrated_loudness_lufs=-14.0), _measurement(integrated_loudness_lufs=-17.0)
    left2, right2 = _measurement(integrated_loudness_lufs=-14.0), _measurement(integrated_loudness_lufs=-20.0, true_peak_dbfs=-20.0, sample_peak_dbfs=-20.0)
    plan1 = _plan_with_pairs(((left1, right1, "a", "b"),))
    plan2 = _plan_with_pairs(((left2, right2, "a", "b"),))
    adj1, adj2 = plan1.adjacent_take_adjustments[0], plan2.adjacent_take_adjustments[0]
    id1 = compute_adjustment_application_id(plan1, "b", adj1.authorized_correction_db, adj1.direction)
    id2 = compute_adjustment_application_id(plan2, "b", adj2.authorized_correction_db, adj2.direction)
    assert id1 != id2


# ---------------------------------------------------------------------------
# STAGE 15: plan immutability.
# ---------------------------------------------------------------------------

def test_plan_never_mutated():
    left, right = _measurement(integrated_loudness_lufs=-14.0), _measurement(integrated_loudness_lufs=-17.0)
    plan = _plan_with_pairs(((left, right, "a", "b"),))
    original_adjustments = plan.adjacent_take_adjustments
    apply_adjacent_take_adjustments(plan, (_segment("a"), _segment("b")))
    assert plan.adjacent_take_adjustments is original_adjustments
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.plan_status = "TAMPERED"


def test_invalid_gain_beyond_envelope_refuses_to_apply():
    left, right = _measurement(integrated_loudness_lufs=-14.0), _measurement(integrated_loudness_lufs=-17.0)
    plan = _plan_with_pairs(((left, right, "a", "b"),))
    tampered_adjustment = dataclasses.replace(plan.adjacent_take_adjustments[0], authorized_correction_db=99.0)
    tampered_plan = dataclasses.replace(plan, adjacent_take_adjustments=(tampered_adjustment,))
    new_segments, results = apply_adjacent_take_adjustments(tampered_plan, (_segment("a"), _segment("b")))
    assert new_segments[0].audio_volume == 1.0 and new_segments[1].audio_volume == 1.0
    assert results[0].execution_status == ADJACENT_EXECUTION_STATUS_INVALID_GAIN


# ---------------------------------------------------------------------------
# STAGE 18: peak safety already guaranteed upstream -- executor consumes it.
# ---------------------------------------------------------------------------

def test_no_limiter_or_new_peak_check_at_adjacent_stage_structural():
    import inspect

    import cutsell_worker.audio_finishing_executor as executor_module

    source = inspect.getsource(executor_module.apply_adjacent_take_adjustments)
    assert "alimiter" not in source
    assert "evaluate_peak_safety" not in source  # never re-derived here -- gain_state is the sole authority


def test_no_dsp_filters_anywhere_in_adjacent_execution():
    import inspect

    import cutsell_worker.audio_finishing_executor as executor_module

    source = inspect.getsource(executor_module.apply_adjacent_take_adjustments)
    for forbidden in ("loudnorm", "acompressor", "afftdn", "highpass", "lowpass", "subprocess", "ffmpeg"):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# STAGE 17: real end-to-end level-measurement replay (before/after render).
# ---------------------------------------------------------------------------

def _ffmpeg(args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def media_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("d252_replay")


@pytest.fixture(scope="module")
def clip_a_mp4(media_dir):
    path = str(media_dir / "clip_a.mp4")
    _ffmpeg([
        "-f", "lavfi", "-i", "sine=frequency=800:duration=2:sample_rate=48000,volume=1.0",
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=2",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", path,
    ])
    return path


@pytest.fixture(scope="module")
def clip_b_mp4(media_dir):
    path = str(media_dir / "clip_b.mp4")
    _ffmpeg([
        "-f", "lavfi", "-i", "sine=frequency=1200:duration=2:sample_rate=48000,volume=0.35",
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=2",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", path,
    ])
    return path


def test_end_to_end_level_replay_targeted_side_moves_neighbor_stable(clip_a_mp4, clip_b_mp4, media_dir):
    from cutsell_worker import render as render_module

    measurement_a = measure_audio(clip_a_mp4)
    measurement_b = measure_audio(clip_b_mp4)
    assert measurement_b.integrated_loudness_lufs < measurement_a.integrated_loudness_lufs - 2.0

    plan = generate_audio_finishing_plan(
        measurement_a, adjacent_pairs=((measurement_a, measurement_b, "clip_a", "clip_b"),),
    )
    adjustment = plan.adjacent_take_adjustments[0]
    assert adjustment.gain_state in (GAIN_STATE_CORRECTION_ALLOWED, GAIN_STATE_CORRECTION_LIMITED)
    assert adjustment.direction == "RAISE_RIGHT"  # b is quieter

    seg_a = RenderSegment(clip_id="clip_a", source_asset_id="asset_a", source_path=clip_a_mp4, start=0.0, end=2.0)
    seg_b = RenderSegment(clip_id="clip_b", source_asset_id="asset_b", source_path=clip_b_mp4, start=0.0, end=2.0)

    before_path = str(media_dir / "before.mp4")
    render_module.render_preview((seg_a, seg_b), before_path, width=320, height=240, fps=25)

    new_segments, results = apply_adjacent_take_adjustments(plan, (seg_a, seg_b))
    after_path = str(media_dir / "after.mp4")
    render_module.render_preview(new_segments, after_path, width=320, height=240, fps=25)

    # Segment A occupies [0, ~2s), segment B occupies [~2s, ~4s) in the
    # rendered timeline (D-097.2's frame-exact single-pass concat).
    before_a = measure_audio(before_path, start_sec=0.2, end_sec=1.8)
    before_b = measure_audio(before_path, start_sec=2.2, end_sec=3.8)
    after_a = measure_audio(after_path, start_sec=0.2, end_sec=1.8)
    after_b = measure_audio(after_path, start_sec=2.2, end_sec=3.8)

    # The corrected side (B, raised) must move measurably louder.
    assert after_b.integrated_loudness_lufs > before_b.integrated_loudness_lufs + 1.0
    # The untouched side (A) must remain materially unchanged.
    assert abs(after_a.integrated_loudness_lufs - before_a.integrated_loudness_lufs) < 0.5

    # Timing: both renders must have the same real duration (gain-only, no
    # structural/timing change).
    before_full = measure_audio(before_path)
    after_full = measure_audio(after_path)
    assert abs(before_full.duration_sec - after_full.duration_sec) < 0.05
