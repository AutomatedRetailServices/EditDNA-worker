"""End-to-End Audio Finishing Composition (D-253).

Real ffmpeg fixtures drive the core positive-path tests (proving Level 1
+ render + Level 2 + verification actually work together on real media,
matching D-251/D-252's own convention); a handful of edge-case plans that
cannot be physically produced by a working ffmpeg pass, and the three
failure-injection tests (render/Level-2/verification failure), use
monkeypatching -- the composition module's own design principle is
"consume already-decided results from each stage," so exercising its
own orchestration/error-propagation logic in isolation from a real
sub-stage failure is a legitimate, standard technique here.

Skipped automatically (not failed) if `ffmpeg`/`ffprobe` are not on PATH.
"""
from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess

import pytest

from cutsell_worker.audio_finishing_composition import (
    COMPOSITION_STATUS_LEVEL1_BLOCKED,
    COMPOSITION_STATUS_LEVEL2_FAILED,
    COMPOSITION_STATUS_NO_CHANGE,
    COMPOSITION_STATUS_PARTIAL,
    COMPOSITION_STATUS_RENDER_FAILED,
    COMPOSITION_STATUS_SUCCESS,
    COMPOSITION_STATUS_VERIFY_FAILED,
    CompositionInput,
    compute_composition_id,
    run_audio_finishing_composition,
)
from cutsell_worker.audio_finishing_executor import (
    ADJACENT_EXECUTION_STATUS_ADJUSTMENT_APPLIED,
    EXECUTION_STATUS_FFMPEG_FAILURE,
    EXECUTION_STATUS_SUCCESS,
    VERIFICATION_STATUS_TECHNICAL_FAILURE,
)
from cutsell_worker.audio_finishing_measurement import (
    CLIPPING_STATUS_NO_CLIPPING_DETECTED,
    MEASUREMENT_STATUS_COMPLETE,
    AudioFinishingMeasurement,
    measure_audio,
)
from cutsell_worker.audio_finishing_policy import (
    GAIN_STATE_ABSTAIN_INSUFFICIENT_EVIDENCE,
    GAIN_STATE_BLOCKED_SILENCE,
    PLAN_STATUS_BLOCKED,
    PLAN_STATUS_READY_FOR_CORRECTION,
    PLAN_STATUS_READY_NO_CHANGE,
    PLAN_STATUS_READY_WITH_LIMITER,
    generate_audio_finishing_plan,
)
from cutsell_worker.render_plan import RenderSegment

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available on this runner",
)


def _ffmpeg(args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def media_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("d253_composition")


def _tone_video(path, *, volume, duration=2.0, sample_rate=48000, channels=2, freq=800):
    _ffmpeg([
        "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={duration}:sample_rate={sample_rate},volume={volume}",
        "-f", "lavfi", "-i", f"testsrc2=size=320x240:rate=25:duration={duration}",
        "-c:v", "libx264", "-c:a", "aac", "-ac", str(channels), "-shortest", path,
    ])


def _measurement(**overrides):
    defaults = dict(
        media_path="/synthetic/fixture.wav", window_start_sec=None, window_end_sec=None,
        duration_sec=2.0, sample_rate_hz=48000, channel_count=2, channel_layout="stereo",
        integrated_loudness_lufs=-14.0, loudness_range_lu=1.0,
        true_peak_dbfs=-6.0, sample_peak_dbfs=-6.5,
        clipping_status=CLIPPING_STATUS_NO_CLIPPING_DETECTED, silence_result=None,
        measurement_status=MEASUREMENT_STATUS_COMPLETE, measurement_errors=(), provenance={},
    )
    defaults.update(overrides)
    return AudioFinishingMeasurement(**defaults)


def _seg(clip_id, source_path, *, start=0.0, end=2.0, audio_volume=1.0):
    return RenderSegment(clip_id=clip_id, source_asset_id=f"asset_{clip_id}", source_path=source_path, start=start, end=end, audio_volume=audio_volume)


@pytest.fixture(scope="module")
def clip_target_mp4(media_dir):
    path = str(media_dir / "clip_target.mp4")
    _tone_video(path, volume=1.0)  # ~-21 LUFS baseline sine, used consistently below
    return path


@pytest.fixture(scope="module")
def clip_quiet_mp4(media_dir):
    path = str(media_dir / "clip_quiet.mp4")
    _tone_video(path, volume=0.35, freq=1200)  # notably quieter, different tone
    return path


# ---------------------------------------------------------------------------
# STAGE 21 core positive paths: no-change, adjacent-only, whole-video-only,
# combined, limiter-authorized/not-authorized.
# ---------------------------------------------------------------------------

def test_no_change_composition(clip_target_mp4, media_dir):
    seg_a = _seg("a", clip_target_mp4)
    seg_b = _seg("b", clip_target_mp4)  # identical source -> identical level, no adjacent mismatch
    m = measure_audio(clip_target_mp4)
    plan = generate_audio_finishing_plan(m)  # whatever this real tone measures, used as-is (no adjacent pairs)
    result = run_audio_finishing_composition(
        CompositionInput(segments=(seg_a, seg_b), plan=plan, output_dir=str(media_dir / "no_change"), render_kwargs={"width": 320, "height": 240, "fps": 25}),
    )
    assert result.render_output_path is not None
    assert os.path.exists(result.render_output_path)
    assert result.composition_status in (COMPOSITION_STATUS_SUCCESS, COMPOSITION_STATUS_NO_CHANGE, COMPOSITION_STATUS_PARTIAL)
    # No Level-1 adjustment was ever authorized (no adjacent_pairs supplied).
    assert result.level1_results == ()


def test_adjacent_mismatch_only_level1_reduces_gap(clip_target_mp4, clip_quiet_mp4, media_dir):
    measurement_a = measure_audio(clip_target_mp4)
    measurement_b = measure_audio(clip_quiet_mp4)
    assert measurement_a.integrated_loudness_lufs - measurement_b.integrated_loudness_lufs > 2.0

    # Whole-video plan input: pretend the composite is already acceptable
    # (isolating this test to the Level-1 effect only).
    whole_video = _measurement(integrated_loudness_lufs=-14.0, true_peak_dbfs=-6.0, sample_peak_dbfs=-6.5)
    plan = generate_audio_finishing_plan(
        whole_video, adjacent_pairs=((measurement_a, measurement_b, "a", "b"),),
    )
    adjustment = plan.adjacent_take_adjustments[0]
    assert adjustment.direction == "RAISE_RIGHT"

    seg_a, seg_b = _seg("a", clip_target_mp4), _seg("b", clip_quiet_mp4)
    result = run_audio_finishing_composition(
        CompositionInput(segments=(seg_a, seg_b), plan=plan, output_dir=str(media_dir / "adjacent_only"), render_kwargs={"width": 320, "height": 240, "fps": 25}),
    )
    applied = [r for r in result.level1_results if r.execution_status == ADJACENT_EXECUTION_STATUS_ADJUSTMENT_APPLIED]
    assert len(applied) == 1
    assert applied[0].segment_id == "b"

    # Real before/after: measure segment B's window in the render before vs.
    # what it would have been unmodified -- proxy via re-measuring the real
    # rendered window and confirming it's louder than the raw source B alone.
    after_b = measure_audio(result.render_output_path, start_sec=2.2, end_sec=3.8)
    after_a = measure_audio(result.render_output_path, start_sec=0.2, end_sec=1.8)
    assert after_b.integrated_loudness_lufs > measurement_b.integrated_loudness_lufs + 1.0
    assert abs(after_a.integrated_loudness_lufs - measurement_a.integrated_loudness_lufs) < 0.5
    assert result.composition_status in (COMPOSITION_STATUS_SUCCESS, COMPOSITION_STATUS_PARTIAL)


def test_level2_only_positive_gain(clip_quiet_mp4, media_dir):
    seg_a, seg_b = _seg("a", clip_quiet_mp4), _seg("b", clip_quiet_mp4)
    real_measurement = measure_audio(clip_quiet_mp4)
    plan = generate_audio_finishing_plan(real_measurement)  # too quiet -> positive whole-video gain
    assert plan.authorized_whole_video_gain_db > 0

    before = measure_audio(clip_quiet_mp4)
    result = run_audio_finishing_composition(
        CompositionInput(segments=(seg_a, seg_b), plan=plan, output_dir=str(media_dir / "level2_pos"), render_kwargs={"width": 320, "height": 240, "fps": 25}),
    )
    assert result.whole_video_execution_record.execution_status == EXECUTION_STATUS_SUCCESS
    final_path = result.provenance["final_output_path"]
    after = measure_audio(final_path)
    assert after.integrated_loudness_lufs > before.integrated_loudness_lufs + 1.0


def test_level2_only_negative_gain(clip_target_mp4, media_dir):
    loud_path = str(media_dir / "loud_for_neg.mp4")
    _tone_video(loud_path, volume=6.0, freq=900)
    seg_a, seg_b = _seg("a", loud_path), _seg("b", loud_path)
    real_measurement = measure_audio(loud_path)
    plan = generate_audio_finishing_plan(real_measurement)
    assert plan.authorized_whole_video_gain_db < 0

    before = measure_audio(loud_path)
    result = run_audio_finishing_composition(
        CompositionInput(segments=(seg_a, seg_b), plan=plan, output_dir=str(media_dir / "level2_neg"), render_kwargs={"width": 320, "height": 240, "fps": 25}),
    )
    final_path = result.provenance["final_output_path"]
    after = measure_audio(final_path)
    assert after.integrated_loudness_lufs < before.integrated_loudness_lufs - 1.0


def test_level1_and_level2_composition_together(clip_target_mp4, clip_quiet_mp4, media_dir):
    measurement_a = measure_audio(clip_target_mp4)
    measurement_b = measure_audio(clip_quiet_mp4)
    whole_video = _measurement(integrated_loudness_lufs=-19.0, true_peak_dbfs=-10.0, sample_peak_dbfs=-10.5)  # too quiet overall
    plan = generate_audio_finishing_plan(
        whole_video, adjacent_pairs=((measurement_a, measurement_b, "a", "b"),),
    )
    assert plan.authorized_whole_video_gain_db > 0
    assert plan.adjacent_take_adjustments[0].authorized_correction_db is not None

    seg_a, seg_b = _seg("a", clip_target_mp4), _seg("b", clip_quiet_mp4)
    result = run_audio_finishing_composition(
        CompositionInput(segments=(seg_a, seg_b), plan=plan, output_dir=str(media_dir / "both_levels"), render_kwargs={"width": 320, "height": 240, "fps": 25}),
    )
    applied_level1 = [r for r in result.level1_results if r.execution_status == ADJACENT_EXECUTION_STATUS_ADJUSTMENT_APPLIED]
    assert len(applied_level1) == 1
    assert result.whole_video_execution_record.execution_status == EXECUTION_STATUS_SUCCESS
    assert result.verification is not None


def test_limiter_authorized_composition(clip_target_mp4, media_dir):
    m = _measurement(integrated_loudness_lufs=-17.5, true_peak_dbfs=-2.5, sample_peak_dbfs=-3.0)
    plan = generate_audio_finishing_plan(m)
    assert plan.limiter_authorized is True
    assert plan.plan_status == PLAN_STATUS_READY_WITH_LIMITER

    seg_a, seg_b = _seg("a", clip_target_mp4), _seg("b", clip_target_mp4)
    result = run_audio_finishing_composition(
        CompositionInput(segments=(seg_a, seg_b), plan=plan, output_dir=str(media_dir / "limiter_yes"), render_kwargs={"width": 320, "height": 240, "fps": 25}),
    )
    assert any(f.startswith("alimiter=") for f in result.whole_video_execution_record.filters_applied)
    volume_idx = next(i for i, f in enumerate(result.whole_video_execution_record.filters_applied) if f.startswith("volume="))
    limiter_idx = next(i for i, f in enumerate(result.whole_video_execution_record.filters_applied) if f.startswith("alimiter="))
    assert volume_idx < limiter_idx


def test_limiter_not_authorized_composition(clip_target_mp4, media_dir):
    seg_a, seg_b = _seg("a", clip_target_mp4), _seg("b", clip_target_mp4)
    m = measure_audio(clip_target_mp4)
    plan = generate_audio_finishing_plan(m)
    result = run_audio_finishing_composition(
        CompositionInput(segments=(seg_a, seg_b), plan=plan, output_dir=str(media_dir / "limiter_no"), render_kwargs={"width": 320, "height": 240, "fps": 25}),
    )
    if result.whole_video_execution_record.execution_status == EXECUTION_STATUS_SUCCESS:
        assert not any(f.startswith("alimiter=") for f in result.whole_video_execution_record.filters_applied)


# ---------------------------------------------------------------------------
# Blocked / abstain / short-window / three-segment / conflict / pre-existing gain.
# ---------------------------------------------------------------------------

def test_blocked_level1_no_mutation_but_composition_proceeds(clip_target_mp4, media_dir):
    # D-249's `evaluate_adjacent_take_continuity` has no dedicated silence
    # firewall of its own (that check exists only on the whole-video path,
    # `evaluate_whole_video_loudness` -- confirmed empirically this gate:
    # feeding a real -inf loudness into an adjacent PAIR does not resolve
    # to BLOCKED_SILENCE, it resolves the (infinite) delta straight into
    # CORRECTION_LIMITED at the +6dB envelope instead. This is an honest,
    # out-of-scope finding for a future policy gate -- D-253 fixes no
    # policy, so this test instead uses the adjacent-stage block that DOES
    # exist and is already proven, D-252's own BLOCKED_PEAK_RISK).
    louder_path = str(media_dir / "louder_for_block.mp4")
    _tone_video(louder_path, volume=8.9, freq=1500)
    # "a" (clip_target_mp4, the quieter side) is the one that would be
    # RAISED toward "b" (louder) -- so the peak-at-ceiling override must be
    # on "a" (the side D-249's evaluate_peak_safety actually checks).
    measurement_a = dataclasses.replace(measure_audio(clip_target_mp4), true_peak_dbfs=-0.5, sample_peak_dbfs=-0.5)
    measurement_b = measure_audio(louder_path)
    whole_video = measurement_a
    plan = generate_audio_finishing_plan(whole_video, adjacent_pairs=((measurement_a, measurement_b, "a", "b"),))
    from cutsell_worker.audio_finishing_policy import GAIN_STATE_BLOCKED_PEAK_RISK
    assert plan.adjacent_take_adjustments[0].direction == "RAISE_LEFT"
    assert plan.adjacent_take_adjustments[0].gain_state == GAIN_STATE_BLOCKED_PEAK_RISK

    seg_a, seg_b = _seg("a", clip_target_mp4), _seg("b", louder_path)
    result = run_audio_finishing_composition(
        CompositionInput(segments=(seg_a, seg_b), plan=plan, output_dir=str(media_dir / "blocked_l1"), render_kwargs={"width": 320, "height": 240, "fps": 25}),
    )
    assert all(r.execution_status != ADJACENT_EXECUTION_STATUS_ADJUSTMENT_APPLIED for r in result.level1_results)
    assert result.render_output_path is not None  # composition still proceeds


def test_silence_never_boosted_at_whole_video_level(media_dir):
    # The real, proven silence firewall (GAIN_STATE_BLOCKED_SILENCE /
    # PLAN_STATUS_BLOCKED) lives on the whole-video path -- see
    # test_blocked_level2_plan_does_not_mutate_final_media for the
    # composition-level proof that a real silent source's plan never
    # authorizes a whole-video gain.
    silent_path = str(media_dir / "silent_never_boosted.mp4")
    _ffmpeg(["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo:d=2", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=2", "-c:v", "libx264", "-c:a", "aac", "-shortest", silent_path])
    m = measure_audio(silent_path)
    # Real AAC-encoded digital silence measures as a very low but finite
    # floor (e.g. -70 LUFS) rather than exact -inf on this ffmpeg build --
    # the reused `check_accidental_silence` (silencedetect) result, not a
    # literal -inf comparison, is what actually classifies it as silence.
    assert m.integrated_loudness_lufs < -60.0
    plan = generate_audio_finishing_plan(m)
    assert plan.plan_status == PLAN_STATUS_BLOCKED
    assert plan.authorized_whole_video_gain_db in (None, 0.0)


def test_blocked_level2_plan_does_not_mutate_final_media(clip_target_mp4, media_dir):
    silent_path = str(media_dir / "silent_whole.mp4")
    _ffmpeg(["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo:d=2", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=2", "-c:v", "libx264", "-c:a", "aac", "-shortest", silent_path])
    m = measure_audio(silent_path)
    plan = generate_audio_finishing_plan(m)
    assert plan.plan_status == PLAN_STATUS_BLOCKED

    seg_a, seg_b = _seg("a", silent_path), _seg("b", silent_path)
    result = run_audio_finishing_composition(
        CompositionInput(segments=(seg_a, seg_b), plan=plan, output_dir=str(media_dir / "blocked_l2"), render_kwargs={"width": 320, "height": 240, "fps": 25}),
    )
    # no separate whole-video output file since Level 2 refused to execute
    assert result.whole_video_execution_record.output_path is None
    assert result.provenance["final_output_path"] == result.render_output_path


def test_short_window_abstain_safe(clip_target_mp4, media_dir):
    measurement_short = measure_audio(clip_target_mp4, start_sec=0.0, end_sec=0.3)
    measurement_normal = measure_audio(clip_target_mp4)
    whole_video = measurement_normal
    plan = generate_audio_finishing_plan(whole_video, adjacent_pairs=((measurement_normal, measurement_short, "a", "b"),))
    assert plan.adjacent_take_adjustments[0].gain_state == GAIN_STATE_ABSTAIN_INSUFFICIENT_EVIDENCE
    seg_a, seg_b = _seg("a", clip_target_mp4), _seg("b", clip_target_mp4)
    result = run_audio_finishing_composition(
        CompositionInput(segments=(seg_a, seg_b), plan=plan, output_dir=str(media_dir / "short_abstain"), render_kwargs={"width": 320, "height": 240, "fps": 25}),
    )
    assert result.render_output_path is not None


def test_three_segment_chain_and_dual_adjacency_conflict(clip_target_mp4, clip_quiet_mp4, media_dir):
    measurement_a = measure_audio(clip_target_mp4)
    measurement_b = measure_audio(clip_quiet_mp4)
    measurement_c = measure_audio(clip_target_mp4)
    whole_video = measurement_a
    # Both pairs authorize raising "b" -> a genuine dual-adjacency conflict.
    plan = generate_audio_finishing_plan(
        whole_video,
        adjacent_pairs=(
            (measurement_a, measurement_b, "a", "b"),
            (measurement_b, measurement_c, "b", "c"),
        ),
    )
    seg_a, seg_b, seg_c = _seg("a", clip_target_mp4), _seg("b", clip_quiet_mp4), _seg("c", clip_target_mp4)
    result = run_audio_finishing_composition(
        CompositionInput(segments=(seg_a, seg_b, seg_c), plan=plan, output_dir=str(media_dir / "three_seg"), render_kwargs={"width": 320, "height": 240, "fps": 25}),
    )
    # neither conflicting adjustment applied to "b"
    assert not any(r.segment_id == "b" and r.execution_status == ADJACENT_EXECUTION_STATUS_ADJUSTMENT_APPLIED for r in result.level1_results)
    assert result.render_output_path is not None


def test_pre_existing_audio_volume_composes_through_composition(clip_quiet_mp4, clip_target_mp4, media_dir):
    measurement_a = measure_audio(clip_target_mp4)
    measurement_b = measure_audio(clip_quiet_mp4)
    whole_video = measurement_a
    plan = generate_audio_finishing_plan(whole_video, adjacent_pairs=((measurement_a, measurement_b, "a", "b"),))
    seg_a = _seg("a", clip_target_mp4)
    seg_b = _seg("b", clip_quiet_mp4, audio_volume=0.9)  # pre-existing manual gain
    result = run_audio_finishing_composition(
        CompositionInput(segments=(seg_a, seg_b), plan=plan, output_dir=str(media_dir / "pre_existing"), render_kwargs={"width": 320, "height": 240, "fps": 25}),
    )
    applied = [r for r in result.level1_results if r.execution_status == ADJACENT_EXECUTION_STATUS_ADJUSTMENT_APPLIED]
    if applied:
        assert applied[0].prior_audio_volume == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Mono / stereo / non-48k / video+audio / audio-only / malformed sources.
# ---------------------------------------------------------------------------

def test_mono_and_stereo_sources_final_format_standardized(media_dir):
    mono_path = str(media_dir / "mono_src.mp4")
    _ffmpeg(["-f", "lavfi", "-i", "sine=frequency=700:duration=2:sample_rate=48000,volume=0.5", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=2", "-c:v", "libx264", "-c:a", "aac", "-ac", "1", "-shortest", mono_path])
    stereo_path = str(media_dir / "stereo_src.mp4")
    _ffmpeg(["-f", "lavfi", "-i", "sine=frequency=700:duration=2:sample_rate=48000,volume=0.5", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=2", "-c:v", "libx264", "-c:a", "aac", "-ac", "2", "-shortest", stereo_path])
    m = measure_audio(mono_path)
    plan = generate_audio_finishing_plan(m)
    seg_a, seg_b = _seg("a", mono_path), _seg("b", stereo_path)
    result = run_audio_finishing_composition(
        CompositionInput(segments=(seg_a, seg_b), plan=plan, output_dir=str(media_dir / "mono_stereo"), render_kwargs={"width": 320, "height": 240, "fps": 25}),
    )
    final_path = result.provenance["final_output_path"]
    final_measurement = measure_audio(final_path)
    assert final_measurement.sample_rate_hz == 48000
    assert final_measurement.channel_count == 2


def test_non_48k_source_final_still_48k(media_dir):
    src_path = str(media_dir / "src_44100.mp4")
    _ffmpeg(["-f", "lavfi", "-i", "sine=frequency=700:duration=2:sample_rate=44100,volume=0.5", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=2", "-c:v", "libx264", "-c:a", "aac", "-shortest", src_path])
    m = measure_audio(src_path)
    plan = generate_audio_finishing_plan(m)
    seg_a, seg_b = _seg("a", src_path), _seg("b", src_path)
    result = run_audio_finishing_composition(
        CompositionInput(segments=(seg_a, seg_b), plan=plan, output_dir=str(media_dir / "non48k"), render_kwargs={"width": 320, "height": 240, "fps": 25}),
    )
    final_measurement = measure_audio(result.provenance["final_output_path"])
    assert final_measurement.sample_rate_hz == 48000


def test_video_and_audio_source_video_stream_preserved(clip_target_mp4, media_dir):
    m = measure_audio(clip_target_mp4)
    plan = generate_audio_finishing_plan(m)
    seg_a, seg_b = _seg("a", clip_target_mp4), _seg("b", clip_target_mp4)
    result = run_audio_finishing_composition(
        CompositionInput(segments=(seg_a, seg_b), plan=plan, output_dir=str(media_dir / "video_preserve"), render_kwargs={"width": 320, "height": 240, "fps": 25}),
    )
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_type", "-of", "csv=p=0", result.provenance["final_output_path"]],
        capture_output=True, text=True, check=True,
    )
    assert "video" in probe.stdout


def test_malformed_source_render_fails_bounded(media_dir):
    m = _measurement()
    plan = generate_audio_finishing_plan(m)
    seg_a = _seg("a", "/nonexistent/does_not_exist.mp4")
    seg_b = _seg("b", "/nonexistent/also_missing.mp4")
    result = run_audio_finishing_composition(
        CompositionInput(segments=(seg_a, seg_b), plan=plan, output_dir=str(media_dir / "malformed"), render_kwargs={"width": 320, "height": 240, "fps": 25}),
    )
    assert result.composition_status == COMPOSITION_STATUS_RENDER_FAILED
    assert result.whole_video_execution_record is None
    assert result.verification is None


def test_empty_segments_level1_blocked(media_dir):
    m = _measurement()
    plan = generate_audio_finishing_plan(m)
    result = run_audio_finishing_composition(
        CompositionInput(segments=(), plan=plan, output_dir=str(media_dir / "empty")),
    )
    assert result.composition_status == COMPOSITION_STATUS_LEVEL1_BLOCKED
    assert result.render_output_path is None


# ---------------------------------------------------------------------------
# Failure injection: render failure, Level-2 failure, verification failure.
# ---------------------------------------------------------------------------

def test_render_failure_injection_stops_before_level2(clip_target_mp4, media_dir, monkeypatch):
    import cutsell_worker.audio_finishing_composition as composition_module

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated render failure")

    monkeypatch.setattr(composition_module.render_module, "render_preview", _boom)

    m = measure_audio(clip_target_mp4)
    plan = generate_audio_finishing_plan(m)
    seg_a, seg_b = _seg("a", clip_target_mp4), _seg("b", clip_target_mp4)
    result = run_audio_finishing_composition(
        CompositionInput(segments=(seg_a, seg_b), plan=plan, output_dir=str(media_dir / "render_fail_inject")),
    )
    assert result.composition_status == COMPOSITION_STATUS_RENDER_FAILED
    assert result.whole_video_execution_record is None
    assert result.verification is None


def test_level2_failure_injection_never_claims_verify_success(clip_target_mp4, media_dir, monkeypatch):
    import cutsell_worker.audio_finishing_composition as composition_module
    from cutsell_worker.audio_finishing_executor import AudioFinishingExecutionRecord

    def _fake_execute(plan, input_path, output_path, *, existing_record=None):
        record = AudioFinishingExecutionRecord(
            execution_id="fake", policy_version=plan.policy_version, plan_status=plan.plan_status,
            input_path=input_path, output_path=None,
            requested_whole_video_gain_db=plan.requested_whole_video_gain_db,
            authorized_whole_video_gain_db=plan.authorized_whole_video_gain_db,
            limiter_authorized=False, true_peak_ceiling_dbtp=plan.true_peak_ceiling_dbtp,
            peak_evidence_source=plan.peak_evidence_source, filters_applied=(),
            execution_status=EXECUTION_STATUS_FFMPEG_FAILURE, ffmpeg_return_code=1,
            errors=("simulated ffmpeg failure",), provenance={},
        )
        return record, None

    monkeypatch.setattr(composition_module, "execute_audio_finishing_plan", _fake_execute)

    m = measure_audio(clip_target_mp4)
    plan = generate_audio_finishing_plan(m)
    seg_a, seg_b = _seg("a", clip_target_mp4), _seg("b", clip_target_mp4)
    result = run_audio_finishing_composition(
        CompositionInput(segments=(seg_a, seg_b), plan=plan, output_dir=str(media_dir / "level2_fail_inject"), render_kwargs={"width": 320, "height": 240, "fps": 25}),
    )
    assert result.composition_status == COMPOSITION_STATUS_LEVEL2_FAILED
    assert result.verification is None  # never claims verification success


def test_post_measurement_failure_injection_reports_verify_failed(clip_target_mp4, media_dir, monkeypatch):
    import cutsell_worker.audio_finishing_composition as composition_module
    from cutsell_worker.audio_finishing_executor import ExecutionVerificationResult

    def _fake_verify(plan, record):
        return ExecutionVerificationResult(
            measurement_status="MEASUREMENT_ERROR", integrated_loudness_lufs=None, true_peak_dbfs=None,
            sample_peak_dbfs=None, peak_evidence_source="UNAVAILABLE", audio_present=False,
            duration_preserved=None, duration_delta_sec=None, sample_rate_expected=False,
            channel_count_expected=False, loudness_in_target_range=None, true_peak_within_ceiling=None,
            verification_status=VERIFICATION_STATUS_TECHNICAL_FAILURE, errors=("simulated measurement failure",), provenance={},
        )

    monkeypatch.setattr(composition_module, "_verify_execution", _fake_verify)

    # Force a plan that resolves to NO_ACTION_NEEDED for the whole-video
    # stage (real clip_target_mp4 is not necessarily near -14 LUFS) so the
    # composition is guaranteed to go through `_verify_final_output`
    # (and therefore the patched `_verify_execution`) rather than a real
    # Level-2 execution's own internal verification call.
    m = dataclasses.replace(measure_audio(clip_target_mp4), integrated_loudness_lufs=-14.0)
    plan = generate_audio_finishing_plan(m)
    assert plan.plan_status == PLAN_STATUS_READY_NO_CHANGE
    seg_a, seg_b = _seg("a", clip_target_mp4), _seg("b", clip_target_mp4)
    result = run_audio_finishing_composition(
        CompositionInput(segments=(seg_a, seg_b), plan=plan, output_dir=str(media_dir / "verify_fail_inject"), render_kwargs={"width": 320, "height": 240, "fps": 25}),
    )
    assert result.composition_status == COMPOSITION_STATUS_VERIFY_FAILED


# ---------------------------------------------------------------------------
# Idempotence: same composition twice, different plan gives distinct id.
# ---------------------------------------------------------------------------

def test_same_composition_replay_is_safe(clip_target_mp4, media_dir):
    m = measure_audio(clip_target_mp4)
    plan = generate_audio_finishing_plan(m)
    seg_a, seg_b = _seg("a", clip_target_mp4), _seg("b", clip_target_mp4)
    ci = CompositionInput(segments=(seg_a, seg_b), plan=plan, output_dir=str(media_dir / "replay"), render_kwargs={"width": 320, "height": 240, "fps": 25})

    first = run_audio_finishing_composition(ci)
    level1_ids = frozenset(
        r.adjustment_application_id for r in first.level1_results if r.adjustment_application_id
    )
    second = run_audio_finishing_composition(
        ci, already_applied_level1_ids=level1_ids, existing_level2_record=first.whole_video_execution_record,
    )
    assert first.composition_id == second.composition_id
    if first.whole_video_execution_record.execution_status == EXECUTION_STATUS_SUCCESS:
        assert second.whole_video_execution_record is first.whole_video_execution_record


def test_different_plan_gives_distinct_composition_id(clip_target_mp4, clip_quiet_mp4):
    m1 = measure_audio(clip_target_mp4)
    m2 = measure_audio(clip_quiet_mp4)
    plan1 = generate_audio_finishing_plan(m1)
    plan2 = generate_audio_finishing_plan(m2)
    seg_a, seg_b = _seg("a", clip_target_mp4), _seg("b", clip_target_mp4)
    id1 = compute_composition_id(plan1, (seg_a, seg_b))
    id2 = compute_composition_id(plan2, (seg_a, seg_b))
    assert id1 != id2


# ---------------------------------------------------------------------------
# Structural firewalls: no policy recomputation, no live integration, no DSP.
# ---------------------------------------------------------------------------

def test_no_new_dsp_or_policy_recomputation_structural():
    import inspect

    import cutsell_worker.audio_finishing_composition as composition_module

    source = inspect.getsource(composition_module)
    for forbidden in ("loudnorm", "acompressor", "afftdn", "highpass", "lowpass", "evaluate_whole_video_loudness", "evaluate_adjacent_take_continuity"):
        assert forbidden not in source


def test_composition_module_never_imported_by_production_entry_points():
    import inspect

    import cutsell_worker.render as render_module
    import cutsell_worker.pipeline as pipeline_module

    assert "audio_finishing_composition" not in inspect.getsource(render_module)
    assert "audio_finishing_composition" not in inspect.getsource(pipeline_module)


def test_composition_record_is_frozen_and_has_full_provenance_schema(clip_target_mp4, media_dir):
    m = measure_audio(clip_target_mp4)
    plan = generate_audio_finishing_plan(m)
    seg_a, seg_b = _seg("a", clip_target_mp4), _seg("b", clip_target_mp4)
    result = run_audio_finishing_composition(
        CompositionInput(segments=(seg_a, seg_b), plan=plan, output_dir=str(media_dir / "schema_check"), render_kwargs={"width": 320, "height": 240, "fps": 25}),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.composition_status = "TAMPERED"
    for field_name in ("composition_id", "input_segment_ids", "policy_version", "plan_status", "level1_results", "render_output_path", "whole_video_execution_record", "post_measurement", "verification", "composition_status", "errors", "provenance"):
        assert hasattr(result, field_name)
