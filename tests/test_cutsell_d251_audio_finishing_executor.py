"""Audio Finishing WHOLE-VIDEO EXECUTOR FOUNDATION (D-251).

Real ffmpeg fixtures for the core end-to-end gain/limiter/verify path
(matching D-247's own synthetic-media convention); a few edge-case plans
that cannot be physically produced by a working ffmpeg pass (e.g. "true
peak entirely unavailable") are directly-constructed `AudioFinishingPlan`/
`AudioFinishingMeasurement` objects executed against a real (but
measurement-unrelated) media file -- the executor's own design principle
is "consume the plan's already-decided numbers, never recompute them
from the file," so this is a legitimate way to exercise those specific
plan states without needing a broken ffmpeg build.

Skipped automatically (not failed) if `ffmpeg`/`ffprobe` are not on PATH.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from cutsell_worker.audio_finishing_executor import (
    EXECUTION_STATUS_FFMPEG_FAILURE,
    EXECUTION_STATUS_INVALID_GAIN,
    EXECUTION_STATUS_MEASUREMENT_REFERENCE_MISSING,
    EXECUTION_STATUS_NO_ACTION_NEEDED,
    EXECUTION_STATUS_PEAK_SAFETY_UNVERIFIED,
    EXECUTION_STATUS_PLAN_NOT_EXECUTABLE,
    EXECUTION_STATUS_SUCCESS,
    VERIFICATION_STATUS_PASS,
    VERIFICATION_STATUS_POLICY_OUT_OF_RANGE,
    AudioFinishingExecutionRecord,
    compute_execution_id,
    execute_audio_finishing_plan,
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
    PEAK_EVIDENCE_FALLBACK_SAMPLE_PEAK,
    PEAK_EVIDENCE_UNAVAILABLE,
    PLAN_STATUS_ABSTAIN,
    PLAN_STATUS_BLOCKED,
    PLAN_STATUS_READY_FOR_CORRECTION,
    PLAN_STATUS_READY_NO_CHANGE,
    PLAN_STATUS_READY_WITH_LIMITER,
    generate_audio_finishing_plan,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available on this runner",
)


def _ffmpeg(args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def media_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("audio_finishing_executor")


def _tone(path: str, *, volume: float, duration: float = 3.0, sample_rate: int = 48000, channels: int = 2):
    _ffmpeg([
        "-f", "lavfi",
        "-i", f"sine=frequency=1000:duration={duration}:sample_rate={sample_rate},volume={volume}",
        "-ac", str(channels), path,
    ])


@pytest.fixture(scope="module")
def already_acceptable_wav(media_dir):
    # Empirically ~-14 LUFS at this sine amplitude*volume combination
    # (D-247/D-249 both confirmed volume=1 sine ~ -21 LUFS; volume≈2.24 ~ -14).
    path = str(media_dir / "acceptable.wav")
    _tone(path, volume=2.24)
    return path


@pytest.fixture(scope="module")
def too_quiet_wav(media_dir):
    path = str(media_dir / "too_quiet.wav")
    # ~-18 LUFS -- outside the [-15,-13] band but well within the 6dB
    # automatic envelope (requested ~4dB), so this must resolve to
    # CORRECTION_ALLOWED, not CORRECTION_LIMITED.
    _tone(path, volume=1.4)
    return path


@pytest.fixture(scope="module")
def too_loud_wav(media_dir):
    path = str(media_dir / "too_loud.wav")
    _tone(path, volume=6.0)  # well above -14 LUFS, still under clipping
    return path


@pytest.fixture(scope="module")
def very_quiet_wav(media_dir):
    path = str(media_dir / "very_quiet.wav")
    # ~-28 LUFS -- needs >6dB correction (-> CORRECTION_LIMITED), but RMS
    # stays well above silencedetect's -35dB noise floor so this must
    # NOT be classified as silence.
    _tone(path, volume=0.45)
    return path


@pytest.fixture(scope="module")
def near_ceiling_wav(media_dir):
    # loud + close to the ceiling so a positive correction (from a paired
    # too-quiet measurement swapped in) would cross -1.0 dBTP
    path = str(media_dir / "near_ceiling.wav")
    _tone(path, volume=8.9)
    return path


@pytest.fixture(scope="module")
def silence_wav(media_dir):
    path = str(media_dir / "silence.wav")
    _ffmpeg(["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo:d=3", path])
    return path


@pytest.fixture(scope="module")
def mono_wav(media_dir):
    path = str(media_dir / "mono.wav")
    _tone(path, volume=0.5, channels=1)
    return path


@pytest.fixture(scope="module")
def non48k_wav(media_dir):
    path = str(media_dir / "tone_44100.wav")
    _tone(path, volume=0.5, sample_rate=44100)
    return path


@pytest.fixture(scope="module")
def video_and_audio_mp4(media_dir):
    path = str(media_dir / "clip.mp4")
    _ffmpeg([
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=2:sample_rate=48000,volume=0.5",
        "-f", "lavfi", "-i", "testsrc=size=64x64:rate=25:duration=2",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", path,
    ])
    return path


def _plan_for(path, tmp_path_factory=None, **measure_kwargs):
    measurement = measure_audio(path, **measure_kwargs)
    return generate_audio_finishing_plan(measurement), measurement


# ---------------------------------------------------------------------------
# Core end-to-end: real gain application, real re-measurement.
# ---------------------------------------------------------------------------

def test_too_quiet_real_gain_increases_measured_loudness(too_quiet_wav, media_dir):
    plan, before = _plan_for(too_quiet_wav)
    assert plan.plan_status in (PLAN_STATUS_READY_FOR_CORRECTION, PLAN_STATUS_READY_WITH_LIMITER)
    assert plan.authorized_whole_video_gain_db > 0

    output = str(media_dir / "too_quiet_out.wav")
    record, verification = execute_audio_finishing_plan(plan, too_quiet_wav, output)

    assert record.execution_status == EXECUTION_STATUS_SUCCESS
    assert any(f.startswith("volume=") for f in record.filters_applied)
    assert verification is not None
    assert verification.integrated_loudness_lufs > before.integrated_loudness_lufs


def test_too_loud_real_gain_decreases_measured_loudness(too_loud_wav, media_dir):
    plan, before = _plan_for(too_loud_wav)
    assert plan.authorized_whole_video_gain_db < 0

    output = str(media_dir / "too_loud_out.wav")
    record, verification = execute_audio_finishing_plan(plan, too_loud_wav, output)

    assert record.execution_status == EXECUTION_STATUS_SUCCESS
    assert verification.integrated_loudness_lufs < before.integrated_loudness_lufs


def test_exact_authorized_gain_applied_no_recomputation(too_quiet_wav, media_dir):
    plan, _ = _plan_for(too_quiet_wav)
    output = str(media_dir / "exact_gain_out.wav")
    record, _ = execute_audio_finishing_plan(plan, too_quiet_wav, output)
    expected_filter = f"volume={plan.authorized_whole_video_gain_db:.6f}dB"
    assert expected_filter in record.filters_applied
    assert record.authorized_whole_video_gain_db == plan.authorized_whole_video_gain_db
    assert record.requested_whole_video_gain_db == plan.requested_whole_video_gain_db


def test_over_envelope_correction_limited_still_executes(very_quiet_wav, media_dir):
    plan, _ = _plan_for(very_quiet_wav)
    assert abs(plan.authorized_whole_video_gain_db) == pytest.approx(6.0)
    assert abs(plan.requested_whole_video_gain_db) > 6.0

    output = str(media_dir / "very_quiet_out.wav")
    record, verification = execute_audio_finishing_plan(plan, very_quiet_wav, output)
    assert record.execution_status == EXECUTION_STATUS_SUCCESS
    assert record.authorized_whole_video_gain_db == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# Plan-status gating: no-change / blocked / abstain never execute (or
# execute nothing) per D-251's explicit scope.
# ---------------------------------------------------------------------------

def test_no_change_plan_returns_no_action_needed_without_transcode(already_acceptable_wav, media_dir):
    plan, _ = _plan_for(already_acceptable_wav)
    assert plan.plan_status == PLAN_STATUS_READY_NO_CHANGE

    output = str(media_dir / "no_change_out.wav")
    record, verification = execute_audio_finishing_plan(plan, already_acceptable_wav, output)

    assert record.execution_status == EXECUTION_STATUS_NO_ACTION_NEEDED
    assert record.output_path is None
    assert not os.path.exists(output)
    assert verification is None


def test_blocked_plan_never_executes(silence_wav, media_dir):
    # a silence measurement resolves to BLOCKED_SILENCE -> top-level BLOCKED
    plan, _ = _plan_for(silence_wav)
    assert plan.plan_status == PLAN_STATUS_BLOCKED

    output = str(media_dir / "blocked_out.wav")
    record, verification = execute_audio_finishing_plan(plan, silence_wav, output)

    assert record.execution_status == EXECUTION_STATUS_PLAN_NOT_EXECUTABLE
    assert record.output_path is None
    assert not os.path.exists(output)
    assert verification is None


def test_abstain_plan_never_executes(media_dir, too_quiet_wav):
    # Directly construct a measurement whose duration is below the
    # minimum reliable window -- a real, deterministic ABSTAIN trigger.
    measurement = AudioFinishingMeasurement(
        media_path=too_quiet_wav, window_start_sec=None, window_end_sec=None,
        duration_sec=0.5, sample_rate_hz=48000, channel_count=2, channel_layout="stereo",
        integrated_loudness_lufs=-20.0, loudness_range_lu=1.0,
        true_peak_dbfs=-6.0, sample_peak_dbfs=-6.5,
        clipping_status=CLIPPING_STATUS_NO_CLIPPING_DETECTED, silence_result=None,
        measurement_status=MEASUREMENT_STATUS_COMPLETE, measurement_errors=(), provenance={},
    )
    plan = generate_audio_finishing_plan(measurement)
    assert plan.plan_status == PLAN_STATUS_ABSTAIN
    assert plan.whole_video_state == GAIN_STATE_ABSTAIN_INSUFFICIENT_EVIDENCE

    output = str(media_dir / "abstain_out.wav")
    record, verification = execute_audio_finishing_plan(plan, too_quiet_wav, output)
    assert record.execution_status == EXECUTION_STATUS_PLAN_NOT_EXECUTABLE
    assert not os.path.exists(output)
    assert verification is None


def test_unknown_plan_status_never_executes(too_quiet_wav, media_dir):
    measurement = AudioFinishingMeasurement(
        media_path=too_quiet_wav, window_start_sec=None, window_end_sec=None,
        duration_sec=3.0, sample_rate_hz=48000, channel_count=2, channel_layout="stereo",
        integrated_loudness_lufs=-14.0, loudness_range_lu=1.0,
        true_peak_dbfs=-6.0, sample_peak_dbfs=-6.5,
        clipping_status=CLIPPING_STATUS_NO_CLIPPING_DETECTED, silence_result=None,
        measurement_status=MEASUREMENT_STATUS_COMPLETE, measurement_errors=(), provenance={},
    )
    real_plan = generate_audio_finishing_plan(measurement)
    import dataclasses
    tampered_plan = dataclasses.replace(real_plan, plan_status="UNKNOWN")

    output = str(media_dir / "unknown_out.wav")
    record, verification = execute_audio_finishing_plan(tampered_plan, too_quiet_wav, output)
    assert record.execution_status == EXECUTION_STATUS_PLAN_NOT_EXECUTABLE
    assert not os.path.exists(output)
    assert verification is None


# ---------------------------------------------------------------------------
# Limiter: only when authorized, no compressor, no loudnorm, after volume.
# ---------------------------------------------------------------------------

def test_limiter_appears_only_when_authorized_and_after_volume(media_dir, too_quiet_wav):
    measurement = AudioFinishingMeasurement(
        media_path=too_quiet_wav, window_start_sec=None, window_end_sec=None,
        duration_sec=3.0, sample_rate_hz=48000, channel_count=2, channel_layout="stereo",
        integrated_loudness_lufs=-17.5, loudness_range_lu=1.0,
        true_peak_dbfs=-2.5, sample_peak_dbfs=-3.0,
        clipping_status=CLIPPING_STATUS_NO_CLIPPING_DETECTED, silence_result=None,
        measurement_status=MEASUREMENT_STATUS_COMPLETE, measurement_errors=(), provenance={},
    )
    plan = generate_audio_finishing_plan(measurement)
    assert plan.limiter_authorized is True
    assert plan.plan_status == PLAN_STATUS_READY_WITH_LIMITER

    output = str(media_dir / "limiter_out.wav")
    record, verification = execute_audio_finishing_plan(plan, too_quiet_wav, output)

    assert record.execution_status == EXECUTION_STATUS_SUCCESS
    volume_idx = next(i for i, f in enumerate(record.filters_applied) if f.startswith("volume="))
    limiter_idx = next(i for i, f in enumerate(record.filters_applied) if f.startswith("alimiter="))
    assert volume_idx < limiter_idx


def test_limiter_absent_when_not_authorized(too_quiet_wav, media_dir):
    plan, _ = _plan_for(too_quiet_wav)  # ample headroom, no limiter expected
    assert plan.limiter_authorized is False

    output = str(media_dir / "no_limiter_out.wav")
    record, _ = execute_audio_finishing_plan(plan, too_quiet_wav, output)
    assert not any(f.startswith("alimiter=") for f in record.filters_applied)


def test_no_compressor_or_loudnorm_ever_in_filter_chain(too_quiet_wav, very_quiet_wav, media_dir):
    for path, name in ((too_quiet_wav, "a"), (very_quiet_wav, "b")):
        plan, _ = _plan_for(path)
        output = str(media_dir / f"no_dsp_check_{name}.wav")
        record, _ = execute_audio_finishing_plan(plan, path, output)
        joined = ",".join(record.filters_applied)
        assert "acompressor" not in joined
        assert "loudnorm" not in joined
        assert "afftdn" not in joined
        assert "highpass" not in joined
        assert "lowpass" not in joined


# ---------------------------------------------------------------------------
# Peak / sample-peak fallback / true-peak-unavailable defense in depth.
# ---------------------------------------------------------------------------

def test_true_peak_unavailable_with_positive_gain_refuses_to_execute(too_quiet_wav, media_dir):
    measurement = AudioFinishingMeasurement(
        media_path=too_quiet_wav, window_start_sec=None, window_end_sec=None,
        duration_sec=3.0, sample_rate_hz=48000, channel_count=2, channel_layout="stereo",
        integrated_loudness_lufs=-20.0, loudness_range_lu=1.0,
        true_peak_dbfs=None, sample_peak_dbfs=None,
        clipping_status=CLIPPING_STATUS_NO_CLIPPING_DETECTED, silence_result=None,
        measurement_status=MEASUREMENT_STATUS_COMPLETE, measurement_errors=(), provenance={},
    )
    plan = generate_audio_finishing_plan(measurement)
    # policy layer already refuses this (BLOCKED_PEAK_RISK/ABSTAIN); confirm
    # the executor's own defense-in-depth also never executes regardless.
    output = str(media_dir / "peak_unavailable_out.wav")
    record, verification = execute_audio_finishing_plan(plan, too_quiet_wav, output)
    assert record.execution_status in (EXECUTION_STATUS_PLAN_NOT_EXECUTABLE, EXECUTION_STATUS_PEAK_SAFETY_UNVERIFIED)
    assert not os.path.exists(output)
    assert verification is None


def test_sample_peak_fallback_honestly_tagged(too_quiet_wav, media_dir):
    measurement = AudioFinishingMeasurement(
        media_path=too_quiet_wav, window_start_sec=None, window_end_sec=None,
        duration_sec=3.0, sample_rate_hz=48000, channel_count=2, channel_layout="stereo",
        integrated_loudness_lufs=-18.0, loudness_range_lu=1.0,
        true_peak_dbfs=None, sample_peak_dbfs=-10.0,
        clipping_status=CLIPPING_STATUS_NO_CLIPPING_DETECTED, silence_result=None,
        measurement_status=MEASUREMENT_STATUS_COMPLETE, measurement_errors=(), provenance={},
    )
    plan = generate_audio_finishing_plan(measurement)
    assert plan.peak_evidence_source == PEAK_EVIDENCE_FALLBACK_SAMPLE_PEAK

    output = str(media_dir / "sample_peak_fallback_out.wav")
    record, _ = execute_audio_finishing_plan(plan, too_quiet_wav, output)
    assert record.peak_evidence_source == PEAK_EVIDENCE_FALLBACK_SAMPLE_PEAK
    assert record.execution_status == EXECUTION_STATUS_SUCCESS


# ---------------------------------------------------------------------------
# Format handling: mono/stereo, non-48k, video+audio, audio-only.
# ---------------------------------------------------------------------------

def test_mono_source_does_not_itself_alter_gain_decision(mono_wav, media_dir):
    plan, _ = _plan_for(mono_wav)
    assert plan.authorized_whole_video_gain_db is not None and plan.authorized_whole_video_gain_db > 0
    output = str(media_dir / "mono_out.wav")
    record, verification = execute_audio_finishing_plan(plan, mono_wav, output)
    assert record.execution_status == EXECUTION_STATUS_SUCCESS
    # aformat standardizes to stereo regardless of mono input, per design.
    assert verification.channel_count_expected is True


def test_non_48k_source_output_standardized_to_48k(non48k_wav, media_dir):
    plan, _ = _plan_for(non48k_wav)
    output = str(media_dir / "non48k_out.wav")
    record, verification = execute_audio_finishing_plan(plan, non48k_wav, output)
    assert record.execution_status == EXECUTION_STATUS_SUCCESS
    assert verification.sample_rate_expected is True


def test_video_and_audio_file_preserves_video_stream_via_copy(video_and_audio_mp4, media_dir):
    plan, _ = _plan_for(video_and_audio_mp4)
    output = str(media_dir / "video_out.mp4")
    record, verification = execute_audio_finishing_plan(plan, video_and_audio_mp4, output)
    assert record.execution_status == EXECUTION_STATUS_SUCCESS
    assert os.path.exists(output)
    # decode-integrity: the output must still contain a real, playable video stream
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_type", "-of", "csv=p=0", output],
        capture_output=True, text=True, check=True,
    )
    assert "video" in probe.stdout


def test_audio_only_file_does_not_request_video_copy(too_quiet_wav, media_dir):
    plan, _ = _plan_for(too_quiet_wav)
    output = str(media_dir / "audio_only_out.wav")
    record, _ = execute_audio_finishing_plan(plan, too_quiet_wav, output)
    assert record.execution_status == EXECUTION_STATUS_SUCCESS
    assert "-map" not in record.provenance.get("ffmpeg_args", [])


# ---------------------------------------------------------------------------
# Failure modes: malformed input, simulated ffmpeg failure.
# ---------------------------------------------------------------------------

def test_malformed_input_path_bounded_failure(too_quiet_wav, media_dir):
    plan, _ = _plan_for(too_quiet_wav)
    output = str(media_dir / "malformed_out.wav")
    record, verification = execute_audio_finishing_plan(plan, "/nonexistent/does_not_exist.wav", output)
    assert record.execution_status == EXECUTION_STATUS_FFMPEG_FAILURE
    assert record.ffmpeg_return_code is not None and record.ffmpeg_return_code != 0
    assert not os.path.exists(output)
    assert verification is None


def test_measurement_reference_missing_refuses_to_execute(too_quiet_wav, media_dir):
    plan, _ = _plan_for(too_quiet_wav)
    import dataclasses
    tampered = dataclasses.replace(plan, measurement_reference=None)
    output = str(media_dir / "missing_ref_out.wav")
    record, verification = execute_audio_finishing_plan(tampered, too_quiet_wav, output)
    assert record.execution_status == EXECUTION_STATUS_MEASUREMENT_REFERENCE_MISSING
    assert verification is None


def test_invalid_gain_beyond_envelope_refuses_to_execute(too_quiet_wav, media_dir):
    plan, _ = _plan_for(too_quiet_wav)
    import dataclasses
    tampered = dataclasses.replace(plan, authorized_whole_video_gain_db=99.0)
    output = str(media_dir / "invalid_gain_out.wav")
    record, verification = execute_audio_finishing_plan(tampered, too_quiet_wav, output)
    assert record.execution_status == EXECUTION_STATUS_INVALID_GAIN
    assert not os.path.exists(output)
    assert verification is None


# ---------------------------------------------------------------------------
# Idempotence: identical plan re-execution is a no-op; a different plan is not.
# ---------------------------------------------------------------------------

def test_identical_plan_same_execution_id(too_quiet_wav):
    plan, _ = _plan_for(too_quiet_wav)
    id1 = compute_execution_id(plan, too_quiet_wav)
    id2 = compute_execution_id(plan, too_quiet_wav)
    assert id1 == id2


def test_different_plan_creates_distinct_execution_id(too_quiet_wav, too_loud_wav):
    plan_a, _ = _plan_for(too_quiet_wav)
    plan_b, _ = _plan_for(too_loud_wav)
    assert compute_execution_id(plan_a, too_quiet_wav) != compute_execution_id(plan_b, too_loud_wav)


def test_reexecuting_same_plan_is_a_noop_via_existing_record(too_quiet_wav, media_dir, monkeypatch):
    plan, _ = _plan_for(too_quiet_wav)
    output = str(media_dir / "idempotent_out.wav")
    record1, _ = execute_audio_finishing_plan(plan, too_quiet_wav, output)
    assert record1.execution_status == EXECUTION_STATUS_SUCCESS

    calls = []
    import cutsell_worker.audio_finishing_executor as executor_module
    real_ffmpeg_execute = executor_module._ffmpeg_execute

    def _counting_ffmpeg_execute(*args, **kwargs):
        calls.append(1)
        return real_ffmpeg_execute(*args, **kwargs)

    monkeypatch.setattr(executor_module, "_ffmpeg_execute", _counting_ffmpeg_execute)

    record2, verification2 = execute_audio_finishing_plan(plan, too_quiet_wav, output, existing_record=record1)
    assert calls == []  # ffmpeg never invoked the second time
    assert record2.execution_id == record1.execution_id
    assert record2 is record1
    assert verification2 is None


# ---------------------------------------------------------------------------
# Structural guards: no adjacent-take execution, no live integration.
# ---------------------------------------------------------------------------

def test_adjacent_take_adjustments_never_read_or_executed():
    # Structural guard via AST, not a naive substring scan (the module's
    # own docstring legitimately explains this exclusion in prose): no
    # real attribute-access node anywhere in the module ever reads
    # `.adjacent_take_adjustments` off a plan.
    import ast
    import inspect

    import cutsell_worker.audio_finishing_executor as executor_module

    tree = ast.parse(inspect.getsource(executor_module))
    attribute_accesses = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "adjacent_take_adjustments" not in attribute_accesses


def test_executor_module_never_imported_by_render_or_pipeline():
    import cutsell_worker.render as render_module
    import inspect

    render_source = inspect.getsource(render_module)
    assert "audio_finishing_executor" not in render_source


def test_execution_record_and_verification_are_frozen(too_quiet_wav, media_dir):
    plan, _ = _plan_for(too_quiet_wav)
    output = str(media_dir / "frozen_check_out.wav")
    record, verification = execute_audio_finishing_plan(plan, too_quiet_wav, output)
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.execution_status = "TAMPERED"
    with pytest.raises(dataclasses.FrozenInstanceError):
        verification.verification_status = "TAMPERED"
