"""Audio Finishing MEASUREMENT-ONLY foundation (D-247).

Every media fixture below is SYNTHETIC, generated locally by ffmpeg itself
(lavfi sine/anullsrc sources) -- no provider, no internet, no real RAW,
matching the exact convention `test_cutsell_post_render_media_qc.py` (D-028)
already established. Skipped automatically (not failed) if `ffmpeg`/
`ffprobe` are not on PATH.

Two test classes:
- Parser-only tests (`Test*Parsing`) exercise the private ebur128/astats
  text parsers directly against real, previously-captured ffmpeg 6.1.1
  stderr text -- fast, no subprocess, and pin down the exact real-world
  output shape this module depends on.
- Fixture-backed tests exercise `measure_audio` end to end against real
  generated media, proving the whole module measures real files
  correctly and applies zero correction/policy.
"""
import shutil
import subprocess

import pytest

from cutsell_worker.audio_finishing_measurement import (
    CLIPPING_STATUS_CLIPPING_DETECTED,
    CLIPPING_STATUS_NO_CLIPPING_DETECTED,
    CLIPPING_STATUS_UNKNOWN,
    MEASUREMENT_STATUS_COMPLETE,
    MEASUREMENT_STATUS_MEASUREMENT_ERROR,
    MEASUREMENT_STATUS_PARTIAL,
    MEASUREMENT_STATUS_UNAVAILABLE,
    _classify_clipping,
    _parse_astats_sample_peak,
    _parse_ebur128_summary,
    attach_audio_finishing_diagnostics,
    measure_audio,
)
from cutsell_worker.post_render_watch_listen_qc import PostRenderQCResult

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available on this runner",
)

# Real ffmpeg 6.1.1-3ubuntu5 ebur128=peak=true stderr, captured empirically
# against a locally generated 1kHz/48kHz/stereo sine tone this session.
_REAL_EBUR128_OUTPUT = """
[Parsed_ebur128_0 @ 0x1] Summary:

  Integrated loudness:
    I:         -21.1 LUFS
    Threshold: -31.1 LUFS

  Loudness range:
    LRA:         0.0 LU
    Threshold:   0.0 LUFS
    LRA low:     0.0 LUFS
    LRA high:    0.0 LUFS

  True peak:
    Peak:      -21.1 dBFS
"""

# Real ffmpeg 6.1.1-3ubuntu5 astats stderr shape (per-channel blocks then a
# terminal Overall block), captured empirically against the same tone.
_REAL_ASTATS_OUTPUT = """
[Parsed_astats_0 @ 0x1] Channel: 1
[Parsed_astats_0 @ 0x1] Peak level dB: -21.072762
[Parsed_astats_0 @ 0x1] Channel: 2
[Parsed_astats_0 @ 0x1] Peak level dB: -21.072762
[Parsed_astats_0 @ 0x1] Overall
[Parsed_astats_0 @ 0x1] Peak level dB: -21.072762
[Parsed_astats_0 @ 0x1] Number of samples: 96000
"""

_REAL_ASTATS_CLIPPED_OUTPUT = """
[Parsed_astats_0 @ 0x1] Channel: 1
[Parsed_astats_0 @ 0x1] Peak level dB: 0.000265
[Parsed_astats_0 @ 0x1] Channel: 2
[Parsed_astats_0 @ 0x1] Peak level dB: 0.000265
[Parsed_astats_0 @ 0x1] Overall
[Parsed_astats_0 @ 0x1] Peak level dB: 0.000265
"""


class TestEbur128SummaryParsing:
    def test_parses_real_summary_block(self):
        integrated, lra, true_peak, errors = _parse_ebur128_summary(_REAL_EBUR128_OUTPUT)
        assert integrated == pytest.approx(-21.1)
        assert lra == pytest.approx(0.0)
        assert true_peak == pytest.approx(-21.1)
        assert errors == []

    def test_ignores_per_frame_lines_before_summary(self):
        streaming_noise = "t: 0.5 TARGET:-23 LUFS M: -18.0 S:-18.0 I: -99.0 LUFS LRA: 99.0 LU FTPK: -1 -1 dBFS TPK: -1 -1 dBFS\n"
        combined = streaming_noise + _REAL_EBUR128_OUTPUT
        integrated, lra, true_peak, errors = _parse_ebur128_summary(combined)
        # Must read the real Summary values, never the streaming noise above it.
        assert integrated == pytest.approx(-21.1)
        assert lra == pytest.approx(0.0)
        assert true_peak == pytest.approx(-21.1)

    def test_missing_summary_block_reports_error_not_fabricated_value(self):
        integrated, lra, true_peak, errors = _parse_ebur128_summary("no summary here at all")
        assert integrated is None
        assert lra is None
        assert true_peak is None
        assert errors

    def test_negative_infinity_loudness_parses_as_real_infinity(self):
        silent_summary = _REAL_EBUR128_OUTPUT.replace("-21.1 LUFS\n    Threshold", "-inf LUFS\n    Threshold").replace(
            "Peak:      -21.1 dBFS", "Peak:      -inf dBFS"
        )
        integrated, lra, true_peak, errors = _parse_ebur128_summary(silent_summary)
        assert integrated == float("-inf")
        assert true_peak == float("-inf")


class TestAstatsSamplePeakParsing:
    def test_parses_overall_section_only(self):
        peak, errors = _parse_astats_sample_peak(_REAL_ASTATS_OUTPUT)
        assert peak == pytest.approx(-21.072762)
        assert errors == []

    def test_reads_last_overall_not_per_channel_lines(self):
        # Per-channel lines report a different (wrong-if-read) value; only
        # the line after the LAST "Overall" marker is correct.
        tampered = _REAL_ASTATS_OUTPUT.replace(
            "[Parsed_astats_0 @ 0x1] Channel: 1\n[Parsed_astats_0 @ 0x1] Peak level dB: -21.072762",
            "[Parsed_astats_0 @ 0x1] Channel: 1\n[Parsed_astats_0 @ 0x1] Peak level dB: -99.0",
        )
        peak, errors = _parse_astats_sample_peak(tampered)
        assert peak == pytest.approx(-21.072762)

    def test_missing_overall_section_reports_error(self):
        peak, errors = _parse_astats_sample_peak("no overall section")
        assert peak is None
        assert errors


class TestClippingClassification:
    def test_clean_peak_is_not_clipping(self):
        assert _classify_clipping(-21.07) == CLIPPING_STATUS_NO_CLIPPING_DETECTED

    def test_full_scale_peak_is_clipping(self):
        assert _classify_clipping(0.000265) == CLIPPING_STATUS_CLIPPING_DETECTED

    def test_unknown_peak_is_unknown_never_a_default_guess(self):
        assert _classify_clipping(None) == CLIPPING_STATUS_UNKNOWN

    def test_real_clipped_astats_output_classifies_as_clipping(self):
        peak, _ = _parse_astats_sample_peak(_REAL_ASTATS_CLIPPED_OUTPUT)
        assert _classify_clipping(peak) == CLIPPING_STATUS_CLIPPING_DETECTED


def _ffmpeg(args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def media_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("audio_finishing_measurement")


@pytest.fixture(scope="module")
def clean_tone(media_dir):
    path = str(media_dir / "clean_tone.wav")
    _ffmpeg(["-f", "lavfi", "-i", "sine=frequency=1000:duration=2:sample_rate=48000", "-ac", "2", path])
    return path


@pytest.fixture(scope="module")
def quiet_tone(media_dir):
    path = str(media_dir / "quiet_tone.wav")
    _ffmpeg(["-f", "lavfi", "-i", "sine=frequency=1000:duration=2:sample_rate=48000,volume=0.05", "-ac", "2", path])
    return path


@pytest.fixture(scope="module")
def loud_tone(media_dir):
    path = str(media_dir / "loud_tone.wav")
    _ffmpeg(["-f", "lavfi", "-i", "sine=frequency=1000:duration=2:sample_rate=48000,volume=5", "-ac", "2", path])
    return path


@pytest.fixture(scope="module")
def clipped_tone(media_dir):
    path = str(media_dir / "clipped_tone.wav")
    _ffmpeg(["-f", "lavfi", "-i", "sine=frequency=1000:duration=2:sample_rate=48000,volume=20", "-ac", "2", path])
    return path


@pytest.fixture(scope="module")
def silence(media_dir):
    path = str(media_dir / "silence.wav")
    _ffmpeg(["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo:d=2", path])
    return path


@pytest.fixture(scope="module")
def mono_tone(media_dir):
    path = str(media_dir / "mono_tone.wav")
    _ffmpeg(["-f", "lavfi", "-i", "sine=frequency=1000:duration=2:sample_rate=48000", "-ac", "1", path])
    return path


@pytest.fixture(scope="module")
def non_48k_tone(media_dir):
    path = str(media_dir / "tone_44100.wav")
    _ffmpeg(["-f", "lavfi", "-i", "sine=frequency=1000:duration=2:sample_rate=44100", "-ac", "2", path])
    return path


@pytest.fixture(scope="module")
def short_tone(media_dir):
    path = str(media_dir / "short_tone.wav")
    _ffmpeg(["-f", "lavfi", "-i", "sine=frequency=1000:duration=0.05:sample_rate=48000", "-ac", "2", path])
    return path


@pytest.fixture(scope="module")
def adjacent_levels(media_dir):
    """A single file whose first half is quiet and second half is loud --
    the join-level/windowed-measurement fixture (Stage 8/9's data-contract
    foundation)."""
    quiet_path = str(media_dir / "_adj_quiet.wav")
    loud_path = str(media_dir / "_adj_loud.wav")
    _ffmpeg(["-f", "lavfi", "-i", "sine=frequency=1000:duration=1:sample_rate=48000,volume=0.05", "-ac", "2", quiet_path])
    _ffmpeg(["-f", "lavfi", "-i", "sine=frequency=1000:duration=1:sample_rate=48000,volume=5", "-ac", "2", loud_path])
    path = str(media_dir / "adjacent_levels.wav")
    _ffmpeg([
        "-i", quiet_path, "-i", loud_path,
        "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[a]",
        "-map", "[a]", path,
    ])
    return path


def test_clean_tone_measures_complete_with_no_clipping(clean_tone):
    result = measure_audio(clean_tone)
    assert result.measurement_status == MEASUREMENT_STATUS_COMPLETE
    assert result.sample_rate_hz == 48000
    assert result.channel_count == 2
    assert result.duration_sec == pytest.approx(2.0, abs=0.05)
    assert result.clipping_status == CLIPPING_STATUS_NO_CLIPPING_DETECTED
    assert result.integrated_loudness_lufs is not None
    assert result.true_peak_dbfs is not None
    assert result.sample_peak_dbfs is not None
    assert result.true_peak_dbfs < -1.0
    assert result.measurement_errors == ()


def test_quiet_tone_measures_lower_loudness_than_loud_tone(quiet_tone, loud_tone):
    quiet = measure_audio(quiet_tone)
    loud = measure_audio(loud_tone)
    assert quiet.integrated_loudness_lufs < loud.integrated_loudness_lufs
    assert quiet.sample_peak_dbfs < loud.sample_peak_dbfs


def test_clipped_tone_reports_clipping_detected(clipped_tone):
    result = measure_audio(clipped_tone)
    assert result.clipping_status == CLIPPING_STATUS_CLIPPING_DETECTED
    assert result.sample_peak_dbfs >= -0.1


def test_loud_but_unclipped_tone_reports_no_clipping(loud_tone):
    result = measure_audio(loud_tone)
    assert result.clipping_status == CLIPPING_STATUS_NO_CLIPPING_DETECTED


def test_silence_fixture_is_flagged_by_reused_silence_check(silence):
    result = measure_audio(silence)
    assert result.silence_result is not None
    assert isinstance(result.silence_result, PostRenderQCResult)
    assert result.silence_result.status == "FAIL"  # a 2s silent file exceeds the default 1.2s threshold
    assert len(result.silence_result.findings) >= 1


def test_mono_file_reports_channel_count_one(mono_tone):
    result = measure_audio(mono_tone)
    assert result.channel_count == 1


def test_non_48k_sample_rate_is_measured_not_assumed(non_48k_tone):
    result = measure_audio(non_48k_tone)
    assert result.sample_rate_hz == 44100


def test_short_file_still_measures_duration_and_loudness(short_tone):
    result = measure_audio(short_tone)
    assert result.duration_sec == pytest.approx(0.05, abs=0.03)
    # Loudness/peak measurement on a very short window may legitimately be
    # partial (ebur128 needs enough samples for a stable integrated value)
    # but must never silently fabricate a number -- status must say so.
    assert result.measurement_status in {MEASUREMENT_STATUS_COMPLETE, MEASUREMENT_STATUS_PARTIAL}


def test_windowed_measurement_distinguishes_adjacent_different_levels(adjacent_levels):
    first_half = measure_audio(adjacent_levels, start_sec=0.0, end_sec=1.0)
    second_half = measure_audio(adjacent_levels, start_sec=1.0, end_sec=2.0)
    assert first_half.window_start_sec == 0.0
    assert first_half.window_end_sec == 1.0
    assert first_half.duration_sec == pytest.approx(1.0)
    # The quiet half must measure meaningfully quieter than the loud half --
    # proving the window args actually scope ffmpeg's measurement, not the
    # whole file both times.
    assert first_half.sample_peak_dbfs < second_half.sample_peak_dbfs - 10.0
    assert first_half.integrated_loudness_lufs < second_half.integrated_loudness_lufs - 10.0


def test_nonexistent_file_reports_error_status_never_raises():
    result = measure_audio("/nonexistent/path/does_not_exist.wav")
    assert result.measurement_status in {MEASUREMENT_STATUS_UNAVAILABLE, MEASUREMENT_STATUS_MEASUREMENT_ERROR}
    assert result.measurement_errors


def test_measurement_never_modifies_the_source_file_on_disk(clean_tone):
    import hashlib

    def _hash(path):
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()

    before = _hash(clean_tone)
    measure_audio(clean_tone)
    after = _hash(clean_tone)
    assert before == after


def test_no_gain_correction_field_or_verdict_exists_on_the_result(clean_tone):
    result = measure_audio(clean_tone)
    # MEASUREMENT ONLY: no pass/fail verdict, no target, no applied-gain
    # field anywhere on the result -- structural proof this stays a
    # measurement, never a policy/correction object.
    field_names = set(result.__dataclass_fields__.keys())
    forbidden = {"status", "verdict", "applied_gain_db", "target_loudness_lufs", "corrected_path"}
    assert field_names.isdisjoint(forbidden)


def test_attach_audio_finishing_diagnostics_never_changes_qc_status(clean_tone):
    from cutsell_worker.post_render_media_qc import run_post_render_media_qc

    qc_result = run_post_render_media_qc(clean_tone)
    diagnostics = attach_audio_finishing_diagnostics(qc_result, clean_tone)
    assert diagnostics["post_render_qc_status"] == qc_result.status
    assert diagnostics["audio_finishing_measurement"].measurement_status == MEASUREMENT_STATUS_COMPLETE
    # Re-running the underlying QC independently must still produce the
    # exact same verdict -- proves attaching diagnostics has zero side effect.
    assert run_post_render_media_qc(clean_tone).status == qc_result.status
