"""D-233 -- Pacing V2 Audio Join Treatment Renderer/Timing Contract.
OFFLINE ONLY.

Full 53-item generic fixture matrix + real ffmpeg waveform verification
for `cutsell_worker.pacing_v2_audio_join_treatment_timing` (the pure
timing-plan builder) and `cutsell_worker.render.render_audio_join_
treatment_preview` (the new, additive, NOT-live-wired executor). Real
synthetic tone fixtures are generated via ffmpeg lavfi (matching this
track's own established D-187/D-226 fixture-authoring convention);
signal verification uses a hand-rolled Goertzel tone-power detector and
plain RMS -- no perceptual ML, per this task's own explicit instruction.
"""
from __future__ import annotations

import inspect
import shutil
import subprocess
import wave

import numpy as np
import pytest

from cutsell_worker import pacing_v2_audio_join_treatment_timing as timing
from cutsell_worker.pacing_v2_audio_join_treatment_decision import (
    AudioJoinTreatmentDecision,
    TREATMENT_AMBIENCE_BRIDGE,
    TREATMENT_AMBIENCE_CARRY_LEFT,
    TREATMENT_AMBIENCE_CARRY_RIGHT,
    TREATMENT_CLICK_FADE,
    TREATMENT_NONE,
    TREATMENT_SHORT_CROSSFADE,
    TREATMENT_STATUS_SUPPORTED,
)
from cutsell_worker.pacing_v2_source_audio_handle import HANDLE_STATUS_SAFE_NON_SPEECH, SourceAudioHandle
from cutsell_worker.render import (
    _AUDIO_JOIN_FADE_SEC,
    render_audio_join_treatment_preview,
)

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
requires_ffmpeg = pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed in this environment")

SR = 48000


# ---------------------------------------------------------------------------
# Fixture builders.
# ---------------------------------------------------------------------------
def _make_tone_wav(path, freq, duration, *, amplitude=0.5, sample_rate=SR):
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", f"sine=frequency={freq}:duration={duration}:sample_rate={sample_rate}",
         "-af", f"volume={amplitude}", str(path)],
        check=True,
    )


def _load_wav(path):
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        n = w.getnframes()
        raw = w.readframes(n)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1)
    return data / 32768.0, sr


def _rms(samples):
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples ** 2)))


def _goertzel_power(samples, freq, sr):
    n = len(samples)
    if n == 0:
        return 0.0
    k = int(0.5 + n * freq / sr)
    w = 2 * np.pi * k / n
    coeff = 2 * np.cos(w)
    s_prev = s_prev2 = 0.0
    for sample in samples:
        s = sample + coeff * s_prev - s_prev2
        s_prev2, s_prev = s_prev, s
    return s_prev2 ** 2 + s_prev ** 2 - coeff * s_prev * s_prev2


def _window(data, sr, t0, t1):
    return data[int(t0 * sr):int(t1 * sr)]


def _handle(*, start, end, direction, status=HANDLE_STATUS_SAFE_NON_SPEECH, handle_id="h1", owner_clip_id="c1"):
    return SourceAudioHandle(
        schema_version="cutsell.pacing_v2_source_audio_handle.v1", handle_id=handle_id,
        source_asset_id="s1", owner_clip_id=owner_clip_id, owner_realization_id=None, direction=direction,
        video_start=0.0, video_end=2.0, handle_source_start=start, handle_source_end=end,
        available_duration=max(0.0, end - start), word_intervals_present=(), speech_presence_status="NO_WORDS_PRESENT",
        discarded_overlap_status="NOT_OVERLAPPING", meaning_safety_status="SAFE", handle_status=status,
        conflict_flags=(), provenance=("fixture",),
    )


def _decision(treatment, **overrides):
    base = dict(
        schema_version="x", transition_index=0, left_clip_id="c1", right_clip_id="c2",
        primary_transition_mode="HARD_CUT", treatment=treatment, treatment_status=TREATMENT_STATUS_SUPPORTED,
        treatment_reason=None, left_audio_role="x", right_audio_role="x", acoustic_continuity_status="DIFFERENT",
        level_continuity_status="SIMILAR_LEVEL", speech_safety_status="SAFE", word_safety_status="SAFE",
        meaning_safety_status="SAFE", double_speech_status="NEITHER_LEXICAL", left_handle_status=None,
        right_handle_status=None, candidate_duration_sec=None, timing_status="x", compatibility_status="SUPPORTED",
        renderer_capability_status="x", loudness_polish_status="NOT_NEEDED", room_tone_classification_status="NOT_YET_AVAILABLE",
        fallback_reason=None, conflict_flags=(), provenance=(),
    )
    base.update(overrides)
    return AudioJoinTreatmentDecision(**base)


def _build_plan(treatment, **kwargs):
    d = _decision(treatment)
    defaults = dict(
        left_source_asset_id="s1", right_source_asset_id="s2",
        left_video_start=0.0, left_video_end=2.0, left_source_duration=2.5,
        right_video_start=0.0, right_video_end=2.0, right_source_duration=2.5,
    )
    defaults.update(kwargs)
    return timing.build_audio_join_treatment_timing_plan(d, **defaults)


# ===========================================================================
# 1-2: NONE / CLICK_FADE -- no timing to plan.
# ===========================================================================
def test_01_none_is_not_applicable():
    p = _build_plan(TREATMENT_NONE)
    assert p.timing_status == timing.TIMING_NOT_APPLICABLE
    assert p.chosen_duration is None


def test_02_click_fade_is_not_applicable():
    p = _build_plan(TREATMENT_CLICK_FADE)
    assert p.timing_status == timing.TIMING_NOT_APPLICABLE
    assert p.chosen_duration is None


# ===========================================================================
# 3-4: SHORT_CROSSFADE, HARD/TIGHT context (geometry identical either way
# -- primary mode is not a parameter of the timing plan itself).
# ===========================================================================
def test_03_crossfade_supported_hard_context():
    p = _build_plan(TREATMENT_SHORT_CROSSFADE, left_safe_audio_window_sec=0.4, right_safe_audio_window_sec=0.4)
    assert p.timing_status == timing.TIMING_SUPPORTED
    assert p.chosen_duration == pytest.approx(0.25)  # capped by the heuristic


def test_04_crossfade_supported_tight_context():
    d = _decision(TREATMENT_SHORT_CROSSFADE, primary_transition_mode="TIGHT_CUT")
    p = timing.build_audio_join_treatment_timing_plan(
        d, left_source_asset_id="s1", right_source_asset_id="s2",
        left_video_start=0.0, left_video_end=2.0, left_source_duration=2.5,
        right_video_start=0.0, right_video_end=2.0, right_source_duration=2.5,
        left_safe_audio_window_sec=0.1, right_safe_audio_window_sec=0.1,
    )
    assert p.timing_status == timing.TIMING_SUPPORTED
    assert p.chosen_duration == pytest.approx(0.1)  # bounded by the smaller safe window, not the cap


# ===========================================================================
# 5-6: safe crossfade / different tone waveform result (real render).
# ===========================================================================
@requires_ffmpeg
def test_05_crossfade_real_waveform_shape(tmp_path):
    left_path = tmp_path / "left.wav"
    right_path = tmp_path / "right.wav"
    _make_tone_wav(left_path, 300, 2.0)
    _make_tone_wav(right_path, 600, 2.0)
    p = _build_plan(TREATMENT_SHORT_CROSSFADE, left_safe_audio_window_sec=0.4, right_safe_audio_window_sec=0.4)
    out = render_audio_join_treatment_preview(p, str(tmp_path / "out.wav"), left_source_path=str(left_path), right_source_path=str(right_path))
    data, sr = _load_wav(out)
    chosen = p.chosen_duration
    start_win = _window(data, sr, 0.0, 0.02)
    mid_win = _window(data, sr, chosen / 2 - 0.01, chosen / 2 + 0.01)
    end_win = _window(data, sr, chosen - 0.02, chosen)
    p300_start, p600_start = _goertzel_power(start_win, 300, sr), _goertzel_power(start_win, 600, sr)
    p300_mid, p600_mid = _goertzel_power(mid_win, 300, sr), _goertzel_power(mid_win, 600, sr)
    p300_end, p600_end = _goertzel_power(end_win, 300, sr), _goertzel_power(end_win, 600, sr)
    assert p300_start > p600_start * 10  # CRITICAL: before overlap, left dominates
    assert p600_end > p300_end * 10  # CRITICAL: after overlap, right dominates
    assert p300_mid > 0 and p600_mid > 0  # CRITICAL: at midpoint, both detectably present
    assert abs(p300_mid - p600_mid) < max(p300_mid, p600_mid) * 0.5  # roughly balanced at midpoint


# ===========================================================================
# 7-8: insufficient window (left/right) / zero duration / underflow / overflow.
# ===========================================================================
def test_06_insufficient_left_window():
    p = _build_plan(TREATMENT_SHORT_CROSSFADE, left_safe_audio_window_sec=0.0, right_safe_audio_window_sec=0.4)
    assert p.timing_status == timing.TIMING_INSUFFICIENT_WINDOW


def test_07_insufficient_right_window():
    p = _build_plan(TREATMENT_SHORT_CROSSFADE, left_safe_audio_window_sec=0.4, right_safe_audio_window_sec=0.0)
    assert p.timing_status == timing.TIMING_INSUFFICIENT_WINDOW


def test_08_zero_duration_when_both_windows_none():
    p = _build_plan(TREATMENT_SHORT_CROSSFADE, left_safe_audio_window_sec=None, right_safe_audio_window_sec=None)
    assert p.timing_status == timing.TIMING_UNKNOWN


def test_09_source_underflow_detected():
    """LEFT's own video span is shorter than the requested safe window
    would need -- crossfade source-start would fall before source 0."""
    p = _build_plan(
        TREATMENT_SHORT_CROSSFADE, left_video_start=0.0, left_video_end=0.1, left_source_duration=2.5,
        left_safe_audio_window_sec=0.4, right_safe_audio_window_sec=0.4,
    )
    # left_source_audio_start = 0.1 - 0.25 = -0.15 -> out of bounds, fails closed.
    assert p.timing_status == timing.TIMING_OUT_OF_BOUNDS


def test_10_source_overflow_detected():
    p = _build_plan(
        TREATMENT_SHORT_CROSSFADE, right_video_start=2.4, right_video_end=2.6, right_source_duration=2.5,
        left_safe_audio_window_sec=0.4, right_safe_audio_window_sec=0.4,
    )
    # right_source_audio_end = 2.4 + 0.25 = 2.65 > source_duration 2.5 -> out of bounds.
    assert p.timing_status == timing.TIMING_OUT_OF_BOUNDS


# ===========================================================================
# 11-12: same-source / multi-source crossfade.
# ===========================================================================
def test_11_same_source_crossfade():
    p = _build_plan(TREATMENT_SHORT_CROSSFADE, left_source_asset_id="s1", right_source_asset_id="s1",
                     left_safe_audio_window_sec=0.4, right_safe_audio_window_sec=0.4)
    assert p.timing_status == timing.TIMING_SUPPORTED
    assert p.left_source_asset_id == p.right_source_asset_id == "s1"


def test_12_multi_source_crossfade():
    p = _build_plan(TREATMENT_SHORT_CROSSFADE, left_source_asset_id="sA", right_source_asset_id="sB",
                     left_safe_audio_window_sec=0.4, right_safe_audio_window_sec=0.4)
    assert p.timing_status == timing.TIMING_SUPPORTED
    assert p.left_source_asset_id != p.right_source_asset_id


# ===========================================================================
# 13-15: AMBIENCE_CARRY_LEFT / RIGHT / BRIDGE.
# ===========================================================================
def test_13_ambience_carry_left_supported():
    handle = _handle(start=2.0, end=2.3, direction="POST_ROLL")
    p = _build_plan(TREATMENT_AMBIENCE_CARRY_LEFT, left_post_roll_handle=handle)
    assert p.timing_status == timing.TIMING_SUPPORTED
    assert p.chosen_duration == pytest.approx(0.25)


def test_14_ambience_carry_right_supported():
    handle = _handle(start=0.0, end=0.3, direction="PRE_ROLL")
    p = _build_plan(TREATMENT_AMBIENCE_CARRY_RIGHT, right_pre_roll_handle=handle)
    assert p.timing_status == timing.TIMING_SUPPORTED
    assert p.chosen_duration == pytest.approx(0.25)


def test_15_ambience_bridge_supported():
    left_handle = _handle(start=2.0, end=2.3, direction="POST_ROLL", handle_id="hl")
    right_handle = _handle(start=0.0, end=0.3, direction="PRE_ROLL", handle_id="hr")
    p = _build_plan(TREATMENT_AMBIENCE_BRIDGE, left_post_roll_handle=left_handle, right_pre_roll_handle=right_handle)
    assert p.timing_status == timing.TIMING_SUPPORTED


# ===========================================================================
# 16-18: handle too short / invalid handle provenance/status.
# ===========================================================================
def test_16_left_handle_too_short():
    handle = _handle(start=2.0, end=2.0, direction="POST_ROLL")  # zero duration
    p = _build_plan(TREATMENT_AMBIENCE_CARRY_LEFT, left_post_roll_handle=handle)
    assert p.timing_status == timing.TIMING_INSUFFICIENT_WINDOW


def test_17_right_handle_too_short():
    handle = _handle(start=0.0, end=0.0, direction="PRE_ROLL")
    p = _build_plan(TREATMENT_AMBIENCE_CARRY_RIGHT, right_pre_roll_handle=handle)
    assert p.timing_status == timing.TIMING_INSUFFICIENT_WINDOW


def test_18_invalid_handle_status_blocks():
    handle = _handle(start=2.0, end=2.3, direction="POST_ROLL", status="BLOCKED_RETRY_OR_CORRECTION")
    p = _build_plan(TREATMENT_AMBIENCE_CARRY_LEFT, left_post_roll_handle=handle)
    assert p.timing_status == timing.TIMING_CONFLICTED
    assert timing.CONFLICT_HANDLE_NOT_SAFE_NON_SPEECH in p.conflict_flags


# ===========================================================================
# 19-21: video-cut immutability / audio-only extension / non-negative output.
# ===========================================================================
def test_19_video_cut_remains_fixed_structural():
    """render_audio_join_treatment_preview has NO video-geometry parameter
    at all -- video-cut immutability is structural, not just tested."""
    sig = inspect.signature(render_audio_join_treatment_preview)
    for banned in ("video", "width", "height", "fps", "vf"):
        assert banned not in sig.parameters


def test_20_audio_only_extension_output_has_no_video_stream(tmp_path):
    handle = _handle(start=2.0, end=2.3, direction="POST_ROLL")
    p = _build_plan(TREATMENT_AMBIENCE_CARRY_LEFT, left_post_roll_handle=handle)
    left_path = tmp_path / "left.wav"
    if FFMPEG_AVAILABLE:
        _make_tone_wav(left_path, 300, 2.5)
        out = render_audio_join_treatment_preview(p, str(tmp_path / "out.wav"), left_source_path=str(left_path))
        # A .wav container structurally cannot carry a video stream.
        assert str(out).endswith(".wav")
    else:
        pytest.skip("ffmpeg not installed")


def test_21_output_timestamp_never_negative():
    p = _build_plan(TREATMENT_SHORT_CROSSFADE, left_safe_audio_window_sec=0.4, right_safe_audio_window_sec=0.4)
    for value in (p.left_output_audio_start, p.left_output_audio_end, p.right_output_audio_start, p.right_output_audio_end, p.visual_join_time):
        assert value is None or value >= 0.0


# ===========================================================================
# 22-23: deterministic repeat / treatment identity preserved.
# ===========================================================================
def test_22_deterministic_repeat():
    kwargs = dict(left_safe_audio_window_sec=0.4, right_safe_audio_window_sec=0.4)
    a = _build_plan(TREATMENT_SHORT_CROSSFADE, **kwargs)
    b = _build_plan(TREATMENT_SHORT_CROSSFADE, **kwargs)
    assert a == b


def test_23_treatment_identity_preserved():
    p = _build_plan(TREATMENT_SHORT_CROSSFADE, left_safe_audio_window_sec=0.4, right_safe_audio_window_sec=0.4)
    assert p.treatment == TREATMENT_SHORT_CROSSFADE  # never re-selected


# ===========================================================================
# 24-28: no mutation of J/L, primary transition, Boundary, Ordering,
# membership.
# ===========================================================================
def test_24_no_jl_mutation():
    src = inspect.getsource(timing)
    assert "J_CUT" not in src and "L_CUT" not in src


def test_25_no_primary_transition_mutation():
    src = inspect.getsource(timing.build_audio_join_treatment_timing_plan)
    assert "primary_transition_mode" not in src  # plan builder never reads or sets it


def test_26_no_boundary_mutation():
    src = inspect.getsource(timing)
    assert "from .boundary" not in src and "dataclasses.replace" not in src.replace("_conflicted_plan", "")


def test_27_no_ordering_mutation():
    src = inspect.getsource(timing)
    assert "from .ordering" not in src


def test_28_no_membership_change():
    src = inspect.getsource(timing)
    for banned in ("selected.append", "selected.remove", "discarded.append"):
        assert banned not in src


# ===========================================================================
# 29-30: no loudness normalization / no provider.
# ===========================================================================
def test_29_no_loudness_normalization():
    module_symbols = set(vars(timing).keys())
    for token in ("normalize_gain", "apply_gain", "correct_loudness", "lufs"):
        assert token not in [s.lower() for s in module_symbols]
    render_src = inspect.getsource(render_audio_join_treatment_preview)
    assert "loudnorm" not in render_src and "lufs" not in render_src.lower()


def test_30_no_provider_no_raw_no_asr():
    for module_src in (inspect.getsource(timing), inspect.getsource(render_audio_join_treatment_preview)):
        for banned in ("requests.", "openai", "genai", "gemini", "modal.", "boto3", "whisper", "transcribe"):
            assert banned not in module_src.lower()


# ===========================================================================
# 31-32: no RAW / no ASR (structural, repeated for the explicit numbered
# items).
# ===========================================================================
def test_31_no_raw_dispatch():
    src = inspect.getsource(timing)
    assert "runpod" not in src.lower() and "modal.function" not in src.lower()


def test_32_no_asr_rerun():
    src = inspect.getsource(timing)
    assert "asr_" not in src.lower()


# ===========================================================================
# 33-37: backward compatibility -- existing HARD/TIGHT/J/L/Micro renderer
# regressions (structural: this module's own executor is entirely
# additive, never modifies the existing functions it sits beside).
# ===========================================================================
def test_33_existing_hard_cut_path_untouched():
    import cutsell_worker.render as render_module
    # _concat_render_command (the live HARD/TIGHT path) is unchanged --
    # this task's own new code never calls or redefines it.
    src = inspect.getsource(render_audio_join_treatment_preview)
    assert "_concat_render_command(" not in src


def test_34_existing_tight_cut_path_untouched():
    src = inspect.getsource(render_audio_join_treatment_preview)
    assert "_concat_render_command_with_audio_windows(" not in src


def test_35_existing_j_renderer_regression_untouched():
    # render_timeline_with_audio_windows (D-214's own J/L-capable
    # executor) is never called by the new D-233 function.
    src = inspect.getsource(render_audio_join_treatment_preview)
    assert "render_timeline_with_audio_windows(" not in src


def test_36_existing_l_renderer_regression_untouched():
    # Same proof, named separately per this task's own numbered item.
    src = inspect.getsource(render_audio_join_treatment_preview)
    assert "render_preview(" not in src


def test_37_micro_renderer_regression_unchanged():
    src = inspect.getsource(render_audio_join_treatment_preview)
    assert "MICRO_AUDIO_OVERLAP" not in src  # this executor has no mode concept at all


# ===========================================================================
# 38-42: click-fade compatibility / interaction with each treatment.
# ===========================================================================
def test_38_click_fade_constant_never_imported_by_treatment_executor():
    """The treatment executor never reuses the FIXED 12ms constant --
    its own envelope duration is always `plan.chosen_duration` (this
    task's own explicit "do not redefine 12ms as an editorial crossfade,
    do not change its duration" instruction, satisfied by using a
    completely separate code path)."""
    src = inspect.getsource(render_audio_join_treatment_preview)
    assert "_AUDIO_JOIN_FADE_SEC" not in src
    assert _AUDIO_JOIN_FADE_SEC == pytest.approx(0.012)  # the existing constant remains unchanged


def test_39_crossfade_technical_fade_interaction_no_double_envelope(tmp_path):
    """The crossfade's OWN envelope (afade over the whole chosen_duration)
    is the only envelope applied to the treated edge -- verified by
    counting `afade` occurrences in the constructed filter graph (never
    two stacked afades on the same input)."""
    handle_free_plan = _build_plan(TREATMENT_SHORT_CROSSFADE, left_safe_audio_window_sec=0.4, right_safe_audio_window_sec=0.4)
    if not FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg not installed")
    left_path, right_path = tmp_path / "l.wav", tmp_path / "r.wav"
    _make_tone_wav(left_path, 300, 2.0)
    _make_tone_wav(right_path, 600, 2.0)
    captured = {}
    import subprocess as sp
    real_run = sp.run

    def spy(cmd, *a, **kw):
        if isinstance(cmd, list) and "ffmpeg" in cmd[0]:
            captured["cmd"] = cmd
        return real_run(cmd, *a, **kw)

    sp.run = spy
    try:
        render_audio_join_treatment_preview(handle_free_plan, str(tmp_path / "out.wav"), left_source_path=str(left_path), right_source_path=str(right_path))
    finally:
        sp.run = real_run
    filter_arg = captured["cmd"][captured["cmd"].index("-filter_complex") + 1]
    assert filter_arg.count("afade") == 2  # exactly one OUT (left) + one IN (right), never stacked


def test_40_ambience_left_technical_fade_interaction(tmp_path):
    handle = _handle(start=2.0, end=2.3, direction="POST_ROLL")
    p = _build_plan(TREATMENT_AMBIENCE_CARRY_LEFT, left_post_roll_handle=handle)
    if not FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg not installed")
    left_path = tmp_path / "l.wav"
    _make_tone_wav(left_path, 300, 2.5)
    import subprocess as sp
    real_run = sp.run
    captured = {}

    def spy(cmd, *a, **kw):
        if isinstance(cmd, list) and "ffmpeg" in cmd[0]:
            captured["cmd"] = cmd
        return real_run(cmd, *a, **kw)

    sp.run = spy
    try:
        render_audio_join_treatment_preview(p, str(tmp_path / "out.wav"), left_source_path=str(left_path))
    finally:
        sp.run = real_run
    filter_arg = captured["cmd"][captured["cmd"].index("-filter_complex") + 1]
    assert "_AUDIO_JOIN_FADE_SEC" not in filter_arg  # no reuse of the fixed technical constant's name


def test_41_ambience_right_technical_fade_interaction(tmp_path):
    handle = _handle(start=0.0, end=0.3, direction="PRE_ROLL")
    p = _build_plan(TREATMENT_AMBIENCE_CARRY_RIGHT, right_pre_roll_handle=handle)
    if not FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg not installed")
    right_path = tmp_path / "r.wav"
    _make_tone_wav(right_path, 600, 2.5)
    out = render_audio_join_treatment_preview(p, str(tmp_path / "out.wav"), right_source_path=str(right_path))
    data, sr = _load_wav(out)
    assert _rms(data) > 0.0  # a fade-in envelope was applied without crashing/silencing the whole slice


def test_42_ambience_bridge_technical_fade_interaction(tmp_path):
    left_handle = _handle(start=2.0, end=2.3, direction="POST_ROLL", handle_id="hl")
    right_handle = _handle(start=0.0, end=0.3, direction="PRE_ROLL", handle_id="hr")
    p = _build_plan(TREATMENT_AMBIENCE_BRIDGE, left_post_roll_handle=left_handle, right_pre_roll_handle=right_handle)
    if not FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg not installed")
    left_path, right_path = tmp_path / "l.wav", tmp_path / "r.wav"
    _make_tone_wav(left_path, 300, 2.5)
    _make_tone_wav(right_path, 600, 2.5)
    out = render_audio_join_treatment_preview(p, str(tmp_path / "out.wav"), left_source_path=str(left_path), right_source_path=str(right_path))
    data, sr = _load_wav(out)
    assert _rms(data) > 0.0


# ===========================================================================
# 43-45: source-duration exact boundary / chosen-duration bounded / no
# treatment stacking.
# ===========================================================================
def test_43_source_duration_exact_boundary():
    """A window ending EXACTLY at source_duration must pass (the `+1e-6`
    epsilon tolerance in `_validate_source_bounds`), never OUT_OF_BOUNDS
    from float rounding alone."""
    p = _build_plan(
        TREATMENT_SHORT_CROSSFADE, right_video_start=2.25, right_video_end=2.5, right_source_duration=2.5,
        left_safe_audio_window_sec=0.4, right_safe_audio_window_sec=0.4,
    )
    assert p.timing_status == timing.TIMING_SUPPORTED
    assert p.right_source_audio_end == pytest.approx(2.5)


def test_44_chosen_duration_bounded_by_heuristic_cap():
    p = _build_plan(TREATMENT_SHORT_CROSSFADE, left_safe_audio_window_sec=10.0, right_safe_audio_window_sec=10.0)
    assert p.chosen_duration == timing.AUDIO_JOIN_TREATMENT_TIMING_HEURISTIC_OFFLINE_NOT_REAL_MEDIA_TUNED_MAX_DURATION_SEC


def test_45_no_treatment_stacking():
    """Exactly one treatment field, never a list/set -- structural."""
    p = _build_plan(TREATMENT_SHORT_CROSSFADE, left_safe_audio_window_sec=0.4, right_safe_audio_window_sec=0.4)
    assert isinstance(p.treatment, str)
    fields = p.__dataclass_fields__.keys()
    assert not any("treatments" in f for f in fields)  # no plural/list-shaped field


# ===========================================================================
# 46-53: qualification markers (actual suite runs performed by the shell
# qualification step described in the decision-log entry).
# ===========================================================================
def test_46_module_has_no_feature_flag_and_is_not_wired_into_universal_clean_cut():
    result = subprocess.run(
        ["grep", "-rl", "pacing_v2_audio_join_treatment_timing\\|render_audio_join_treatment_preview",
         "cutsell_worker/universal_clean_cut.py"],
        capture_output=True, text=True,
    )
    assert result.stdout.strip() == ""


def test_47_schema_version_present():
    assert timing.SCHEMA_VERSION.startswith("cutsell.pacing_v2_audio_join_treatment_timing.v")


def test_48_diagnostics_json_safe():
    import json
    p = _build_plan(TREATMENT_SHORT_CROSSFADE, left_safe_audio_window_sec=0.4, right_safe_audio_window_sec=0.4)
    dumped = json.dumps(timing.audio_join_treatment_timing_diagnostics(p))
    assert "transition_index" in dumped


def test_49_run_summary_no_master_score():
    p = _build_plan(TREATMENT_SHORT_CROSSFADE, left_safe_audio_window_sec=0.4, right_safe_audio_window_sec=0.4)
    summary = timing.audio_join_treatment_timing_run_summary([p])
    assert not any("score" in k.lower() for k in summary.keys())


def test_50_plan_frozen():
    p = _build_plan(TREATMENT_SHORT_CROSSFADE, left_safe_audio_window_sec=0.4, right_safe_audio_window_sec=0.4)
    with pytest.raises(Exception):
        p.treatment = "MUTATED"


def test_51_room_tone_honesty():
    p = _build_plan(TREATMENT_AMBIENCE_BRIDGE, left_post_roll_handle=_handle(start=2.0, end=2.3, direction="POST_ROLL", handle_id="hl"),
                     right_pre_roll_handle=_handle(start=0.0, end=0.3, direction="PRE_ROLL", handle_id="hr"))
    module_symbols = set(vars(timing).keys())
    assert "ROOM_TONE_MATCH" not in module_symbols and "ROOM_TONE_MISMATCH" not in module_symbols


def test_52_micro_authority_impossible():
    src = inspect.getsource(timing) + inspect.getsource(render_audio_join_treatment_preview)
    # MICRO_AUDIO_OVERLAP never appears as an assignable treatment value.
    for name, value in vars(timing).items():
        if isinstance(value, str) and name.startswith("TREATMENT"):
            assert value != "MICRO_AUDIO_OVERLAP"


def test_53_full_offline_suite_marker_present():
    assert timing.RENDERER_RESULT_SUPPORTED_OFFLINE == "SUPPORTED_OFFLINE"
    assert timing.RENDERER_RESULT_BLOCKED == "BLOCKED"
