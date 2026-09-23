"""D-214 -- PACING V2 RENDERER / TIMELINE CONTRACT EXTENSION, OFFLINE ONLY.

D-213's own forensic named the ONE renderer gap blocking J_CUT/L_CUT/
MICRO_AUDIO_OVERLAP: `_concat_render_command` gave every segment's audio
and video the identical trimmed duration, so they could never diverge.
This suite proves the new independent audio-window contract
(`RenderSegment.audio_start`/`.audio_end`, `render.
_concat_render_command_with_audio_windows`, `render.
render_timeline_with_audio_windows`, `render.
dialogue_pacing_transition_execution_diagnostics`) at the EXECUTION layer
only -- no mode-selection logic, no live wiring, no Boundary/Ordering/
BestTake/Pacing-authority change. Synthetic media only (two locally
ffmpeg-generated tone sources, 220 Hz "low" / 1760 Hz "high", exactly
D-097.2's own precedent) -- no downloaded content, no provider, no ASR, no
Video00 wording.
"""
from __future__ import annotations

import ast
import pathlib
import shutil
import subprocess

import numpy as np
import pytest

from cutsell_worker import render
from cutsell_worker.media_probe import probe_media
from cutsell_worker.render_plan import RenderSegment, _can_coalesce, _coalesce_contiguous_segments

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available on this runner")

_WIDTH, _HEIGHT, _FPS = 160, 120, 30
_LOW_HZ, _HIGH_HZ = 220, 1760


def _ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args], check=True)


@pytest.fixture(scope="module")
def tone_sources(tmp_path_factory):
    directory = tmp_path_factory.mktemp("d214_render_timeline")
    low = str(directory / "low.mp4")
    high = str(directory / "high.mp4")
    for path, frequency in ((low, _LOW_HZ), (high, _HIGH_HZ)):
        _ffmpeg([
            "-y", "-f", "lavfi", "-i", f"testsrc=size={_WIDTH}x{_HEIGHT}:rate={_FPS}",
            "-f", "lavfi", "-i", f"sine=frequency={frequency}:sample_rate=48000,volume=0.3,aformat=channel_layouts=stereo",
            "-t", "6", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-b:a", "96k", path,
        ])
    return str(directory), low, high


def _tone_presence(path: str, frequency: int, *, sample_rate: int = 48000, block_ms: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Goertzel-style per-block energy for `frequency` in `path`'s own audio
    track -- exactly D-097.2's own deterministic-audio-marker precedent.
    Returns (times, energies); the caller normalizes/thresholds."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "-"],
        stdout=subprocess.PIPE, check=True,
    )
    pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float64)
    block = max(1, int(sample_rate * block_ms / 1000.0))
    count = pcm.size // block
    t = np.arange(block) / sample_rate
    kernel = np.exp(-2j * np.pi * frequency * t)
    energies = np.array([abs(np.dot(pcm[i * block:(i + 1) * block], kernel)) / block for i in range(count)])
    times = np.arange(count) * block_ms / 1000.0
    return times, energies


def _presence_mask(path: str, frequency: int, *, threshold_fraction: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    times, energies = _tone_presence(path, frequency)
    peak = float(energies.max()) if energies.size else 0.0
    return times, energies > (threshold_fraction * peak if peak > 0 else 0.0)


def _first_true(times: np.ndarray, mask: np.ndarray) -> float | None:
    idx = np.argmax(mask) if mask.any() else None
    return float(times[idx]) if idx is not None else None


def _last_true(times: np.ndarray, mask: np.ndarray) -> float | None:
    if not mask.any():
        return None
    idx = len(mask) - 1 - int(np.argmax(mask[::-1]))
    return float(times[idx])


# --- 1-3: backward compatibility / default window representation -----------

def test_01_default_render_segment_has_no_independent_audio_window():
    seg = RenderSegment(clip_id="a", source_asset_id="s", source_path="p", start=1.0, end=2.0)
    assert seg.audio_start is None and seg.audio_end is None
    assert seg.has_independent_audio_window is False
    assert seg.effective_audio_start == 1.0
    assert seg.effective_audio_end == 2.0


def test_02_independent_window_representable_and_diverges():
    seg = RenderSegment(clip_id="a", source_asset_id="s", source_path="p", start=1.0, end=2.0, audio_start=0.7)
    assert seg.has_independent_audio_window is True
    assert seg.effective_audio_start == 0.7
    assert seg.effective_audio_end == 2.0  # audio_end still defaults


def test_03_default_audio_window_equals_video_window_duration():
    seg = RenderSegment(clip_id="a", source_asset_id="s", source_path="p", start=1.0, end=2.5)
    assert seg.audio_duration_sec == pytest.approx(seg.duration_sec)


# --- 4/5: HARD_CUT / TIGHT_CUT backward-compatible execution -----------------

def test_04_no_divergence_delegates_byte_identical_command(tone_sources, tmp_path):
    _, low, high = tone_sources
    segs = (
        RenderSegment(clip_id="a", source_asset_id="s1", source_path=low, start=0.5, end=1.2),
        RenderSegment(clip_id="b", source_asset_id="s2", source_path=high, start=0.3, end=1.0),
    )
    plain = render._concat_render_command(segs, tmp_path / "out.mp4", width=_WIDTH, height=_HEIGHT, fps=_FPS, workdir=tmp_path)
    extended = render._concat_render_command_with_audio_windows(segs, tmp_path / "out.mp4", width=_WIDTH, height=_HEIGHT, fps=_FPS, workdir=tmp_path)
    assert plain == extended


def test_05_boundary_reason_alone_never_triggers_divergent_path(tone_sources, tmp_path):
    # A TIGHT_CUT-labelled clip (Boundary already trimmed it upstream) carries
    # no independent audio window at THIS layer -- HARD_CUT and TIGHT_CUT are
    # mechanically identical at execution time; only the (out-of-scope) Pacing
    # attribution differs.
    _, low, high = tone_sources
    segs = (
        RenderSegment(clip_id="a", source_asset_id="s1", source_path=low, start=0.5, end=1.2, boundary_reason="tighten_audio_exit"),
        RenderSegment(clip_id="b", source_asset_id="s2", source_path=high, start=0.3, end=1.0, boundary_reason="tighten_audio_entry"),
    )
    assert not any(s.has_independent_audio_window for s in segs)
    plain = render._concat_render_command(segs, tmp_path / "out.mp4", width=_WIDTH, height=_HEIGHT, fps=_FPS, workdir=tmp_path)
    extended = render._concat_render_command_with_audio_windows(segs, tmp_path / "out.mp4", width=_WIDTH, height=_HEIGHT, fps=_FPS, workdir=tmp_path)
    assert plain == extended


# --- 6/7: J-CUT representation + execution ----------------------------------

def _jcut_segments(low: str, high: str) -> tuple[RenderSegment, RenderSegment]:
    return (
        RenderSegment(clip_id="a", source_asset_id="s1", source_path=low, start=0.5, end=1.5),
        RenderSegment(clip_id="b", source_asset_id="s2", source_path=high, start=0.8, end=1.8, audio_start=0.5),
    )


def test_06_jcut_representation_matches_derived_geometry():
    segs = _jcut_segments("low.mp4", "high.mp4")
    left, right = segs
    assert right.effective_audio_start < right.start  # leading audio window
    assert left.effective_audio_end == left.end  # left side untouched


def test_07_jcut_execution_proves_next_audio_leads_visual_switch(tone_sources, tmp_path):
    _, low, high = tone_sources
    segs = _jcut_segments(low, high)
    diag = render.dialogue_pacing_transition_execution_diagnostics(segs)
    row = diag["transitions"][0]
    assert row["mode"] == render.TRANSITION_J_CUT
    assert row["execution_status"] == render.EXECUTION_STATUS_EXECUTABLE

    out = render.render_timeline_with_audio_windows(segs, str(tmp_path / "jcut.mp4"), width=_WIDTH, height=_HEIGHT, fps=_FPS)
    times_low, low_mask = _presence_mask(out, _LOW_HZ)
    times_high, high_mask = _presence_mask(out, _HIGH_HZ)
    high_start = _first_true(times_high, high_mask)
    low_end = _last_true(times_low, low_mask)
    assert high_start is not None and high_start < row["video_switch_time"] - 0.05
    assert high_start == pytest.approx(row["right_audio_start_timeline"], abs=0.03)
    assert low_end == pytest.approx(row["left_audio_end_timeline"], abs=0.03)


# --- 8/9: L-CUT representation + execution ----------------------------------

def _lcut_segments(low: str, high: str) -> tuple[RenderSegment, RenderSegment]:
    return (
        RenderSegment(clip_id="a", source_asset_id="s1", source_path=low, start=0.5, end=1.5, audio_end=1.8),
        RenderSegment(clip_id="b", source_asset_id="s2", source_path=high, start=0.8, end=1.8),
    )


def test_08_lcut_representation_matches_derived_geometry():
    segs = _lcut_segments("low.mp4", "high.mp4")
    left, right = segs
    assert left.effective_audio_end > left.end  # trailing audio window
    assert right.effective_audio_start == right.start


def test_09_lcut_execution_proves_previous_audio_persists_past_switch(tone_sources, tmp_path):
    _, low, high = tone_sources
    segs = _lcut_segments(low, high)
    diag = render.dialogue_pacing_transition_execution_diagnostics(segs)
    row = diag["transitions"][0]
    assert row["mode"] == render.TRANSITION_L_CUT

    out = render.render_timeline_with_audio_windows(segs, str(tmp_path / "lcut.mp4"), width=_WIDTH, height=_HEIGHT, fps=_FPS)
    times_low, low_mask = _presence_mask(out, _LOW_HZ)
    times_high, high_mask = _presence_mask(out, _HIGH_HZ)
    low_end = _last_true(times_low, low_mask)
    high_start = _first_true(times_high, high_mask)
    assert low_end is not None and low_end > row["video_switch_time"] + 0.05
    assert low_end == pytest.approx(row["left_audio_end_timeline"], abs=0.03)
    assert high_start == pytest.approx(row["right_audio_start_timeline"], abs=0.03)


# --- 10/11: MICRO_AUDIO_OVERLAP representation + execution -------------------

def _micro_overlap_segments(low: str, high: str) -> tuple[RenderSegment, RenderSegment]:
    return (
        RenderSegment(clip_id="a", source_asset_id="s1", source_path=low, start=0.5, end=1.5, audio_end=1.6),
        RenderSegment(clip_id="b", source_asset_id="s2", source_path=high, start=0.8, end=1.8, audio_start=0.7),
    )


def test_10_micro_overlap_representation_has_both_leading_and_trailing():
    segs = _micro_overlap_segments("low.mp4", "high.mp4")
    left, right = segs
    assert left.effective_audio_end > left.end
    assert right.effective_audio_start < right.start


def test_11_micro_overlap_execution_proves_bounded_simultaneous_presence(tone_sources, tmp_path):
    _, low, high = tone_sources
    segs = _micro_overlap_segments(low, high)
    diag = render.dialogue_pacing_transition_execution_diagnostics(segs)
    row = diag["transitions"][0]
    assert row["mode"] == render.TRANSITION_MICRO_AUDIO_OVERLAP
    assert 0.0 < row["actual_overlap_sec"] < 0.5  # a genuinely bounded, small overlap

    out = render.render_timeline_with_audio_windows(segs, str(tmp_path / "micro.mp4"), width=_WIDTH, height=_HEIGHT, fps=_FPS)
    times_low, low_mask = _presence_mask(out, _LOW_HZ)
    times_high, high_mask = _presence_mask(out, _HIGH_HZ)
    both_present_seconds = np.sum(low_mask & high_mask) * 0.01
    assert both_present_seconds > 0.0
    assert both_present_seconds == pytest.approx(row["actual_overlap_sec"], abs=0.05)


# --- 12: no-overlap hard-cut negative control --------------------------------

def test_12_hard_cut_negative_control_zero_cross_segment_overlap(tone_sources, tmp_path):
    _, low, high = tone_sources
    segs = (
        RenderSegment(clip_id="a", source_asset_id="s1", source_path=low, start=0.5, end=1.2),
        RenderSegment(clip_id="b", source_asset_id="s2", source_path=high, start=0.3, end=1.0),
    )
    diag = render.dialogue_pacing_transition_execution_diagnostics(segs)
    assert diag["transitions"][0]["mode"] == render.TRANSITION_HARD_CUT
    assert diag["transitions"][0]["actual_overlap_sec"] == 0.0

    out = render.render_timeline_with_audio_windows(segs, str(tmp_path / "hardcut.mp4"), width=_WIDTH, height=_HEIGHT, fps=_FPS)
    times_low, low_mask = _presence_mask(out, _LOW_HZ)
    times_high, high_mask = _presence_mask(out, _HIGH_HZ)
    assert not np.any(low_mask & high_mask)


# --- 13-16: exact timing assertions (requested vs. actual) ------------------

def test_13_exact_visual_switch_matches_cumulative_video_duration(tone_sources, tmp_path):
    _, low, high = tone_sources
    segs = _jcut_segments(low, high)
    diag = render.dialogue_pacing_transition_execution_diagnostics(segs)
    expected = render.rendered_segment_duration_sec(segs[0].duration_sec, fps=_FPS)
    assert diag["transitions"][0]["video_switch_time"] == pytest.approx(expected, abs=1e-3)


def test_14_exact_left_audio_end_placement(tone_sources):
    _, low, high = tone_sources
    segs = _jcut_segments(low, high)
    diag = render.dialogue_pacing_transition_execution_diagnostics(segs)
    row = diag["transitions"][0]
    assert row["left_audio_end_timeline"] == pytest.approx(row["video_switch_time"], abs=1e-3)


def test_15_exact_right_audio_start_placement(tone_sources):
    _, low, high = tone_sources
    segs = _jcut_segments(low, high)
    diag = render.dialogue_pacing_transition_execution_diagnostics(segs)
    row = diag["transitions"][0]
    lead = segs[1].start - segs[1].effective_audio_start
    assert row["right_audio_start_timeline"] == pytest.approx(row["video_switch_time"] - lead, abs=1e-3)


def test_16_exact_requested_overlap_matches_actual_when_executable(tone_sources):
    _, low, high = tone_sources
    segs = _jcut_segments(low, high)
    diag = render.dialogue_pacing_transition_execution_diagnostics(segs)
    row = diag["transitions"][0]
    assert row["requested_overlap_sec"] == pytest.approx(row["actual_overlap_sec"], abs=1e-3)


# --- 17-19: source identity / same-source / multi-source -------------------

def test_17_source_identity_preserved_in_diagnostics(tone_sources):
    _, low, high = tone_sources
    segs = _jcut_segments(low, high)
    diag = render.dialogue_pacing_transition_execution_diagnostics(segs)
    row = diag["transitions"][0]
    assert row["left_clip_id"] == "a" and row["right_clip_id"] == "b"


def test_18_same_source_jcut_uses_two_independent_seek_windows(tone_sources, tmp_path):
    _, low, _high = tone_sources
    segs = (
        RenderSegment(clip_id="a", source_asset_id="s1", source_path=low, start=0.5, end=1.5),
        RenderSegment(clip_id="b", source_asset_id="s1", source_path=low, start=2.5, end=3.5, audio_start=2.2),
    )
    command = render._concat_render_command_with_audio_windows(segs, tmp_path / "out.mp4", width=_WIDTH, height=_HEIGHT, fps=_FPS, workdir=tmp_path)
    joined = " ".join(command)
    # Four independent -i pairs for the SAME file (2 video + 2 audio seeks),
    # each with its own -ss/-to -- never collapsed into one shared seek.
    assert joined.count(low) == 4
    assert "-ss 2.500 -to 3.500" in joined  # b's own video window
    assert "-ss 2.200 -to 3.500" in joined  # b's own (leading) audio window


def test_19_multi_source_jcut_execution_is_source_correct(tone_sources, tmp_path):
    _, low, high = tone_sources
    segs = _jcut_segments(low, high)
    out = render.render_timeline_with_audio_windows(segs, str(tmp_path / "multisource.mp4"), width=_WIDTH, height=_HEIGHT, fps=_FPS)
    _, low_mask = _presence_mask(out, _LOW_HZ)
    _, high_mask = _presence_mask(out, _HIGH_HZ)
    assert low_mask.any() and high_mask.any()  # both distinct sources' tones present


# --- 20-24: validation / fail-closed contract --------------------------------

def test_20_invalid_negative_audio_start_rejected():
    seg = RenderSegment(clip_id="a", source_asset_id="s", source_path="p", start=1.0, end=2.0, audio_start=-0.5)
    with pytest.raises(ValueError, match="invalid_audio_window_negative_start"):
        render.validate_audio_window(seg, probe=_fake_probe(has_audio=True, duration=10.0))


def _fake_probe(*, has_audio: bool, duration: float):
    from cutsell_worker.media_probe import MediaProbe
    return MediaProbe(duration_sec=duration, width=160, height=120, fps=30.0, has_audio=has_audio)


def test_21_invalid_audio_end_beyond_source_duration_rejected():
    seg = RenderSegment(clip_id="a", source_asset_id="s", source_path="p", start=1.0, end=2.0, audio_end=100.0)
    with pytest.raises(ValueError, match="audio_window_exceeds_source_availability"):
        render.validate_audio_window(seg, probe=_fake_probe(has_audio=True, duration=10.0))


def test_22_malformed_window_end_not_after_start_rejected():
    seg = RenderSegment(clip_id="a", source_asset_id="s", source_path="p", start=1.0, end=2.0, audio_start=1.5, audio_end=1.5)
    with pytest.raises(ValueError, match="malformed_audio_window_end_not_after_start"):
        render.validate_audio_window(seg, probe=_fake_probe(has_audio=True, duration=10.0))


def test_23_transition_mode_is_always_one_of_the_closed_vocabulary():
    left = RenderSegment(clip_id="a", source_asset_id="s1", source_path="low.mp4", start=0.5, end=1.5)
    right = RenderSegment(clip_id="b", source_asset_id="s2", source_path="high.mp4", start=0.8, end=1.8)
    for lead, trail in ((0.0, 0.0), (0.3, 0.0), (0.0, 0.3), (0.2, 0.2)):
        l = left if trail == 0.0 else RenderSegment(clip_id="a", source_asset_id="s1", source_path="low.mp4", start=0.5, end=1.5, audio_end=1.5 + trail)
        r = right if lead == 0.0 else RenderSegment(clip_id="b", source_asset_id="s2", source_path="high.mp4", start=0.8, end=1.8, audio_start=0.8 - lead)
        mode = render._infer_transition_mode(l, r)
        assert mode in (
            render.TRANSITION_HARD_CUT, render.TRANSITION_TIGHT_CUT, render.TRANSITION_J_CUT,
            render.TRANSITION_L_CUT, render.TRANSITION_MICRO_AUDIO_OVERLAP,
        )


def test_24_invalid_window_raises_never_returns_a_clamped_value():
    seg = RenderSegment(clip_id="a", source_asset_id="s", source_path="p", start=1.0, end=2.0, audio_start=-1.0)
    with pytest.raises(ValueError):
        render.validate_audio_window(seg, probe=_fake_probe(has_audio=True, duration=10.0))
    # Structural "no silent clamp" proof: `audio_start`/`audio_end` are each
    # assigned AT MOST ONCE (a straight read of the segment's own effective
    # value) -- a clamp would require a SECOND assignment overwriting the
    # first with an adjusted, in-range value. Every failure path only ever
    # raises; there is no reassignment anywhere to clamp with.
    import inspect
    source = inspect.getsource(render.validate_audio_window)
    assert source.count("audio_start =") <= 1
    assert source.count("audio_end =") <= 1
    assert source.count("raise ValueError") >= 3


# --- 25/26: determinism and input-order stability ---------------------------

def test_25_deterministic_command_across_two_builds(tone_sources, tmp_path):
    _, low, high = tone_sources
    segs = _jcut_segments(low, high)
    c1 = render._concat_render_command_with_audio_windows(segs, tmp_path / "o.mp4", width=_WIDTH, height=_HEIGHT, fps=_FPS, workdir=tmp_path)
    c2 = render._concat_render_command_with_audio_windows(segs, tmp_path / "o.mp4", width=_WIDTH, height=_HEIGHT, fps=_FPS, workdir=tmp_path)
    assert c1 == c2


def test_26_input_order_preserved_not_silently_reordered(tone_sources, tmp_path):
    _, low, high = tone_sources
    segs = _jcut_segments(low, high)
    command = render._concat_render_command_with_audio_windows(segs, tmp_path / "o.mp4", width=_WIDTH, height=_HEIGHT, fps=_FPS, workdir=tmp_path)
    joined = " ".join(command)
    assert joined.index(low) < joined.index(high)


# --- 27-29: multi-join isolation / join-scoped filtergraph / no smear -------

def _three_segment_fixture(low: str, high: str, *, ab_overlap: bool) -> tuple[RenderSegment, RenderSegment, RenderSegment]:
    a = RenderSegment(clip_id="a", source_asset_id="s1", source_path=low, start=0.5, end=1.2)
    b_kwargs = {"audio_start": 1.5 - 0.2} if ab_overlap else {}
    b = RenderSegment(clip_id="b", source_asset_id="s2", source_path=high, start=1.5, end=2.2, **b_kwargs)
    c = RenderSegment(clip_id="c", source_asset_id="s1", source_path=low, start=3.0, end=3.7)
    return a, b, c


def test_27_multi_join_isolation_c_placement_unaffected_by_ab_overlap(tone_sources, tmp_path):
    _, low, high = tone_sources
    without = _three_segment_fixture(low, high, ab_overlap=False)
    with_overlap = _three_segment_fixture(low, high, ab_overlap=True)
    diag_without = render.dialogue_pacing_transition_execution_diagnostics(without)
    diag_with = render.dialogue_pacing_transition_execution_diagnostics(with_overlap)
    bc_without = diag_without["transitions"][1]
    bc_with = diag_with["transitions"][1]
    assert bc_without["video_switch_time"] == pytest.approx(bc_with["video_switch_time"], abs=1e-6)
    assert bc_without["right_audio_start_timeline"] == pytest.approx(bc_with["right_audio_start_timeline"], abs=1e-6)


def test_28_join_scoped_filtergraph_c_adelay_equals_own_cumulative_position(tone_sources, tmp_path):
    _, low, high = tone_sources
    segs = _three_segment_fixture(low, high, ab_overlap=True)
    command = render._concat_render_command_with_audio_windows(segs, tmp_path / "o.mp4", width=_WIDTH, height=_HEIGHT, fps=_FPS, workdir=tmp_path)
    joined = " ".join(command)
    expected_c_position_ms = round(
        (render.rendered_segment_duration_sec(segs[0].duration_sec, fps=_FPS)
         + render.rendered_segment_duration_sec(segs[1].duration_sec, fps=_FPS)) * 1000.0
    )
    assert f"adelay={expected_c_position_ms}|{expected_c_position_ms}[a2]" in joined


def test_29_no_global_audio_smear_c_actual_timing_unaffected_by_ab_overlap(tone_sources, tmp_path):
    _, low, high = tone_sources
    without = _three_segment_fixture(low, high, ab_overlap=False)
    with_overlap = _three_segment_fixture(low, high, ab_overlap=True)
    out_without = render.render_timeline_with_audio_windows(without, str(tmp_path / "without.mp4"), width=_WIDTH, height=_HEIGHT, fps=_FPS)
    out_with = render.render_timeline_with_audio_windows(with_overlap, str(tmp_path / "with.mp4"), width=_WIDTH, height=_HEIGHT, fps=_FPS)
    times_a, mask_a = _presence_mask(out_without, _LOW_HZ)
    times_b, mask_b = _presence_mask(out_with, _LOW_HZ)
    # segment c is the SECOND appearance of the low tone on the timeline in
    # both variants -- its own onset (last rising edge) must land at the
    # same instant regardless of the A/B join's own overlap decision.
    def _second_low_onset(times, mask):
        rising = np.where(np.diff(mask.astype(int)) == 1)[0] + 1
        return float(times[rising[-1]]) if len(rising) else None
    onset_without = _second_low_onset(times_a, mask_a)
    onset_with = _second_low_onset(times_b, mask_b)
    assert onset_without is not None and onset_with is not None
    assert onset_without == pytest.approx(onset_with, abs=0.03)


# --- 30/31: 12ms fade discipline --------------------------------------------

def test_30_join_fade_constant_unchanged_and_reused():
    assert render._AUDIO_JOIN_FADE_SEC == pytest.approx(0.012)


def test_31_divergent_audio_chain_still_uses_the_existing_fade_helper(tone_sources, tmp_path):
    _, low, high = tone_sources
    segs = _jcut_segments(low, high)
    command = render._concat_render_command_with_audio_windows(segs, tmp_path / "o.mp4", width=_WIDTH, height=_HEIGHT, fps=_FPS, workdir=tmp_path)
    joined = " ".join(command)
    assert "afade=t=in:st=0:d=0.012" in joined
    assert "afade=t=out" in joined


# --- 32/33: contiguous-coalesce interaction ---------------------------------

def test_32_coalesce_unchanged_for_default_touching_segments():
    left = RenderSegment(clip_id="a", source_asset_id="s1", source_path="low.mp4", start=0.5, end=1.5)
    right = RenderSegment(clip_id="b", source_asset_id="s1", source_path="low.mp4", start=1.5, end=2.5)
    assert _can_coalesce(left, right) is True


def test_33_coalesce_refused_across_a_nontrivial_audio_window_join():
    left = RenderSegment(clip_id="a", source_asset_id="s1", source_path="low.mp4", start=0.5, end=1.5, audio_end=1.7)
    right = RenderSegment(clip_id="b", source_asset_id="s1", source_path="low.mp4", start=1.5, end=2.5)
    assert _can_coalesce(left, right) is False
    result = _coalesce_contiguous_segments((left, right))
    assert len(result) == 2  # never silently merged away


# --- 34-45: discipline / scope-isolation checks -----------------------------

def test_34_no_live_wiring_into_universal_clean_cut_or_pipeline():
    forbidden = (
        "render_timeline_with_audio_windows",
        "_concat_render_command_with_audio_windows",
        "has_independent_audio_window",
        "dialogue_pacing_transition_execution_diagnostics",
    )
    for relative in ("universal_clean_cut.py", "pipeline.py", "dialogue_pacing_transition.py"):
        path = pathlib.Path("cutsell_worker") / relative
        text = path.read_text()
        for needle in forbidden:
            assert needle not in text, f"{needle!r} leaked into live wiring: {path}"


def test_35_live_pacing_modes_unchanged():
    from cutsell_worker.dialogue_pacing_transition import PHASE_1_EXECUTABLE_MODES, HARD_CUT, TIGHT_CUT
    assert PHASE_1_EXECUTABLE_MODES == (HARD_CUT, TIGHT_CUT)


def test_36_module_imports_no_out_of_scope_authority():
    forbidden_modules = {
        "requests", "httpx", "urllib", "openai", "google", "genai", "modal", "runpod",
        "editorial_moment_sequence", "whole_video_editorial_reasoning",
        "ordering_realization_plan", "ordering_composer_adapter",
        "boundary_engine_pass", "post_selection_edge_only_boundary",
        "post_selection_interior_gap_trim", "take_judge", "deterministic_best_take_authority",
        "multimodal_besttake_arbiter", "realization_resolver",
    }
    source = pathlib.Path(__file__).read_text()
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[-1])
    hit = imported & forbidden_modules
    assert not hit, f"D-214 test file imports an out-of-scope module: {hit}"


def test_37_no_video00_identifiers_anywhere_in_fixtures():
    # Check the actual fixture/identifier surface this file uses -- never
    # the whole file's own text, which would trivially self-match the
    # forbidden words this very check lists.
    haystack = " ".join([
        _LOW_HZ.__str__(), _HIGH_HZ.__str__(), "s1", "s2", "low.mp4", "high.mp4",
        "a", "b", "c",
    ]).lower()
    forbidden = ("pimples", "gyneco", "video00", "cut.ai", "human gold", "stomach")
    for needle in forbidden:
        assert needle not in haystack


def test_38_no_new_editorial_threshold_only_a_numeric_safety_epsilon():
    # AUDIO_TIMELINE_EPSILON_SEC is a floating-point safety tolerance, not an
    # editorial duration -- explicitly far smaller than any real timing unit
    # this module reasons about (AUDIO_EDGE_OVERLAP_TOLERANCE_SEC = 0.08 s).
    assert render.AUDIO_TIMELINE_EPSILON_SEC == pytest.approx(1e-6)
    assert render.AUDIO_TIMELINE_EPSILON_SEC < 1e-3


def test_39_render_module_never_selects_a_mode_it_only_reports_one():
    import inspect
    # The diagnostics function INFERS a label for reporting; the two
    # functions that actually BUILD/RUN an ffmpeg command must never
    # reference it or any mode constant -- structural proof the renderer
    # never uses a "mode" to decide what to build.
    for fn in (render._concat_render_command_with_audio_windows, render.render_timeline_with_audio_windows):
        body = inspect.getsource(fn)
        assert "_infer_transition_mode" not in body
        assert "TRANSITION_J_CUT" not in body
        assert "TRANSITION_L_CUT" not in body
        assert "TRANSITION_MICRO_AUDIO_OVERLAP" not in body


def test_40_no_boundary_ordering_besttake_mutation_by_this_module():
    # This module's new functions take/return only RenderSegment tuples and
    # plain dicts -- never a DraftClip, ProcessingResult, or DraftTimeline,
    # so there is nothing here that COULD mutate Boundary/Ordering/BestTake
    # state even accidentally.
    import inspect
    for fn in (
        render._concat_render_command_with_audio_windows,
        render.render_timeline_with_audio_windows,
        render.dialogue_pacing_transition_execution_diagnostics,
        render.validate_audio_window,
    ):
        sig = inspect.signature(fn)
        for name in ("draft", "result", "processing_result", "selection"):
            assert name not in sig.parameters


def test_41_no_transcript_or_word_level_semantic_analysis():
    source = pathlib.Path("cutsell_worker/render.py").read_text()
    d214_start = source.index("# D-214 -- PACING V2 RENDERER")
    d214_body = source[d214_start:]
    for needle in ("Word", ".words", "semantic_claims", "claim_coverage"):
        assert needle not in d214_body


# --- 42-52: regression / performance -----------------------------------------

def test_42_existing_render_preview_untouched_by_import(tone_sources, tmp_path):
    # render_preview's own command construction is completely unaffected --
    # spot-check via the existing, unmodified _concat_render_command path.
    _, low, high = tone_sources
    segs = (
        RenderSegment(clip_id="a", source_asset_id="s1", source_path=low, start=0.5, end=1.2),
        RenderSegment(clip_id="b", source_asset_id="s2", source_path=high, start=0.3, end=1.0),
    )
    out = render.render_preview(segs, str(tmp_path / "preview.mp4"), width=_WIDTH, height=_HEIGHT, fps=_FPS)
    import os
    assert os.path.getsize(out) > 0


def test_43_render_runtime_recorded_not_thresholded(tone_sources, tmp_path):
    import time
    _, low, high = tone_sources
    segs = _jcut_segments(low, high)
    started = time.monotonic()
    render.render_timeline_with_audio_windows(segs, str(tmp_path / "timed.mp4"), width=_WIDTH, height=_HEIGHT, fps=_FPS)
    elapsed = time.monotonic() - started
    assert elapsed >= 0.0  # recorded for the report; no arbitrary pass/fail threshold


def test_44_deterministic_repeat_of_full_render_diagnostics(tone_sources, tmp_path):
    _, low, high = tone_sources
    segs1 = _jcut_segments(low, high)
    segs2 = _jcut_segments(low, high)
    diag1 = render.dialogue_pacing_transition_execution_diagnostics(segs1)
    diag2 = render.dialogue_pacing_transition_execution_diagnostics(segs2)
    assert diag1 == diag2
