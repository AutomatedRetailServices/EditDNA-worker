"""D-278: V1 Manual B-roll + Layered Audio/Voice-Over Composition
Foundation -- executor tests. Every test here runs REAL ffmpeg against
REAL, distinguishable-signal synthetic fixtures (Stage 45/46's own
"do not rely only on command inspection / exit code" instruction):
solid-color video (red base / blue broll-A / green broll-B) and
distinct sine-tone audio (440Hz primary / 880Hz broll-A / 990Hz
broll-B / 1320Hz VO-A / 1760Hz VO-B), verified by direct pixel and FFT
frequency measurement of the actual composed output.
"""
import ast
import inspect
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cutsell_worker import render as render_mod
from cutsell_worker import timeline_composition as tc
from cutsell_worker import timeline_composition_executor as tce


def _source_without_docstrings(obj) -> str:
    """Strip module/function/class docstrings before scanning for a
    literal, so this module's own prose explaining what it does NOT do
    (which legitimately mentions the forbidden words) never produces a
    false positive -- the same technique D-274F-A's own test suite
    established for exactly this failure mode."""
    tree = ast.parse(inspect.getsource(obj))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(getattr(node.body[0], "value", None), ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)

pytestmark_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available on this runner")

OUTPUT_WIDTH = tce.CANONICAL_OUTPUT_WIDTH
OUTPUT_HEIGHT = tce.CANONICAL_OUTPUT_HEIGHT


def _ffmpeg(args):
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args], check=True)


def _frame_rgb(path: str, t: float) -> np.ndarray:
    raw_path = str(Path(path).with_suffix(".rgb.tmp"))
    _ffmpeg(["-ss", f"{t}", "-i", path, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", raw_path])
    data = np.fromfile(raw_path, dtype=np.uint8)
    Path(raw_path).unlink(missing_ok=True)
    data = data[: OUTPUT_WIDTH * OUTPUT_HEIGHT * 3].reshape(OUTPUT_HEIGHT, OUTPUT_WIDTH, 3)
    return data.mean(axis=(0, 1))


def _dominant_color(rgb: np.ndarray) -> str:
    r, g, b = rgb
    if r > g and r > b:
        return "red"
    if g > r and g > b:
        return "green"
    if b > r and b > g:
        return "blue"
    return "unknown"


def _peak_freq(path: str, t: float, window: float = 0.4) -> float:
    raw_path = str(Path(path).with_suffix(".pcm.tmp"))
    _ffmpeg(["-ss", f"{t}", "-t", f"{window}", "-i", path, "-f", "s16le", "-ar", "48000", "-ac", "1", raw_path])
    data = np.fromfile(raw_path, dtype=np.int16).astype(np.float64)
    Path(raw_path).unlink(missing_ok=True)
    if len(data) < 64:
        return 0.0
    windowed = data * np.hanning(len(data))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(data), d=1 / 48000)
    return float(freqs[int(np.argmax(spectrum))])


def _sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# Stage 44/45/46 -- real, distinguishable-signal synthetic fixtures.
# =============================================================================

@pytest.fixture(scope="module")
def base_edit(tmp_path_factory):
    d = tmp_path_factory.mktemp("d278_base")
    path = str(d / "base.mp4")
    _ffmpeg(["-f", "lavfi", "-i", "color=c=red:s=640x360:d=10:r=30",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", path])
    return path


@pytest.fixture(scope="module")
def base_edit_no_audio(tmp_path_factory):
    d = tmp_path_factory.mktemp("d278_base_silent")
    path = str(d / "base_silent.mp4")
    _ffmpeg(["-f", "lavfi", "-i", "color=c=red:s=640x360:d=6:r=30",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", path])
    return path


@pytest.fixture(scope="module")
def broll_a(tmp_path_factory):
    d = tmp_path_factory.mktemp("d278_broll_a")
    path = str(d / "broll_a.mp4")
    _ffmpeg(["-f", "lavfi", "-i", "color=c=blue:s=640x360:d=4:r=30",
             "-f", "lavfi", "-i", "sine=frequency=880:duration=4",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", path])
    return path


@pytest.fixture(scope="module")
def broll_b(tmp_path_factory):
    d = tmp_path_factory.mktemp("d278_broll_b")
    path = str(d / "broll_b.mp4")
    _ffmpeg(["-f", "lavfi", "-i", "color=c=green:s=640x360:d=3:r=30",
             "-f", "lavfi", "-i", "sine=frequency=990:duration=3",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", path])
    return path


@pytest.fixture(scope="module")
def voice_over_a(tmp_path_factory):
    d = tmp_path_factory.mktemp("d278_vo_a")
    path = str(d / "vo_a.m4a")
    _ffmpeg(["-f", "lavfi", "-i", "sine=frequency=1320:duration=2", "-c:a", "aac", path])
    return path


@pytest.fixture(scope="module")
def voice_over_b(tmp_path_factory):
    d = tmp_path_factory.mktemp("d278_vo_b")
    path = str(d / "vo_b.m4a")
    _ffmpeg(["-f", "lavfi", "-i", "sine=frequency=1760:duration=2", "-c:a", "aac", path])
    return path


def _base_asset(base_edit, duration=10.0, has_audio=True):
    return tce.ResolvedTimelineAsset(asset_id="base", local_path=base_edit, duration_sec=duration, has_audio=has_audio)


def _broll_placement(placement_id, asset_id, start, end, src_in, src_out, mode, asset_duration):
    ref = tc.TimelineAssetReference(asset_id=asset_id, source_media_identity=f"s3://bucket/{asset_id}.mp4", duration_sec=asset_duration)
    return tc.BrollPlacement(placement_id=placement_id, asset=ref, timeline_start_sec=start, timeline_end_sec=end,
                              source_in_sec=src_in, source_out_sec=src_out, audio_mode=mode)


def _vo_placement(placement_id, asset_id, start, end, src_in, src_out, asset_duration):
    ref = tc.TimelineAssetReference(asset_id=asset_id, source_media_identity=f"s3://bucket/{asset_id}.m4a", duration_sec=asset_duration)
    return tc.VoiceOverPlacement(placement_id=placement_id, asset=ref, timeline_start_sec=start, timeline_end_sec=end,
                                  source_in_sec=src_in, source_out_sec=src_out)


def _resolved(asset_id, local_path, duration_sec, has_audio=True):
    return tce.ResolvedTimelineAsset(asset_id=asset_id, local_path=local_path, duration_sec=duration_sec, has_audio=has_audio)


def _run(composition, base_asset, broll_assets, vo_assets, tmp_path, timeout_sec=None):
    plan_result = tce.build_timeline_render_plan(composition, base_asset, broll_assets, vo_assets)
    return plan_result, (
        tce.execute_timeline_composition(plan_result.plan, output_directory=str(tmp_path), timeout_sec=timeout_sec)
        if plan_result.plan is not None else None
    )


# =============================================================================
# Stage 1 -- dedicated composition owner (structural checks)
# =============================================================================

def test_dedicated_composition_executor_module_never_imports_engine_authorities():
    source = _source_without_docstrings(tce)
    for forbidden in ("best_take", "pacing", "boundary", "freeze", "visual_finishing",
                       "audio_finishing_policy", "audio_join"):
        assert forbidden not in source.lower(), f"timeline_composition_executor.py must not import {forbidden}"


def test_no_shell_true_anywhere_in_executor():
    source = _source_without_docstrings(tce)
    assert "shell=True" not in source


def test_render_timeout_reused_by_reference_not_duplicated_literal():
    import inspect
    source = inspect.getsource(tce.execute_timeline_composition)
    assert "1200" not in source
    assert "render_mod.RENDER_FFMPEG_TIMEOUT_SEC" in source


def test_no_ai_placement_scoring_or_ranking_code_present():
    source = _source_without_docstrings(tce)
    for forbidden in ("similarity", "ranking", "auto_place", "sales_beat", "clip_score"):
        assert forbidden not in source.lower()


def test_executor_never_inspects_face_product_hands_or_scene_mode():
    source = _source_without_docstrings(tce)
    for forbidden in ("face", "product_bbox", "hand_bbox", "scene_mode", "visual_mode"):
        assert forbidden not in source.lower()


# =============================================================================
# Stage 38 -- base-only bypass
# =============================================================================

@pytestmark_ffmpeg
def test_base_only_bypass_returns_base_path_unchanged(base_edit, tmp_path):
    composition = tc.TimelineComposition(contract_version=tc.TIMELINE_CONTRACT_VERSION,
                                          base_edit_identity="ai_edit_1", timeline_duration_sec=10.0)
    base_asset = _base_asset(base_edit)
    plan_result, result = _run(composition, base_asset, {}, {}, tmp_path)
    assert plan_result.outcome == tce.PLAN_BUILD_SUCCEEDED
    assert result.outcome == tce.BASE_ONLY_BYPASS
    assert result.output_path == base_edit


@pytestmark_ffmpeg
def test_base_only_bypass_does_not_mutate_original(base_edit, tmp_path):
    before = _sha256(base_edit)
    composition = tc.TimelineComposition(contract_version=tc.TIMELINE_CONTRACT_VERSION,
                                          base_edit_identity="ai_edit_1", timeline_duration_sec=10.0)
    _run(composition, _base_asset(base_edit), {}, {}, tmp_path)
    assert _sha256(base_edit) == before


# =============================================================================
# Stage 6/7/8 -- the three B-roll audio modes, real visual+audio evidence
# =============================================================================

@pytestmark_ffmpeg
def test_one_broll_keep_primary_voice_visual_and_audio_evidence(base_edit, broll_a, tmp_path):
    broll = _broll_placement("b1", "broll_a", 2.0, 6.0, 0.0, 4.0, tc.TimelineAudioMode.KEEP_PRIMARY_VOICE, 4.0)
    composition = tc.TimelineComposition(contract_version=tc.TIMELINE_CONTRACT_VERSION,
                                          base_edit_identity="ai_edit_2", timeline_duration_sec=10.0,
                                          broll_placements=(broll,))
    _, result = _run(composition, _base_asset(base_edit), {"broll_a": _resolved("broll_a", broll_a, 4.0)}, {}, tmp_path)
    assert result.outcome == tce.COMPOSITION_SUCCEEDED
    out = result.output_path
    assert _dominant_color(_frame_rgb(out, 1.0)) == "red"     # before broll
    assert _dominant_color(_frame_rgb(out, 3.0)) == "blue"    # inside broll
    assert _dominant_color(_frame_rgb(out, 7.0)) == "red"     # after broll -- base resumes
    assert abs(_peak_freq(out, 1.0) - 440) < 15               # primary before
    assert abs(_peak_freq(out, 3.0) - 440) < 15               # KEEP_PRIMARY_VOICE: primary continues
    assert abs(_peak_freq(out, 7.0) - 440) < 15               # primary after


@pytestmark_ffmpeg
def test_one_broll_mute_broll_audio_visual_switches_audio_stays_primary(base_edit, broll_a, tmp_path):
    broll = _broll_placement("b1", "broll_a", 2.0, 6.0, 0.0, 4.0, tc.TimelineAudioMode.MUTE_BROLL_AUDIO, 4.0)
    composition = tc.TimelineComposition(contract_version=tc.TIMELINE_CONTRACT_VERSION,
                                          base_edit_identity="ai_edit_3", timeline_duration_sec=10.0,
                                          broll_placements=(broll,))
    _, result = _run(composition, _base_asset(base_edit), {"broll_a": _resolved("broll_a", broll_a, 4.0)}, {}, tmp_path)
    assert result.outcome == tce.COMPOSITION_SUCCEEDED
    out = result.output_path
    assert _dominant_color(_frame_rgb(out, 3.0)) == "blue"
    assert abs(_peak_freq(out, 3.0) - 440) < 15  # MUTE_BROLL_AUDIO: broll's own 880Hz never heard


@pytestmark_ffmpeg
def test_one_broll_use_broll_audio_visual_and_audio_switch(base_edit, broll_a, tmp_path):
    broll = _broll_placement("b1", "broll_a", 2.0, 6.0, 0.0, 4.0, tc.TimelineAudioMode.USE_BROLL_AUDIO, 4.0)
    composition = tc.TimelineComposition(contract_version=tc.TIMELINE_CONTRACT_VERSION,
                                          base_edit_identity="ai_edit_4", timeline_duration_sec=10.0,
                                          broll_placements=(broll,))
    _, result = _run(composition, _base_asset(base_edit), {"broll_a": _resolved("broll_a", broll_a, 4.0)}, {}, tmp_path)
    assert result.outcome == tce.COMPOSITION_SUCCEEDED
    out = result.output_path
    assert _dominant_color(_frame_rgb(out, 3.0)) == "blue"
    assert abs(_peak_freq(out, 1.0) - 440) < 15   # primary before window
    assert abs(_peak_freq(out, 3.0) - 880) < 15   # broll's own audio inside window
    assert abs(_peak_freq(out, 7.0) - 440) < 15   # primary resumes after


# =============================================================================
# Stage 24 -- multiple non-overlapping B-roll placements, one generation
# =============================================================================

@pytestmark_ffmpeg
def test_two_non_overlapping_broll_placements_one_generation(base_edit, broll_a, broll_b, tmp_path):
    b1 = _broll_placement("b1", "broll_a", 1.0, 3.0, 0.0, 2.0, tc.TimelineAudioMode.KEEP_PRIMARY_VOICE, 4.0)
    b2 = _broll_placement("b2", "broll_b", 6.0, 9.0, 0.0, 3.0, tc.TimelineAudioMode.USE_BROLL_AUDIO, 3.0)
    composition = tc.TimelineComposition(contract_version=tc.TIMELINE_CONTRACT_VERSION,
                                          base_edit_identity="ai_edit_5", timeline_duration_sec=10.0,
                                          broll_placements=(b1, b2))
    resolved = {"broll_a": _resolved("broll_a", broll_a, 4.0), "broll_b": _resolved("broll_b", broll_b, 3.0)}
    _, result = _run(composition, _base_asset(base_edit), resolved, {}, tmp_path)
    assert result.outcome == tce.COMPOSITION_SUCCEEDED
    out = result.output_path
    assert _dominant_color(_frame_rgb(out, 2.0)) == "blue"
    assert _dominant_color(_frame_rgb(out, 4.5)) == "red"
    assert _dominant_color(_frame_rgb(out, 7.5)) == "green"
    assert abs(_peak_freq(out, 7.5) - 990) < 15


# =============================================================================
# Stage 9/10 -- voice-over
# =============================================================================

@pytestmark_ffmpeg
def test_voice_over_over_primary_switches_audio_suppresses_primary(base_edit, voice_over_a, tmp_path):
    vo = _vo_placement("v1", "vo_a", 6.0, 8.0, 0.0, 2.0, 2.0)
    composition = tc.TimelineComposition(contract_version=tc.TIMELINE_CONTRACT_VERSION,
                                          base_edit_identity="ai_edit_6", timeline_duration_sec=10.0,
                                          voice_over_placements=(vo,))
    _, result = _run(composition, _base_asset(base_edit), {}, {"vo_a": _resolved("vo_a", voice_over_a, 2.0)}, tmp_path)
    assert result.outcome == tce.COMPOSITION_SUCCEEDED
    out = result.output_path
    assert abs(_peak_freq(out, 2.0) - 440) < 15    # primary before VO
    assert abs(_peak_freq(out, 7.0) - 1320) < 15   # VO active, primary suppressed
    assert abs(_peak_freq(out, 9.0) - 440) < 15    # primary resumes after VO
    # Stage 22: visual is untouched by a VO-only placement.
    assert _dominant_color(_frame_rgb(out, 7.0)) == "red"


# =============================================================================
# Stage 26/27 -- B-roll + VO same region: VO wins
# =============================================================================

@pytestmark_ffmpeg
def test_broll_keep_voice_and_vo_same_interval_vo_wins(base_edit, broll_a, voice_over_a, tmp_path):
    broll = _broll_placement("b1", "broll_a", 3.0, 7.0, 0.0, 4.0, tc.TimelineAudioMode.KEEP_PRIMARY_VOICE, 4.0)
    vo = _vo_placement("v1", "vo_a", 3.0, 5.0, 0.0, 2.0, 2.0)
    composition = tc.TimelineComposition(contract_version=tc.TIMELINE_CONTRACT_VERSION,
                                          base_edit_identity="ai_edit_7", timeline_duration_sec=10.0,
                                          broll_placements=(broll,), voice_over_placements=(vo,))
    resolved_broll = {"broll_a": _resolved("broll_a", broll_a, 4.0)}
    resolved_vo = {"vo_a": _resolved("vo_a", voice_over_a, 2.0)}
    _, result = _run(composition, _base_asset(base_edit), resolved_broll, resolved_vo, tmp_path)
    assert result.outcome == tce.COMPOSITION_SUCCEEDED
    out = result.output_path
    assert _dominant_color(_frame_rgb(out, 4.0)) == "blue"   # B-roll visual still shows
    assert abs(_peak_freq(out, 4.0) - 1320) < 15             # VOICE_OVER wins over KEEP_PRIMARY_VOICE
    assert abs(_peak_freq(out, 6.0) - 440) < 15              # after VO, back to primary (still inside broll window)


@pytestmark_ffmpeg
def test_broll_use_audio_and_vo_conflict_vo_wins(base_edit, broll_a, voice_over_a, tmp_path):
    broll = _broll_placement("b1", "broll_a", 3.0, 7.0, 0.0, 4.0, tc.TimelineAudioMode.USE_BROLL_AUDIO, 4.0)
    vo = _vo_placement("v1", "vo_a", 3.0, 5.0, 0.0, 2.0, 2.0)
    composition = tc.TimelineComposition(contract_version=tc.TIMELINE_CONTRACT_VERSION,
                                          base_edit_identity="ai_edit_8", timeline_duration_sec=10.0,
                                          broll_placements=(broll,), voice_over_placements=(vo,))
    resolved_broll = {"broll_a": _resolved("broll_a", broll_a, 4.0)}
    resolved_vo = {"vo_a": _resolved("vo_a", voice_over_a, 2.0)}
    _, result = _run(composition, _base_asset(base_edit), resolved_broll, resolved_vo, tmp_path)
    assert result.outcome == tce.COMPOSITION_SUCCEEDED
    out = result.output_path
    assert abs(_peak_freq(out, 4.0) - 1320) < 15   # VOICE_OVER wins over BROLL_SOURCE_AUDIO
    assert abs(_peak_freq(out, 6.0) - 880) < 15    # after VO ends, broll's own USE_BROLL_AUDIO resumes


# =============================================================================
# Stage 25 -- multiple VO segments
# =============================================================================

@pytestmark_ffmpeg
def test_multiple_voice_over_segments(base_edit, voice_over_a, voice_over_b, tmp_path):
    vo1 = _vo_placement("v1", "vo_a", 1.0, 3.0, 0.0, 2.0, 2.0)
    vo2 = _vo_placement("v2", "vo_b", 6.0, 8.0, 0.0, 2.0, 2.0)
    composition = tc.TimelineComposition(contract_version=tc.TIMELINE_CONTRACT_VERSION,
                                          base_edit_identity="ai_edit_9", timeline_duration_sec=10.0,
                                          voice_over_placements=(vo1, vo2))
    resolved_vo = {"vo_a": _resolved("vo_a", voice_over_a, 2.0), "vo_b": _resolved("vo_b", voice_over_b, 2.0)}
    _, result = _run(composition, _base_asset(base_edit), {}, resolved_vo, tmp_path)
    assert result.outcome == tce.COMPOSITION_SUCCEEDED
    out = result.output_path
    assert abs(_peak_freq(out, 2.0) - 1320) < 15
    assert abs(_peak_freq(out, 4.5) - 440) < 15
    assert abs(_peak_freq(out, 7.0) - 1760) < 15


# =============================================================================
# Stage 11 -- caption authority (pure computation, no ASR, no rendering)
# =============================================================================

def test_caption_authority_original_when_nothing_covers_region():
    composition = tc.TimelineComposition(contract_version=tc.TIMELINE_CONTRACT_VERSION,
                                          base_edit_identity="x", timeline_duration_sec=10.0)
    regions = tce.resolve_caption_regions(composition)
    assert regions == (tce.CaptionRegion(0.0, 10.0, "ORIGINAL_PRIMARY_VOICE"),)


def test_caption_authority_broll_source_audio_region():
    broll = _broll_placement("b1", "a1", 2.0, 5.0, 0.0, 3.0, tc.TimelineAudioMode.USE_BROLL_AUDIO, 3.0)
    composition = tc.TimelineComposition(contract_version=tc.TIMELINE_CONTRACT_VERSION,
                                          base_edit_identity="x", timeline_duration_sec=10.0,
                                          broll_placements=(broll,))
    regions = tce.resolve_caption_regions(composition)
    assert any(r.authority == "BROLL_SOURCE_AUDIO" and r.start_sec == 2.0 and r.end_sec == 5.0 for r in regions)


def test_caption_authority_voice_over_region_wins_over_broll():
    broll = _broll_placement("b1", "a1", 2.0, 6.0, 0.0, 4.0, tc.TimelineAudioMode.USE_BROLL_AUDIO, 4.0)
    vo = _vo_placement("v1", "vo1", 3.0, 5.0, 0.0, 2.0, 2.0)
    composition = tc.TimelineComposition(contract_version=tc.TIMELINE_CONTRACT_VERSION,
                                          base_edit_identity="x", timeline_duration_sec=10.0,
                                          broll_placements=(broll,), voice_over_placements=(vo,))
    regions = tce.resolve_caption_regions(composition)
    assert any(r.authority == "VOICE_OVER" and r.start_sec == 3.0 and r.end_sec == 5.0 for r in regions)


def test_no_caption_text_ever_fabricated():
    """Stage 13: CaptionRegion carries only an authority label, never a
    text field."""
    fields = tce.CaptionRegion.__dataclass_fields__.keys()
    assert set(fields) == {"start_sec", "end_sec", "authority"}


# =============================================================================
# Stage 19/28/29 -- moved/trimmed/replaced B-roll, identity behavior
# =============================================================================

@pytestmark_ffmpeg
def test_moved_broll_shows_visual_at_new_position(base_edit, broll_a, tmp_path):
    broll = _broll_placement("b1", "broll_a", 2.0, 6.0, 0.0, 4.0, tc.TimelineAudioMode.KEEP_PRIMARY_VOICE, 4.0)
    composition, _ = tc.add_broll(
        tc.TimelineComposition(contract_version=tc.TIMELINE_CONTRACT_VERSION,
                                base_edit_identity="ai_edit_10", timeline_duration_sec=10.0),
        broll,
    )
    moved, move_result = tc.move_broll(composition, "b1", timeline_start_sec=6.0, timeline_end_sec=10.0)
    assert move_result.valid
    resolved = {"broll_a": _resolved("broll_a", broll_a, 4.0)}
    _, result = _run(moved, _base_asset(base_edit), resolved, {}, tmp_path)
    assert result.outcome == tce.COMPOSITION_SUCCEEDED
    out = result.output_path
    assert _dominant_color(_frame_rgb(out, 3.0)) == "red"    # old position now shows base again
    assert _dominant_color(_frame_rgb(out, 8.0)) == "blue"   # new position shows broll


@pytestmark_ffmpeg
def test_trimmed_broll_composition_uses_new_source_window(base_edit, broll_a, tmp_path):
    broll = _broll_placement("b1", "broll_a", 2.0, 6.0, 0.0, 4.0, tc.TimelineAudioMode.KEEP_PRIMARY_VOICE, 4.0)
    composition, _ = tc.add_broll(
        tc.TimelineComposition(contract_version=tc.TIMELINE_CONTRACT_VERSION,
                                base_edit_identity="ai_edit_11", timeline_duration_sec=10.0),
        broll,
    )
    trimmed, trim_result = tc.trim_broll(composition, "b1", source_in_sec=1.0, source_out_sec=3.0,
                                          timeline_start_sec=2.0, timeline_end_sec=4.0)
    assert trim_result.valid
    resolved = {"broll_a": _resolved("broll_a", broll_a, 4.0)}
    _, result = _run(trimmed, _base_asset(base_edit), resolved, {}, tmp_path)
    assert result.outcome == tce.COMPOSITION_SUCCEEDED
    assert _dominant_color(_frame_rgb(result.output_path, 3.0)) == "blue"
    assert _dominant_color(_frame_rgb(result.output_path, 5.0)) == "red"  # broll now ends at 4.0, not 6.0


@pytestmark_ffmpeg
def test_replaced_broll_asset_composition(base_edit, broll_a, broll_b, tmp_path):
    broll = _broll_placement("b1", "broll_a", 2.0, 5.0, 0.0, 3.0, tc.TimelineAudioMode.KEEP_PRIMARY_VOICE, 4.0)
    composition, _ = tc.add_broll(
        tc.TimelineComposition(contract_version=tc.TIMELINE_CONTRACT_VERSION,
                                base_edit_identity="ai_edit_12", timeline_duration_sec=10.0),
        broll,
    )
    new_placement = _broll_placement("b1", "broll_b", 2.0, 5.0, 0.0, 3.0, tc.TimelineAudioMode.KEEP_PRIMARY_VOICE, 3.0)
    replaced, replace_result = tc.replace_broll(composition, "b1", new_placement)
    assert replace_result.valid
    resolved = {"broll_b": _resolved("broll_b", broll_b, 3.0)}
    _, result = _run(replaced, _base_asset(base_edit), resolved, {}, tmp_path)
    assert result.outcome == tce.COMPOSITION_SUCCEEDED
    assert _dominant_color(_frame_rgb(result.output_path, 3.0)) == "green"  # broll_b's own color, not broll_a's blue


# =============================================================================
# Stage 39 -- executor must not bypass D-277 validation
# =============================================================================

def test_invalid_overlap_rejected_at_plan_build():
    composition = tc.TimelineComposition(
        contract_version=tc.TIMELINE_CONTRACT_VERSION, base_edit_identity="x", timeline_duration_sec=10.0,
        broll_placements=(
            _broll_placement("b1", "a1", 0.0, 5.0, 0.0, 5.0, tc.TimelineAudioMode.KEEP_PRIMARY_VOICE, 5.0),
            _broll_placement("b2", "a1", 3.0, 8.0, 0.0, 5.0, tc.TimelineAudioMode.KEEP_PRIMARY_VOICE, 5.0),
        ),
    )
    plan_result = tce.build_timeline_render_plan(composition, _resolved("base", "x.mp4", 10.0), {"a1": _resolved("a1", "a.mp4", 5.0)}, {})
    assert plan_result.outcome == tce.TIMELINE_INVALID
    assert plan_result.plan is None


def test_invalid_bounds_rejected_at_plan_build():
    composition = tc.TimelineComposition(
        contract_version=tc.TIMELINE_CONTRACT_VERSION, base_edit_identity="x", timeline_duration_sec=5.0,
        broll_placements=(_broll_placement("b1", "a1", 0.0, 8.0, 0.0, 8.0, tc.TimelineAudioMode.KEEP_PRIMARY_VOICE, 8.0),),
    )
    plan_result = tce.build_timeline_render_plan(composition, _resolved("base", "x.mp4", 5.0), {"a1": _resolved("a1", "a.mp4", 8.0)}, {})
    assert plan_result.outcome == tce.TIMELINE_INVALID


def test_missing_broll_asset_rejected():
    composition = tc.TimelineComposition(
        contract_version=tc.TIMELINE_CONTRACT_VERSION, base_edit_identity="x", timeline_duration_sec=10.0,
        broll_placements=(_broll_placement("b1", "a1", 0.0, 3.0, 0.0, 3.0, tc.TimelineAudioMode.KEEP_PRIMARY_VOICE, 3.0),),
    )
    plan_result = tce.build_timeline_render_plan(composition, _resolved("base", "x.mp4", 10.0), {}, {})
    assert plan_result.outcome == tce.ASSET_MISSING


def test_missing_voice_over_asset_rejected():
    composition = tc.TimelineComposition(
        contract_version=tc.TIMELINE_CONTRACT_VERSION, base_edit_identity="x", timeline_duration_sec=10.0,
        voice_over_placements=(_vo_placement("v1", "vo1", 0.0, 2.0, 0.0, 2.0, 2.0),),
    )
    plan_result = tce.build_timeline_render_plan(composition, _resolved("base", "x.mp4", 10.0), {}, {})
    assert plan_result.outcome == tce.ASSET_MISSING


def test_broll_duration_mismatch_rejected():
    composition = tc.TimelineComposition(
        contract_version=tc.TIMELINE_CONTRACT_VERSION, base_edit_identity="x", timeline_duration_sec=10.0,
        broll_placements=(_broll_placement("b1", "a1", 0.0, 3.0, 0.0, 2.0, tc.TimelineAudioMode.KEEP_PRIMARY_VOICE, 2.0),),
    )
    plan_result = tce.build_timeline_render_plan(composition, _resolved("base", "x.mp4", 10.0), {"a1": _resolved("a1", "a.mp4", 2.0)}, {})
    assert plan_result.outcome == tce.BROLL_DURATION_MISMATCH


def test_voiceover_duration_mismatch_rejected():
    composition = tc.TimelineComposition(
        contract_version=tc.TIMELINE_CONTRACT_VERSION, base_edit_identity="x", timeline_duration_sec=10.0,
        voice_over_placements=(_vo_placement("v1", "vo1", 0.0, 3.0, 0.0, 1.0, 1.0),),
    )
    plan_result = tce.build_timeline_render_plan(composition, _resolved("base", "x.mp4", 10.0), {}, {"vo1": _resolved("vo1", "vo.m4a", 1.0)})
    assert plan_result.outcome == tce.VOICEOVER_DURATION_MISMATCH


def test_resolved_asset_duration_mismatch_rejected():
    """The caller's own resolved-asset duration must agree with the
    duration the TimelineAssetReference in the placement itself
    claims -- a stale/incorrect resolution is rejected, not trusted."""
    composition = tc.TimelineComposition(
        contract_version=tc.TIMELINE_CONTRACT_VERSION, base_edit_identity="x", timeline_duration_sec=10.0,
        broll_placements=(_broll_placement("b1", "a1", 0.0, 3.0, 0.0, 3.0, tc.TimelineAudioMode.KEEP_PRIMARY_VOICE, 3.0),),
    )
    plan_result = tce.build_timeline_render_plan(composition, _resolved("base", "x.mp4", 10.0), {"a1": _resolved("a1", "a.mp4", 99.0)}, {})
    assert plan_result.outcome == tce.ASSET_BOUNDS_INVALID


# =============================================================================
# Paths with spaces / unicode (Stage 44 items 25/26)
# =============================================================================

@pytestmark_ffmpeg
def test_base_only_bypass_with_path_containing_spaces(tmp_path_factory, tmp_path):
    d = tmp_path_factory.mktemp("d278_space")
    path = str(d / "my base edit.mp4")
    _ffmpeg(["-f", "lavfi", "-i", "color=c=red:s=640x360:d=3:r=30",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", path])
    composition = tc.TimelineComposition(contract_version=tc.TIMELINE_CONTRACT_VERSION,
                                          base_edit_identity="x", timeline_duration_sec=3.0)
    _, result = _run(composition, _base_asset(path, duration=3.0), {}, {}, tmp_path)
    assert result.outcome == tce.BASE_ONLY_BYPASS
    assert result.output_path == path


@pytestmark_ffmpeg
def test_broll_composition_with_unicode_asset_path(tmp_path_factory, tmp_path):
    d = tmp_path_factory.mktemp("d278_unicode")
    base_path = str(d / "base.mp4")
    broll_path = str(d / "b-roll_café_日本語.mp4")
    _ffmpeg(["-f", "lavfi", "-i", "color=c=red:s=640x360:d=6:r=30",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", base_path])
    _ffmpeg(["-f", "lavfi", "-i", "color=c=blue:s=640x360:d=2:r=30",
             "-f", "lavfi", "-i", "sine=frequency=880:duration=2",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", broll_path])
    broll = _broll_placement("b1", "u1", 1.0, 3.0, 0.0, 2.0, tc.TimelineAudioMode.KEEP_PRIMARY_VOICE, 2.0)
    composition = tc.TimelineComposition(contract_version=tc.TIMELINE_CONTRACT_VERSION,
                                          base_edit_identity="x", timeline_duration_sec=6.0,
                                          broll_placements=(broll,))
    resolved = {"u1": _resolved("u1", broll_path, 2.0)}
    _, result = _run(composition, _base_asset(base_path, duration=6.0), resolved, {}, tmp_path)
    assert result.outcome == tce.COMPOSITION_SUCCEEDED
    assert _dominant_color(_frame_rgb(result.output_path, 2.0)) == "blue"


# =============================================================================
# Original-asset preservation, format QC, identity, timeout
# =============================================================================

@pytestmark_ffmpeg
def test_original_assets_unchanged_after_composition(base_edit, broll_a, voice_over_a, tmp_path):
    base_sha_before, broll_sha_before, vo_sha_before = _sha256(base_edit), _sha256(broll_a), _sha256(voice_over_a)
    broll = _broll_placement("b1", "broll_a", 1.0, 3.0, 0.0, 2.0, tc.TimelineAudioMode.KEEP_PRIMARY_VOICE, 4.0)
    vo = _vo_placement("v1", "vo_a", 5.0, 7.0, 0.0, 2.0, 2.0)
    composition = tc.TimelineComposition(contract_version=tc.TIMELINE_CONTRACT_VERSION,
                                          base_edit_identity="x", timeline_duration_sec=10.0,
                                          broll_placements=(broll,), voice_over_placements=(vo,))
    resolved_broll = {"broll_a": _resolved("broll_a", broll_a, 4.0)}
    resolved_vo = {"vo_a": _resolved("vo_a", voice_over_a, 2.0)}
    _, result = _run(composition, _base_asset(base_edit), resolved_broll, resolved_vo, tmp_path)
    assert result.outcome == tce.COMPOSITION_SUCCEEDED
    assert _sha256(base_edit) == base_sha_before
    assert _sha256(broll_a) == broll_sha_before
    assert _sha256(voice_over_a) == vo_sha_before


@pytestmark_ffmpeg
def test_output_format_qc_pass(base_edit, broll_a, tmp_path):
    broll = _broll_placement("b1", "broll_a", 1.0, 3.0, 0.0, 2.0, tc.TimelineAudioMode.KEEP_PRIMARY_VOICE, 4.0)
    composition = tc.TimelineComposition(contract_version=tc.TIMELINE_CONTRACT_VERSION,
                                          base_edit_identity="x", timeline_duration_sec=10.0,
                                          broll_placements=(broll,))
    _, result = _run(composition, _base_asset(base_edit), {"broll_a": _resolved("broll_a", broll_a, 4.0)}, {}, tmp_path)
    assert result.outcome == tce.COMPOSITION_SUCCEEDED
    assert result.diagnostics["format_qc_status"] == "PASS"


def test_plan_identity_deterministic_for_identical_semantic_state():
    composition = tc.TimelineComposition(
        contract_version=tc.TIMELINE_CONTRACT_VERSION, base_edit_identity="x", timeline_duration_sec=10.0,
        broll_placements=(_broll_placement("b1", "a1", 0.0, 3.0, 0.0, 3.0, tc.TimelineAudioMode.KEEP_PRIMARY_VOICE, 3.0),),
    )
    resolved = {"a1": _resolved("a1", "a.mp4", 3.0)}
    r1 = tce.build_timeline_render_plan(composition, _resolved("base", "x.mp4", 10.0), resolved, {})
    r2 = tce.build_timeline_render_plan(composition, _resolved("base", "y.mp4", 10.0), resolved, {})  # different local path
    assert r1.plan.plan_identity == r2.plan.plan_identity  # Stage 29: never derived from a local path


def test_timeline_edit_changes_plan_identity():
    composition = tc.TimelineComposition(
        contract_version=tc.TIMELINE_CONTRACT_VERSION, base_edit_identity="x", timeline_duration_sec=10.0,
        broll_placements=(_broll_placement("b1", "a1", 0.0, 3.0, 0.0, 3.0, tc.TimelineAudioMode.KEEP_PRIMARY_VOICE, 3.0),),
    )
    moved, _ = tc.move_broll(composition, "b1", timeline_start_sec=5.0, timeline_end_sec=8.0)
    resolved = {"a1": _resolved("a1", "a.mp4", 3.0)}
    r1 = tce.build_timeline_render_plan(composition, _resolved("base", "x.mp4", 10.0), resolved, {})
    r2 = tce.build_timeline_render_plan(moved, _resolved("base", "x.mp4", 10.0), resolved, {})
    assert r1.plan.plan_identity != r2.plan.plan_identity


@pytestmark_ffmpeg
def test_asset_with_no_audio_falls_back_to_silence_not_a_crash(base_edit_no_audio, broll_a, tmp_path):
    broll = _broll_placement("b1", "broll_a", 1.0, 3.0, 0.0, 2.0, tc.TimelineAudioMode.KEEP_PRIMARY_VOICE, 4.0)
    composition = tc.TimelineComposition(contract_version=tc.TIMELINE_CONTRACT_VERSION,
                                          base_edit_identity="x", timeline_duration_sec=6.0,
                                          broll_placements=(broll,))
    base_asset = _base_asset(base_edit_no_audio, duration=6.0, has_audio=False)
    _, result = _run(composition, base_asset, {"broll_a": _resolved("broll_a", broll_a, 4.0)}, {}, tmp_path)
    assert result.outcome == tce.COMPOSITION_SUCCEEDED
    # Contract still requires a (silent) stereo audio stream -- confirmed by
    # the existing format-QC pass, not re-asserted redundantly here.
    assert result.diagnostics["format_qc_status"] == "PASS"


@pytestmark_ffmpeg
def test_composition_uses_canonical_render_timeout_by_default(base_edit, broll_a, tmp_path):
    broll = _broll_placement("b1", "broll_a", 1.0, 3.0, 0.0, 2.0, tc.TimelineAudioMode.KEEP_PRIMARY_VOICE, 4.0)
    composition = tc.TimelineComposition(contract_version=tc.TIMELINE_CONTRACT_VERSION,
                                          base_edit_identity="x", timeline_duration_sec=10.0,
                                          broll_placements=(broll,))
    _, result = _run(composition, _base_asset(base_edit), {"broll_a": _resolved("broll_a", broll_a, 4.0)}, {}, tmp_path)
    assert result.diagnostics["timeout_sec"] == render_mod.RENDER_FFMPEG_TIMEOUT_SEC == 1200.0
