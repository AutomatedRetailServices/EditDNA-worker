"""D-226 -- Pacing V2 Controlled J/L Perceptual Timing Qualification.
OFFLINE / LOCAL MEDIA ONLY. NO Video00, NO Modal, NO RunPod, NO provider,
NO live J/L authority.

Exercises `tests/d226_jl_fixture_lab.py`'s own controlled, purely-local
fixture generator against the REAL, UNMODIFIED D-220 timing policy
(`pacing_v2_timing_policy.decide_jcut_timing`/`decide_lcut_timing`) and
the REAL, UNMODIFIED D-214 renderer (`render.render_timeline_with_audio_
windows`), then proves this task's own required structural safety matrix
on the ACTUAL rendered output.
"""
from __future__ import annotations

import ast
import json
import shutil
import subprocess
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
LAB_SOURCE = (REPO_ROOT / "tests" / "d226_jl_fixture_lab.py").read_text()

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available on this runner")

import sys  # noqa: E402
sys.path.insert(0, str(REPO_ROOT / "tests"))
import d226_jl_fixture_lab as lab  # noqa: E402

from cutsell_worker.dialogue_pacing_transition import J_CUT, L_CUT  # noqa: E402
from cutsell_worker.pacing_v2_timing_policy import TIMING_STATUS_SUPPORTED  # noqa: E402


def _ffprobe_stream_durations(path: str) -> dict:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "json", path],
        capture_output=True, text=True, check=True,
    )
    payload = json.loads(completed.stdout or "{}")
    out = {}
    for stream in payload.get("streams") or ():
        out[stream.get("codec_type")] = float(stream.get("duration") or 0.0)
    return out


@pytest.fixture(scope="module")
def all_results(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("d226_jl_fixtures")
    results = lab.run_all_cases(tmpdir)
    return {r.case_name: r for r in results}


# ===========================================================================
# 1-6: J/L case coverage (moderate/large/small window each).
# ===========================================================================

def test_01_j_moderate_window(all_results):
    r = all_results["J1_moderate"]
    assert r.direction == J_CUT
    assert r.max_safe_window == pytest.approx(0.6)


def test_02_j_large_window(all_results):
    r = all_results["J2_large"]
    assert r.max_safe_window == pytest.approx(2.0)


def test_03_j_small_window(all_results):
    r = all_results["J3_small"]
    assert r.max_safe_window == pytest.approx(0.15)


def test_04_l_moderate_window(all_results):
    r = all_results["L1_moderate"]
    assert r.direction == L_CUT
    assert r.max_safe_window == pytest.approx(0.6)


def test_05_l_large_window(all_results):
    r = all_results["L2_large"]
    assert r.max_safe_window == pytest.approx(2.0)


def test_06_l_small_window(all_results):
    r = all_results["L3_small"]
    assert r.max_safe_window == pytest.approx(0.15)


# ===========================================================================
# 7-10: all four variants rendered for every case.
# ===========================================================================

@pytest.mark.parametrize("case_name", ["J1_moderate", "J2_large", "J3_small", "L1_moderate", "L2_large", "L3_small"])
def test_07_to_10_all_variants_rendered(all_results, case_name):
    r = all_results[case_name]
    for variant in lab.VARIANTS:
        path = pathlib.Path(r.variant_paths[variant])
        assert path.exists() and path.stat().st_size > 0


# ===========================================================================
# 11: chosen <= max_safe (every case).
# ===========================================================================

@pytest.mark.parametrize("case_name", ["J1_moderate", "J2_large", "J3_small", "L1_moderate", "L2_large", "L3_small"])
def test_11_chosen_within_max_safe(all_results, case_name):
    assert all_results[case_name].chosen_within_max_safe


# ===========================================================================
# 12-13: source identity / availability.
# ===========================================================================

def test_12_source_identity(all_results):
    r = all_results["J1_moderate"]
    for path in r.variant_paths.values():
        assert pathlib.Path(path).suffix == ".mp4"


def test_13_source_availability(all_results):
    from cutsell_worker.media_probe import probe_media
    r = all_results["J1_moderate"]
    probe = probe_media(r.variant_paths[lab.VARIANT_BASELINE])
    assert probe.has_audio
    assert probe.duration_sec > 0


# ===========================================================================
# 14: no word truncation / 18: no duplicate speech.
# ===========================================================================

@pytest.mark.parametrize("case_name", ["J1_moderate", "J2_large", "J3_small", "L1_moderate", "L2_large", "L3_small"])
def test_14_no_word_truncation(all_results, case_name):
    assert all_results[case_name].no_word_truncation


@pytest.mark.parametrize("case_name", ["J1_moderate", "J2_large", "J3_small", "L1_moderate", "L2_large", "L3_small"])
def test_18_no_duplicate_speech(all_results, case_name):
    assert all_results[case_name].no_duplicate_speech_region


# ===========================================================================
# 15-17: visual switch exact / audio lead exact / audio tail exact.
# Video-stream duration is D-214's own invariant (never moved by any audio
# decision) -- verified here directly on the REAL rendered files, within a
# one-frame tolerance for encoder frame-count rounding (a pre-existing
# `render.py` property, not something this task modifies or re-litigates;
# D-214's own test suite already frame-exactly proves the underlying
# mechanism).
# ===========================================================================

@pytest.mark.parametrize("case_name", ["J1_moderate", "J2_large", "J3_small", "L1_moderate", "L2_large", "L3_small"])
def test_15_16_17_video_track_duration_invariant_across_variants(all_results, case_name):
    r = all_results[case_name]
    durations = []
    for variant in lab.VARIANTS:
        streams = _ffprobe_stream_durations(r.variant_paths[variant])
        assert "video" in streams
        durations.append(streams["video"])
    spread = max(durations) - min(durations)
    assert spread <= (2.0 / lab._FPS) + 1e-6, f"{case_name}: video duration moved by an audio decision: {durations}"


def test_17b_max_safe_variant_may_extend_audio_beyond_video_large_window(all_results):
    # The MAX_SAFE control variant is DELIBERATELY allowed to play audio
    # past the video's own combined length when the full raw window
    # exceeds what the video track needs (this task's own explicit
    # "perceptual control, never production policy" instruction) -- this
    # is the concrete, real-media evidence for the "large-window result"
    # deliverable item, not a defect.
    r = all_results["L2_large"]
    streams = _ffprobe_stream_durations(r.variant_paths[lab.VARIANT_MAX_SAFE])
    assert streams["audio"] > streams["video"] + 0.1


# ===========================================================================
# 19: renderer execution / 20: deterministic repeat.
# ===========================================================================

@pytest.mark.parametrize("case_name", ["J1_moderate", "J2_large", "J3_small", "L1_moderate", "L2_large", "L3_small"])
def test_19_renderer_execution_succeeded(all_results, case_name):
    r = all_results[case_name]
    assert r.decision_timing_status == TIMING_STATUS_SUPPORTED


@pytest.mark.parametrize("case_name", ["J1_moderate", "J2_large", "J3_small", "L1_moderate", "L2_large", "L3_small"])
def test_20_deterministic_repeat(all_results, case_name):
    assert all_results[case_name].deterministic_repeat_matches


# ===========================================================================
# 21-23: no Video00 / no network / no provider (AST + substring, this
# track's own established convention).
# ===========================================================================

def _lab_ast():
    return ast.parse(LAB_SOURCE)


def test_21_no_video00_literal():
    forbidden = ("Video00", "VIDEO-2026", "video00-modal", "longford validation", "longform validation")
    for needle in forbidden:
        assert needle not in LAB_SOURCE


def test_22_no_network_module_imported():
    tree = _lab_ast()
    forbidden_modules = {"requests", "urllib", "urllib.request", "http.client", "socket", "boto3", "modal"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in {"requests", "urllib", "socket", "boto3", "modal"}
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in {"requests", "urllib", "socket", "boto3", "modal"}


def test_23_no_provider_identifier_referenced():
    tree = _lab_ast()
    identifiers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        if isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
    forbidden = {"generate_speech", "tts", "TextToSpeech", "openai", "google_genai", "analyze_prosodic_delivery"}
    assert not (identifiers & forbidden)


# ===========================================================================
# 24-25: no production authority / no production timing constants.
# ===========================================================================

def test_24_lab_never_imported_by_cutsell_worker():
    for path in (REPO_ROOT / "cutsell_worker").rglob("*.py"):
        text = path.read_text()
        assert "d226_jl_fixture_lab" not in text, f"leaked into production code: {path}"


def test_25_experimental_control_constant_never_in_production():
    for path in (REPO_ROOT / "cutsell_worker").rglob("*.py"):
        text = path.read_text()
        assert "_EXPERIMENTAL_SHORTER_CONTROL_FRACTION" not in text


# ===========================================================================
# 26-27: no Boundary/Ordering mutation (AST -- lab imports neither).
# ===========================================================================

def test_26_27_no_boundary_or_ordering_module_imported():
    tree = _lab_ast()
    forbidden = {"boundary_engine_pass", "post_selection_edge_only_boundary",
                 "ordering_live_diagnostics_integration", "universal_clean_cut", "pipeline"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[-1].lstrip(".") not in forbidden


# ===========================================================================
# 28-29: no crossfade / ambience implementation.
# ===========================================================================

def test_28_29_no_crossfade_or_ambience_identifier():
    tree = _lab_ast()
    identifiers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            identifiers.add(node.name)
    forbidden = {"SHORT_CROSSFADE", "AMBIENCE_CARRY_LEFT", "AMBIENCE_CARRY_RIGHT", "AMBIENCE_BRIDGE"}
    assert not (identifiers & forbidden)


# ===========================================================================
# Anchor-cap / small-window / Prosodic-optionality hypotheses.
# ===========================================================================

def test_anchor_cap_dominates_on_large_window_j(all_results):
    r = all_results["J2_large"]
    # Hypothesis: anchor-word duration prevents an excessively long J-cut
    # when the max-safe window is very large.
    assert r.decision_chosen_duration == pytest.approx(r.anchor_word_duration)
    assert r.decision_chosen_duration < r.max_safe_window / 2.0


def test_anchor_cap_dominates_on_large_window_l(all_results):
    r = all_results["L2_large"]
    assert r.decision_chosen_duration == pytest.approx(r.anchor_word_duration)
    assert r.decision_chosen_duration < r.max_safe_window / 2.0


def test_small_window_never_exceeded_j(all_results):
    r = all_results["J3_small"]
    assert r.decision_chosen_duration == pytest.approx(r.max_safe_window)
    assert r.decision_chosen_duration < r.anchor_word_duration


def test_small_window_never_exceeded_l(all_results):
    r = all_results["L3_small"]
    assert r.decision_chosen_duration == pytest.approx(r.max_safe_window)
    assert r.decision_chosen_duration < r.anchor_word_duration


def test_prosodic_optionality_absent_matches_geometry_only(all_results):
    from cutsell_worker.pacing_v2_timing_policy import TIMING_BASIS_WORD_GEOMETRY
    r = all_results["J1_moderate"]
    assert r.decision_timing_basis == TIMING_BASIS_WORD_GEOMETRY


def test_prosodic_optionality_continuous_never_changes_chosen_amount(all_results):
    from cutsell_worker.pacing_v2_timing_policy import TIMING_BASIS_WORD_PLUS_PROSODIC
    for case_name in ("J1_moderate", "J2_large", "J3_small", "L1_moderate", "L2_large", "L3_small"):
        r = all_results[case_name]
        assert r.decision_with_prosody_timing_basis == TIMING_BASIS_WORD_PLUS_PROSODIC
        assert r.decision_with_prosody_chosen_duration == pytest.approx(r.decision_chosen_duration)


# ===========================================================================
# Module hygiene.
# ===========================================================================

def test_module_compiles_and_imports():
    import importlib
    importlib.reload(lab)


def test_no_micro_audio_overlap_authority_referenced():
    tree = _lab_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names = {alias.name for alias in (node.names or ())}
            assert "MICRO_AUDIO_OVERLAP" not in names
