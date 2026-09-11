"""D-223 -- Pacing V2 Source Audio Handle Foundation. Offline only.

Fixture matrix proving `pacing_v2_source_audio_handle.py`'s own derivation,
firewalls, and status decision table -- including the D-222 root-cause
replay (the central architecture proof this task names) -- entirely from
already-computed Boundary-adjacent audit fixtures, never a real RAW, never
a renderer call, never a decision-layer call.
"""
from __future__ import annotations

import ast
import re

import pytest

from cutsell_worker.contracts import DraftClip, Word
from cutsell_worker import pacing_v2_source_audio_handle as sah


# ---------------------------------------------------------------------------
# Fixture helpers (same `_clip()`/`_words()` convention D-215's own test file
# established, reused across this whole track).
# ---------------------------------------------------------------------------

def _words(*specs) -> tuple:
    return tuple(Word(text=t, start=s, end=e) for (t, s, e) in specs)


def _clip(clip_id: str, source_asset_id: str, start: float, end: float, *, words=(), realization_id=None) -> DraftClip:
    return DraftClip(
        clip_id=clip_id, source_asset_id=source_asset_id, source_order=0,
        start=start, end=end, text="", caption_text="", words=words,
        realization_id=realization_id,
    )


def _audit_row(clip_id: str, original_start: float, original_end: float,
                result_start: float, result_end: float, actions=()) -> dict:
    return {
        "clip_id": clip_id, "original_start": original_start, "original_end": original_end,
        "result_start": result_start, "result_end": result_end, "actions": list(actions),
    }


def _dead_air_action(duration: float) -> dict:
    return {"action": "trim_locked_leading_non_speech_edge", "duration_sec": duration,
            "evidence": ["event:unintentional_dead_air:0.90"]}


def _retry_action(kind: str, duration: float = 0.5) -> dict:
    return {"action": "trim_locked_leading_non_speech_edge", "duration_sec": duration,
            "evidence": [f"event:{kind}:0.91"]}


# ===========================================================================
# 1-2: basic safe non-speech derivation (boundary_engine_pass provenance)
# ===========================================================================

def test_01_pre_roll_safe_non_speech_from_boundary_engine_pass():
    clip = _clip("c1", "s1", 2.0, 8.0)
    audit = (_audit_row("c1", 1.5, 8.0, 2.0, 8.0),)
    handle = sah.build_pre_roll_audio_handle(clip, boundary_engine_pass_audit=audit)
    assert handle.handle_status == sah.HANDLE_STATUS_SAFE_NON_SPEECH
    assert handle.handle_source_start == 1.5 and handle.handle_source_end == 2.0
    assert handle.available_duration == pytest.approx(0.5)
    assert handle.video_start == 2.0 and handle.video_end == 8.0


def test_02_post_roll_safe_non_speech_from_boundary_engine_pass():
    clip = _clip("c1", "s1", 2.0, 8.0)
    audit = (_audit_row("c1", 2.0, 8.5, 2.0, 8.0),)
    handle = sah.build_post_roll_audio_handle(clip, boundary_engine_pass_audit=audit)
    assert handle.handle_status == sah.HANDLE_STATUS_SAFE_NON_SPEECH
    assert handle.handle_source_start == 8.0 and handle.handle_source_end == 8.5
    assert handle.available_duration == pytest.approx(0.5)


# ===========================================================================
# 3-5: no provenance / no widening / stale mismatch -> UNAVAILABLE
# ===========================================================================

def test_03_no_provenance_recorded_is_unavailable():
    clip = _clip("c1", "s1", 2.0, 8.0)
    handle = sah.build_pre_roll_audio_handle(clip)
    assert handle.handle_status == sah.HANDLE_STATUS_UNAVAILABLE
    assert handle.available_duration == 0.0
    assert sah.CONFLICT_NO_PROVENANCE in handle.conflict_flags


def test_04_provenance_present_but_no_widening_is_unavailable():
    clip = _clip("c1", "s1", 2.0, 8.0)
    audit = (_audit_row("c1", 2.0, 8.0, 2.0, 8.0),)
    handle = sah.build_pre_roll_audio_handle(clip, boundary_engine_pass_audit=audit)
    assert handle.handle_status == sah.HANDLE_STATUS_UNAVAILABLE


def test_05_stale_provenance_mismatch_is_unavailable():
    clip = _clip("c1", "s1", 2.5, 8.0)  # clip.start moved since the audit was recorded
    audit = (_audit_row("c1", 1.5, 8.0, 2.0, 8.0),)
    handle = sah.build_pre_roll_audio_handle(clip, boundary_engine_pass_audit=audit)
    assert handle.handle_status == sah.HANDLE_STATUS_UNAVAILABLE
    assert sah.CONFLICT_STALE_PROVENANCE in handle.conflict_flags


# ===========================================================================
# 6-7: geometry / source-duration invariant
# ===========================================================================

def test_06_invalid_geometry_is_unavailable_never_clamped():
    clip = _clip("c1", "s1", 2.0, 8.0)
    audit = (_audit_row("c1", 2.1, 8.0, 2.0, 8.0),)  # original_start > result_start: malformed
    handle = sah.build_pre_roll_audio_handle(clip, boundary_engine_pass_audit=audit)
    # original_start(2.1) < result_start(2.0) is False -> not widened -> UNAVAILABLE via no-widening path
    assert handle.handle_status == sah.HANDLE_STATUS_UNAVAILABLE


def test_07_exceeds_source_duration_is_unavailable():
    clip = _clip("c1", "s1", 2.0, 8.0)
    audit = (_audit_row("c1", 2.0, 9.0, 2.0, 8.0),)
    handle = sah.build_post_roll_audio_handle(clip, boundary_engine_pass_audit=audit, source_duration_sec=8.5)
    assert handle.handle_status == sah.HANDLE_STATUS_UNAVAILABLE
    assert sah.CONFLICT_EXCEEDS_SOURCE_DURATION in handle.conflict_flags


# ===========================================================================
# 8-11: discarded-span / neighbor-selected-clip firewalls
# ===========================================================================

def test_08_pre_roll_blocked_by_discarded_overlap():
    clip = _clip("c1", "s1", 2.0, 8.0)
    discarded = (_clip("d1", "s1", 1.6, 1.9),)
    audit = (_audit_row("c1", 1.5, 8.0, 2.0, 8.0),)
    handle = sah.build_pre_roll_audio_handle(clip, boundary_engine_pass_audit=audit, discarded=discarded)
    assert handle.handle_status == sah.HANDLE_STATUS_BLOCKED_DISCARDED
    assert handle.discarded_overlap_status == sah.OVERLAP_DISCARDED
    assert sah.CONFLICT_OVERLAPS_DISCARDED in handle.conflict_flags


def test_09_post_roll_blocked_by_discarded_overlap():
    clip = _clip("c1", "s1", 2.0, 8.0)
    discarded = (_clip("d1", "s1", 8.2, 8.4),)
    audit = (_audit_row("c1", 2.0, 8.5, 2.0, 8.0),)
    handle = sah.build_post_roll_audio_handle(clip, boundary_engine_pass_audit=audit, discarded=discarded)
    assert handle.handle_status == sah.HANDLE_STATUS_BLOCKED_DISCARDED


def test_10_pre_roll_blocked_by_neighbor_selected_overlap():
    clip = _clip("c1", "s1", 2.0, 8.0)
    neighbor = _clip("c0", "s1", 1.4, 1.9)
    audit = (_audit_row("c1", 1.5, 8.0, 2.0, 8.0),)
    handle = sah.build_pre_roll_audio_handle(clip, boundary_engine_pass_audit=audit, other_selected=(neighbor,))
    assert handle.handle_status == sah.HANDLE_STATUS_BLOCKED_NEIGHBOR_SELECTED_CLIP
    assert handle.discarded_overlap_status == sah.OVERLAP_NEIGHBOR_SELECTED


def test_11_post_roll_blocked_by_neighbor_selected_overlap():
    clip = _clip("c1", "s1", 2.0, 8.0)
    neighbor = _clip("c2", "s1", 8.2, 9.0)
    audit = (_audit_row("c1", 2.0, 8.5, 2.0, 8.0),)
    handle = sah.build_post_roll_audio_handle(clip, boundary_engine_pass_audit=audit, other_selected=(neighbor,))
    assert handle.handle_status == sah.HANDLE_STATUS_BLOCKED_NEIGHBOR_SELECTED_CLIP


def test_12_discarded_from_a_different_source_asset_never_blocks():
    clip = _clip("c1", "s1", 2.0, 8.0)
    discarded = (_clip("d1", "s2", 1.6, 1.9),)  # different source_asset_id
    audit = (_audit_row("c1", 1.5, 8.0, 2.0, 8.0),)
    handle = sah.build_pre_roll_audio_handle(clip, boundary_engine_pass_audit=audit, discarded=discarded)
    assert handle.handle_status == sah.HANDLE_STATUS_SAFE_NON_SPEECH


# ===========================================================================
# 13-22: post_selection_edge_only_boundary evidence-kind classification
# (unintentional_dead_air safe; every retry/BTS kind blocked)
# ===========================================================================

def test_13_post_selection_dead_air_is_safe_non_speech():
    clip = _clip("c1", "s1", 2.0, 8.0)
    audit = (_audit_row("c1", 1.5, 8.0, 2.0, 8.0, actions=[_dead_air_action(0.5)]),)
    handle = sah.build_pre_roll_audio_handle(clip, post_selection_edge_only_boundary_audit=audit)
    assert handle.handle_status == sah.HANDLE_STATUS_SAFE_NON_SPEECH


@pytest.mark.parametrize("kind", [
    "retry_setup", "searching_for_words", "false_start", "wrong_take",
    "breaking_character", "camera_adjustment",
    "body_reset_candidate", "hand_motion_reset_candidate",
    "camera_disengagement_candidate", "facial_expression_shift_candidate",
])
def test_14_to_23_post_selection_retry_or_bts_kind_blocks(kind):
    clip = _clip("c1", "s1", 2.0, 8.0)
    audit = (_audit_row("c1", 1.5, 8.0, 2.0, 8.0, actions=[_retry_action(kind)]),)
    handle = sah.build_pre_roll_audio_handle(clip, post_selection_edge_only_boundary_audit=audit)
    assert handle.handle_status == sah.HANDLE_STATUS_BLOCKED_RETRY_OR_CORRECTION
    assert sah.CONFLICT_RETRY_OR_BTS_EVIDENCE in handle.conflict_flags
    assert any(kind in p for p in handle.provenance)


# ===========================================================================
# 24: both audit sources present for the same clip -> widest safe candidate
# used, never a crash, never silently ignored.
# ===========================================================================

def test_24_both_audit_sources_present_picks_widest_available():
    clip = _clip("c1", "s1", 2.0, 8.0)
    boundary = (_audit_row("c1", 1.8, 8.0, 2.0, 8.0),)
    post_selection = (_audit_row("c1", 1.5, 8.0, 2.0, 8.0, actions=[_dead_air_action(0.5)]),)
    handle = sah.build_pre_roll_audio_handle(
        clip, boundary_engine_pass_audit=boundary, post_selection_edge_only_boundary_audit=post_selection,
    )
    assert handle.handle_status == sah.HANDLE_STATUS_SAFE_NON_SPEECH
    assert handle.handle_source_start == 1.5  # widest candidate wins
    assert handle.available_duration == pytest.approx(0.5)


# ===========================================================================
# 25-28: broader_word_timings extensibility -- meaning/word-presence firewalls
# ===========================================================================

def test_25_broader_word_timings_non_critical_is_speech_present_not_authoritative():
    clip = _clip("c1", "s1", 2.0, 8.0)
    audit = (_audit_row("c1", 1.5, 8.0, 2.0, 8.0),)
    broader = {"s1": _words(("yeah", 1.6, 1.8))}
    handle = sah.build_pre_roll_audio_handle(
        clip, boundary_engine_pass_audit=audit, broader_word_timings=broader,
    )
    assert handle.speech_presence_status == sah.SPEECH_PRESENCE_WORDS_PRESENT
    assert handle.meaning_safety_status == sah.MEANING_SAFETY_SAFE
    assert handle.handle_status == sah.HANDLE_STATUS_SPEECH_PRESENT_NOT_AUTHORITATIVE


def test_26_broader_word_timings_negation_blocks_on_meaning():
    clip = _clip("c1", "s1", 2.0, 8.0)
    audit = (_audit_row("c1", 1.5, 8.0, 2.0, 8.0),)
    broader = {"s1": _words(("no", 1.6, 1.7), ("funciona", 1.7, 1.9))}
    handle = sah.build_pre_roll_audio_handle(
        clip, boundary_engine_pass_audit=audit, broader_word_timings=broader,
    )
    assert handle.meaning_safety_status == sah.MEANING_SAFETY_BLOCKED_CRITICAL
    assert handle.handle_status == sah.HANDLE_STATUS_BLOCKED_MEANING_CRITICAL
    assert sah.CONFLICT_MEANING_CRITICAL in handle.conflict_flags


def test_27_broader_word_timings_measurement_quantity_blocks_on_meaning():
    clip = _clip("c1", "s1", 2.0, 8.0)
    audit = (_audit_row("c1", 1.5, 8.0, 2.0, 8.0),)
    broader = {"s1": _words(("30", 1.6, 1.8), ("por", 1.8, 1.9), ("ciento", 1.9, 2.0))}
    handle = sah.build_pre_roll_audio_handle(
        clip, boundary_engine_pass_audit=audit, broader_word_timings=broader,
    )
    assert handle.handle_status == sah.HANDLE_STATUS_BLOCKED_MEANING_CRITICAL


def test_28_broader_word_timings_outside_window_never_counted():
    clip = _clip("c1", "s1", 2.0, 8.0)
    audit = (_audit_row("c1", 1.5, 8.0, 2.0, 8.0),)
    broader = {"s1": _words(("no", 0.0, 0.2))}  # entirely outside [1.5, 2.0)
    handle = sah.build_pre_roll_audio_handle(
        clip, boundary_engine_pass_audit=audit, broader_word_timings=broader,
    )
    assert handle.speech_presence_status == sah.SPEECH_PRESENCE_NO_WORDS
    assert handle.handle_status == sah.HANDLE_STATUS_SAFE_NON_SPEECH


# ===========================================================================
# 29: D-222 ROOT-CAUSE REPLAY -- the central architecture proof this task
# names explicitly: source 0.0->10.0, Boundary-finalized clip 2.0->8.0,
# safe non-speech pre-roll 1.5->2.0 and post-roll 8.0->8.5 in the ORIGINAL
# source, WITHOUT mutating clip.start/clip.end.
# ===========================================================================

def test_29_d222_root_cause_replay_central_proof():
    clip = _clip("c1", "s1", 2.0, 8.0)
    audit = (_audit_row("c1", 1.5, 8.5, 2.0, 8.0),)
    pre = sah.build_pre_roll_audio_handle(clip, boundary_engine_pass_audit=audit, source_duration_sec=10.0)
    post = sah.build_post_roll_audio_handle(clip, boundary_engine_pass_audit=audit, source_duration_sec=10.0)
    assert pre.available_duration == pytest.approx(0.5)
    assert post.available_duration == pytest.approx(0.5)
    assert pre.handle_status == sah.HANDLE_STATUS_SAFE_NON_SPEECH
    assert post.handle_status == sah.HANDLE_STATUS_SAFE_NON_SPEECH
    # clip.start/clip.end themselves are untouched -- the Boundary-finalized
    # VIDEO WINDOW is never mutated by this foundation.
    assert clip.start == 2.0 and clip.end == 8.0
    assert pre.video_start == 2.0 and pre.video_end == 8.0
    assert post.video_start == 2.0 and post.video_end == 8.0


# ===========================================================================
# 30-31: per-clip, not per-pair -- edge clips (no left/right neighbor) still
# get a correct handle on the side that has one.
# ===========================================================================

def test_30_first_clip_in_sequence_still_gets_pre_roll_handle():
    first = _clip("c1", "s1", 2.0, 8.0)
    second = _clip("c2", "s1", 8.0, 12.0)
    audit = (_audit_row("c1", 1.5, 8.0, 2.0, 8.0),)
    handles = sah.build_source_audio_handles((first, second), boundary_engine_pass_audit=audit)
    pre = [h for h in handles if h.owner_clip_id == "c1" and h.direction == sah.DIRECTION_PRE_ROLL][0]
    assert pre.handle_status == sah.HANDLE_STATUS_SAFE_NON_SPEECH


def test_31_last_clip_in_sequence_still_gets_post_roll_handle():
    first = _clip("c1", "s1", 2.0, 8.0)
    second = _clip("c2", "s1", 8.0, 12.0)
    audit = (_audit_row("c2", 8.0, 12.5, 8.0, 12.0),)
    handles = sah.build_source_audio_handles((first, second), boundary_engine_pass_audit=audit)
    post = [h for h in handles if h.owner_clip_id == "c2" and h.direction == sah.DIRECTION_POST_ROLL][0]
    assert post.handle_status == sah.HANDLE_STATUS_SAFE_NON_SPEECH


# ===========================================================================
# 32-33: build_source_audio_handles shape / run-summary aggregation
# ===========================================================================

def test_32_build_source_audio_handles_produces_two_per_clip():
    clips = tuple(_clip(f"c{i}", "s1", float(i * 3), float(i * 3 + 2)) for i in range(4))
    handles = sah.build_source_audio_handles(clips)
    assert len(handles) == 2 * len(clips)
    assert sum(1 for h in handles if h.direction == sah.DIRECTION_PRE_ROLL) == len(clips)
    assert sum(1 for h in handles if h.direction == sah.DIRECTION_POST_ROLL) == len(clips)


def test_33_run_summary_counts_are_consistent_no_master_score():
    clip_safe = _clip("c1", "s1", 2.0, 8.0)
    clip_blocked = _clip("c2", "s1", 20.0, 26.0)
    audit = (
        _audit_row("c1", 1.5, 8.0, 2.0, 8.0),
        _audit_row("c2", 19.5, 26.0, 20.0, 26.0, actions=[_retry_action("false_start")]),
    )
    handles = sah.build_source_audio_handles((clip_safe, clip_blocked), boundary_engine_pass_audit=audit)
    summary = sah.source_audio_handle_run_summary(handles)
    assert summary["pre_roll_handle_count"] == 2
    assert summary["post_roll_handle_count"] == 2
    assert summary["safe_non_speech_handle_count"] == 1
    assert summary["blocked_retry_or_correction_count"] == 1
    assert summary["total_safe_handle_duration"] == pytest.approx(0.5)
    assert summary["unavailable_count"] == 2  # both post-roll handles: no widening recorded


# ===========================================================================
# 34-36: handle identity -- stable, deterministic, never a random UUID
# ===========================================================================

def test_34_handle_id_deterministic_for_same_inputs():
    clip = _clip("c1", "s1", 2.0, 8.0)
    audit = (_audit_row("c1", 1.5, 8.0, 2.0, 8.0),)
    h1 = sah.build_pre_roll_audio_handle(clip, boundary_engine_pass_audit=audit)
    h2 = sah.build_pre_roll_audio_handle(clip, boundary_engine_pass_audit=audit)
    assert h1.handle_id == h2.handle_id


def test_35_handle_id_differs_for_different_geometry():
    clip = _clip("c1", "s1", 2.0, 8.0)
    audit_a = (_audit_row("c1", 1.5, 8.0, 2.0, 8.0),)
    audit_b = (_audit_row("c1", 1.2, 8.0, 2.0, 8.0),)
    h1 = sah.build_pre_roll_audio_handle(clip, boundary_engine_pass_audit=audit_a)
    h2 = sah.build_pre_roll_audio_handle(clip, boundary_engine_pass_audit=audit_b)
    assert h1.handle_id != h2.handle_id


def test_36_handle_id_is_never_a_random_uuid_shape():
    clip = _clip("c1", "s1", 2.0, 8.0)
    audit = (_audit_row("c1", 1.5, 8.0, 2.0, 8.0),)
    handle = sah.build_pre_roll_audio_handle(clip, boundary_engine_pass_audit=audit)
    uuid_re = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
    assert not uuid_re.match(handle.handle_id)
    assert handle.handle_id.startswith("handle:s1:c1:PRE_ROLL:")


# ===========================================================================
# 37: clip.start/clip.end are never mutated by any builder call
# ===========================================================================

def test_37_clip_start_end_never_mutated():
    clip = _clip("c1", "s1", 2.0, 8.0)
    audit = (_audit_row("c1", 1.5, 8.5, 2.0, 8.0),)
    before = (clip.start, clip.end)
    sah.build_pre_roll_audio_handle(clip, boundary_engine_pass_audit=audit)
    sah.build_post_roll_audio_handle(clip, boundary_engine_pass_audit=audit)
    assert (clip.start, clip.end) == before


# ===========================================================================
# 38-41: layering discipline -- AST-based structural audits (this track's
# own established convention: substring scans false-positive on docstring
# prose, so every "no X" check below inspects real import/identifier nodes).
# ===========================================================================

def _module_ast():
    import inspect
    source = inspect.getsource(sah)
    return ast.parse(source), source


def test_38_no_renderer_module_imported():
    tree, _ = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in ("render", "render_plan", ".render", ".render_plan")
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "render" not in alias.name


def test_39_no_decision_layer_or_timing_policy_imported():
    tree, _ = _module_ast()
    forbidden_modules = {
        "pacing_transition_decision", ".pacing_transition_decision",
        "pacing_v2_timing_policy", ".pacing_v2_timing_policy",
        "pacing_v2_evidence_adapter", ".pacing_v2_evidence_adapter",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in forbidden_modules


def test_40_no_j_cut_l_cut_identifier_defined_in_this_module():
    tree, _ = _module_ast()
    identifiers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            identifiers.add(node.name)
    assert "J_CUT" not in identifiers and "L_CUT" not in identifiers
    assert "decide_transition" not in identifiers


def test_41_no_master_or_global_score_identifier():
    tree, _ = _module_ast()
    for node in ast.walk(tree):
        name = None
        if isinstance(node, ast.Name):
            name = node.id
        elif isinstance(node, ast.arg):
            name = node.arg
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Attribute):
            name = node.attr
        if name:
            lowered = name.lower()
            assert "master" not in lowered
            assert "score" not in lowered


# ===========================================================================
# 42: SAFE_SPEECH_HANDLE is defined but never assigned by the builder --
# proven by direct decision-table sweep, including the WORD_COVERAGE_UNKNOWN
# combination the two current provenance sources never themselves produce
# (see `_decide_handle_status`'s own docstring).
# ===========================================================================

def test_42_decision_table_never_assigns_safe_speech_handle():
    overlaps = (sah.OVERLAP_NONE, sah.OVERLAP_DISCARDED, sah.OVERLAP_NEIGHBOR_SELECTED)
    retry_options = ((), ("retry_setup",))
    meanings = (
        sah.MEANING_SAFETY_NOT_APPLICABLE_NO_WORDS, sah.MEANING_SAFETY_SAFE,
        sah.MEANING_SAFETY_BLOCKED_CRITICAL, sah.MEANING_SAFETY_UNKNOWN,
    )
    presences = (sah.SPEECH_PRESENCE_NO_WORDS, sah.SPEECH_PRESENCE_WORDS_PRESENT, sah.SPEECH_PRESENCE_UNKNOWN)
    seen_statuses = set()
    for overlap in overlaps:
        for retry in retry_options:
            for meaning in meanings:
                for presence in presences:
                    status = sah._decide_handle_status(
                        overlap_status=overlap, retry_evidence_kinds=retry,
                        meaning_safety_status=meaning, speech_presence_status=presence,
                    )
                    seen_statuses.add(status)
                    assert status != sah.HANDLE_STATUS_SAFE_SPEECH
    # The WORD_COVERAGE_UNKNOWN outcome IS reachable in the decision table
    # (reserved for a future provenance source lacking today's positive
    # proof) even though today's two real provenance sources never produce
    # `SPEECH_PRESENCE_UNKNOWN` themselves.
    assert sah.HANDLE_STATUS_UNKNOWN_WORD_COVERAGE in seen_statuses
    assert sah.HANDLE_STATUS_SAFE_SPEECH in sah.CLOSED_HANDLE_STATUSES  # defined, just never assigned


def test_43_diagnostics_row_shape():
    clip = _clip("c1", "s1", 2.0, 8.0)
    audit = (_audit_row("c1", 1.5, 8.0, 2.0, 8.0),)
    handle = sah.build_pre_roll_audio_handle(clip, boundary_engine_pass_audit=audit)
    row = sah.source_audio_handle_diagnostics(handle)
    for key in (
        "handle_id", "direction", "source_asset_id", "owner_clip_id", "video_start", "video_end",
        "handle_start", "handle_end", "available_duration", "word_coverage_status",
        "words_present_count", "discarded_overlap_status", "meaning_safety_status",
        "handle_status", "conflict_flags", "provenance",
    ):
        assert key in row


def test_44_owner_realization_id_carried_when_present():
    clip = _clip("c1", "s1", 2.0, 8.0, realization_id="r_abc")
    audit = (_audit_row("c1", 1.5, 8.0, 2.0, 8.0),)
    handle = sah.build_pre_roll_audio_handle(clip, boundary_engine_pass_audit=audit)
    assert handle.owner_realization_id == "r_abc"


def test_45_cross_source_clips_never_compared():
    clip = _clip("c1", "s1", 2.0, 8.0)
    neighbor_other_source = _clip("cX", "s2", 1.0, 1.9)
    discarded_other_source = _clip("dX", "s2", 1.0, 1.9)
    audit = (_audit_row("c1", 1.5, 8.0, 2.0, 8.0),)
    handle = sah.build_pre_roll_audio_handle(
        clip, boundary_engine_pass_audit=audit,
        other_selected=(neighbor_other_source,), discarded=(discarded_other_source,),
    )
    assert handle.handle_status == sah.HANDLE_STATUS_SAFE_NON_SPEECH
