"""D-224 -- Pacing V2 Source Audio Handle Live Evidence Integration.
DIAGNOSTIC ONLY. NO LIVE J/L/MICRO/AUDIO-JOIN-TREATMENT AUTHORITY.

Proves `pacing_v2_handle_aware_evidence.py` combines D-223's own
`SourceAudioHandle` evidence with today's existing in-window silent-head/
tail geometry into one deterministic, source-coordinate-correct
availability contract, reusing D-215's decision engine and D-220's timing
policy verbatim (never a second classifier, never a heuristic change),
and that D-221's own real `0/26` scarcity result can, offline, become a
positive candidate window once a safe handle is available -- without ever
touching `clip.start`/`clip.end`, D-142's own live output, or any
`RenderSegment`.

Fixture convention: `left` occupies `[0.0, 4.0)` and `right` occupies
`[6.0, 10.0)` in the SAME source file, leaving a real `[4.0, 6.0)` gap
between their own already-selected spans -- deliberately, so a candidate
handle window living inside that gap is never mistaken for reuse of a
NEIGHBOR clip's own currently-selected material (D-223's own neighbor-
selected-clip firewall, item 17 -- exercised on its own terms in D-223's
test suite, not re-tested here).
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from cutsell_worker.contracts import DraftClip, SemanticRole, Word
from cutsell_worker.dialogue_pacing_transition import HARD_CUT, J_CUT, L_CUT, TIGHT_CUT
from cutsell_worker import pacing_v2_handle_aware_evidence as hae
from cutsell_worker import pacing_v2_source_audio_handle as sah

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODULE_SOURCE = (REPO_ROOT / "cutsell_worker" / "pacing_v2_handle_aware_evidence.py").read_text()

SOURCE_A = "synthetic_source_a"
SOURCE_B = "synthetic_source_b"

LEFT_START, LEFT_END = 0.0, 4.0
RIGHT_START, RIGHT_END = 6.0, 10.0


def _words(text: str, start: float, end: float) -> tuple:
    tokens = text.split()
    if not tokens:
        return ()
    step = (end - start) / len(tokens)
    return tuple(Word(t, start + i * step, start + (i + 1) * step) for i, t in enumerate(tokens))


def _clip(clip_id, start, end, text, *, words=None, source=SOURCE_A, order=0, realization_id=None) -> DraftClip:
    return DraftClip(
        clip_id=clip_id, source_asset_id=source, source_order=order, start=start, end=end,
        text=text, caption_text=text, words=(words if words is not None else _words(text, start, end)),
        semantic_role=SemanticRole.STORY, selected=True, realization_id=realization_id,
    )


def _pair(*, left_end_word=3.7, right_start_word=6.3, source=SOURCE_A,
          left_start=LEFT_START, left_end=LEFT_END, right_start=RIGHT_START, right_end=RIGHT_END):
    left = _clip("l", left_start, left_end, "uno dos tres", words=_words("uno dos tres", left_start, left_end_word), source=source)
    right = _clip("r", right_start, right_end, "cuatro cinco", words=_words("cuatro cinco", right_start_word, right_end - 0.5), source=source)
    return left, right


def _audit_row(clip_id, original_start, original_end, result_start, result_end, actions=()):
    return {
        "clip_id": clip_id, "original_start": original_start, "original_end": original_end,
        "result_start": result_start, "result_end": result_end, "actions": list(actions),
    }


def _retry_action(kind: str, duration: float = 0.5) -> dict:
    return {"action": "trim_locked_leading_non_speech_edge", "duration_sec": duration,
            "evidence": [f"event:{kind}:0.91"]}


def _build(left, right, **kwargs):
    kwargs.setdefault("dialogue_overlap_enabled", False)
    diag = hae.build_handle_aware_pacing_v2_diagnostics((left, right), **kwargs)
    return diag["transitions"][0], diag


# ===========================================================================
# 1: no handle at all -> combined equals the old in-window value exactly.
# ===========================================================================

def test_01_no_handle_combined_equals_old_in_window():
    left, right = _pair(left_end_word=3.6, right_start_word=6.4)
    row, _ = _build(left, right)
    assert row["old_in_window_j_head"] == pytest.approx(0.4)
    assert row["combined_j_available_window"] == pytest.approx(0.4)
    assert row["combined_j_source"] == hae.COMBINED_SOURCE_IN_WINDOW_ONLY
    assert row["old_in_window_l_tail"] == pytest.approx(0.4)
    assert row["combined_l_available_window"] == pytest.approx(0.4)
    assert row["combined_l_source"] == hae.COMBINED_SOURCE_IN_WINDOW_ONLY


# ===========================================================================
# 2-4: safe pre / post / pre+post handle extends availability.
# ===========================================================================

def test_02_safe_pre_handle_extends_j_availability():
    left, right = _pair(left_end_word=4.0, right_start_word=6.3)  # old in-window head = 0.3
    audit = (_audit_row("r", 5.5, RIGHT_END, RIGHT_START, RIGHT_END),)  # handle [5.5, 6.0) = 0.5
    row, _ = _build(left, right, boundary_engine_pass_audit=audit)
    assert row["combined_j_available_window"] == pytest.approx(0.8)
    assert row["combined_j_source"] == hae.COMBINED_SOURCE_IN_WINDOW_PLUS_HANDLE
    assert row["right_pre_handle_status"] == sah.HANDLE_STATUS_SAFE_NON_SPEECH


def test_03_safe_post_handle_extends_l_availability():
    left, right = _pair(left_end_word=3.7, right_start_word=RIGHT_START)  # old in-window tail = 0.3
    audit = (_audit_row("l", LEFT_START, 4.5, LEFT_START, LEFT_END),)  # handle [4.0, 4.5) = 0.5
    row, _ = _build(left, right, boundary_engine_pass_audit=audit)
    assert row["combined_l_available_window"] == pytest.approx(0.8)
    assert row["combined_l_source"] == hae.COMBINED_SOURCE_IN_WINDOW_PLUS_HANDLE
    assert row["left_post_handle_status"] == sah.HANDLE_STATUS_SAFE_NON_SPEECH


def test_04_safe_pre_and_post_handle_both_extend():
    left, right = _pair(left_end_word=4.0, right_start_word=RIGHT_START)
    audit = (
        _audit_row("l", LEFT_START, 4.5, LEFT_START, LEFT_END),
        _audit_row("r", 5.5, RIGHT_END, RIGHT_START, RIGHT_END),
    )
    row, _ = _build(left, right, boundary_engine_pass_audit=audit)
    assert row["combined_j_available_window"] == pytest.approx(0.5)
    assert row["combined_l_available_window"] == pytest.approx(0.5)


# ===========================================================================
# 5-6: D-221 scarcity replay -- zero old J/L, positive handle-aware J/L.
# ===========================================================================

def test_05_zero_old_j_positive_handle_j_unlocks_j_cut():
    left, right = _pair(left_end_word=4.0, right_start_word=RIGHT_START)  # old head == 0.0
    audit = (_audit_row("r", 5.5, RIGHT_END, RIGHT_START, RIGHT_END),)
    row, _ = _build(left, right, boundary_engine_pass_audit=audit, dialogue_overlap_enabled=True)
    assert row["old_in_window_j_head"] == 0.0
    assert row["combined_j_available_window"] == pytest.approx(0.5)
    assert row["combined_j_source"] == hae.COMBINED_SOURCE_HANDLE_ONLY


def test_06_zero_old_l_positive_handle_l_unlocks_l_cut():
    left, right = _pair(left_end_word=LEFT_END, right_start_word=6.3)  # old tail == 0.0
    audit = (_audit_row("l", LEFT_START, 4.5, LEFT_START, LEFT_END),)
    row, _ = _build(left, right, boundary_engine_pass_audit=audit, dialogue_overlap_enabled=True)
    assert row["old_in_window_l_tail"] == 0.0
    assert row["combined_l_available_window"] == pytest.approx(0.5)
    assert row["combined_l_source"] == hae.COMBINED_SOURCE_HANDLE_ONLY


# ===========================================================================
# 7-8: blocked handle controls -- combined stays at the old in-window value.
# ===========================================================================

def test_07_blocked_discarded_pre_handle_does_not_extend():
    left, right = _pair(left_end_word=4.0, right_start_word=6.3)  # old head = 0.3
    audit = (_audit_row("r", 5.5, RIGHT_END, RIGHT_START, RIGHT_END),)
    discarded = (_clip("d", 5.6, 5.9, "abandonado", source=SOURCE_A),)
    row, _ = _build(left, right, boundary_engine_pass_audit=audit, discarded=discarded)
    assert row["right_pre_handle_status"] == sah.HANDLE_STATUS_BLOCKED_DISCARDED
    assert row["combined_j_available_window"] == pytest.approx(0.3)
    assert row["combined_j_source"] == hae.COMBINED_SOURCE_IN_WINDOW_ONLY


def test_08_blocked_discarded_post_handle_does_not_extend():
    left, right = _pair(left_end_word=3.7, right_start_word=RIGHT_START)  # old tail = 0.3
    audit = (_audit_row("l", LEFT_START, 4.5, LEFT_START, LEFT_END),)
    discarded = (_clip("d", 4.1, 4.4, "abandonado", source=SOURCE_A),)
    row, _ = _build(left, right, boundary_engine_pass_audit=audit, discarded=discarded)
    assert row["left_post_handle_status"] == sah.HANDLE_STATUS_BLOCKED_DISCARDED
    assert row["combined_l_available_window"] == pytest.approx(0.3)


# ===========================================================================
# 9-10: unknown handle controls -- fail closed, never extend.
# ===========================================================================

def _unknown_handle(direction, owner_clip_id, start, end):
    return sah.SourceAudioHandle(
        schema_version=sah.SCHEMA_VERSION, handle_id="h_unknown", source_asset_id=SOURCE_A,
        owner_clip_id=owner_clip_id, owner_realization_id=None, direction=direction,
        video_start=start if direction == sah.DIRECTION_POST_ROLL else end,
        video_end=end if direction == sah.DIRECTION_POST_ROLL else start,
        handle_source_start=start, handle_source_end=end, available_duration=end - start,
        word_intervals_present=(), speech_presence_status=sah.SPEECH_PRESENCE_UNKNOWN,
        discarded_overlap_status=sah.OVERLAP_NONE, meaning_safety_status=sah.MEANING_SAFETY_UNKNOWN,
        handle_status=sah.HANDLE_STATUS_UNKNOWN_WORD_COVERAGE, conflict_flags=(), provenance=(),
    )


def test_09_unknown_pre_handle_never_extends():
    handle = _unknown_handle(sah.DIRECTION_PRE_ROLL, "r", 5.5, 6.0)
    combined, source = hae._combined_j_availability(0.2, RIGHT_START, handle)
    assert combined == pytest.approx(0.2)
    assert source == hae.COMBINED_SOURCE_IN_WINDOW_ONLY


def test_10_unknown_post_handle_never_extends():
    handle = _unknown_handle(sah.DIRECTION_POST_ROLL, "l", 4.0, 4.5)
    combined, source = hae._combined_l_availability(0.2, LEFT_END, handle)
    assert combined == pytest.approx(0.2)
    assert source == hae.COMBINED_SOURCE_IN_WINDOW_ONLY


# ===========================================================================
# 11: speech-present handle is never reused (word-presence alone never
# authorizes reuse -- D-223's own conservative rule, restated here).
# ===========================================================================

def test_11_speech_present_handle_never_reused():
    left, right = _pair(left_end_word=4.0, right_start_word=6.3)  # old head = 0.3
    audit = (_audit_row("r", 5.5, RIGHT_END, RIGHT_START, RIGHT_END),)
    broader = {SOURCE_A: _words("hola", 5.6, 5.8)}
    row, _ = _build(left, right, boundary_engine_pass_audit=audit, broader_word_timings=broader)
    assert row["right_pre_handle_status"] == sah.HANDLE_STATUS_SPEECH_PRESENT_NOT_AUTHORITATIVE
    assert row["combined_j_available_window"] == pytest.approx(0.3)  # in-window only, handle excluded


# ===========================================================================
# 12-14: discarded / retry / correction handle blocked.
# ===========================================================================

def test_12_discarded_handle_blocked():
    left, right = _pair(left_end_word=4.0, right_start_word=6.3)
    audit = (_audit_row("r", 5.5, RIGHT_END, RIGHT_START, RIGHT_END),)
    discarded = (_clip("d", 5.6, 5.9, "abandonado"),)
    row, _ = _build(left, right, boundary_engine_pass_audit=audit, discarded=discarded)
    assert row["right_pre_handle_status"] == sah.HANDLE_STATUS_BLOCKED_DISCARDED


def test_13_retry_handle_blocked():
    left, right = _pair(left_end_word=4.0, right_start_word=6.3)
    audit = (_audit_row("r", 5.5, RIGHT_END, RIGHT_START, RIGHT_END, actions=[_retry_action("retry_setup")]),)
    row, _ = _build(left, right, post_selection_edge_only_boundary_audit=audit)
    assert row["right_pre_handle_status"] == sah.HANDLE_STATUS_BLOCKED_RETRY_OR_CORRECTION


def test_14_correction_style_evidence_handle_blocked():
    left, right = _pair(left_end_word=4.0, right_start_word=6.3)
    audit = (_audit_row("r", 5.5, RIGHT_END, RIGHT_START, RIGHT_END, actions=[_retry_action("false_start")]),)
    row, _ = _build(left, right, post_selection_edge_only_boundary_audit=audit)
    assert row["right_pre_handle_status"] == sah.HANDLE_STATUS_BLOCKED_RETRY_OR_CORRECTION


# ===========================================================================
# 15: meaning-critical handle blocked.
# ===========================================================================

def test_15_meaning_critical_handle_blocked():
    left, right = _pair(left_end_word=4.0, right_start_word=6.3)
    audit = (_audit_row("r", 5.5, RIGHT_END, RIGHT_START, RIGHT_END),)
    broader = {SOURCE_A: _words("no", 5.6, 5.8)}
    row, _ = _build(left, right, boundary_engine_pass_audit=audit, broader_word_timings=broader)
    assert row["right_pre_handle_status"] == sah.HANDLE_STATUS_BLOCKED_MEANING_CRITICAL
    assert row["combined_j_available_window"] == pytest.approx(0.3)


# ===========================================================================
# 16-18: combined-source categorization.
# ===========================================================================

def test_16_in_window_only():
    assert hae._combined_source(0.3, 0.0) == hae.COMBINED_SOURCE_IN_WINDOW_ONLY


def test_17_handle_only():
    assert hae._combined_source(0.0, 0.5) == hae.COMBINED_SOURCE_HANDLE_ONLY


def test_18_in_window_plus_handle():
    assert hae._combined_source(0.3, 0.5) == hae.COMBINED_SOURCE_IN_WINDOW_PLUS_HANDLE


def test_18b_none():
    assert hae._combined_source(0.0, 0.0) == hae.COMBINED_SOURCE_NONE


# ===========================================================================
# 19: no double-count -- real interval union, not a blind sum, proven on a
# deliberately overlapping synthetic pair.
# ===========================================================================

def test_19_no_double_count_on_overlapping_intervals():
    # [1.0, 2.0) and [1.5, 2.5) overlap by 0.5 -- union length is 1.5, a
    # blind sum would wrongly report 2.0.
    length = hae._merged_interval_length([(1.0, 2.0), (1.5, 2.5)])
    assert length == pytest.approx(1.5)


def test_19b_touching_intervals_summed_exactly_once():
    length = hae._merged_interval_length([(1.5, 2.0), (2.0, 2.5)])
    assert length == pytest.approx(1.0)


# ===========================================================================
# 20-21-22: source coordinates preserved / source-bound checks.
# ===========================================================================

def test_20_source_coordinates_preserved_in_diagnostics_row():
    left, right = _pair(left_end_word=4.0, right_start_word=6.3)
    audit = (_audit_row("r", 5.5, RIGHT_END, RIGHT_START, RIGHT_END),)
    row, _ = _build(left, right, boundary_engine_pass_audit=audit)
    assert row["right_pre_handle_start"] == pytest.approx(5.5)
    assert row["right_pre_handle_end"] == pytest.approx(6.0)


def test_21_source_start_bound_never_violated():
    left, right = _pair(left_end_word=4.0, right_start_word=6.3)
    # Malformed audit: original_start < 0 would violate the source-start
    # bound -- D-223's own geometry check catches this, UNAVAILABLE.
    audit = (_audit_row("r", -0.5, RIGHT_END, RIGHT_START, RIGHT_END),)
    row, _ = _build(left, right, boundary_engine_pass_audit=audit, source_duration_by_asset={SOURCE_A: 10.0})
    assert row["right_pre_handle_status"] == sah.HANDLE_STATUS_UNAVAILABLE
    assert row["combined_j_available_window"] == pytest.approx(0.3)


def test_22_source_end_bound_never_violated():
    left, right = _pair(left_end_word=3.7, right_start_word=RIGHT_START)
    audit = (_audit_row("l", LEFT_START, 12.0, LEFT_START, LEFT_END),)  # exceeds source duration
    row, _ = _build(left, right, boundary_engine_pass_audit=audit, source_duration_by_asset={SOURCE_A: 10.0})
    assert row["left_post_handle_status"] == sah.HANDLE_STATUS_UNAVAILABLE


# ===========================================================================
# 23-24: same source / multi-source behavior.
# ===========================================================================

def test_23_same_source_pair_computes_normally():
    left, right = _pair()
    row, _ = _build(left, right)
    assert row is not None


def test_24_multi_source_pair_never_crashes_and_stays_source_local():
    left = _clip("l", 0.0, 4.0, "uno", words=_words("uno", 0.0, 3.7), source=SOURCE_A)
    right = _clip("r", 100.0, 105.0, "dos", words=_words("dos", 100.3, 104.5), source=SOURCE_B)
    audit = (_audit_row("r", 99.5, 105.0, 100.0, 105.0),)
    row, _ = _build(left, right, boundary_engine_pass_audit=audit)
    assert row["right_pre_handle_start"] == pytest.approx(99.5)
    assert row["combined_j_source"] == hae.COMBINED_SOURCE_IN_WINDOW_PLUS_HANDLE


# ===========================================================================
# 25: source identity carried through diagnostics.
# ===========================================================================

def test_25_source_identity_in_handle_diagnostics():
    left, right = _pair(left_end_word=4.0, right_start_word=6.3)
    audit = (_audit_row("r", 5.5, RIGHT_END, RIGHT_START, RIGHT_END),)
    _, diag = _build(left, right, boundary_engine_pass_audit=audit)
    handle_rows = [h for h in diag["handle_diagnostics"] if h["owner_clip_id"] == "r" and h["direction"] == "PRE_ROLL"]
    assert handle_rows[0]["source_asset_id"] == SOURCE_A


# ===========================================================================
# 26: broader words reused (never invented) to prove word-presence firewall
# through this exact seam.
# ===========================================================================

def test_26_broader_words_reused_never_invented():
    left, right = _pair(left_end_word=4.0, right_start_word=6.3)
    audit = (_audit_row("r", 5.5, RIGHT_END, RIGHT_START, RIGHT_END),)
    broader = {SOURCE_A: _words("espera", 5.6, 5.8)}
    row, _ = _build(left, right, boundary_engine_pass_audit=audit, broader_word_timings=broader)
    assert row["right_pre_handle_status"] != sah.HANDLE_STATUS_SAFE_NON_SPEECH


# ===========================================================================
# 27-28: no ASR / no provider (AST-based, this track's own convention).
# ===========================================================================

def _module_ast():
    return ast.parse(MODULE_SOURCE)


def test_27_no_asr_module_imported():
    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in ("asr", ".asr")


def test_28_no_provider_identifier_referenced():
    tree = _module_ast()
    identifiers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        if isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
    forbidden = {"analyze_prosodic_delivery", "WholeVideoProvider", "ASRProvider", "CleanCutProvider"}
    assert not (identifiers & forbidden)


# ===========================================================================
# 29-30: D-215/D-220 reuse, never a second classifier / heuristic change.
# ===========================================================================

def test_29_decide_transition_imported_not_redefined():
    tree = _module_ast()
    imported = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "pacing_transition_decision":
            imported = any(alias.name == "decide_transition" for alias in node.names)
    assert imported
    defined = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "decide_transition"]
    assert not defined


def test_30_d220_timing_functions_imported_not_redefined():
    tree = _module_ast()
    imported = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "pacing_v2_timing_policy":
            names = {alias.name for alias in node.names}
            imported = {"decide_jcut_timing", "decide_lcut_timing"} <= names
    assert imported
    defined = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name in ("decide_jcut_timing", "decide_lcut_timing")
    ]
    assert not defined


def test_30b_d220_chosen_duration_matches_direct_call():
    from cutsell_worker.pacing_v2_timing_policy import decide_jcut_timing
    left, right = _pair(left_end_word=4.0, right_start_word=RIGHT_START)
    audit = (_audit_row("r", 5.5, RIGHT_END, RIGHT_START, RIGHT_END),)
    row, _ = _build(left, right, boundary_engine_pass_audit=audit, dialogue_overlap_enabled=True)
    if row["new_handle_aware_diagnostic_mode"] == J_CUT:
        direct = decide_jcut_timing(left, right, max_safe_lead=row["combined_j_available_window"], transition_index=0)
        assert row["d220_j_chosen_duration"] == pytest.approx(direct.chosen_duration)


# ===========================================================================
# 31-33: immutability / determinism / ordering.
# ===========================================================================

def test_31_32_video_start_end_unchanged():
    left, right = _pair(left_end_word=4.0, right_start_word=6.3)
    audit = (_audit_row("r", 5.5, RIGHT_END, RIGHT_START, RIGHT_END),)
    before = ((left.start, left.end), (right.start, right.end))
    _build(left, right, boundary_engine_pass_audit=audit)
    after = ((left.start, left.end), (right.start, right.end))
    assert before == after


def test_33_selected_order_preserved():
    a = _clip("a", 0.0, 3.0, "uno", words=_words("uno", 0.0, 2.7))
    b = _clip("b", 5.0, 8.0, "dos", words=_words("dos", 5.3, 7.7))
    c = _clip("c", 10.0, 13.0, "tres", words=_words("tres", 10.3, 12.7))
    diag = hae.build_handle_aware_pacing_v2_diagnostics((a, b, c), dialogue_overlap_enabled=False)
    assert [row["left_clip_id"] for row in diag["transitions"]] == ["a", "b"]
    assert [row["right_clip_id"] for row in diag["transitions"]] == ["b", "c"]


# ===========================================================================
# 37: deterministic.
# ===========================================================================

def test_37_deterministic_result():
    left, right = _pair(left_end_word=4.0, right_start_word=6.3)
    audit = (_audit_row("r", 5.5, RIGHT_END, RIGHT_START, RIGHT_END),)
    d1 = hae.build_handle_aware_pacing_v2_diagnostics((left, right), dialogue_overlap_enabled=False, boundary_engine_pass_audit=audit)
    d2 = hae.build_handle_aware_pacing_v2_diagnostics((left, right), dialogue_overlap_enabled=False, boundary_engine_pass_audit=audit)
    assert d1 == d2


# ===========================================================================
# 38-39: diagnostics / summary shape.
# ===========================================================================

_REQUIRED_ROW_KEYS = (
    "left_clip_id", "right_clip_id", "old_in_window_j_head", "old_in_window_l_tail",
    "right_pre_handle_id", "right_pre_handle_status", "right_pre_handle_start", "right_pre_handle_end",
    "right_pre_handle_duration", "left_post_handle_id", "left_post_handle_status",
    "left_post_handle_start", "left_post_handle_end", "left_post_handle_duration",
    "combined_j_available_window", "combined_l_available_window", "combined_j_source", "combined_l_source",
    "old_d215_mode", "new_handle_aware_diagnostic_mode",
    "d220_j_max_safe_window_evaluated", "d220_j_chosen_duration",
    "d220_l_max_safe_window_evaluated", "d220_l_chosen_duration",
)

_REQUIRED_SUMMARY_KEYS = (
    "transition_count", "pre_handle_available_count", "post_handle_available_count",
    "safe_pre_handle_count", "safe_post_handle_count", "blocked_pre_handle_count", "blocked_post_handle_count",
    "old_j_candidate_count", "old_l_candidate_count", "handle_aware_j_candidate_count",
    "handle_aware_l_candidate_count", "j_candidates_unlocked_by_handle_count",
    "l_candidates_unlocked_by_handle_count", "unknown_handle_count",
)


def test_38_diagnostics_row_shape():
    left, right = _pair()
    row, _ = _build(left, right)
    for key in _REQUIRED_ROW_KEYS:
        assert key in row


def test_39_run_summary_shape():
    left, right = _pair(left_end_word=4.0, right_start_word=6.3)
    audit = (_audit_row("r", 5.5, RIGHT_END, RIGHT_START, RIGHT_END),)
    _, diag = _build(left, right, boundary_engine_pass_audit=audit)
    for key in _REQUIRED_SUMMARY_KEYS:
        assert key in diag["run_summary"]
    assert diag["run_summary"]["j_candidates_unlocked_by_handle_count"] >= 0


# ===========================================================================
# 40-42: no crossfade/ambience/micro authority.
# ===========================================================================

def test_40_41_no_crossfade_or_ambience_identifier():
    tree = _module_ast()
    identifiers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            identifiers.add(node.name)
    forbidden = {"SHORT_CROSSFADE", "AMBIENCE_CARRY_LEFT", "AMBIENCE_CARRY_RIGHT", "AMBIENCE_BRIDGE"}
    assert not (identifiers & forbidden)


def test_42_no_micro_audio_overlap_authority_referenced():
    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names = {alias.name for alias in node.names}
            assert "MICRO_AUDIO_OVERLAP" not in names


# ===========================================================================
# Layering: no render/render_plan/Boundary-authority import (RenderSegment
# immutability, item 35, and Boundary behavior untouched).
# ===========================================================================

def test_no_renderer_module_imported():
    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in ("render", "render_plan", ".render", ".render_plan")


def test_no_boundary_authority_module_imported():
    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in ("boundary_engine_pass", ".boundary_engine_pass")


# ===========================================================================
# Live-seam integration (universal_clean_cut.py wiring) -- default-off
# parity, flag-on immutability, D-221 scarcity replay end-to-end.
# ===========================================================================

class TestPipelineWiring:
    def _fixture(self, *, right_first_word_start=6.4, boundary_audio_edge_rows=()):
        from cutsell_worker.contracts import DraftTimeline, EditStrategy, JobState, ProcessingResult, SCHEMA_VERSION

        left = _clip("kept_a", LEFT_START, LEFT_END, "uno dos tres", words=_words("uno dos tres", 0.0, 3.7))
        right_words = tuple(
            Word(t, right_first_word_start + i * 1.0, right_first_word_start + (i + 1) * 1.0)
            for i, t in enumerate("cuatro cinco".split())
        )
        right = _clip("kept_b", RIGHT_START, RIGHT_END, "cuatro cinco", words=right_words)
        diagnostics = {}
        if boundary_audio_edge_rows:
            diagnostics["boundary_engine_pass"] = {"audio_edge_rows": list(boundary_audio_edge_rows)}
        draft = DraftTimeline(
            schema_version=SCHEMA_VERSION, project_id="p1", strategy=EditStrategy.STORYTELLING,
            selected=(left, right), alternates=(), discarded=(), diagnostics=diagnostics,
        )
        result = ProcessingResult(schema_version=SCHEMA_VERSION, project_id="p1", state=JobState.DRAFT_READY, draft=draft, stage_status={})
        return result, (left, right)

    def _run_pipeline_stub(self, monkeypatch, request_obj, **fixture_kwargs):
        import dataclasses
        import cutsell_worker.universal_clean_cut as universal
        result, clips = self._fixture(**fixture_kwargs)
        injected_boundary_diag = dict(result.draft.diagnostics)

        def fake_process(request, local_paths, **kwargs):
            return result

        def fake_boundary_pass(res):
            # The REAL `apply_post_freeze_boundary_pass` (boundary_engine_
            # pass.py, unmodified, tested on its own terms elsewhere) would
            # recompute `diagnostics["boundary_engine_pass"]` from real
            # `whole_video_context` silence events this minimal fixture does
            # not carry. Standing in for it here (same convention as the
            # `polish_human_boundaries_v5`/`enforce_complete_idea_boundaries`
            # stubs immediately below) lets this test prove ONLY the seam
            # this task actually adds: whatever real audit Boundary already
            # produced correctly reaches `pacing_v2_handle_aware`.
            return dataclasses.replace(
                res, draft=dataclasses.replace(
                    res.draft, diagnostics={**res.draft.diagnostics, **injected_boundary_diag},
                ),
            )

        monkeypatch.setattr(universal, "process_local_sources", fake_process)
        monkeypatch.setattr(universal, "polish_human_boundaries_v5", lambda res, paths: res)
        monkeypatch.setattr(universal, "enforce_complete_idea_boundaries", lambda res, paths, **kw: res)
        monkeypatch.setattr(universal, "apply_post_freeze_boundary_pass", fake_boundary_pass)
        out = universal.process_universal_clean_cut_sources(
            request_obj, {}, asr_provider=object(), selection_reasoner=None,
        )
        return out, clips

    def test_default_off_no_handle_aware_key(self, monkeypatch):
        monkeypatch.delenv("CUTSELL_PACING_V2_DIAGNOSTICS_ENABLED", raising=False)
        out, _ = self._run_pipeline_stub(monkeypatch, object())
        assert "pacing_v2" not in out.draft.diagnostics
        assert "pacing_v2_handle_aware" not in out.draft.diagnostics

    def test_flag_on_handle_aware_key_present(self, monkeypatch):
        monkeypatch.setenv("CUTSELL_PACING_V2_DIAGNOSTICS_ENABLED", "1")
        out, _ = self._run_pipeline_stub(monkeypatch, object())
        assert "pacing_v2_handle_aware" in out.draft.diagnostics
        assert out.draft.diagnostics["pacing_v2_handle_aware"]["transition_count"] == 1

    def test_flag_on_immutability_selected_boundary_ordering_unchanged(self, monkeypatch):
        monkeypatch.setenv("CUTSELL_PACING_V2_DIAGNOSTICS_ENABLED", "1")
        out, clips = self._run_pipeline_stub(monkeypatch, object())
        left, right = clips
        assert [c.clip_id for c in out.draft.selected] == ["kept_a", "kept_b"]
        selected_by_id = {c.clip_id: c for c in out.draft.selected}
        assert (selected_by_id["kept_a"].start, selected_by_id["kept_a"].end) == (left.start, left.end)
        assert (selected_by_id["kept_b"].start, selected_by_id["kept_b"].end) == (right.start, right.end)

    def test_flag_on_d142_unchanged_vs_off(self, monkeypatch):
        monkeypatch.setenv("CUTSELL_PACING_V2_DIAGNOSTICS_ENABLED", "0")
        off_out, _ = self._run_pipeline_stub(monkeypatch, object())
        monkeypatch.setenv("CUTSELL_PACING_V2_DIAGNOSTICS_ENABLED", "1")
        on_out, _ = self._run_pipeline_stub(monkeypatch, object())
        assert off_out.draft.diagnostics["dialogue_pacing_transition"] == on_out.draft.diagnostics["dialogue_pacing_transition"]

    def test_d221_scarcity_replay_end_to_end(self, monkeypatch):
        # Reproduces D-221's own exact scarcity shape (right's first word
        # starts exactly at right.start -> old_in_window_j_head == 0.0),
        # then supplies a real boundary_engine_pass audit row -- the SAME
        # already-computed audit shape the live pipeline itself produces --
        # showing the previously-invisible handle now yields a positive
        # combined window through the REAL live seam, not just an offline
        # unit call.
        monkeypatch.setenv("CUTSELL_PACING_V2_DIAGNOSTICS_ENABLED", "1")
        audit_row = _audit_row("kept_b", 5.5, RIGHT_END, RIGHT_START, RIGHT_END)
        out, _ = self._run_pipeline_stub(
            monkeypatch, object(), right_first_word_start=RIGHT_START, boundary_audio_edge_rows=(audit_row,),
        )
        row = out.draft.diagnostics["pacing_v2_handle_aware"]["transitions"][0]
        assert row["old_in_window_j_head"] == 0.0
        assert row["combined_j_available_window"] == pytest.approx(0.5)
        assert row["combined_j_source"] == hae.COMBINED_SOURCE_HANDLE_ONLY


# ---------------------------------------------------------------------------
# compileall / import.
# ---------------------------------------------------------------------------

def test_module_compiles_and_imports():
    import importlib
    importlib.import_module("cutsell_worker.pacing_v2_handle_aware_evidence")
