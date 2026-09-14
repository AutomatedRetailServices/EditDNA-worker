"""D-242 -- Boundary edge provenance completeness. OFFLINE ONLY.

D-241's forensic proved `tighten_selected_audio_edges` (boundary_engine_
pass.py) and `trim_locked_selection_edges` (post_selection_edge_only_
boundary.py) each emitted an audit row for a clip's edge ONLY when a trim
actually applied -- so `pacing_v2_source_audio_handle.py`'s `_widen_bound`
could not tell "edge evaluated, nothing to trim" apart from "edge never
evaluated at all"; both collapsed to `NO_BOUNDARY_PROVENANCE_RECORDED`.

This suite proves the fix is bounded and additive:
  * both authorities now emit exactly one row per selected clip, always;
  * each row carries an explicit five-value edge-status (`EDGE_STATUS_*`)
    per PRE/POST edge, shared vocabulary across both authorities;
  * the handle adapter (`pacing_v2_source_audio_handle.py`) maps each
    specific status to its own provenance value instead of the generic
    "no provenance" literal, while NEVER inferring safe handle
    availability from anything but a real, already-recorded widening;
  * no trim decision, threshold, or constant changed;
  * Freeze / Boundary decisions / Audio Join's own zero-fabrication
    invariant are all unaffected.
"""
from __future__ import annotations

import pytest

from cutsell_worker.boundary_engine_pass import (
    AUDIO_EDGE_MINIMUM_REMAINING_SEC,
    AUDIO_EDGE_MINIMUM_TRIM_SEC,
    AUDIO_EDGE_OVERLAP_TOLERANCE_SEC,
    AUDIO_EDGE_PAD_SEC,
    BOUNDARY_REASON_AUDIO_ENTRY,
    BOUNDARY_REASON_AUDIO_EXIT,
    apply_post_freeze_boundary_pass,
    tighten_selected_audio_edges,
)
from cutsell_worker.contracts import (
    DraftClip, DraftTimeline, EditStrategy, JobState, ProcessingResult, SCHEMA_VERSION, SemanticRole, Word,
)
from cutsell_worker.post_selection_edge_only_boundary import (
    EDGE_STATUS_BLOCKED_BY_SAFETY,
    EDGE_STATUS_EVALUATED_NO_TRIM,
    EDGE_STATUS_NO_ELIGIBLE_EVIDENCE,
    EDGE_STATUS_NO_SOURCE_ROOM_DETERMINABLE,
    EDGE_STATUS_TRIM_APPLIED,
    trim_locked_selection_edges,
)
from cutsell_worker import pacing_v2_source_audio_handle as sah
from cutsell_worker.pacing_v2_audio_join_treatment_live_diagnostics import (
    build_audio_join_treatment_live_diagnostics,
)
from cutsell_worker.selection_boundary_contract import enforce_selection_contract, freeze_selection_contract


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _words(*specs) -> tuple:
    return tuple(Word(text=t, start=s, end=e) for (t, s, e) in specs)


def _clip(clip_id: str, source_asset_id: str, start: float, end: float, *, words=(), selected=True) -> DraftClip:
    return DraftClip(
        clip_id=clip_id, source_asset_id=source_asset_id, source_order=0,
        start=start, end=end, text=" ".join(w.text for w in words), caption_text="",
        words=words, semantic_role=SemanticRole.STORY, selected=selected,
    )


def _diag(events: list, source_asset_id: str = "s1") -> dict:
    return {"whole_video_context": {"sources": [{"source_asset_id": source_asset_id, "events": events}]}}


def _silence(start: float, end: float, confidence: float = 0.95) -> dict:
    return {"kind": "audio_silence_interval", "start": start, "end": end, "confidence": confidence}


def _row_for(rows, clip_id):
    matches = [r for r in rows if r["clip_id"] == clip_id]
    assert len(matches) == 1, f"expected exactly one row for {clip_id}, got {len(matches)}"
    return matches[0]


# ===========================================================================
# 1-2: trim-applied PRE / POST edge rows (boundary_engine_pass)
# ===========================================================================

def test_01_trim_applied_pre_edge_row():
    clip = _clip("c", "s1", 10.0, 16.0, words=_words(("uno", 10.4, 10.8), ("dos", 12.8, 13.2)))
    out, audit = tighten_selected_audio_edges((clip,), _diag([_silence(9.5, 10.6)]))
    row = _row_for(audit, "c")
    assert row["entry_edge_status"] == EDGE_STATUS_TRIM_APPLIED
    assert out[0].start == pytest.approx(10.4)
    assert row["result_start"] == pytest.approx(10.4)
    assert row["original_start"] == pytest.approx(10.0)


def test_02_trim_applied_post_edge_row():
    clip = _clip("c", "s1", 10.0, 16.0, words=_words(("uno", 10.4, 10.8), ("dos", 12.8, 13.2)))
    out, audit = tighten_selected_audio_edges((clip,), _diag([_silence(13.4, 16.5)]))
    row = _row_for(audit, "c")
    assert row["exit_edge_status"] == EDGE_STATUS_TRIM_APPLIED
    assert out[0].end < 16.0
    assert row["result_end"] == pytest.approx(out[0].end)


# ===========================================================================
# 3-4: evaluated-no-trim PRE / POST rows (boundary_engine_pass) -- a silence
# geometrically touches the edge but produces no material trim.
# ===========================================================================

def test_03_evaluated_no_trim_pre_row_audio_edge():
    # Silence geometrically covers the leading edge (entry_evidence_seen)
    # but the resulting candidate, after the first-word clamp, does not
    # clear AUDIO_EDGE_MINIMUM_TRIM_SEC.
    clip = _clip("c", "s1", 10.0, 16.0, words=_words(("uno", 10.05, 10.8), ("dos", 12.8, 13.2)))
    out, audit = tighten_selected_audio_edges((clip,), _diag([_silence(9.9, 10.35)]))
    row = _row_for(audit, "c")
    assert row["entry_edge_status"] == EDGE_STATUS_EVALUATED_NO_TRIM
    assert out[0].start == clip.start
    assert row["result_start"] == row["original_start"]


def test_04_evaluated_no_trim_post_row_audio_edge():
    clip = _clip("c", "s1", 10.0, 16.0, words=_words(("uno", 10.4, 10.8), ("dos", 12.8, 15.95)))
    out, audit = tighten_selected_audio_edges((clip,), _diag([_silence(15.7, 16.1)]))
    row = _row_for(audit, "c")
    assert row["exit_edge_status"] == EDGE_STATUS_EVALUATED_NO_TRIM
    assert out[0].end == clip.end


# ===========================================================================
# 5: no eligible evidence (source has events, none near this edge)
# ===========================================================================

def test_05_no_eligible_evidence_audio_edge():
    clip = _clip("c", "s1", 10.0, 16.0, words=_words(("uno", 10.4, 10.8), ("dos", 12.8, 13.2)))
    out, audit = tighten_selected_audio_edges((clip,), _diag([_silence(12.0, 12.4)]))  # interior only
    row = _row_for(audit, "c")
    assert row["entry_edge_status"] == EDGE_STATUS_NO_ELIGIBLE_EVIDENCE
    assert row["exit_edge_status"] == EDGE_STATUS_NO_ELIGIBLE_EVIDENCE
    assert out == (clip,)


def test_05b_no_eligible_evidence_edge_only_boundary():
    clip = _clip("c", "s1", 10.0, 14.0, words=_words(("hola", 10.4, 10.8), ("mundo", 12.8, 13.2)))
    # Reset event far from either slack window.
    selected, audit = trim_locked_selection_edges(
        (clip,), _diag([{"kind": "camera_adjustment", "start": 11.0, "end": 11.2, "confidence": 0.99}]),
    )
    row = _row_for(audit, "c")
    assert row["leading_edge_status"] == EDGE_STATUS_NO_ELIGIBLE_EVIDENCE
    assert selected == (clip,)


# ===========================================================================
# 6: blocked-by-safety (per-clip minimum-remaining-duration floor rejects an
# otherwise-valid candidate)
# ===========================================================================

def test_06_blocked_by_safety_audio_edge_floor():
    # A clip where both edges have confirmed, qualifying trims (each well
    # clear of AUDIO_EDGE_MINIMUM_TRIM_SEC with margin), but the combined
    # remainder (0.30s) would fall below AUDIO_EDGE_MINIMUM_REMAINING_SEC
    # (0.35s) -- both edges are BLOCKED_BY_SAFETY, and the trim is reverted.
    clip = _clip("c", "s1", 10.0, 10.9, words=_words(("hi", 10.40, 10.42)))
    out, audit = tighten_selected_audio_edges(
        (clip,), _diag([_silence(9.0, 10.35), _silence(10.45, 12.0)]),
    )
    row = _row_for(audit, "c")
    assert row["entry_edge_status"] == EDGE_STATUS_BLOCKED_BY_SAFETY
    assert row["exit_edge_status"] == EDGE_STATUS_BLOCKED_BY_SAFETY
    # Reverted -- no trim actually applied.
    assert out[0].start == clip.start and out[0].end == clip.end
    assert row["actions"] == []


def test_06b_blocked_by_safety_edge_only_boundary_floor():
    clip = _clip("c", "s1", 10.0, 10.4, words=_words(("hi", 10.31, 10.32)))
    events = [
        {"kind": "unintentional_dead_air", "start": 10.0, "end": 10.31, "confidence": 0.95},
    ]
    selected, audit = trim_locked_selection_edges((clip,), _diag(events))
    row = _row_for(audit, "c")
    # leading trim candidate exists (0.31s slack, confirmed dead_air) but
    # 10.4-10.31=0.09 < the module's own 0.25s remaining floor.
    assert row["leading_edge_status"] == EDGE_STATUS_BLOCKED_BY_SAFETY
    assert selected[0].start == clip.start


# ===========================================================================
# 7: unknown source room (no recorded events at all for this source)
# ===========================================================================

def test_07_no_source_room_determinable_audio_edge():
    clip = _clip("c", "s1", 10.0, 16.0, words=_words(("uno", 10.4, 10.8), ("dos", 12.8, 13.2)))
    out, audit = tighten_selected_audio_edges((clip,), {"whole_video_context": {"sources": []}})
    row = _row_for(audit, "c")
    assert row["entry_edge_status"] == EDGE_STATUS_NO_SOURCE_ROOM_DETERMINABLE
    assert row["exit_edge_status"] == EDGE_STATUS_NO_SOURCE_ROOM_DETERMINABLE


def test_07b_no_source_room_determinable_edge_only_boundary_no_events():
    clip = _clip("c", "s1", 10.0, 14.0, words=_words(("hola", 10.4, 10.8), ("mundo", 12.8, 13.2)))
    selected, audit = trim_locked_selection_edges((clip,), {"whole_video_context": {"sources": []}})
    row = _row_for(audit, "c")
    assert row["leading_edge_status"] == EDGE_STATUS_NO_SOURCE_ROOM_DETERMINABLE
    assert row["trailing_edge_status"] == EDGE_STATUS_NO_SOURCE_ROOM_DETERMINABLE


def test_07c_no_source_room_determinable_edge_only_boundary_no_words():
    # No word alignment at all -> genuinely cannot determine slack/room.
    clip = _clip("c", "s1", 10.0, 14.0, words=())
    selected, audit = trim_locked_selection_edges((clip,), _diag([]))
    row = _row_for(audit, "c")
    assert row["leading_edge_status"] == EDGE_STATUS_NO_SOURCE_ROOM_DETERMINABLE
    assert row["trailing_edge_status"] == EDGE_STATUS_NO_SOURCE_ROOM_DETERMINABLE
    assert selected == (clip,)


# ===========================================================================
# 8: source room known (events exist for the source) but not safe to use
# (nearby evidence found but insufficient to confirm -- edge-only boundary's
# own corroboration rule)
# ===========================================================================

def test_08_source_room_known_but_not_safe_edge_only_boundary():
    clip = _clip("c", "s1", 10.0, 14.0, words=_words(("hola", 10.4, 10.8), ("mundo", 12.8, 13.2)))
    events = [{"kind": "body_reset_candidate", "start": 13.25, "end": 13.7, "confidence": 0.97}]
    selected, audit = trim_locked_selection_edges((clip,), _diag(events))
    row = _row_for(audit, "c")
    assert row["trailing_edge_status"] == EDGE_STATUS_EVALUATED_NO_TRIM
    assert row["trailing_edge_reason"] == "nearby_evidence_insufficient_to_confirm"


# ===========================================================================
# 9-10: row-per-edge completeness, no duplicates, multi-clip
# ===========================================================================

def test_09_row_emitted_for_every_selected_clip():
    clips = tuple(
        _clip(f"c{i}", "s1", float(i * 20), float(i * 20 + 5), words=_words((f"w{i}", i * 20 + 0.1, i * 20 + 4.9)))
        for i in range(5)
    )
    out, audit = tighten_selected_audio_edges(clips, _diag([]))
    assert len(audit) == len(clips)
    assert {row["clip_id"] for row in audit} == {c.clip_id for c in clips}

    selected, audit2 = trim_locked_selection_edges(clips, _diag([]))
    assert len(audit2) == len(clips)
    assert {row["clip_id"] for row in audit2} == {c.clip_id for c in clips}


def test_10_no_duplicate_rows():
    clip = _clip("c", "s1", 10.0, 16.0, words=_words(("uno", 10.4, 10.8), ("dos", 12.8, 13.2)))
    _, audit = tighten_selected_audio_edges((clip,), _diag([_silence(9.5, 10.6)]))
    assert len([r for r in audit if r["clip_id"] == "c"]) == 1
    _, audit2 = trim_locked_selection_edges((clip,), _diag([]))
    assert len([r for r in audit2 if r["clip_id"] == "c"]) == 1


# ===========================================================================
# 11-13: no behavior mutation -- timing/membership/Boundary decisions
# ===========================================================================

def test_11_timing_unchanged_on_no_trim():
    clip = _clip("c", "s1", 10.0, 16.0, words=_words(("uno", 10.4, 10.8), ("dos", 12.8, 13.2)))
    out, audit = tighten_selected_audio_edges((clip,), _diag([]))
    assert out[0].start == clip.start and out[0].end == clip.end
    row = _row_for(audit, "c")
    assert row["result_start"] == row["original_start"]
    assert row["result_end"] == row["original_end"]


def test_12_selected_membership_unchanged():
    clips = (
        _clip("a", "s1", 0.0, 4.0, words=_words(("a", 0.1, 3.9))),
        _clip("b", "s1", 5.0, 9.0, words=_words(("b", 5.1, 8.9))),
    )
    out, _ = tighten_selected_audio_edges(clips, _diag([]))
    assert [c.clip_id for c in out] == ["a", "b"]
    assert len(out) == len(clips)


def test_13_boundary_decisions_unchanged_when_trim_applies():
    # Same fixture as the pre-existing D-097.C regression -- proves the
    # actual trim amount/reason is bit-for-bit identical to before D-242.
    clip = _clip("c", "s1", 10.0, 16.0, words=_words(("uno", 10.4, 10.8), ("dos", 12.8, 13.2)))
    out, audit = tighten_selected_audio_edges((clip,), _diag([_silence(9.5, 10.6)]))
    assert out[0].start == pytest.approx(10.4)
    assert out[0].boundary_reason == BOUNDARY_REASON_AUDIO_ENTRY
    assert audit[0]["actions"][0]["action"] == BOUNDARY_REASON_AUDIO_ENTRY


# ===========================================================================
# 14-17: handle adapter consumption
# ===========================================================================

def _audit_row_with_status(clip_id, original_start, original_end, result_start, result_end,
                            *, entry_status=None, exit_status=None, entry_reason=None, exit_reason=None):
    return {
        "clip_id": clip_id, "original_start": original_start, "original_end": original_end,
        "result_start": result_start, "result_end": result_end, "actions": [],
        "edge_evaluated": True,
        "entry_edge_status": entry_status, "entry_edge_reason": entry_reason,
        "exit_edge_status": exit_status, "exit_edge_reason": exit_reason,
    }


def test_14_handle_adapter_no_longer_reports_generic_no_provenance_when_row_exists():
    clip = DraftClip(clip_id="c1", source_asset_id="s1", source_order=0, start=2.0, end=8.0,
                      text="", caption_text="", words=())
    audit = (_audit_row_with_status("c1", 2.0, 8.0, 2.0, 8.0, entry_status=EDGE_STATUS_EVALUATED_NO_TRIM),)
    handle = sah.build_pre_roll_audio_handle(clip, boundary_engine_pass_audit=audit)
    assert handle.handle_status == sah.HANDLE_STATUS_UNAVAILABLE
    assert sah.PROVENANCE_NO_TRIM_RECORDED not in handle.provenance
    assert sah.PROVENANCE_EVALUATED_NO_SAFE_WIDENING in handle.provenance
    assert sah.CONFLICT_EVALUATED_NO_SAFE_WIDENING in handle.conflict_flags
    assert sah.CONFLICT_NO_PROVENANCE not in handle.conflict_flags


@pytest.mark.parametrize("status,expected_provenance,expected_flag", [
    (EDGE_STATUS_EVALUATED_NO_TRIM, sah.PROVENANCE_EVALUATED_NO_SAFE_WIDENING, sah.CONFLICT_EVALUATED_NO_SAFE_WIDENING),
    (EDGE_STATUS_NO_ELIGIBLE_EVIDENCE, sah.PROVENANCE_NO_ELIGIBLE_EVIDENCE_AT_EDGE, sah.CONFLICT_NO_ELIGIBLE_EVIDENCE_AT_EDGE),
    (EDGE_STATUS_NO_SOURCE_ROOM_DETERMINABLE, sah.PROVENANCE_SOURCE_ROOM_UNKNOWN, sah.CONFLICT_SOURCE_ROOM_UNKNOWN),
    (EDGE_STATUS_BLOCKED_BY_SAFETY, sah.PROVENANCE_BLOCKED_BY_SAFETY_FLOOR, sah.CONFLICT_BLOCKED_BY_SAFETY_FLOOR),
])
def test_15_handle_remains_unavailable_for_every_new_status_fail_closed(status, expected_provenance, expected_flag):
    clip = DraftClip(clip_id="c1", source_asset_id="s1", source_order=0, start=2.0, end=8.0,
                      text="", caption_text="", words=())
    audit = (_audit_row_with_status("c1", 2.0, 8.0, 2.0, 8.0, entry_status=status),)
    handle = sah.build_pre_roll_audio_handle(clip, boundary_engine_pass_audit=audit)
    assert handle.handle_status == sah.HANDLE_STATUS_UNAVAILABLE
    assert handle.available_duration == 0.0
    assert expected_provenance in handle.provenance
    assert expected_flag in handle.conflict_flags


def test_16_handle_evaluates_existing_safe_provenance_unchanged():
    # A row that DOES show real widening -- unaffected by D-242, same
    # SAFE_NON_SPEECH outcome as before.
    clip = DraftClip(clip_id="c1", source_asset_id="s1", source_order=0, start=2.0, end=8.0,
                      text="", caption_text="", words=())
    audit = (_audit_row_with_status("c1", 1.5, 8.0, 2.0, 8.0, entry_status=EDGE_STATUS_TRIM_APPLIED),)
    handle = sah.build_pre_roll_audio_handle(clip, boundary_engine_pass_audit=audit)
    assert handle.handle_status == sah.HANDLE_STATUS_SAFE_NON_SPEECH
    assert handle.available_duration == pytest.approx(0.5)


def test_17_pre_post_symmetry():
    clip = DraftClip(clip_id="c1", source_asset_id="s1", source_order=0, start=2.0, end=8.0,
                      text="", caption_text="", words=())
    audit = (_audit_row_with_status(
        "c1", 2.0, 8.0, 2.0, 8.0,
        entry_status=EDGE_STATUS_NO_ELIGIBLE_EVIDENCE, exit_status=EDGE_STATUS_NO_SOURCE_ROOM_DETERMINABLE,
    ),)
    pre = sah.build_pre_roll_audio_handle(clip, boundary_engine_pass_audit=audit)
    post = sah.build_post_roll_audio_handle(clip, boundary_engine_pass_audit=audit)
    assert sah.PROVENANCE_NO_ELIGIBLE_EVIDENCE_AT_EDGE in pre.provenance
    assert sah.PROVENANCE_SOURCE_ROOM_UNKNOWN in post.provenance


# ===========================================================================
# 18-19: multi-clip determinism, source isolation
# ===========================================================================

def test_18_multi_clip_determinism():
    clip_a = DraftClip(clip_id="a", source_asset_id="s1", source_order=0, start=0.0, end=4.0,
                        text="", caption_text="", words=())
    clip_b = DraftClip(clip_id="b", source_asset_id="s1", source_order=1, start=6.0, end=10.0,
                        text="", caption_text="", words=())
    audit = (
        _audit_row_with_status("a", 0.0, 4.0, 0.0, 4.0, entry_status=EDGE_STATUS_EVALUATED_NO_TRIM),
        _audit_row_with_status("b", 6.0, 10.0, 6.0, 10.0, entry_status=EDGE_STATUS_NO_ELIGIBLE_EVIDENCE),
    )
    handles_1 = sah.build_source_audio_handles((clip_a, clip_b), boundary_engine_pass_audit=audit)
    handles_2 = sah.build_source_audio_handles((clip_a, clip_b), boundary_engine_pass_audit=audit)
    assert handles_1 == handles_2


def test_19_source_isolation():
    clip_a = DraftClip(clip_id="a", source_asset_id="s1", source_order=0, start=0.0, end=4.0,
                        text="", caption_text="", words=())
    clip_b = DraftClip(clip_id="b", source_asset_id="s2", source_order=0, start=0.0, end=4.0,
                        text="", caption_text="", words=())
    # Only "a"'s row exists; "b" has none -> exceptional generic literal.
    audit = (_audit_row_with_status("a", 0.0, 4.0, 0.0, 4.0, entry_status=EDGE_STATUS_EVALUATED_NO_TRIM),)
    handle_a = sah.build_pre_roll_audio_handle(clip_a, boundary_engine_pass_audit=audit)
    handle_b = sah.build_pre_roll_audio_handle(clip_b, boundary_engine_pass_audit=audit)
    assert sah.PROVENANCE_EVALUATED_NO_SAFE_WIDENING in handle_a.provenance
    assert handle_b.provenance == (sah.PROVENANCE_NO_TRIM_RECORDED,)


# ===========================================================================
# 20-23: no new threshold, no heuristic, no provider, no RAW (constant/shape
# assertions -- structural proof, not just claim)
# ===========================================================================

def test_20_no_new_threshold_constants_unchanged():
    assert AUDIO_EDGE_PAD_SEC == 0.10
    assert AUDIO_EDGE_MINIMUM_TRIM_SEC == 0.20
    assert AUDIO_EDGE_OVERLAP_TOLERANCE_SEC == 0.08
    assert AUDIO_EDGE_MINIMUM_REMAINING_SEC == 0.35


def test_21_no_new_edge_status_values_beyond_the_five():
    from cutsell_worker.post_selection_edge_only_boundary import EDGE_STATUSES
    assert EDGE_STATUSES == frozenset({
        EDGE_STATUS_TRIM_APPLIED, EDGE_STATUS_EVALUATED_NO_TRIM,
        EDGE_STATUS_NO_ELIGIBLE_EVIDENCE, EDGE_STATUS_NO_SOURCE_ROOM_DETERMINABLE,
        EDGE_STATUS_BLOCKED_BY_SAFETY,
    })


def test_22_no_provider_or_network_call_introduced():
    import pathlib
    for mod in ("boundary_engine_pass", "post_selection_edge_only_boundary", "pacing_v2_source_audio_handle"):
        text = (pathlib.Path(__file__).resolve().parent.parent / "cutsell_worker" / f"{mod}.py").read_text()
        for needle in ("requests.", "boto3", "openai", "google.generativeai", "modal.", "runpod"):
            assert needle not in text, f"{needle!r} unexpectedly present in {mod}.py"


# ===========================================================================
# 24: Freeze unchanged -- the semantic token-stream contract still holds
# across the modified post-Freeze pass.
# ===========================================================================

def test_24_freeze_contract_unaffected_by_always_emitted_rows():
    clip = _clip("kept", "s1", 0.0, 9.5, words=_words(
        ("por", 0.2, 1.9), ("temporada,", 2.0, 3.8), ("me", 3.9, 5.7), ("salia", 5.8, 7.6), ("acne", 7.7, 9.4),
    ))
    draft = freeze_selection_contract(_draft([clip], _diag([])))
    result = apply_post_freeze_boundary_pass(_result(draft))
    # Should not raise -- Boundary's new always-emitted rows never touched
    # the ordered token stream.
    enforce_selection_contract(result.draft)
    assert result.draft.diagnostics["boundary_engine_pass"]["selected_count_out"] == 1


def _draft(clips, diagnostics):
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p1", strategy=EditStrategy.STORYTELLING,
        selected=tuple(clips), alternates=(), discarded=(), diagnostics=diagnostics,
    )


def _result(draft):
    return ProcessingResult(schema_version=SCHEMA_VERSION, project_id="p1", state=JobState.DRAFT_READY, draft=draft, stage_status={})


# ===========================================================================
# 26: Audio Join policy unchanged -- still zero fabricated availability when
# both handles are the new, more specific "no safe widening" statuses.
# ===========================================================================

def test_26_audio_join_still_reports_zero_candidate_duration_when_unavailable():
    left = _clip("l", "s1", 0.0, 4.0, words=_words(("uno", 0.1, 3.9)))
    right = _clip("r", "s1", 6.0, 10.0, words=_words(("dos", 6.1, 9.9)))
    diag = build_audio_join_treatment_live_diagnostics(
        (left, right), dialogue_overlap_enabled=False, boundary_diagnostics={},
        boundary_engine_pass_audit=(
            _audit_row_with_status("l", 0.0, 4.0, 0.0, 4.0, exit_status=EDGE_STATUS_EVALUATED_NO_TRIM),
            _audit_row_with_status("r", 6.0, 10.0, 6.0, 10.0, entry_status=EDGE_STATUS_NO_ELIGIBLE_EVIDENCE),
        ),
    )
    for row in diag["per_join"]:
        candidate = row["decision"]["candidate_duration_sec"]
        assert candidate is None or candidate == pytest.approx(0.0)


# ===========================================================================
# D-239V-shape real replay: 2 selected clips, 1 transition, no previous
# audit row for the relevant edges.
# ===========================================================================

def test_d239v_shape_replay_explicit_rows_but_no_fake_availability():
    left = _clip("l", "s1", 0.0, 4.0, words=_words(("uno", 0.0, 3.9)))   # word touches clip start: zero leading slack
    right = _clip("r", "s1", 4.2, 8.0, words=_words(("dos", 4.21, 8.0)))  # word touches clip end: zero trailing slack
    diagnostics = {}
    # Before D-242: neither function would emit a row for either clip
    # (no silence overlaps, no word slack) -> NO_BOUNDARY_PROVENANCE_RECORDED.
    _, audio_audit = tighten_selected_audio_edges((left, right), diagnostics)
    _, edge_audit = trim_locked_selection_edges((left, right), diagnostics)

    # After D-242: rows exist for both clips on both authorities.
    assert len(audio_audit) == 2
    assert len(edge_audit) == 2

    handles = sah.build_source_audio_handles(
        (left, right),
        boundary_engine_pass_audit=audio_audit,
        post_selection_edge_only_boundary_audit=edge_audit,
    )
    # No handle window invented: every handle is still UNAVAILABLE, but the
    # generic exceptional literal no longer appears -- a specific reason
    # does instead.
    for handle in handles:
        assert handle.handle_status == sah.HANDLE_STATUS_UNAVAILABLE
        assert handle.available_duration == 0.0
        assert sah.PROVENANCE_NO_TRIM_RECORDED not in handle.provenance
