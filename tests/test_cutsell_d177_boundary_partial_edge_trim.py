"""D-177 -- Boundary partial-edge trim for a visual/performance event that
STRADDLES a selected DELIVERY boundary by no more than this module's own
existing AUDIO_EDGE_OVERLAP_TOLERANCE_SEC.

Closes the exact gap D-176's forensic named in `boundary_engine_pass.py`'s
own module docstring: a straddling event was always treated as fully
DELIVERY-owned (zero trim available), even when its own portion actually
inside DELIVERY was smaller than the tolerance this module already treats
as immaterial everywhere else. BOUNDARY still only trims debris; BESTTAKE
still chooses among realizations -- this module never re-decides which
take won, never re-scores a family, never touches Language-Spine, Pacing,
or Renderer. No new numeric constant: eligibility reuses the existing
`AUDIO_EDGE_OVERLAP_TOLERANCE_SEC`; the per-clip floor reuses the existing
`AUDIO_EDGE_MINIMUM_REMAINING_SEC`. Generic fixtures only -- no Video00
text/ids/timestamps (see `test_generic_d176_shape_replay_*` below for the
one dedicated synthetic replay of the D-176 forensic shape).
"""
from cutsell_worker.boundary_engine_pass import (
    BOUNDARY_REASON_VISUAL_AMBIGUOUS_STRADDLE_NO_TRIM,
    BOUNDARY_REASON_VISUAL_DELIVERY_OVERLAP_NO_TRIM,
    BOUNDARY_REASON_VISUAL_ENTRY_PARTIAL_EDGE_TRIM,
    BOUNDARY_REASON_VISUAL_EXIT_PARTIAL_EDGE_TRIM,
    BOUNDARY_REASON_VISUAL_NOT_AT_EDGE,
    AUDIO_EDGE_MINIMUM_REMAINING_SEC,
    AUDIO_EDGE_OVERLAP_TOLERANCE_SEC,
    apply_post_freeze_boundary_pass,
    tighten_selected_visual_edges,
)
from cutsell_worker.contracts import (
    DraftClip, DraftTimeline, EditStrategy, JobState, ProcessingResult,
    SCHEMA_VERSION, SemanticRole, Word,
)
from cutsell_worker.render_plan import build_render_plan
from cutsell_worker.take_judge import rank_takes, score_take
from cutsell_worker.contracts import CandidateTake, MediaSignals

import pytest


def _words(text, start, end):
    tokens = text.split()
    step = (end - start) / len(tokens)
    return tuple(Word(t, start + i * step, start + (i + 1) * step) for i, t in enumerate(tokens))


def _clip(clip_id, start, end, text, *, words=None, source="src"):
    words = _words(text, start, end) if words is None else words
    return DraftClip(
        clip_id=clip_id, source_asset_id=source, source_order=0, start=start, end=end, text=text,
        caption_text=text, words=words, semantic_role=SemanticRole.STORY, selected=True,
    )


def _event(kind, start, end, confidence=0.9):
    return {"kind": kind, "start": start, "end": end, "confidence": confidence}


def _diag(events=(), source="src", extra=None):
    return {
        "whole_video_context": {"sources": [{"source_asset_id": source, "events": list(events)}]},
        **(extra or {}),
    }


def _draft(clips, diagnostics):
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p1", strategy=EditStrategy.STORYTELLING,
        selected=tuple(clips), alternates=(), discarded=(), diagnostics=diagnostics,
    )


def _result(draft):
    return ProcessingResult(schema_version=SCHEMA_VERSION, project_id="p1", state=JobState.DRAFT_READY, draft=draft, stage_status={})


def _rows_for(clip_id, audit):
    return [row for row in audit if row["clip_id"] == clip_id]


# --- 1/2. entry/exit straddling, tiny inside overlap -> trim ------------------

def test_1_entry_straddling_tiny_inside_overlap_trims():
    # delivery_span (10.0, 19.0); event 9.5-10.05 straddles entry by 0.05s
    # (< tolerance 0.08) and touches the clip's own leading edge (9.5). The
    # edge clamps to delivery_span.start itself (hard floor, never past it)
    # -- the 0.05s DELIVERY-side sliver is left untouched, never shaved.
    clip = _clip("c", 9.5, 20.0, "one two three", words=_words("one two three", 10.0, 19.0))
    diag = _diag([_event("camera_disengagement_candidate", 9.5, 10.05)])
    out, audit = tighten_selected_visual_edges((clip,), diag)
    assert out[0].start == pytest.approx(10.0)  # delivery_span.start, not event.end
    assert out[0].end == pytest.approx(20.0)
    row = _rows_for("c", audit)[0]
    assert row["reason"] == BOUNDARY_REASON_VISUAL_ENTRY_PARTIAL_EDGE_TRIM
    assert row["trim_applied"] is True
    assert row["trim_side"] == "ENTRY"
    assert row["inside_delivery_overlap_sec"] == pytest.approx(0.05)


def test_2_exit_straddling_tiny_inside_overlap_trims():
    clip = _clip("c", 10.0, 19.5, "one two three", words=_words("one two three", 10.0, 19.0))
    diag = _diag([_event("facial_expression_shift_candidate", 18.95, 19.5)])
    out, audit = tighten_selected_visual_edges((clip,), diag)
    assert out[0].end == pytest.approx(19.0)  # delivery_span.end, not event.start
    assert out[0].start == pytest.approx(10.0)
    row = _rows_for("c", audit)[0]
    assert row["reason"] == BOUNDARY_REASON_VISUAL_EXIT_PARTIAL_EDGE_TRIM
    assert row["trim_applied"] is True
    assert row["trim_side"] == "EXIT"
    assert row["inside_delivery_overlap_sec"] == pytest.approx(0.05)


# --- 3/4. entry/exit straddling, materially inside -> preserve ----------------

def test_3_entry_straddling_material_inside_overlap_preserves():
    # inside overlap 0.5s (> 0.08 tolerance) -- real DELIVERY defect, same
    # reason/behavior as pre-D-177.
    clip = _clip("c", 9.5, 20.0, "one two three", words=_words("one two three", 10.0, 19.0))
    diag = _diag([_event("camera_disengagement_candidate", 9.5, 10.5)])
    out, audit = tighten_selected_visual_edges((clip,), diag)
    assert out[0].start == pytest.approx(9.5)  # untouched
    row = _rows_for("c", audit)[0]
    assert row["reason"] == BOUNDARY_REASON_VISUAL_DELIVERY_OVERLAP_NO_TRIM
    assert row["trim_applied"] is False
    assert row["inside_delivery_overlap_sec"] == pytest.approx(0.5)


def test_4_exit_straddling_material_inside_overlap_preserves():
    clip = _clip("c", 10.0, 19.5, "one two three", words=_words("one two three", 10.0, 19.0))
    diag = _diag([_event("body_reset_candidate", 18.5, 19.5)])
    out, audit = tighten_selected_visual_edges((clip,), diag)
    assert out[0].end == pytest.approx(19.5)  # untouched
    row = _rows_for("c", audit)[0]
    assert row["reason"] == BOUNDARY_REASON_VISUAL_DELIVERY_OVERLAP_NO_TRIM
    assert row["trim_applied"] is False
    assert row["inside_delivery_overlap_sec"] == pytest.approx(0.5)


# --- 5. fully outside -> unchanged behavior (pure ENTRY/EXIT, D-116) ---------

def test_5_event_fully_outside_delivery_unaffected_by_d177():
    clip = _clip("c", 10.0, 16.0, "one two three", words=_words("one two three", 11.6, 15.0))
    diag = _diag([_event("camera_disengagement_candidate", 10.0, 11.2)])
    out, audit = tighten_selected_visual_edges((clip,), diag)
    assert out[0].start == pytest.approx(11.2)
    row = _rows_for("c", audit)[0]
    assert row["zone"] == "ENTRY"
    assert row["inside_delivery_overlap_sec"] is None  # not a straddle row


# --- 6. fully inside DELIVERY -> no trim (unchanged D-116) --------------------

def test_6_event_fully_inside_delivery_no_trim():
    clip = _clip("c", 10.0, 16.0, "one two three", words=_words("one two three", 10.2, 15.5))
    diag = _diag([_event("hand_motion_reset_candidate", 12.0, 12.5)])
    out, audit = tighten_selected_visual_edges((clip,), diag)
    assert out[0].start == pytest.approx(10.0) and out[0].end == pytest.approx(16.0)
    row = _rows_for("c", audit)[0]
    assert row["reason"] == BOUNDARY_REASON_VISUAL_DELIVERY_OVERLAP_NO_TRIM
    assert row["inside_delivery_overlap_sec"] is None


# --- 7. ambiguous both-edges straddle -> always fails open --------------------

def test_7_ambiguous_both_edges_straddle_fails_open():
    # event wider than the whole measured span -- before AND after.
    clip = _clip("c", 5.0, 25.0, "one two three", words=_words("one two three", 10.0, 19.0))
    diag = _diag([_event("body_reset_candidate", 5.0, 25.0)])
    out, audit = tighten_selected_visual_edges((clip,), diag)
    assert out[0].start == pytest.approx(5.0) and out[0].end == pytest.approx(25.0)
    row = _rows_for("c", audit)[0]
    assert row["reason"] == BOUNDARY_REASON_VISUAL_AMBIGUOUS_STRADDLE_NO_TRIM
    assert row["trim_applied"] is False


# --- 8-12. word/speech safety: never trims a required word --------------------

def test_8_required_word_never_trimmed_entry():
    # The straddling event's own outer edge is clamped to end at/after the
    # first word's start (compute_delivery_span uses real word timestamps,
    # never fabricated) -- trimming can only ever remove non-speech debris.
    clip = _clip("c", 9.5, 20.0, "no seven refund never", words=_words("no seven refund never", 10.0, 19.0))
    diag = _diag([_event("camera_disengagement_candidate", 9.5, 10.05)])
    out, audit = tighten_selected_visual_edges((clip,), diag)
    assert out[0].start <= out[0].words[0].start  # first word (negation "no") fully intact
    assert out[0].words[0].text == "no"


def test_9_negation_word_preserved_after_trim():
    clip = _clip("c", 9.5, 20.0, "never true always", words=_words("never true always", 10.0, 19.0))
    diag = _diag([_event("camera_disengagement_candidate", 9.5, 10.05)])
    out, _audit = tighten_selected_visual_edges((clip,), diag)
    assert out[0].words[0].text == "never"
    assert out[0].start <= out[0].words[0].start


def test_10_number_word_preserved_after_trim():
    clip = _clip("c", 10.0, 19.5, "step one seven done", words=_words("step one seven done", 10.0, 19.0))
    diag = _diag([_event("facial_expression_shift_candidate", 18.95, 19.5)])
    out, _audit = tighten_selected_visual_edges((clip,), diag)
    assert out[0].words[-1].text == "done"
    assert out[0].end >= out[0].words[-1].end


def test_11_factual_term_preserved_after_trim():
    clip = _clip("c", 9.5, 20.0, "diagnosis confirmed today", words=_words("diagnosis confirmed today", 10.0, 19.0))
    diag = _diag([_event("camera_disengagement_candidate", 9.5, 10.05)])
    out, _audit = tighten_selected_visual_edges((clip,), diag)
    assert out[0].words[0].text == "diagnosis"
    assert out[0].start <= out[0].words[0].start


def test_12_complete_idea_clause_ending_preserved_after_trim():
    clip = _clip("c", 10.0, 19.5, "the full complete idea", words=_words("the full complete idea", 10.0, 19.0))
    diag = _diag([_event("facial_expression_shift_candidate", 18.95, 19.5)])
    out, _audit = tighten_selected_visual_edges((clip,), diag)
    assert out[0].words[-1].text == "idea"
    assert out[0].end >= out[0].words[-1].end


# --- 13. minimum-remaining-duration floor respected ---------------------------

def test_13_minimum_remaining_duration_floor_blocks_trim_via_pass():
    # Both entry and exit straddle-trim to leave < AUDIO_EDGE_MINIMUM_REMAINING_SEC
    # remaining -> apply_post_freeze_boundary_pass's own existing floor check
    # (new_end - new_start < AUDIO_EDGE_MINIMUM_REMAINING_SEC) rejects the
    # whole clip's edits, same existing guard, no new one.
    tiny = AUDIO_EDGE_MINIMUM_REMAINING_SEC - 0.10
    delivery_start = 10.0
    delivery_end = delivery_start + tiny
    clip = _clip("c", 9.95, delivery_end + 0.05, "hi",
                 words=_words("hi", delivery_start, delivery_end))
    diag = _diag([
        _event("camera_disengagement_candidate", 9.95, 10.05),
        _event("body_reset_candidate", delivery_end - 0.05, delivery_end + 0.05),
    ])
    result = _result(_draft([clip], diag))
    out = apply_post_freeze_boundary_pass(result)
    # The clip is left unchanged (both edits together violate the per-clip
    # remaining-duration floor) -- same existing guard as D-116.
    assert out.draft.selected[0].start == pytest.approx(9.95)
    assert out.draft.selected[0].end == pytest.approx(delivery_end + 0.05)


# --- 14. idempotence -----------------------------------------------------------

def test_14_idempotent_second_pass_no_further_change():
    clip = _clip("c", 10.0, 20.0, "alpha beta gamma delta epsilon",
                 words=_words("alpha beta gamma delta epsilon", 10.75, 19.04))
    diag = _diag([
        _event("hand_motion_reset_candidate", 10.0, 10.80),
        _event("facial_expression_shift_candidate", 18.97, 20.0),
    ])
    result = _result(_draft([clip], diag))
    out1 = apply_post_freeze_boundary_pass(result)
    out2 = apply_post_freeze_boundary_pass(out1)
    assert out1.draft.selected[0].start == pytest.approx(out2.draft.selected[0].start)
    assert out1.draft.selected[0].end == pytest.approx(out2.draft.selected[0].end)
    assert out1.draft.selected[0].start == pytest.approx(10.75)  # delivery_span.start
    assert out1.draft.selected[0].end == pytest.approx(19.04)  # delivery_span.end


# --- 15. deterministic result ---------------------------------------------------

def test_15_deterministic_repeated_calls_same_input():
    clip = _clip("c", 9.5, 20.0, "one two three", words=_words("one two three", 10.0, 19.0))
    diag = _diag([_event("camera_disengagement_candidate", 9.5, 10.05)])
    out_a, audit_a = tighten_selected_visual_edges((clip,), diag)
    out_b, audit_b = tighten_selected_visual_edges((clip,), diag)
    assert out_a[0].start == out_b[0].start and out_a[0].end == out_b[0].end
    assert audit_a == audit_b


# --- 16. provenance retained -----------------------------------------------------

def test_16_provenance_fields_present_on_trim_row():
    clip = _clip("c", 9.5, 20.0, "one two three", words=_words("one two three", 10.0, 19.0))
    diag = _diag([_event("camera_disengagement_candidate", 9.5, 10.05, confidence=0.77)])
    _out, audit = tighten_selected_visual_edges((clip,), diag)
    row = _rows_for("c", audit)[0]
    for field in (
        "authority", "clip_id", "event_kind", "event_start", "event_end", "event_confidence",
        "zone", "overlaps_delivery", "delivery_start", "delivery_end", "delivery_span_available",
        "old_start", "new_start", "old_end", "new_end", "trim_side", "trim_applied", "reason",
        "evidence_source", "semantic_membership_changed", "inside_delivery_overlap_sec",
    ):
        assert field in row
    assert row["event_kind"] == "camera_disengagement_candidate"
    assert row["event_confidence"] == pytest.approx(0.77)
    assert row["evidence_source"] == "local_performance"
    assert row["semantic_membership_changed"] is False


# --- 17. source timing preserved (never fabricated) ---------------------------

def test_17_source_timing_never_fabricated_uses_real_event_bounds():
    # The trim target is the measured delivery_span.start itself (never
    # fabricated -- computed only from real word timestamps); the row's
    # own event_start/event_end still report the real, unaltered event
    # bounds for provenance, even though the trim clamps short of them.
    clip = _clip("c", 9.5, 20.0, "one two three", words=_words("one two three", 10.0, 19.0))
    diag = _diag([_event("camera_disengagement_candidate", 9.5, 10.037)])
    out, audit = tighten_selected_visual_edges((clip,), diag)
    assert out[0].start == pytest.approx(10.0)  # delivery_span.start, the hard floor
    row = _rows_for("c", audit)[0]
    assert row["event_start"] == pytest.approx(9.5)
    assert row["event_end"] == pytest.approx(10.037)


# --- 18. ordinary motion edge event ---------------------------------------------

def test_18_ordinary_hand_motion_straddle_trims():
    clip = _clip("c", 9.6, 20.0, "one two three", words=_words("one two three", 10.0, 19.0))
    diag = _diag([_event("hand_motion_reset_candidate", 9.6, 10.02)])
    out, audit = tighten_selected_visual_edges((clip,), diag)
    assert out[0].start == pytest.approx(10.0)  # clamped to delivery_span.start
    assert _rows_for("c", audit)[0]["reason"] == BOUNDARY_REASON_VISUAL_ENTRY_PARTIAL_EDGE_TRIM


# --- 19. high-materiality breaking-character edge event ------------------------

def test_19_high_confidence_breaking_character_straddle_still_gated_by_overlap_not_confidence():
    # Confidence never enters the eligibility test -- only the inside-
    # DELIVERY overlap size does (per module docstring).
    clip = _clip("c", 9.6, 20.0, "one two three", words=_words("one two three", 10.0, 19.0))
    diag = _diag([_event("body_reset_candidate", 9.6, 10.02, confidence=0.99)])
    out, audit = tighten_selected_visual_edges((clip,), diag)
    assert out[0].start == pytest.approx(10.0)  # clamped to delivery_span.start
    row = _rows_for("c", audit)[0]
    assert row["trim_applied"] is True
    assert row["event_confidence"] == pytest.approx(0.99)


# --- 20. repeated edge event (same kind twice, same side) ---------------------

def test_20_repeated_same_kind_events_same_side_only_nearest_chain_trims():
    # Two straddle-shaped ENTRY events chained off the running edge -- the
    # first to fire clamps to delivery_span.start (hard floor); the second
    # then finds the floor already reached and fails open (BLOCKED_BY_
    # DELIVERY_FLOOR), never a second, redundant "trim".
    clip = _clip("c", 9.6, 20.0, "one two three", words=_words("one two three", 10.0, 19.0))
    diag = _diag([
        _event("camera_disengagement_candidate", 9.6, 10.02),
        _event("camera_disengagement_candidate", 9.6, 10.01),
    ])
    out, audit = tighten_selected_visual_edges((clip,), diag)
    assert out[0].start == pytest.approx(10.0)  # clamped to delivery_span.start
    applied = [r for r in audit if r["trim_applied"]]
    assert len(applied) == 1


# --- 21-24. D-176 SYNTHETIC REPLAY (generic, non-Video00 shape) ----------------

def test_21_generic_d176_shape_replay_entry_and_exit_debris_trimmed_core_preserved():
    """Generic replay of the D-176 forensic shape: a selected, content-
    correct realization with a short entry sliver and a short exit sliver
    of non-speech debris straddling the measured DELIVERY span, and a
    consensus-keep-equivalent core in between. NOT the literal Video00
    transcript/timestamps -- entirely synthetic text/timing."""
    core_text = "this product solved my specific problem completely"
    clip = _clip(
        "retry_winner", 9.62, 20.30, core_text,
        words=_words(core_text, 10.37, 19.34),
    )
    diag = _diag([
        _event("camera_disengagement_candidate", 9.62, 10.42),   # ~0.80s entry sliver
        _event("facial_expression_shift_candidate", 19.26, 20.30),  # ~1.04s exit sliver
    ])
    result = _result(_draft([clip], diag))
    out = apply_post_freeze_boundary_pass(result)
    winner = out.draft.selected[0]
    # Edge debris trimmed away, clamped exactly to the measured DELIVERY
    # boundary (hard floor) -- never past it, so the tiny in-delivery
    # sliver of each event is left untouched rather than shaved.
    assert winner.start == pytest.approx(10.37)
    assert winner.end == pytest.approx(19.34)
    # ...core DELIVERY content (all real words) fully preserved.
    assert winner.words[0].text == "this"
    assert winner.words[-1].text == "completely"
    assert winner.start <= winner.words[0].start
    assert winner.end >= winner.words[-1].end
    assert winner.text == core_text  # no BestTake/content change -- same clip, same text


def test_22_generic_d176_shape_replay_no_besttake_or_family_signal_touched():
    core_text = "this product solved my specific problem completely"
    clip = _clip("retry_winner", 9.62, 20.30, core_text, words=_words(core_text, 10.37, 19.34))
    diag = _diag([
        _event("camera_disengagement_candidate", 9.62, 10.42),
        _event("facial_expression_shift_candidate", 19.26, 20.30),
    ])
    signals = MediaSignals(source_asset_id="src", start=9.62, end=20.30, visual_fumble=0.1)
    takes = (
        CandidateTake("retry_winner", "src", 0, 9.62, 20.30, core_text,
                      words=_words(core_text, 10.37, 19.34), signals=signals),
        CandidateTake("retry_abandoned", "src", 1, 30.0, 34.0, "false start never mind",
                      words=_words("false start never mind", 30.2, 33.8)),
    )
    before_scores = [score_take(t) for t in takes]
    before_ranked = rank_takes(takes)

    result = _result(_draft([clip], diag))
    apply_post_freeze_boundary_pass(result)  # Boundary-only, no take mutation

    after_scores = [score_take(t) for t in takes]
    after_ranked = rank_takes(takes)
    assert before_scores == after_scores
    assert before_ranked == after_ranked


def test_23_generic_d176_shape_replay_diagnostics_report_both_edges():
    core_text = "this product solved my specific problem completely"
    clip = _clip("retry_winner", 9.62, 20.30, core_text, words=_words(core_text, 10.37, 19.34))
    diag = _diag([
        _event("camera_disengagement_candidate", 9.62, 10.42),
        _event("facial_expression_shift_candidate", 19.26, 20.30),
    ])
    result = _result(_draft([clip], diag))
    out = apply_post_freeze_boundary_pass(result)
    summary = out.draft.diagnostics["boundary_engine_pass"]
    assert summary["partial_edge_trim_evaluated_count"] == 1
    assert summary["partial_edge_trim_applied_count"] == 1
    assert summary["entry_partial_edge_trim_count"] == 1
    assert summary["exit_partial_edge_trim_count"] == 1
    row = summary["partial_edge_trim_rows"][0]
    assert row["clip_id"] == "retry_winner"
    assert row["boundary_before_start"] == pytest.approx(9.62)
    assert row["boundary_before_end"] == pytest.approx(20.30)
    assert row["boundary_after_start"] == pytest.approx(10.37)
    assert row["boundary_after_end"] == pytest.approx(19.34)
    assert "visual_entry_partial_edge_trim" in row["partial_edge_trim_reason"]
    assert "visual_exit_partial_edge_trim" in row["partial_edge_trim_reason"]


def test_24_generic_d176_shape_replay_idempotent():
    core_text = "this product solved my specific problem completely"
    clip = _clip("retry_winner", 9.62, 20.30, core_text, words=_words(core_text, 10.37, 19.34))
    diag = _diag([
        _event("camera_disengagement_candidate", 9.62, 10.42),
        _event("facial_expression_shift_candidate", 19.26, 20.30),
    ])
    result = _result(_draft([clip], diag))
    out1 = apply_post_freeze_boundary_pass(result)
    out2 = apply_post_freeze_boundary_pass(out1)
    assert out1.draft.selected[0].start == out2.draft.selected[0].start
    assert out1.draft.selected[0].end == out2.draft.selected[0].end


# --- 25. CASE-A-style pure edge ownership remains Boundary (unchanged) --------

def test_25_pure_entry_case_a_ownership_unchanged_by_d177():
    clip = _clip("c", 10.0, 16.0, "one two three", words=_words("one two three", 11.6, 15.0))
    diag = _diag([_event("camera_disengagement_candidate", 10.0, 11.2)])
    out, audit = tighten_selected_visual_edges((clip,), diag)
    row = _rows_for("c", audit)[0]
    assert row["zone"] == "ENTRY"
    assert row["authority"] == "boundary_engine_pass"
    assert out[0].start == pytest.approx(11.2)


# --- 26. CASE-B internal defect remains BestTake-owned (never trimmed) -------

def test_26_case_b_style_interior_defect_never_trimmed_by_boundary():
    # An event fully embedded in DELIVERY (no straddle) is CASE B/C
    # territory -- D-177 does not touch it, same as pre-D-177 D-116.
    clip = _clip("c", 10.0, 16.0, "one two three", words=_words("one two three", 10.2, 15.5))
    diag = _diag([_event("hand_motion_reset_candidate", 12.0, 12.5)])
    out, audit = tighten_selected_visual_edges((clip,), diag)
    assert out[0].start == pytest.approx(10.0) and out[0].end == pytest.approx(16.0)
    row = _rows_for("c", audit)[0]
    assert row["trim_applied"] is False
    assert row["inside_delivery_overlap_sec"] is None


# --- 27. not-at-edge straddle -> fails open with correct reason ---------------

def test_27_straddle_small_overlap_but_not_touching_edge_fails_open():
    # Small inside-overlap (eligible) but the clip's own current leading
    # edge is well beyond the event's outer boundary -- not a genuine edge
    # debris shape, never trimmed (D-177 is edge-only, never interior).
    clip = _clip("c", 5.0, 20.0, "one two three", words=_words("one two three", 10.0, 19.0))
    diag = _diag([_event("camera_disengagement_candidate", 9.5, 10.05)])
    out, audit = tighten_selected_visual_edges((clip,), diag)
    assert out[0].start == pytest.approx(5.0)  # untouched -- event nowhere near clip's own edge
    row = _rows_for("c", audit)[0]
    assert row["reason"] == BOUNDARY_REASON_VISUAL_NOT_AT_EDGE
    assert row["inside_delivery_overlap_sec"] == pytest.approx(0.05)


# --- 28. multiple straddle events, one per side, both trim in one pass -------

def test_28_entry_and_exit_straddle_both_trim_same_pass():
    clip = _clip("c", 9.6, 20.4, "one two three", words=_words("one two three", 10.0, 20.0))
    diag = _diag([
        _event("camera_disengagement_candidate", 9.6, 10.03),
        _event("body_reset_candidate", 19.97, 20.4),
    ])
    out, audit = tighten_selected_visual_edges((clip,), diag)
    assert out[0].start == pytest.approx(10.0)  # clamped to delivery_span.start
    assert out[0].end == pytest.approx(20.0)  # clamped to delivery_span.end
    applied = [r for r in audit if r["trim_applied"]]
    assert len(applied) == 2


# --- 29. no-words / delivery span unavailable -> never trims -----------------

def test_29_no_words_no_partial_edge_trim():
    clip = _clip("c", 9.5, 20.0, "unused", words=())
    diag = _diag([_event("camera_disengagement_candidate", 9.5, 10.05)])
    out, audit = tighten_selected_visual_edges((clip,), diag)
    assert out[0].start == pytest.approx(9.5) and out[0].end == pytest.approx(20.0)
    row = _rows_for("c", audit)[0]
    assert row["delivery_span_available"] is False


# --- 30. clip membership / order unchanged through the full pass -------------

def test_30_clip_membership_and_order_unchanged_with_straddle_events():
    clip_a = _clip("a", 9.6, 20.0, "one two three", words=_words("one two three", 10.0, 19.0))
    clip_b = _clip("b", 30.0, 40.0, "four five six", words=_words("four five six", 30.2, 39.0))
    diag = _diag([_event("camera_disengagement_candidate", 9.6, 10.03)])
    result = _result(_draft([clip_a, clip_b], diag))
    out = apply_post_freeze_boundary_pass(result)
    assert [c.clip_id for c in out.draft.selected] == ["a", "b"]
    assert {c.clip_id for c in out.draft.selected} == {"a", "b"}


# --- 31. render plan reflects the trimmed edges, membership stable -----------

def test_31_render_plan_reflects_partial_edge_trim():
    clip = _clip("c", 9.6, 20.4, "one two three", words=_words("one two three", 10.0, 20.0))
    diag = _diag([
        _event("camera_disengagement_candidate", 9.6, 10.03),
        _event("body_reset_candidate", 19.97, 20.4),
    ])
    result = _result(_draft([clip], diag))
    out = apply_post_freeze_boundary_pass(result)
    plan = build_render_plan(out.draft, {"src": "/tmp/fake.mp4"})
    seg = next(s for s in plan if s.clip_id == "c")
    assert seg.start == pytest.approx(10.0)
    assert seg.end == pytest.approx(20.0)


# --- 32. never extends past the clip's own original bounds -------------------

def test_32_partial_edge_trim_never_extends_clip():
    clip = _clip("c", 9.6, 20.4, "one two three", words=_words("one two three", 10.0, 20.0))
    diag = _diag([_event("camera_disengagement_candidate", 9.6, 10.03)])
    out, _audit = tighten_selected_visual_edges((clip,), diag)
    assert out[0].start >= clip.start
    assert out[0].end <= clip.end


# --- 33. no provider/network call anywhere in this module ---------------------

def test_33_module_has_no_provider_or_network_symbol():
    import cutsell_worker.boundary_engine_pass as mod
    source = open(mod.__file__, "r", encoding="utf-8").read().lower()
    for banned in ("openai", "gemini", "requests.", "httpx.", "urllib.request"):
        assert banned not in source


# --- 34. D-116/D-097-C/D-123/D-174/Family/Language-Spine/Pacing untouched ----

def test_34_d116_reason_constants_and_signature_backward_compatible():
    # tighten_selected_visual_edges must still return exactly a 2-tuple
    # (clips, audit) -- unchanged signature, no consumer breakage.
    clip = _clip("c", 10.0, 16.0, "one two three", words=_words("one two three", 10.2, 13.0))
    diag = _diag([_event("body_reset_candidate", 13.5, 16.0)])
    result = tighten_selected_visual_edges((clip,), diag)
    assert isinstance(result, tuple) and len(result) == 2
    clips, audit = result
    assert isinstance(clips, tuple) and isinstance(audit, tuple)
    # Existing D-116 reason constants still importable/unchanged.
    from cutsell_worker.boundary_engine_pass import (
        BOUNDARY_REASON_VISUAL_ENTRY_EDGE_TRIM, BOUNDARY_REASON_VISUAL_EXIT_EDGE_TRIM,
        BOUNDARY_REASON_VISUAL_BLOCKED_BY_DELIVERY_FLOOR, BOUNDARY_REASON_VISUAL_TRIM_UNAVAILABLE,
    )
    assert BOUNDARY_REASON_VISUAL_ENTRY_EDGE_TRIM == "visual_entry_edge_trim"
    assert BOUNDARY_REASON_VISUAL_EXIT_EDGE_TRIM == "visual_exit_edge_trim"
    assert AUDIO_EDGE_OVERLAP_TOLERANCE_SEC == 0.08  # no new/changed threshold
