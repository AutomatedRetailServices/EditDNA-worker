"""D-116 -- Boundary CASE A visual edge consumption of the D-115 canonical
ENTRY/DELIVERY/EXIT evidence.

Boundary never recomputes delivery start/end, ENTRY/DELIVERY/EXIT, or
"event overlap" itself -- it calls `positioned_performance_evidence.
compute_delivery_span`/`classify_event_zone` directly. A DELIVERY-zone
event (including any straddling event, which D-115 classifies as DELIVERY
on any overlap) is never trimmed -- that is reserved for a future,
separately-authorized BestTake/DeliveryScorer consumer (CASE B). Generic
fixtures only -- no Video00 text/ids/timestamps.
"""
from cutsell_worker.boundary_engine_pass import (
    BOUNDARY_REASON_VISUAL_BLOCKED_BY_DELIVERY_FLOOR,
    BOUNDARY_REASON_VISUAL_DELIVERY_OVERLAP_NO_TRIM,
    BOUNDARY_REASON_VISUAL_ENTRY_EDGE_TRIM,
    BOUNDARY_REASON_VISUAL_EXIT_EDGE_TRIM,
    BOUNDARY_REASON_VISUAL_NOT_AT_EDGE,
    BOUNDARY_REASON_VISUAL_TRIM_UNAVAILABLE,
    apply_post_freeze_boundary_pass,
    tighten_selected_audio_edges,
    tighten_selected_visual_edges,
)
from cutsell_worker.contracts import (
    CandidateTake, DraftClip, DraftTimeline, EditStrategy, JobState,
    MediaSignals, ProcessingResult, SCHEMA_VERSION, SemanticRole, Word,
)
from cutsell_worker.positioned_performance_evidence import (
    build_positioned_performance_evidence_for_takes,
    positioned_performance_evidence_diagnostics,
)
from cutsell_worker.post_selection_interior_gap_trim import AUDIO_SILENCE_EVENT_KIND
from cutsell_worker.render_plan import build_render_plan
from cutsell_worker.take_judge import rank_takes, score_take

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


def _visual_rows_for(clip_id, audit):
    return [row for row in audit if row["clip_id"] == clip_id]


# --- CASE A: EXIT ------------------------------------------------------------

def test_exit_body_reset_fully_after_delivery_trims_end_safely():
    # The reset continues to (or past) the clip's own current trailing edge --
    # a valid contiguous suffix removal, exactly the shape the directive's
    # own Pimples-C QA fixture describes.
    clip = _clip("c", 10.0, 16.0, "one two three", words=_words("one two three", 10.2, 13.0))
    diag = _diag([_event("body_reset_candidate", 13.5, 16.0)])
    out, audit = tighten_selected_visual_edges((clip,), diag)
    assert out[0].end == pytest.approx(13.5)  # trimmed exactly to the event's own start
    assert out[0].start == pytest.approx(10.0)  # entry untouched
    assert out[0].boundary_reason == BOUNDARY_REASON_VISUAL_EXIT_EDGE_TRIM
    row = _visual_rows_for("c", audit)[0]
    assert row["reason"] == BOUNDARY_REASON_VISUAL_EXIT_EDGE_TRIM
    assert row["trim_applied"] is True
    assert row["trim_side"] == "EXIT"
    assert row["zone"] == "EXIT"


def test_entry_camera_disengagement_fully_before_delivery_trims_start_safely():
    clip = _clip("c", 10.0, 16.0, "one two three", words=_words("one two three", 11.6, 15.0))
    diag = _diag([_event("camera_disengagement_candidate", 10.0, 11.2)])
    out, audit = tighten_selected_visual_edges((clip,), diag)
    assert out[0].start == pytest.approx(11.2)  # trimmed exactly to the event's own end
    assert out[0].end == pytest.approx(16.0)  # exit untouched
    assert out[0].boundary_reason == BOUNDARY_REASON_VISUAL_ENTRY_EDGE_TRIM
    row = _visual_rows_for("c", audit)[0]
    assert row["reason"] == BOUNDARY_REASON_VISUAL_ENTRY_EDGE_TRIM
    assert row["trim_applied"] is True
    assert row["zone"] == "ENTRY"
    # first spoken word remains fully intact
    assert out[0].start <= out[0].words[0].start


def test_exit_trim_never_crosses_delivery_end():
    # The event starts BEFORE the last spoken word ends -> would require
    # cutting into DELIVERY to honor the event's own start; the hard floor
    # (delivery_span.end) wins instead.
    clip = _clip("c", 10.0, 16.0, "one two three", words=_words("one two three", 10.2, 15.5))
    diag = _diag([_event("body_reset_candidate", 14.0, 15.9)])
    out, audit = tighten_selected_visual_edges((clip,), diag)
    zone = _visual_rows_for("c", audit)[0]["zone"]
    assert zone == "DELIVERY"  # overlaps the delivery span -> D-115 classifies DELIVERY
    assert out[0].end == pytest.approx(16.0)  # untouched
    assert _visual_rows_for("c", audit)[0]["reason"] == BOUNDARY_REASON_VISUAL_DELIVERY_OVERLAP_NO_TRIM


def test_entry_trim_never_crosses_delivery_start():
    clip = _clip("c", 10.0, 16.0, "one two three", words=_words("one two three", 10.5, 15.0))
    diag = _diag([_event("camera_disengagement_candidate", 10.0, 10.8)])
    out, audit = tighten_selected_visual_edges((clip,), diag)
    row = _visual_rows_for("c", audit)[0]
    assert row["zone"] == "DELIVERY"  # 10.8 > delivery_start(10.5) -> overlaps
    assert row["reason"] == BOUNDARY_REASON_VISUAL_DELIVERY_OVERLAP_NO_TRIM
    assert out[0].start == pytest.approx(10.0)  # untouched, first word intact


# --- CASE B/C exclusion: no trim into DELIVERY --------------------------------

def test_event_fully_inside_delivery_no_visual_trim():
    clip = _clip("c", 10.0, 16.0, "one two three", words=_words("one two three", 10.2, 15.5))
    diag = _diag([_event("hand_motion_reset_candidate", 12.0, 12.5)])
    out, audit = tighten_selected_visual_edges((clip,), diag)
    assert out[0].start == pytest.approx(10.0) and out[0].end == pytest.approx(16.0)
    row = _visual_rows_for("c", audit)[0]
    assert row["zone"] == "DELIVERY"
    assert row["reason"] == BOUNDARY_REASON_VISUAL_DELIVERY_OVERLAP_NO_TRIM
    assert row["trim_applied"] is False


def test_event_crossing_delivery_start_preserves_delivery():
    clip = _clip("c", 10.0, 16.0, "one two three four", words=_words("one two three four", 12.0, 15.0))
    diag = _diag([_event("facial_expression_shift_candidate", 11.5, 12.5)])
    out, audit = tighten_selected_visual_edges((clip,), diag)
    assert out[0].start == pytest.approx(10.0)  # never cut past the entry side either
    row = _visual_rows_for("c", audit)[0]
    assert row["zone"] == "DELIVERY"
    assert row["trim_applied"] is False


def test_event_crossing_delivery_end_preserves_delivery():
    clip = _clip("c", 10.0, 16.0, "one two three four", words=_words("one two three four", 11.0, 14.0))
    diag = _diag([_event("body_reset_candidate", 13.5, 14.5)])
    out, audit = tighten_selected_visual_edges((clip,), diag)
    assert out[0].end == pytest.approx(16.0)
    row = _visual_rows_for("c", audit)[0]
    assert row["zone"] == "DELIVERY"
    assert row["trim_applied"] is False


# --- edge-only, never interior -----------------------------------------------

def test_interior_event_not_at_edge_no_trim():
    # ENTRY-zone (well before delivery) but nowhere near the clip's own
    # current leading edge -> D-116 is edge-tightening only, never a split.
    clip = _clip("c", 0.0, 16.0, "one two three", words=_words("one two three", 12.0, 15.0))
    diag = _diag([_event("body_reset_candidate", 4.0, 4.5)])
    out, audit = tighten_selected_visual_edges((clip,), diag)
    assert out[0].start == pytest.approx(0.0) and out[0].end == pytest.approx(16.0)
    row = _visual_rows_for("c", audit)[0]
    assert row["reason"] == BOUNDARY_REASON_VISUAL_NOT_AT_EDGE
    assert row["trim_applied"] is False


# --- multiple events -----------------------------------------------------------

def test_multiple_exit_events_safe_deterministic_tightening():
    clip = _clip("c", 10.0, 20.0, "one two three", words=_words("one two three", 10.2, 13.0))
    # Two contiguous exit events chained from the clip's own trailing edge
    # inward -- the tighter (leftmost) safe cut wins.
    diag = _diag([
        _event("body_reset_candidate", 18.0, 20.0),
        _event("hand_motion_reset_candidate", 15.0, 18.05),
    ])
    out, audit = tighten_selected_visual_edges((clip,), diag)
    assert out[0].end == pytest.approx(15.0)  # chained through both contiguous events
    rows = _visual_rows_for("c", audit)
    applied = [r for r in rows if r["trim_applied"]]
    assert len(applied) == 2  # both events contributed to the chain


def test_multiple_entry_events_safe_deterministic_tightening():
    clip = _clip("c", 0.0, 20.0, "one two three", words=_words("one two three", 4.0, 8.0))
    diag = _diag([
        _event("camera_disengagement_candidate", 0.0, 1.0),
        _event("facial_expression_shift_candidate", 1.0, 3.8),
    ])
    out, audit = tighten_selected_visual_edges((clip,), diag)
    assert out[0].start == pytest.approx(3.8)
    rows = _visual_rows_for("c", audit)
    applied = [r for r in rows if r["trim_applied"]]
    assert len(applied) == 2


# --- audio/visual coexistence, one pass --------------------------------------

def test_audio_only_behavior_remains_unchanged():
    clip = _clip("c", 10.0, 16.0, "uno dos tres", words=_words("uno dos tres", 11.2, 16.0))
    diag = _diag([_event(AUDIO_SILENCE_EVENT_KIND, 9.4, 11.0, confidence=1.0)])
    audio_only, audio_audit = tighten_selected_audio_edges((clip,), diag)
    full_pass_out, full_pass_audit = tighten_selected_visual_edges(audio_only, diag)
    # No visual event kinds present -> the visual pass is a no-op on top of audio.
    assert full_pass_out[0].start == audio_only[0].start
    assert full_pass_out[0].end == audio_only[0].end
    assert full_pass_audit == ()


def test_audio_and_visual_coexist_without_second_pass():
    clip = _clip("c", 10.0, 20.0, "uno dos tres", words=_words("uno dos tres", 11.2, 15.0))
    diag = _diag([
        _event(AUDIO_SILENCE_EVENT_KIND, 9.4, 11.0, confidence=1.0),
        _event("body_reset_candidate", 15.5, 20.0),
    ])
    result = _result(_draft([clip], diag))
    out = apply_post_freeze_boundary_pass(result)
    selected = out.draft.selected
    assert len(selected) == 1
    # Audio tightened the entry, visual tightened the exit -- ONE pass, both applied.
    assert selected[0].start == pytest.approx(10.9)  # silence end minus the existing audio pad
    assert selected[0].end == pytest.approx(15.5)  # trimmed to the visual event's own start
    summary = out.draft.diagnostics["boundary_engine_pass"]
    assert summary["audio_entry_trim_count"] == 1
    assert summary["visual_exit_trim_count"] == 1
    assert "boundary_visual_edge_trim" in out.draft.diagnostics


# --- membership / order / monotonicity ---------------------------------------

def test_clip_membership_and_order_unchanged():
    clip_a = _clip("a", 0.0, 6.0, "one two", words=_words("one two", 0.5, 5.5))
    clip_b = _clip("b", 10.0, 16.0, "three four", words=_words("three four", 10.2, 13.0))
    diag = _diag([
        _event("camera_disengagement_candidate", 0.0, 0.4),  # clip_a lead-in, safely trimmable
        _event("body_reset_candidate", 13.5, 16.0),
    ])
    result = _result(_draft([clip_a, clip_b], diag))
    out = apply_post_freeze_boundary_pass(result)
    assert [c.clip_id for c in out.draft.selected] == ["a", "b"]  # order preserved
    assert {c.clip_id for c in out.draft.selected} == {"a", "b"}  # membership preserved


def test_boundary_only_tightens_never_extends():
    clip = _clip("c", 10.0, 16.0, "one two three", words=_words("one two three", 10.2, 13.0))
    diag = _diag([_event("body_reset_candidate", 13.5, 15.9)])
    out, _audit = tighten_selected_visual_edges((clip,), diag)
    assert out[0].start >= clip.start
    assert out[0].end <= clip.end


def test_render_plan_membership_unchanged():
    clip_a = _clip("a", 0.0, 6.0, "one two", words=_words("one two", 0.5, 5.5))
    clip_b = _clip("b", 10.0, 16.0, "three four", words=_words("three four", 10.2, 13.0))
    diag = _diag([_event("body_reset_candidate", 13.5, 16.0)])
    result = _result(_draft([clip_a, clip_b], diag))
    before_plan = build_render_plan(result.draft, {"src": "/tmp/fake.mp4"})
    out = apply_post_freeze_boundary_pass(result)
    after_plan = build_render_plan(out.draft, {"src": "/tmp/fake.mp4"})
    assert [seg.clip_id for seg in before_plan] == [seg.clip_id for seg in after_plan] == ["a", "b"]
    # Boundary tightened clip b's exit -- a real, expected physical change.
    after_b = next(seg for seg in after_plan if seg.clip_id == "b")
    assert after_b.end == pytest.approx(13.5)
    assert after_b.end <= 16.0


# --- no-words / defaults -------------------------------------------------------

def test_no_word_delivery_span_no_unsafe_visual_trim():
    clip = _clip("c", 10.0, 16.0, "unused", words=())
    diag = _diag([_event("body_reset_candidate", 13.5, 15.9)])
    out, audit = tighten_selected_visual_edges((clip,), diag)
    assert out[0].start == pytest.approx(10.0) and out[0].end == pytest.approx(16.0)
    row = _visual_rows_for("c", audit)[0]
    assert row["reason"] == BOUNDARY_REASON_VISUAL_TRIM_UNAVAILABLE
    assert row["delivery_span_available"] is False
    assert row["trim_applied"] is False


def test_default_only_mediasignals_have_no_effect():
    # A clip carrying only default-scalar MediaSignals (no real local-
    # performance aggregation at all) trims identically to one with none --
    # the visual path never reads MediaSignals, only real positioned events.
    words = _words("one two three", 10.2, 13.0)
    clip_with_defaults = DraftClip(
        clip_id="c", source_asset_id="src", source_order=0, start=10.0, end=16.0,
        text="one two three", caption_text="one two three", words=words,
        semantic_role=SemanticRole.STORY, selected=True,
        signals=MediaSignals(source_asset_id="src", start=10.0, end=16.0),
    )
    diag = _diag([_event("body_reset_candidate", 13.5, 16.0)])
    out, _audit = tighten_selected_visual_edges((clip_with_defaults,), diag)
    assert out[0].end == pytest.approx(13.5)  # identical to the no-signals case above


# --- D-115 / BestTake / DeliveryScorer regression -----------------------------

def test_d115_diagnostics_remain_valid_after_d116():
    take = CandidateTake(
        "c1", "src", 0, 10.0, 16.0, "one two three",
        words=_words("one two three", 10.2, 13.0),
    )
    context = None
    evidence = build_positioned_performance_evidence_for_takes((take,), context)
    rows = positioned_performance_evidence_diagnostics(evidence)
    assert len(rows) == 1
    assert rows[0]["delivery_span"]["available"] is True


def test_besttake_deliveryscorer_results_unchanged_by_boundary():
    signals = MediaSignals(source_asset_id="src", start=10.0, end=16.0, visual_fumble=0.2)
    takes = (
        CandidateTake("c1", "src", 0, 10.0, 16.0, "one two three",
                      words=_words("one two three", 10.2, 13.0), signals=signals),
        CandidateTake("c2", "src", 1, 20.0, 26.0, "four five six",
                      words=_words("four five six", 20.5, 25.5)),
    )
    before = [score_take(t) for t in takes]
    before_ranked = rank_takes(takes)

    clip = _clip("c1", 10.0, 16.0, "one two three", words=_words("one two three", 10.2, 13.0))
    diag = _diag([_event("body_reset_candidate", 13.5, 15.9)])
    apply_post_freeze_boundary_pass(_result(_draft([clip], diag)))  # D-116, boundary-only

    after = [score_take(t) for t in takes]
    after_ranked = rank_takes(takes)
    assert before == after
    assert before_ranked == after_ranked
