from cutsell_worker.contracts import DraftClip, SemanticRole, Word
from cutsell_worker.post_selection_continuity_coalescer import coalesce_selected_source_continuity


def _clip(clip_id, start, end, text):
    return DraftClip(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
        caption_text=text,
        words=(Word(text, start + 0.05, end - 0.05),),
        semantic_role=SemanticRole.STORY,
        selected=True,
    )


def _diagnostics(events=None):
    return {"whole_video_context": {"sources": [{"source_asset_id": "src", "events": events or []}]}}


def test_same_source_micro_gap_without_reset_is_coalesced():
    left = _clip("a", 10.0, 12.0, "primera")
    right = _clip("b", 12.34, 14.0, "segunda")
    selected, audit = coalesce_selected_source_continuity((left, right), _diagnostics())
    assert len(selected) == 1
    assert selected[0].start == 10.0
    assert selected[0].end == 14.0
    assert selected[0].text == "primera segunda"
    assert len(audit) == 1
    assert audit[0]["source_gap_sec"] == 0.34


def test_strong_reset_blocks_coalescing_for_unrelated_parents():
    left = _clip("a", 10.0, 12.0, "primera")
    right = _clip("b", 12.34, 14.0, "segunda")
    events = [{"kind": "body_reset_candidate", "start": 12.1, "end": 12.4, "confidence": 0.97}]
    selected, audit = coalesce_selected_source_continuity((left, right), _diagnostics(events))
    assert len(selected) == 2
    assert audit == ()


def test_same_parent_micro_gap_ignores_isolated_body_reset():
    left = _clip("clip_parent__psigl123", 10.0, 12.0, "También me salían espinillas.")
    right = _clip("clip_parent__psigr456", 12.50, 14.0, "Era como un rush")
    events = [{"kind": "body_reset_candidate", "start": 12.1, "end": 12.45, "confidence": 0.97}]
    selected, audit = coalesce_selected_source_continuity((left, right), _diagnostics(events))
    assert len(selected) == 1
    assert selected[0].start == 10.0
    assert selected[0].end == 14.0
    assert audit[0]["same_logical_parent"] is True
    assert audit[0]["reason"] == "same_parent_micro_gap_without_explicit_retry_event"


def test_same_parent_micro_gap_still_blocks_explicit_retry():
    left = _clip("clip_parent__psigl123", 10.0, 12.0, "primera")
    right = _clip("clip_parent__psigr456", 12.42, 14.0, "segunda")
    events = [{"kind": "false_start", "start": 12.1, "end": 12.5, "confidence": 0.94}]
    selected, audit = coalesce_selected_source_continuity((left, right), _diagnostics(events))
    assert len(selected) == 2
    assert audit == ()


def test_same_parent_gap_above_micro_limit_fails_open():
    left = _clip("clip_parent__psigl123", 10.0, 12.0, "primera")
    right = _clip("clip_parent__psigr456", 12.8, 14.0, "segunda")
    selected, audit = coalesce_selected_source_continuity((left, right), _diagnostics())
    assert len(selected) == 2
    assert audit == ()


def test_larger_source_gap_fails_open():
    left = _clip("a", 10.0, 12.0, "primera")
    right = _clip("b", 12.8, 14.0, "segunda")
    selected, audit = coalesce_selected_source_continuity((left, right), _diagnostics())
    assert len(selected) == 2
    assert audit == ()
