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


def _diagnostics(events=None, boundary_splits=None):
    return {
        "whole_video_context": {"sources": [{"source_asset_id": "src", "events": events or []}]},
        "post_selection_interior_gap_trim": boundary_splits or [],
    }


def test_same_source_micro_gap_without_reset_is_coalesced():
    left = _clip("a", 10.0, 12.0, "primera")
    right = _clip("b", 12.34, 14.0, "segunda")
    selected, audit = coalesce_selected_source_continuity((left, right), _diagnostics())
    # D-097.3: the micro-gap is restored by extending the leading clip to the
    # next clip's start; both identities survive (no `__continuity__` id).
    assert [clip.clip_id for clip in selected] == ["a", "b"]
    assert selected[0].start == 10.0
    assert selected[0].end == 12.34
    assert selected[1].start == 12.34 and selected[1].end == 14.0
    assert selected[0].text == "primera" and selected[1].text == "segunda"
    assert len(audit) == 1
    assert audit[0]["source_gap_sec"] == 0.34
    assert audit[0]["action"] == "source_gap_restored_identities_preserved"
    assert audit[0]["merged_parent_ids"] == ["a", "b"]


def test_strong_reset_blocks_coalescing():
    left = _clip("a", 10.0, 12.0, "primera")
    right = _clip("b", 12.34, 14.0, "segunda")
    events = [{"kind": "body_reset_candidate", "start": 12.1, "end": 12.4, "confidence": 0.97}]
    selected, audit = coalesce_selected_source_continuity((left, right), _diagnostics(events))
    assert len(selected) == 2
    assert audit == ()


def test_larger_source_gap_fails_open():
    left = _clip("a", 10.0, 12.0, "primera")
    right = _clip("b", 12.8, 14.0, "segunda")
    selected, audit = coalesce_selected_source_continuity((left, right), _diagnostics())
    assert len(selected) == 2
    assert audit == ()


def test_boundary_authorized_micro_gap_is_never_recoalesced():
    left = _clip("left", 10.0, 12.0, "primera")
    right = _clip("right", 12.44, 14.0, "segunda")
    boundary_splits = [{
        "authority": "post_selection_interior_gap_trim",
        "decision": "split",
        "removed_gap_start": 12.0,
        "removed_gap_end": 12.44,
        "removed_gap_sec": 0.44,
        "evidence_mode": "completed_sentence_anticipatory_reset",
    }]
    selected, audit = coalesce_selected_source_continuity(
        (left, right),
        _diagnostics(boundary_splits=boundary_splits),
    )
    assert len(selected) == 2
    assert audit == ()


def test_restored_gap_renders_as_one_contiguous_segment_without_losing_identity():
    # The physical outcome of the old identity merge is preserved by the
    # render plan's mechanical contiguous coalesce (same source, exactly
    # contiguous), while StoryValidator still sees both selected ids.
    from cutsell_worker.contracts import DraftTimeline, EditStrategy, SCHEMA_VERSION
    from cutsell_worker.final_story_coherence_validation import _missing_idea_coverage
    from cutsell_worker.render_plan import build_render_plan

    from dataclasses import replace as _replace

    # Captions off (the RAW harness default): the render plan's mechanical
    # coalesce only joins segments whose playback/caption settings agree.
    left = _replace(_clip("a", 10.0, 12.0, "primera"), caption_text="")
    right = _replace(_clip("b", 12.34, 14.0, "segunda"), caption_text="")
    selected, _ = coalesce_selected_source_continuity((left, right), _diagnostics())
    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=selected, alternates=(), discarded=(),
        diagnostics={"take_judge_groups": [
            {"group_id": "tg_a", "ranked": [{"clip_id": "a"}, {"clip_id": "x"}]},
            {"group_id": "tg_b", "ranked": [{"clip_id": "b"}]},
        ]},
    )
    assert _missing_idea_coverage(draft) == []
    plan = build_render_plan(draft, {"src": "/tmp/does-not-need-to-exist.mp4"})
    assert len(plan) == 1 and plan[0].start == 10.0 and plan[0].end == 14.0
