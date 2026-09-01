"""D-036: human_boundary_polish_v5's micro visual-reset-gap split is the
confirmed root cause of RAW 33415661351's STRUCTURAL_DUPLICATE_SEGMENT --
splitting one already-frozen semantic clip into physical left/right pieces
via `dataclasses.replace`, which kept the SAME `clip_id` for every piece
(and can split repeatedly, producing 3+ pieces sharing one id). These tests
prove the fix: every physical piece gets a fresh `render_fragment_id` while
`clip_id` (the semantic identity CanonicalEditPlan/Freeze reason about)
never changes, and `parent_semantic_clip_id` always points at the true root
semantic clip -- even across a second split of an already-split piece.
"""
from types import SimpleNamespace

from cutsell_worker.contracts import (
    DraftClip,
    DraftTimeline,
    EditStrategy,
    JobState,
    ProcessingResult,
    SCHEMA_VERSION,
    Word,
    effective_parent_semantic_clip_id,
    effective_render_fragment_id,
)
from cutsell_worker.human_boundary_polish_v5 import (
    BOUNDARY_REASON_MICRO_VISUAL_RESET_GAP,
    _remove_micro_visual_reset_word_gaps,
    polish_human_boundaries_v5,
)


def _word(text, start, end):
    return Word(text=text, start=start, end=end, confidence=0.9)


def _reset_event(start, end, confidence=0.9):
    return SimpleNamespace(
        source_asset_id="src", start=start, end=end,
        kind="body_reset_candidate", confidence=confidence, description="",
    )


def _timeline(*events):
    return SimpleNamespace(source_asset_id="src", events=tuple(events))


def _clip(words, *, clip_id="clip_root"):
    text = " ".join(w.text for w in words)
    return DraftClip(
        clip_id=clip_id, source_asset_id="src", source_order=0,
        start=float(words[0].start), end=float(words[-1].end),
        text=text, caption_text=text, words=tuple(words), selected=True,
    )


# ---------------------------------------------------------------------------
# 1. One semantic clip split into 2 legitimate physical fragments
# ---------------------------------------------------------------------------

def test_two_fragment_split_gets_unique_ids_and_shared_parent():
    words = (_word("one", 0.0, 1.0), _word("two", 1.4, 2.0))
    clip = _clip(words)
    # gap = 1.0 -> 1.4 (0.4s, in the >=0.34 band): needs score>=1.20, strong>=1.
    timeline = _timeline(_reset_event(0.9, 1.5, confidence=0.9), _reset_event(0.9, 1.5, confidence=0.9))

    pieces, rows = _remove_micro_visual_reset_word_gaps(clip, timeline)

    assert len(pieces) == 2
    assert len(rows) == 1
    assert rows[0]["action"] == "remove_micro_visual_reset_word_gap"
    assert rows[0]["semantic_membership_changed"] is False

    left, right = pieces
    # Semantic identity (clip_id) is untouched on both pieces.
    assert left.clip_id == "clip_root"
    assert right.clip_id == "clip_root"
    # Physical identity is fresh and distinct per piece.
    ids = {effective_render_fragment_id(left), effective_render_fragment_id(right)}
    assert len(ids) == 2
    assert "clip_root" not in ids  # neither piece's fragment id collides with the bare semantic id either
    # Both point back at the same semantic parent.
    assert effective_parent_semantic_clip_id(left) == "clip_root"
    assert effective_parent_semantic_clip_id(right) == "clip_root"
    assert left.boundary_reason == BOUNDARY_REASON_MICRO_VISUAL_RESET_GAP
    assert right.boundary_reason == BOUNDARY_REASON_MICRO_VISUAL_RESET_GAP
    # fragment_index/fragment_count reflect the FINAL piece count.
    assert (left.fragment_index, left.fragment_count) == (0, 2)
    assert (right.fragment_index, right.fragment_count) == (1, 2)
    # Text/word content is correctly partitioned, not lost or duplicated.
    assert left.text == "one"
    assert right.text == "two"


# ---------------------------------------------------------------------------
# 2. One semantic clip split into 3 legitimate fragments: order, no overlap
# ---------------------------------------------------------------------------

def test_three_fragment_split_preserves_order_and_all_unique_ids():
    words = (
        _word("one", 0.0, 1.0), _word("two", 1.4, 2.0), _word("three", 2.4, 3.4),
    )
    clip = _clip(words)
    timeline = _timeline(
        _reset_event(0.9, 1.5), _reset_event(0.9, 1.5),   # gap 1 (1.0-1.4)
        _reset_event(1.9, 2.5), _reset_event(1.9, 2.5),   # gap 2 (2.0-2.4)
    )

    pieces, rows = _remove_micro_visual_reset_word_gaps(clip, timeline)

    assert len(pieces) == 3
    assert len(rows) == 2
    fragment_ids = [effective_render_fragment_id(p) for p in pieces]
    assert len(set(fragment_ids)) == 3, "all three fragments must have distinct physical identity"
    assert all(effective_parent_semantic_clip_id(p) == "clip_root" for p in pieces)
    assert all(p.clip_id == "clip_root" for p in pieces), "semantic identity must never change"
    # Rendered order must be the source order -- no reordering.
    assert [round(p.start, 3) for p in pieces] == [0.0, 1.4, 2.4]
    assert [round(p.end, 3) for p in pieces] == [1.0, 2.0, 3.4]
    assert [p.text for p in pieces] == ["one", "two", "three"]
    assert [p.fragment_index for p in pieces] == [0, 1, 2]
    assert all(p.fragment_count == 3 for p in pieces)
    # No overlap between consecutive fragments.
    for left, right in zip(pieces, pieces[1:]):
        assert left.end <= right.start


# ---------------------------------------------------------------------------
# Re-splitting an ALREADY-split fragment keeps pointing at the true root
# ---------------------------------------------------------------------------

def test_resplitting_an_already_split_fragment_points_at_the_true_root():
    # Simulate a clip that a PRIOR Boundary pass already split (e.g.
    # post_selection_interior_gap_trim), carrying its own fragment identity
    # and a parent pointer -- human_boundary_polish_v5 must still resolve
    # the TRUE root, not treat the already-split piece as its own root.
    words = (_word("alpha", 0.0, 1.0), _word("beta", 1.4, 2.0))
    already_split_piece = DraftClip(
        clip_id="clip_root__priorchild", source_asset_id="src", source_order=0,
        start=0.0, end=2.0, text="alpha beta", caption_text="alpha beta",
        words=words, selected=True,
        render_fragment_id="clip_root__priorchild", parent_semantic_clip_id="clip_root",
    )
    timeline = _timeline(_reset_event(0.9, 1.5), _reset_event(0.9, 1.5))

    pieces, _rows = _remove_micro_visual_reset_word_gaps(already_split_piece, timeline)

    assert len(pieces) == 2
    for piece in pieces:
        assert piece.clip_id == "clip_root__priorchild"  # semantic identity of THIS piece preserved
        assert effective_parent_semantic_clip_id(piece) == "clip_root"  # true root, not itself
    ids = {effective_render_fragment_id(p) for p in pieces}
    assert len(ids) == 2
    assert "clip_root__priorchild" not in ids


# ---------------------------------------------------------------------------
# No qualifying gap: no split, no fragment fields set at all
# ---------------------------------------------------------------------------

def test_no_qualifying_gap_leaves_clip_untouched():
    words = (_word("one", 0.0, 1.0), _word("two", 1.02, 2.0))  # gap too small
    clip = _clip(words)
    pieces, rows = _remove_micro_visual_reset_word_gaps(clip, _timeline())

    assert pieces == (clip,)
    assert rows == []
    assert pieces[0].render_fragment_id is None
    assert pieces[0].parent_semantic_clip_id is None


# ---------------------------------------------------------------------------
# 13. Boundary physical fragmentation cannot modify semantic membership
# ---------------------------------------------------------------------------

def test_polish_human_boundaries_v5_never_changes_selected_clip_count_intent_or_order():
    # The pass may only ever SPLIT a selected clip into more physical pieces
    # covering the exact same spoken content in the exact same order -- it
    # must never drop, add, or reorder semantic membership.
    words_a = (_word("one", 0.0, 1.0), _word("two", 1.4, 2.0))
    words_b = (_word("three", 5.0, 6.0),)
    clip_a = _clip(words_a, clip_id="clip_a")
    clip_b = _clip(words_b, clip_id="clip_b")
    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(clip_a, clip_b), alternates=(), discarded=(),
        diagnostics={
            "whole_video_context": {
                "sources": [{
                    "source_asset_id": "src",
                    "events": [
                        {"start": 0.9, "end": 1.5, "kind": "body_reset_candidate", "confidence": 0.9},
                        {"start": 0.9, "end": 1.5, "kind": "body_reset_candidate", "confidence": 0.9},
                    ],
                }]
            }
        },
    )
    result = ProcessingResult(schema_version=SCHEMA_VERSION, project_id="p", state=JobState.DRAFT_READY, draft=draft, stage_status={})

    polished = polish_human_boundaries_v5(result, {})

    # clip_a split into 2 physical pieces, clip_b untouched -- total 3.
    assert len(polished.draft.selected) == 3
    assert [c.clip_id for c in polished.draft.selected] == ["clip_a", "clip_a", "clip_b"]
    combined_text = " ".join(c.text for c in polished.draft.selected if c.clip_id == "clip_a")
    assert combined_text == "one two"
    assert polished.draft.selected[-1].text == "three"
