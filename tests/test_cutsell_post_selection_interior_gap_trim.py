from cutsell_worker.contracts import DraftClip, SemanticRole, Word
from cutsell_worker.post_selection_interior_gap_trim import split_selected_interior_performance_gaps


def _clip():
    words = (
        Word("uno", 0.10, 0.40),
        Word("dos", 0.50, 0.80),
        Word("tres", 0.90, 1.20),
        Word("cuatro", 2.20, 2.50),
        Word("cinco", 2.60, 2.90),
        Word("seis", 3.00, 3.30),
    )
    return DraftClip(
        clip_id="clip-a",
        source_asset_id="src",
        source_order=0,
        start=0.0,
        end=3.5,
        text="uno dos tres cuatro cinco seis",
        caption_text="uno dos tres cuatro cinco seis",
        words=words,
        semantic_role=SemanticRole.STORY,
        selected=True,
    )


def _diagnostics(with_reset=True):
    events = []
    if with_reset:
        events = [
            {"kind": "hand_motion_reset_candidate", "start": 1.25, "end": 1.80, "confidence": 0.96},
            {"kind": "body_reset_candidate", "start": 1.35, "end": 1.90, "confidence": 0.95},
            {"kind": "facial_expression_shift_candidate", "start": 1.40, "end": 1.95, "confidence": 0.88},
        ]
    return {
        "whole_video_context": {
            "sources": [
                {"source_asset_id": "src", "events": events}
            ]
        }
    }


def _long_gap_clip(left_terminal=True):
    left_last = "termina." if left_terminal else "termina"
    words = (
        Word("esta", 0.10, 0.35),
        Word("idea", 0.45, 0.70),
        Word(left_last, 0.80, 1.10),
        Word("otra", 2.75, 3.00),
        Word("idea", 3.10, 3.35),
        Word("continúa", 3.45, 3.80),
    )
    return DraftClip(
        clip_id="clip-long",
        source_asset_id="src",
        source_order=0,
        start=0.0,
        end=4.0,
        text=" ".join(word.text for word in words),
        caption_text=" ".join(word.text for word in words),
        words=words,
        semantic_role=SemanticRole.STORY,
        selected=True,
    )


def _physical_only_diagnostics(two_resets=True):
    events = [
        {"kind": "hand_motion_reset_candidate", "start": 1.45, "end": 1.70, "confidence": 0.96},
    ]
    if two_resets:
        events.append(
            {"kind": "body_reset_candidate", "start": 2.10, "end": 2.35, "confidence": 0.95}
        )
    return {
        "whole_video_context": {
            "sources": [
                {"source_asset_id": "src", "events": events}
            ]
        }
    }


def _short_completed_gap_clip(gap_sec=0.44):
    right_start = 3.00 + gap_sec
    words = (
        Word("esta", 0.60, 0.90),
        Word("frase", 1.20, 1.50),
        Word("termina.", 2.70, 3.00),
        Word("otra", right_start, right_start + 0.25),
        Word("frase", right_start + 0.35, right_start + 0.60),
        Word("continúa", right_start + 0.70, right_start + 1.00),
    )
    return DraftClip(
        clip_id="clip-short-reset",
        source_asset_id="src",
        source_order=0,
        start=0.0,
        end=4.8,
        text=" ".join(word.text for word in words),
        caption_text=" ".join(word.text for word in words),
        words=words,
        semantic_role=SemanticRole.STORY,
        selected=True,
    )


def _anticipatory_reset_diagnostics(*, include_near_gap=True):
    events = [
        {"kind": "hand_motion_reset_candidate", "start": 1.10, "end": 1.17, "confidence": 1.0},
    ]
    if include_near_gap:
        events.append(
            {"kind": "hand_motion_reset_candidate", "start": 2.65, "end": 2.72, "confidence": 1.0}
        )
    else:
        events.append(
            {"kind": "hand_motion_reset_candidate", "start": 1.55, "end": 1.62, "confidence": 1.0}
        )
    return {
        "whole_video_context": {
            "sources": [
                {"source_asset_id": "src", "events": events}
            ]
        }
    }


def test_multimodal_speech_free_interior_gap_is_split():
    selected, audit = split_selected_interior_performance_gaps((_clip(),), _diagnostics(True))

    assert len(selected) == 2
    assert selected[0].end == 1.20
    assert selected[1].start == 2.20
    assert [w.text for w in selected[0].words] == ["uno", "dos", "tres"]
    assert [w.text for w in selected[1].words] == ["cuatro", "cinco", "seis"]
    assert len(audit) == 1
    assert audit[0]["removed_gap_sec"] == 1.0
    assert audit[0]["evidence_mode"] == "multimodal_break"


def test_gap_without_multimodal_reset_is_kept():
    selected, audit = split_selected_interior_performance_gaps((_clip(),), _diagnostics(False))

    assert len(selected) == 1
    assert selected[0].clip_id == "clip-a"
    assert selected[0].start == 0.0
    assert selected[0].end == 3.5
    assert audit == ()


def test_long_completed_sentence_gap_with_two_physical_resets_is_split_without_face_break():
    clip = _long_gap_clip(left_terminal=True)
    selected, audit = split_selected_interior_performance_gaps(
        (clip,), _physical_only_diagnostics(two_resets=True)
    )

    assert len(selected) == 2
    assert selected[0].end == 1.10
    assert selected[1].start == 2.75
    assert [w.text for child in selected for w in child.words] == [w.text for w in clip.words]
    assert " ".join(child.text for child in selected) == clip.text
    assert len(audit) == 1
    assert audit[0]["removed_gap_sec"] == 1.65
    assert audit[0]["evidence_mode"] == "long_gap_physical_reset"


def test_long_gap_without_completed_left_sentence_is_kept_without_face_break():
    selected, audit = split_selected_interior_performance_gaps(
        (_long_gap_clip(left_terminal=False),), _physical_only_diagnostics(two_resets=True)
    )

    assert len(selected) == 1
    assert audit == ()


def test_long_gap_with_only_one_physical_reset_is_kept_without_face_break():
    selected, audit = split_selected_interior_performance_gaps(
        (_long_gap_clip(left_terminal=True),), _physical_only_diagnostics(two_resets=False)
    )

    assert len(selected) == 1
    assert audit == ()


def test_short_completed_sentence_gap_with_anticipatory_and_near_gap_resets_is_split():
    clip = _short_completed_gap_clip(gap_sec=0.44)
    selected, audit = split_selected_interior_performance_gaps(
        (clip,), _anticipatory_reset_diagnostics(include_near_gap=True)
    )

    assert len(selected) == 2
    assert selected[0].end == 3.00
    assert selected[1].start == 3.44
    assert [w.text for child in selected for w in child.words] == [w.text for w in clip.words]
    assert len(audit) == 1
    assert audit[0]["removed_gap_sec"] == 0.44
    assert audit[0]["evidence_mode"] == "completed_sentence_anticipatory_reset"


def test_short_completed_sentence_gap_without_near_gap_reset_is_kept():
    selected, audit = split_selected_interior_performance_gaps(
        (_short_completed_gap_clip(gap_sec=0.44),),
        _anticipatory_reset_diagnostics(include_near_gap=False),
    )

    assert len(selected) == 1
    assert audit == ()


def test_sub_040_completed_sentence_gap_is_kept_even_with_anticipatory_resets():
    selected, audit = split_selected_interior_performance_gaps(
        (_short_completed_gap_clip(gap_sec=0.36),),
        _anticipatory_reset_diagnostics(include_near_gap=True),
    )

    assert len(selected) == 1
    assert audit == ()


# --- D-046 FIX A: D-036 fragment provenance must be stamped on a split ----


def test_split_pieces_carry_parent_semantic_clip_id_back_to_the_original():
    selected, _ = split_selected_interior_performance_gaps((_clip(),), _diagnostics(True))

    assert len(selected) == 2
    for piece in selected:
        assert piece.parent_semantic_clip_id == "clip-a"
        assert piece.render_fragment_id == piece.clip_id
        assert piece.boundary_reason == "remove_interior_performance_gap"
    assert selected[0].fragment_index == 0
    assert selected[1].fragment_index == 1
    assert selected[0].fragment_count == 2
    assert selected[1].fragment_count == 2


def test_unsplit_clip_carries_no_fragment_provenance():
    selected, _ = split_selected_interior_performance_gaps((_clip(),), _diagnostics(False))

    assert len(selected) == 1
    assert selected[0].parent_semantic_clip_id is None
    assert selected[0].fragment_index is None
    assert selected[0].fragment_count is None


def test_chained_split_keeps_parent_semantic_clip_id_pointed_at_the_true_root():
    # A clip that already carries fragment provenance from an EARLIER
    # physical pass (e.g. this hook re-splitting a piece human_boundary_
    # polish_v5 already touched, or a second interior gap in the same
    # original) must keep pointing descendants at the ROOT semantic clip,
    # never at the intermediate fragment's own clip_id.
    from dataclasses import replace

    already_fragment = replace(_clip(), clip_id="root__psiglabc", parent_semantic_clip_id="root")
    selected, _ = split_selected_interior_performance_gaps((already_fragment,), _diagnostics(True))

    assert len(selected) == 2
    for piece in selected:
        assert piece.parent_semantic_clip_id == "root"
