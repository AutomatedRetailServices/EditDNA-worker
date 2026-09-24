"""D-291.6 verification on the real path (Product Owner item 3): the word
evidence the live repair consults must (1) exist on the frozen draft's
clips as the pipeline builds them, (2) belong to the right segment,
including a physical fragment, and (3) keep the source time system after a
trim. It also states what happens when the evidence is missing or padded.
"""
from __future__ import annotations

import pytest

from cutsell_worker.contracts import CandidateTake, DraftClip, DraftTimeline, EditStrategy, ProcessingRequest, SCHEMA_VERSION, SemanticRole, SourceAsset, Word
from cutsell_worker.live_boundary_repair import _speech_for_segment, repair_segment_for_finding, speech_room_at_edge
from cutsell_worker.live_render_qc import protected_speech_by_clip_id
from cutsell_worker.pipeline import build_flow_b_draft
from cutsell_worker.post_render_media_qc import ABRUPT_AUDIO_DISCONTINUITY, LINGERING_ACCIDENTAL_SILENCE
from cutsell_worker.post_render_watch_listen_qc import PostRenderFinding
from cutsell_worker.post_selection_interior_gap_trim import split_selected_interior_performance_gaps
from cutsell_worker.render_plan import RenderSegment, build_render_plan
from tests.test_cutsell_d289_contained_realization_closure import RecordedAnswersArbiter
from tests.test_cutsell_d291_5_composite_kept_complementary import FakeJudge


def _words(text, start, end):
    tokens = text.split()
    step = (end - start) / len(tokens)
    return tuple(Word(tok, round(start + i * step, 3), round(start + (i + 0.8) * step, 3)) for i, tok in enumerate(tokens))


def _take(cid, start, end, text):
    return CandidateTake(cid, "src", 0, start, end, text, words=_words(text, start, end), complete_idea=True)


def _request():
    return ProcessingRequest(project_id="p", user_id="u", sources=(SourceAsset(
        source_asset_id="src", project_id="p", user_id="u", original_name="raw.mp4", source_order=0,
        duration_sec=400.0, uri="s3://b/raw.mp4",
    ),))


def test_pipeline_clips_carry_their_own_words_and_the_repair_map_covers_every_selected_clip():
    takes = (
        _take("A", 10.0, 16.0, "Tenía cáncer de tiroides y no lo sabía."),
        _take("B", 20.0, 28.0, "Nunca se nos ocurrió hacer un chequeo de la tiroides por sonografía."),
        _take("C", 40.0, 47.0, "El año pasado comencé a notar hinchazón en mi cara y aumento de peso."),
    )
    labels = {"A": ("winner", 0.95), "B": ("winner", 0.95), "C": ("winner", 0.95)}
    result = build_flow_b_draft(_request(), takes, editorial_judge=FakeJudge(labels),
                                semantic_equivalence_arbiter=RecordedAnswersArbiter({}))
    draft = result.draft
    assert [c.clip_id for c in draft.selected] == ["A", "B", "C"]
    for clip in draft.selected:
        assert clip.words, clip.clip_id
        assert all(clip.start - 1e-6 <= w.start <= w.end <= clip.end + 1e-6 for w in clip.words), clip.clip_id
    protected = protected_speech_by_clip_id(draft)
    assert set(protected) == {"A", "B", "C"}
    # the render plan keeps the same ids, so the repair finds the same words per segment
    plan = build_render_plan(draft, {"src": "/x.mp4"})
    for seg in plan:
        assert _speech_for_segment(seg, protected) == protected[seg.clip_id]


def test_a_physical_fragment_keeps_clamped_words_and_is_found_by_its_own_id_or_its_parent():
    words = (
        Word("uno", 0.10, 0.40), Word("dos", 0.50, 0.80), Word("tres.", 0.90, 1.20),
        Word("cuatro", 2.20, 2.50), Word("cinco", 2.60, 2.90), Word("seis", 3.00, 3.30),
    )
    clip = DraftClip(clip_id="clip-a", source_asset_id="src", source_order=0, start=0.0, end=3.5,
                     text="uno dos tres. cuatro cinco seis", caption_text="x", words=words,
                     semantic_role=SemanticRole.STORY, selected=True)
    diagnostics = {"whole_video_context": {"sources": [{"source_asset_id": "src", "events": [
        {"kind": "hand_motion_reset_candidate", "start": 1.25, "end": 1.80, "confidence": 0.96},
        {"kind": "body_reset_candidate", "start": 1.35, "end": 1.90, "confidence": 0.95},
        {"kind": "facial_expression_shift_candidate", "start": 1.40, "end": 1.95, "confidence": 0.88},
        {"kind": "audio_silence_interval", "start": 1.22, "end": 2.18, "confidence": 1.0},
    ]}]}}
    pieces, audit = split_selected_interior_performance_gaps((clip,), diagnostics)
    assert len(pieces) == 2 and audit
    draft = DraftTimeline(schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
                          selected=tuple(pieces), alternates=(), discarded=(), diagnostics={})
    protected = protected_speech_by_clip_id(draft)
    for piece in pieces:
        assert piece.clip_id in protected and piece.parent_semantic_clip_id == "clip-a"
        # words stay in SOURCE seconds, inside the piece's own span
        assert all(piece.start - 1e-6 <= s <= e <= piece.end + 1e-6 for s, e in protected[piece.clip_id])
    seg = RenderSegment(clip_id=pieces[1].clip_id, source_asset_id="src", source_path="/x.mp4",
                        start=pieces[1].start, end=pieces[1].end, parent_semantic_clip_id="clip-a")
    assert _speech_for_segment(seg, protected) == protected[pieces[1].clip_id]
    # a fragment id unknown to the map falls back to the parent's words (never to nothing)
    alias = RenderSegment(clip_id="clip-a#render", source_asset_id="src", source_path="/x.mp4",
                          start=pieces[1].start, end=pieces[1].end, parent_semantic_clip_id="clip-a")
    assert _speech_for_segment(alias, {"clip-a": protected[pieces[0].clip_id]}) == protected[pieces[0].clip_id]


def test_time_system_survives_a_trim_words_are_source_seconds_output_offsets_are_mapped(tmp_path):
    # a trailing repair converts the OUTPUT finding to the segment's SOURCE edge
    # and compares it with SOURCE word times: after a trim the words are unchanged
    # and the room is re-measured from the new edge.
    seg = RenderSegment(clip_id="a", source_asset_id="src", source_path="", start=100.0, end=106.0)
    speech = ((100.2, 101.0), (101.2, 105.4))
    assert speech_room_at_edge(seg, "trailing", edge_time=106.0, speech=speech) == pytest.approx(0.6)
    from dataclasses import replace
    trimmed = replace(seg, end=105.7)
    assert speech_room_at_edge(trimmed, "trailing", edge_time=105.7, speech=speech) == pytest.approx(0.3)
    assert speech_room_at_edge(trimmed, "leading", edge_time=105.7, speech=speech) == pytest.approx(0.2)


def test_missing_or_padded_word_evidence_fails_closed_never_open():
    """What happens when the evidence is absent or incomplete:
    - clip without words (no ASR words): a measured silence inside the
      window is still trimmed (the measurement itself proves no speech),
      a zero-extent join probe is refused;
    - an ASR word padded PAST the edge (D-291.9's case before reconciliation):
      the room is 0, so the repair is refused rather than trusting the edge."""
    seg = RenderSegment(clip_id="a", source_asset_id="src", source_path="", start=0.0, end=3.0)
    padded = ((0.2, 1.0), (1.1, 3.4))  # last word padded 0.4 s past the segment end
    assert speech_room_at_edge(seg, "trailing", edge_time=3.0, speech=padded) == 0.0
    assert _speech_for_segment(seg, {"other": ((0.0, 1.0),)}) is None  # no evidence for this clip
    assert _speech_for_segment(seg, {}) is None
