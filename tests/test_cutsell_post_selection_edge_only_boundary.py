from cutsell_worker.contracts import DraftClip, SemanticRole, Word
from cutsell_worker.post_selection_edge_only_boundary import trim_locked_selection_edges


def _clip(start=10.0, end=14.0, words=None):
    words = words or (
        Word("Hola", 10.40, 10.80),
        Word("mundo.", 12.80, 13.20),
    )
    return DraftClip(
        clip_id="clip_a",
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text="Hola mundo.",
        caption_text="Hola mundo.",
        words=words,
        semantic_role=SemanticRole.STORY,
        selected=True,
    )


def _diag(events):
    return {"whole_video_context": {"sources": [{"source_asset_id": "src", "events": events}]}}


def test_trims_only_edge_slack_and_preserves_selection_identity():
    clip = _clip()
    events = [
        {"kind": "body_reset_candidate", "start": 13.25, "end": 13.7, "confidence": 0.96},
        {"kind": "camera_disengagement_candidate", "start": 13.3, "end": 13.8, "confidence": 0.91},
    ]
    selected, audit = trim_locked_selection_edges((clip,), _diag(events))
    assert len(selected) == 1
    out = selected[0]
    assert out.clip_id == clip.clip_id
    assert out.text == clip.text
    assert out.words == clip.words
    assert out.start == clip.start
    assert out.end == 13.20
    assert audit[0]["selection_identity_preserved"] is True


def test_single_body_motion_does_not_trim():
    clip = _clip()
    events = [{"kind": "body_reset_candidate", "start": 13.25, "end": 13.7, "confidence": 0.97}]
    selected, audit = trim_locked_selection_edges((clip,), _diag(events))
    assert selected == (clip,)
    assert audit == ()


def test_authoritative_dead_air_can_trim_leading_edge():
    clip = _clip()
    events = [{"kind": "unintentional_dead_air", "start": 10.0, "end": 10.4, "confidence": 0.94}]
    selected, audit = trim_locked_selection_edges((clip,), _diag(events))
    assert selected[0].start == 10.40
    assert selected[0].end == 14.0
    assert len(audit) == 1


def test_never_changes_selection_membership_or_order():
    a = _clip(start=10.0, end=14.0)
    b = DraftClip(
        clip_id="clip_b",
        source_asset_id="src",
        source_order=0,
        start=20.0,
        end=23.0,
        text="Segundo.",
        caption_text="Segundo.",
        words=(Word("Segundo.", 20.2, 22.5),),
        semantic_role=SemanticRole.STORY,
        selected=True,
    )
    selected, _ = trim_locked_selection_edges((a, b), _diag([]))
    assert [clip.clip_id for clip in selected] == ["clip_a", "clip_b"]
