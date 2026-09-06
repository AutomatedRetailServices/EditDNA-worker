from cutsell_worker.contracts import DraftClip, SemanticRole
from cutsell_worker.short_bts_process_cleanup import suppress_short_explicit_bts_process_fragments


def _clip(cid, start, end, text):
    return DraftClip(
        clip_id=cid,
        source_asset_id="src",
        source_order=0,
        start=float(start),
        end=float(end),
        text=text,
        caption_text=text,
        semantic_role=SemanticRole.STORY,
        selected=True,
    )


def _diag(cid, label="bts", confidence=0.75):
    return {"hybrid_editorial_chunks": [{"decisions": [{"clip_id": cid, "label": label, "confidence": confidence}]}]}


def test_round12_video02_drops_trying_to_say_in_character_bts_fragment():
    clip = _clip("process", 109.79, 111.05, "Trying to say in character")
    selected, discarded, audit = suppress_short_explicit_bts_process_fragments((clip,), (), _diag("process"))
    assert selected == ()
    assert discarded[0].clip_id == "process"
    assert audit[0]["reason"] == "short_explicit_recording_process_bts"


def test_does_not_delete_story_phrase_trying_to_stay_in_character():
    clip = _clip("story", 113.14, 114.9, "We were trying to stay in character")
    selected, discarded, audit = suppress_short_explicit_bts_process_fragments((clip,), (), _diag("story"))
    assert selected == (clip,)
    assert discarded == ()
    assert audit == ()


def test_does_not_delete_explicit_process_phrase_without_bts_label():
    clip = _clip("keep", 10.0, 11.1, "Trying to say this")
    selected, discarded, audit = suppress_short_explicit_bts_process_fragments((clip,), (), _diag("keep", label="keep", confidence=0.95))
    assert selected == (clip,)
    assert discarded == ()
    assert audit == ()


def test_does_not_delete_long_bts_discussion_even_with_process_words():
    clip = _clip("long", 10.0, 14.5, "I was trying to say something about how we made the whole scene work")
    selected, discarded, audit = suppress_short_explicit_bts_process_fragments((clip,), (), _diag("long", confidence=0.95))
    assert selected == (clip,)
    assert discarded == ()
    assert audit == ()
