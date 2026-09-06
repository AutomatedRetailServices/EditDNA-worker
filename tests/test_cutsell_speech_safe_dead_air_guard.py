from cutsell_worker.contracts import DraftClip, Word
from cutsell_worker.human_boundary_polish_v3 import _remove_interval


def _clip():
    words = (
        Word("La", 0.40, 0.70),
        Word("biopsia", 0.75, 1.20),
        Word("confirmo", 1.25, 1.80),
        Word("tiroides", 2.00, 2.55),
        Word("despues", 4.00, 4.45),
    )
    text = " ".join(word.text for word in words)
    return DraftClip(
        clip_id="clip",
        source_asset_id="src",
        source_order=0,
        start=0.0,
        end=5.0,
        text=text,
        caption_text=text,
        words=words,
    )


def test_dead_air_cut_that_intersects_spoken_word_fails_open():
    clip = _clip()

    # A silence detector may call this region dead air, but it overlaps the aligned
    # final word of the sentence. Spoken-word authority must win.
    pieces = _remove_interval(clip, 1.90, 2.70)

    assert pieces == (clip,)
    assert "tiroides" in pieces[0].text
    assert pieces[0].end == 5.0


def test_true_dead_air_between_word_envelopes_can_still_be_removed():
    clip = _clip()

    # This interval sits strictly between aligned words and remains eligible for the
    # existing recording-process cleanup behavior.
    pieces = _remove_interval(clip, 2.70, 3.80)

    assert len(pieces) == 2
    assert pieces[0].end == 2.70
    assert pieces[1].start == 3.80
    assert all(not (word.end > 2.70 and word.start < 3.80) for word in clip.words)
