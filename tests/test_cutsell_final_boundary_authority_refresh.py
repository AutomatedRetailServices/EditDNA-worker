from cutsell_worker.contracts import DraftClip, SemanticRole, Word
from cutsell_worker.final_boundary_authority import _clip_from_envelope


def test_complete_idea_envelope_refreshes_text_even_when_timestamps_match():
    source_words = (
        Word(text="También", start=192.40, end=192.70),
        Word(text="me", start=192.72, end=192.84),
        Word(text="salían", start=192.86, end=193.15),
        Word(text="espinillas.", start=193.17, end=194.78),
        Word(text="Era", start=195.14, end=195.30),
        Word(text="como", start=195.32, end=195.48),
        Word(text="un", start=195.50, end=195.62),
        Word(text="rush,", start=195.64, end=196.38),
        Word(text="una", start=196.88, end=197.05),
        Word(text="alergia.", start=197.07, end=197.98),
    )
    stale = DraftClip(
        clip_id="clip_x",
        source_asset_id="src_x",
        source_order=0,
        start=192.40,
        end=197.98,
        text="También me salían espinillas.",
        caption_text="También me salían espinillas.",
        words=source_words[:4],
        semantic_role=SemanticRole.OTHER,
    )

    repaired, diagnostic = _clip_from_envelope(stale, source_words)

    assert repaired.start == stale.start
    assert repaired.end == stale.end
    assert repaired.text == "También me salían espinillas. Era como un rush, una alergia."
    assert repaired.words == source_words
    assert diagnostic["last_word"] == "alergia."
