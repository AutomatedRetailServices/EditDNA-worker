from types import SimpleNamespace
from dataclasses import replace
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, JobState, ProcessingResult, SemanticRole, Word
from cutsell_worker.final_boundary_authority import (
    _clip_from_envelope,
    _trim_spaced_duplicate_from_left,
    _trim_short_post_cta_aside,
    _trim_trailing_aborted_restarts,
    enforce_complete_idea_boundaries,
)


def _words(texts, start=0.0, step=.4):
    return tuple(Word(text, start + i * step, start + i * step + .3) for i, text in enumerate(texts))


def test_trailing_aborted_restart_is_trimmed_at_repeated_prefix():
    words = _words(("your", "muscles", "grow", "and", "your", "mus"))
    clip = DraftClip("a", "src", 0, 0, 2.3, "your muscles grow and your mus", "", words=words)
    output, rows = _trim_trailing_aborted_restarts([clip], {"src": words})
    assert output[0].text == "your muscles grow and"
    assert rows[0]["action"] == "trim_trailing_aborted_restart"


def test_accent_partial_matching_does_not_delete_complete_words():
    words = _words(("energía", "y", "tus", "músculos", "tus", "mus"))
    clip = DraftClip("a", "src", 0, 0, 2.3, " ".join(w.text for w in words), "", words=words)
    assert _trim_trailing_aborted_restarts([clip], {"src": words})[0] == [clip]
    complete = _words(("energía", "y", "tus", "músculos", "tus", "musculos"))
    clip2 = replace(clip, words=complete, text=" ".join(w.text for w in complete))
    assert _trim_trailing_aborted_restarts([clip2], {"src": complete})[0] == [clip2]
    complete = _words(("Yo", "tomo", "cafeína", "y", "también", "tomo", "café"))
    clip3 = replace(clip, end=2.7, words=complete, text=" ".join(w.text for w in complete))
    assert _trim_trailing_aborted_restarts([clip3], {"src": complete})[0] == [clip3]


def test_spaced_exact_retry_phrase_is_kept_on_later_continuing_take():
    left_words = _words(("energy", "daily", "you", "can", "perform", "more"), start=0)
    right_words = _words(("you", "can", "perform", "more", "at", "gym"), start=5)
    left = DraftClip("a", "src", 0, 0, 2.3, "energy daily you can perform more", "", words=left_words)
    right = DraftClip("b", "src", 0, 5, 7.3, "you can perform more at gym", "", words=right_words)
    output, rows = _trim_spaced_duplicate_from_left([left, right], {"src": left_words + right_words})
    assert output[0].text == "energy daily"
    assert output[1] == right
    assert rows[0]["action"] == "trim_spaced_retry_duplicate_from_left"


def test_short_non_cta_aside_after_long_post_cta_pause_is_trimmed():
    head = _words(("find", "it", "in", "the", "cart"), start=0, step=.35)
    tail = _words(("that", "is", "done"), start=4, step=.35)
    words = head + tail
    clip = DraftClip("a", "src", 0, 0, 5, "find it in the cart that is done", "", words=words)
    output, rows = _trim_short_post_cta_aside([clip], {"src": words})
    assert output[0].text == "find it in the cart"
    assert rows[0]["action"] == "trim_short_post_cta_aside"


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


def test_source_refresh_cannot_truncate_a_protected_terminal_word_without_moving_edge():
    selected_words = (
        Word("I", 1.0, 1.2), Word("resolved", 1.2, 1.6),
        Word("it", 1.6, 1.8), Word("with", 1.8, 2.0), Word("cream.", 2.0, 2.4),
    )
    # A later ASR partition ends at the same measured edge but has assigned
    # the last spoken word to its adjacent segment.
    source_words = selected_words[:-1]
    selected = DraftClip(
        "clip", "src", 0, 1.0, 2.4, "I resolved it with cream.",
        "I resolved it with cream.", words=selected_words,
    )
    repaired, row = _clip_from_envelope(selected, source_words)
    assert repaired == selected
    assert repaired.text.endswith("cream.")
    assert row["action"] == "keep_source_envelope_would_truncate_selected_text"
    assert row["source_envelope_word_count"] == 4


def test_complete_idea_recovery_never_reintroduces_a_discarded_neighbor():
    source_words = (
        Word('Use', 0.0, .2), Word('this', .22, .4), Word('product', .42, .7),
        Word('today', .72, 1.0), Word('because', 1.1, 1.35),
        Word('I', 1.37, 1.45), Word('am', 1.47, 1.58), Word('recording.', 1.60, 2.0),
    )
    selected=DraftClip('keep','src',0,0,1.0,'Use this product today','Use this product today',words=source_words[:4])
    discarded=DraftClip('bts','src',0,1.1,2.0,'because I am recording.','because I am recording.',words=source_words[4:])
    draft=DraftTimeline('cutsell.v1','p',EditStrategy.STORYTELLING,(selected,),(),(discarded,),{})
    result=ProcessingResult('cutsell.v1','p',JobState.DRAFT_READY,draft,{})
    class ASR:
        def transcribe(self,*args,**kwargs):
            return (SimpleNamespace(words=source_words),)
    fixed=enforce_complete_idea_boundaries(result,{'src':'unused'},asr_provider=ASR())
    assert fixed.draft.selected[0].end == 1.0
    assert fixed.draft.selected[0].text == 'Use this product today'
    assert fixed.draft.diagnostics['final_boundary_authority'][0]['action']=='limit_envelope_at_discarded_selection'


def test_complete_idea_recovery_blocks_discard_that_overlaps_original_edge():
    source_words = (
        Word('Use', 0.0, .2), Word('this', .22, .4), Word('product', .42, .7),
        Word('today', .72, 1.0), Word('because', 1.1, 1.35),
        Word('I', 1.37, 1.45), Word('am', 1.47, 1.58), Word('recording.', 1.60, 2.0),
    )
    selected=DraftClip('keep','src',0,0,1.0,'Use this product today','Use this product today',words=source_words[:4])
    # Selection authorities can produce partially overlapping source spans.
    # The discarded tail still owns the newly requested interval (1.0, 2.0].
    discarded=DraftClip('bts','src',0,.8,2.0,'today because I am recording.',
                        'today because I am recording.',words=source_words[3:])
    draft=DraftTimeline('cutsell.v1','p',EditStrategy.STORYTELLING,(selected,),(),(discarded,),{})
    result=ProcessingResult('cutsell.v1','p',JobState.DRAFT_READY,draft,{})
    class ASR:
        def transcribe(self,*args,**kwargs):
            return (SimpleNamespace(words=source_words),)
    fixed=enforce_complete_idea_boundaries(result,{'src':'unused'},asr_provider=ASR())
    assert fixed.draft.selected[0].end == 1.0
    assert fixed.draft.selected[0].text == 'Use this product today'
    row=fixed.draft.diagnostics['final_boundary_authority'][0]
    assert row['action']=='limit_envelope_at_discarded_selection'
    assert row['blocked_trailing_recovery'] is True
