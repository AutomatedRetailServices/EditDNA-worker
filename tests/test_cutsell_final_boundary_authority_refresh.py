from types import SimpleNamespace
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, JobState, ProcessingResult, SemanticRole, Word
from cutsell_worker.final_boundary_authority import _clip_from_envelope, enforce_complete_idea_boundaries


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
