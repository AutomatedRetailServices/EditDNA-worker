import copy
import pytest
from cutsell_worker.deepgram_asr import checked_segments, DeepgramASR
from cutsell_worker.universal_clean_cut_validation import _validation_asr
from types import SimpleNamespace


def payload():
    return {'results':{'channels':[{'alternatives':[{'transcript':'Hi. Try again!', 'words':[
        {'word':'hi','punctuated_word':'Hi.','start':0,'end':0.4,'confidence':0.9},
        {'word':'try','punctuated_word':'Try','start':1,'end':1.3,'confidence':0.8},
        {'word':'again','punctuated_word':'again!','start':1.3,'end':2,'confidence':0.95}]}]}]}}


def test_keeps_every_token_and_native_boundaries():
    s=checked_segments(payload(),'source',3)
    assert [x.text for x in s]==['Hi.','Try again!']
    assert [(w.text,w.start,w.end) for x in s for w in x.words]==[('Hi.',0,0.4),('Try',1,1.3),('again!',1.3,2)]


@pytest.mark.parametrize('change',[{'end':0},{'start':float('nan')},{'end':4},{'confidence':2}])
def test_invalid_evidence_fails_closed(change):
    p=payload();p['results']['channels'][0]['alternatives'][0]['words'][0].update(change)
    with pytest.raises(ValueError):checked_segments(p,'source',3)


def test_missing_word_fails_closed():
    p=payload();p['results']['channels'][0]['alternatives'][0]['words'].pop()
    with pytest.raises(ValueError):checked_segments(p,'source',3)


def test_opt_in_only():
    assert isinstance(_validation_asr(SimpleNamespace(asr_model='medium'),env={'CUTSELL_VALIDATION_ASR_PROVIDER':'deepgram-nova-3-multi'}),DeepgramASR)
    assert not isinstance(_validation_asr(SimpleNamespace(asr_model='medium'),env={}),DeepgramASR)


def test_native_overlap_is_preserved_without_inventing_boundaries():
    p=payload();words=p['results']['channels'][0]['alternatives'][0]['words']
    words[1]['end']=2.2
    s=checked_segments(p,'source',3)
    assert s[-1].end==2.2
    assert s[-1].words[-1].start==1.3
    assert s[-1].words[-1].end==2
