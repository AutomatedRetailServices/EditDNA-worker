import json
from types import SimpleNamespace
from pathlib import Path
import pytest
from cutsell_worker import medium_whisperx_asr as mw
from cutsell_worker.contracts import TranscriptSegment
from cutsell_worker.universal_clean_cut_validation import _validation_asr

class Base:
    model_name = 'medium'
    last_detected_language = 'es'
    calls = 0
    def config_fingerprint(self, **kw):
        return SimpleNamespace(fingerprint=lambda: 'decode-'+str(kw))
    def transcribe(self, path, **kw):
        self.calls += 1
        return (TranscriptSegment(kw['source_asset_id'], 10, 12, 'No repitas.', ()),)

def setup(monkeypatch, tmp_path, mutation=None):
    runtime=tmp_path/'python';runtime.touch()
    monkeypatch.setattr(mw,'ALIGNER_PYTHON',str(runtime));monkeypatch.setattr(mw,'ALIGNER_SCRIPT',str(runtime))
    def run(args, **kw):
        if args[0] == str(runtime):
            assert 'OPENAI_API_KEY' not in kw['env']
            words=[{'word':'No','start':.1,'end':.3}, {'word':'repitas.','start':.4,'end':1.5}]
            if mutation == 'missing': words.pop(0)
            if mutation == 'zero': words[0]['end']=.1
            chunk={'index': 0, 'segments':[{'words':words}], 'alignment_model':'test'}
            if mutation == 'order':chunk['index']=1
            Path(args[-1]).write_text(json.dumps({'chunks':[chunk], 'runtime':{'packages':{'whisperx':'3.8.6'}}}))
        return SimpleNamespace(returncode=0,stderr='')
    monkeypatch.setattr(mw.subprocess,'run',run)
    source=tmp_path/'source';source.write_bytes(b'audio')
    return mw.MediumWhisperXASR(Base()),str(source)

def test_explicit_route_and_legacy_unchanged():
    config=SimpleNamespace(asr_model='medium')
    assert _validation_asr(config,env={}).model_name == 'medium'
    assert isinstance(_validation_asr(config,env={'CUTSELL_VALIDATION_ASR_PROVIDER':mw.PROVIDER}), mw.MediumWhisperXASR)

@pytest.mark.parametrize('language',['en','es'])
def test_alignment_preserves_text_absolute_times_and_reuses_source(monkeypatch,tmp_path,language):
    p,path=setup(monkeypatch,tmp_path)
    a=p.transcribe(path,source_asset_id='source',language_hint=language)
    assert a[0].text == 'No repitas.'
    assert a[0].start == 10.1 and a[0].end == 11.5
    assert [w.text for w in a[0].words] == ['No','repitas.']
    assert p.last_audit['status']=='passed'
    assert p.transcribe(path,source_asset_id='source',language_hint=language)==a
    assert p.base.calls==1 and p.last_audit['cache_hit_count']==1
    Path(path).write_bytes(b'changed');p.transcribe(path,source_asset_id='source',language_hint=language)
    assert p.base.calls==2

@pytest.mark.parametrize('mutation',['missing','zero','order'])
def test_bad_alignment_stops_without_fallback(monkeypatch,tmp_path,mutation):
    p,path=setup(monkeypatch,tmp_path,mutation)
    with pytest.raises(mw.AlignmentEvidenceError):p.transcribe(path,source_asset_id='source')
    assert p.last_audit['status']=='failed' and p.last_audit['fallback'] is None
    assert not p._source_cache

def test_unknown_language_stops(monkeypatch,tmp_path):
    p,path=setup(monkeypatch,tmp_path);p.base.last_detected_language='fr'
    with pytest.raises(mw.AlignmentEvidenceError):p.transcribe(path,source_asset_id='source')

def test_fingerprint_tracks_language_and_decode():
    p=mw.MediumWhisperXASR(Base())
    assert p.config_fingerprint(language_hint='es').fingerprint()!=p.config_fingerprint(language_hint='en').fingerprint()


def test_impossible_duplicate_microtail_is_dropped_before_alignment():
    complete = TranscriptSegment('source', 356.21, 361.61,
        'Por eso cuídate, alimentate bien, hidrata y haz ejercicio.', ())
    phantom = TranscriptSegment('source', 362.05, 362.07, complete.text, ())
    kept, dropped = mw._drop_degenerate_duplicate_tail((complete, phantom))
    assert kept == (complete,)
    assert dropped == [{'start': 362.05, 'end': 362.07,
                        'reason': 'degenerate_duplicate_tail'}]


def test_impossible_duplicate_microtail_after_terminal_silence_is_dropped():
    complete = TranscriptSegment('source', 356.71, 361.55,
        'Por eso cuídate, alimentate bien, hídrate y haz ejercicio.', ())
    phantom = TranscriptSegment('source', 366.88, 366.90, complete.text, ())
    kept, dropped = mw._drop_degenerate_duplicate_tail((complete, phantom))
    assert kept == (complete,)
    assert dropped == [{'start': 366.88, 'end': 366.90,
                        'reason': 'degenerate_duplicate_tail'}]


def test_duplicate_microsegment_is_not_dropped_when_it_is_not_the_source_tail():
    repeated = 'Por eso cuídate, alimentate bien, hídrate y haz ejercicio.'
    complete = TranscriptSegment('source', 10.0, 15.0, repeated, ())
    micro = TranscriptSegment('source', 20.0, 20.02, repeated, ())
    later = TranscriptSegment('source', 21.0, 24.0, 'Una idea posterior válida.', ())
    kept, dropped = mw._drop_degenerate_duplicate_tail((complete, micro, later))
    assert kept == (complete, micro, later)
    assert dropped == []


@pytest.mark.parametrize('start,end,text', [
    (362.05, 362.20, 'Por eso cuídate, alimentate bien, hidrata y haz ejercicio.'),
    (362.05, 362.07, 'Una idea nueva que debe conservarse.'),
    (377.00, 377.02, 'Por eso cuídate, alimentate bien, hidrata y haz ejercicio.'),
])
def test_distinct_or_plausible_tail_remains_strictly_aligned(start,end,text):
    complete = TranscriptSegment('source', 356.21, 361.61,
        'Por eso cuídate, alimentate bien, hidrata y haz ejercicio.', ())
    tail = TranscriptSegment('source', start, end, text, ())
    kept, dropped = mw._drop_degenerate_duplicate_tail((complete, tail))
    assert kept == (complete, tail)
    assert dropped == []
