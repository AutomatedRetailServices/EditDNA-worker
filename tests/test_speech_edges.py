import copy
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

for name, attrs in (("requests", {}), ("boto3", {}), ("clip", {}), ("faster_whisper", {"WhisperModel": object})):
    if name not in sys.modules and importlib.util.find_spec(name) is None:
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        sys.modules[name] = module

from worker import pipeline
from worker.models import openai_provider as provider
from worker.models.openai_client import OpenAIResponseValidationError
from worker.semantic_slot_v2 import build_clause_inputs
from worker.speech_edges import SpeechEdgeProposal, apply_speech_edge


def saved(index=0):
    return json.loads((Path(__file__).parent / 'fixtures' / 'august_unpunctuated_candidates.json').read_text())[index]


def proposal(left=0, right=13, confidence=.95):
    return SpeechEdgeProposal(left, right, confidence, 'production_talk')


def test_saved_tail_cut_keeps_exact_contiguous_words_and_source():
    clip = saved()
    clip.update(source_index=2, source_local='source.mov')
    original = copy.deepcopy(clip)
    assert apply_speech_edge(clip, proposal()) == 'applied'
    assert clip['end'] == pytest.approx(4.06)
    assert clip['words'] == original['words'][:13]
    assert clip['text'] == 'shut the front door and rhino is running a crazy deal right now'
    assert clip['start'] == original['start']
    for key in ('id', 'source_index', 'source_local', 'chain_ids'):
        assert clip[key] == original[key]
    assert clip['meta']['speech_edge_edit']['original_end'] == 7.91


@pytest.mark.parametrize('left,right,expected', [(2, 8, (1.96, 5.6)), (2, 6, (1.96, 3.64))])
def test_prefix_and_both_edges_keep_one_contiguous_span(left, right, expected):
    tokens = ['Wait', 'reset', 'It', 'has', 'two', 'sizes', 'stop', 'recording']
    times = [(0,.3),(.4,.7),(2,2.3),(2.4,2.7),(2.8,3.1),(3.2,3.6),(5,5.2),(5.3,5.6)]
    clip = pipeline.make_base_clip('x', 0, 5.6, ' '.join(tokens), [
        {'word': word, 'start': a, 'end': b} for word, (a,b) in zip(tokens,times)])
    assert apply_speech_edge(clip, proposal(left,right)) == 'applied'
    assert (clip['start'],clip['end']) == pytest.approx(expected)
    assert clip['text'] == ' '.join(tokens[left:right])


@pytest.mark.parametrize('mutation,expected', [
    (lambda c: c.pop('words'), 'invalid_range'),
    (lambda c: c.update(text=c['text']+' missing'), 'transcript_timing_mismatch'),
    (lambda c: c['words'][2].update(end=float('nan')), 'invalid_word_timing'),
    (lambda c: c['words'][2].update(end=c['words'][2]['start']), 'invalid_word_timing'),
    (lambda c: c['words'][2].update(start=-1), 'invalid_word_timing'),
    (lambda c: c['words'][13].update(start=4.03), 'no_safe_boundary_gap'),
    (lambda c: c['words'][13].update(start=3.9), 'invalid_word_timing'),
])
def test_bad_alignment_is_preserved(mutation, expected):
    clip = saved()
    mutation(clip)
    # JSON handles NaN for comparison; no rejected cut is allowed to mutate input.
    before = json.dumps(clip, sort_keys=True)
    assert apply_speech_edge(clip, proposal()) == expected
    assert json.dumps(clip, sort_keys=True) == before


def test_actual_incomplete_word_coverage_is_not_repaired_by_guessing():
    clip = saved(8)
    before = copy.deepcopy(clip)
    assert apply_speech_edge(clip, proposal(0,9)) == 'transcript_timing_mismatch'
    assert clip == before


@pytest.mark.parametrize('trim', [proposal(-1,13), proposal(0,99), proposal(0,13,.5), proposal(0,19)])
def test_bad_or_uncertain_range_is_preserved(trim):
    clip = saved()
    before = copy.deepcopy(clip)
    assert apply_speech_edge(clip, trim) != 'applied'
    assert clip == before


def response(clip, **overrides):
    row = dict(id=clip['id'], primary_slot='HOOK', secondary_slot=None,
               confidence=.95, secondary_confidence=None, completeness=.95,
               sales_relevance=.9, standalone_quality=.9, abstain=False,
               reason='Complete opening followed by recording preparation', evidence_tags=[],
               edge_trim=dict(keep_start=0,keep_end=13,confidence=.95,reason='production_talk'))
    row.update(overrides)
    return json.dumps({'results':[row]})


def test_existing_semantic_call_applies_tail_before_composer(monkeypatch):
    clip = saved()
    calls = []
    def chat(operation, model, messages, **kwargs):
        calls.append(operation)
        payload = json.loads(messages[1]['content'][0]['text'])
        assert len(payload['clauses'][0]['word_tokens']) == 19
        return response(clip)
    monkeypatch.setattr(provider, '_chat', chat)
    monkeypatch.setattr(pipeline, 'EDITDNA_USE_LLM', True)
    assert pipeline.enrich_clips_semantic([clip])
    assert calls == ['semantic_classification_v2']
    assert clip['meta']['semantic_v2']['edge_trim_status'] == 'applied'
    assert clip['end'] == pytest.approx(4.06)
    assert clip['id'] in pipeline.build_composer([clip])['used_clip_ids']


@pytest.mark.parametrize('changes', [dict(abstain=True), dict(confidence=.8), dict(completeness=.8), dict(primary_slot='OTHER')])
def test_uncertain_trim_cannot_reclassify_or_exclude_original(monkeypatch, changes):
    clip = saved()
    original = copy.deepcopy(clip)
    monkeypatch.setattr(provider, '_chat', lambda *a, **k: response(clip, **changes))
    monkeypatch.setattr(pipeline, 'EDITDNA_USE_LLM', True)
    pipeline.enrich_clips_semantic([clip])
    assert clip['meta']['semantic_v2']['edge_trim_status'] == 'uncertain_retained_speech'
    for key in ('start','end','words','text'):
        assert clip[key] == original[key]
    assert not clip['meta']['semantic_v2'].get('excluded_from_composer',False)


@pytest.mark.parametrize('trim', [[], {'keep_start':True,'keep_end':13,'confidence':.95,'reason':'production_talk'},
    {'keep_start':0,'keep_end':999,'confidence':.95,'reason':'production_talk'},
    {'keep_start':0,'keep_end':13,'confidence':.95,'reason':'weak_sales'}])
def test_malformed_provider_trim_fails_validation(monkeypatch, trim):
    clip = saved()
    monkeypatch.setattr(provider, '_chat', lambda *a, **k: response(clip, edge_trim=trim))
    with pytest.raises(OpenAIResponseValidationError):
        provider.classify_semantic_v2('unused', build_clause_inputs([clip]))


def test_render_uses_grounded_edges_for_audio_and_video_without_double_trim(monkeypatch, tmp_path):
    clip = saved()
    apply_speech_edge(clip, proposal())
    calls = []
    monkeypatch.setattr(pipeline, 'HEAD_TRIM_SEC', .2)
    monkeypatch.setattr(pipeline, 'TAIL_TRIM_SEC', .2)
    monkeypatch.setattr(pipeline, 'has_audio_stream', lambda _: True)
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda command, **kwargs:
                        calls.append(command) or types.SimpleNamespace(returncode=0, stdout='', stderr=''))
    pipeline.render_funnel_video('source.mov', str(tmp_path), [clip], [clip['id']])
    filters = calls[0][calls[0].index('-filter_complex')+1]
    assert '[0:v]trim=start=0.000:end=4.060' in filters
    assert '[0:a]atrim=start=0.000:end=4.060' in filters


def test_no_proposal_preserves_intentional_profanity_and_humor(monkeypatch):
    clip = saved()
    clip['text'] = 'This damn thing works and that is my terrible joke.'
    before = copy.deepcopy(clip)
    monkeypatch.setattr(provider, '_chat', lambda *a, **k: response(clip, edge_trim=None))
    monkeypatch.setattr(pipeline, 'EDITDNA_USE_LLM', True)
    pipeline.enrich_clips_semantic([clip])
    for key in ('start','end','words','text'):
        assert clip[key] == before[key]
    assert clip['meta']['semantic_v2']['edge_trim_status'] == 'not_requested'
