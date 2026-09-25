import json
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import pytest
from cutsell_worker.watch_listen_runtime import (
    automatic_watch_listen, capability_enabled, runtime_diagnostics, DEPENDENCIES, MASTER,
)
from cutsell_worker.global_editorial_context import with_global_editorial_evidence
from cutsell_worker.whole_video_analysis import WholeVideoContext, SourceVideoContext
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.hybrid_session_cleanup import _source_context


def test_request_scope_enables_every_registered_capability_and_resets():
    @automatic_watch_listen
    def run():
        assert all(capability_enabled('CUTSELL_'+name,{}) for name in DEPENDENCIES)
    run()
    assert not capability_enabled('CUTSELL_WATCH_LISTEN_BESTTAKE_EVIDENCE_ENABLED',{})


def test_exception_and_parallel_request_do_not_leak_profile():
    @automatic_watch_listen
    def run():
        with ThreadPoolExecutor() as executor:
            assert executor.submit(capability_enabled,'CUTSELL_WATCH_LISTEN_BESTTAKE_EVIDENCE_ENABLED',{}).result() is False
        raise RuntimeError('fixture')
    with pytest.raises(RuntimeError): run()
    assert not capability_enabled('CUTSELL_WATCH_LISTEN_BESTTAKE_EVIDENCE_ENABLED',{})


def test_dependency_off_blocks_authority_and_is_observable(monkeypatch):
    monkeypatch.setenv('CUTSELL_WATCH_LISTEN_BESTTAKE_EVIDENCE_ENABLED','0')
    @automatic_watch_listen
    def run():
        rows=runtime_diagnostics()['capabilities']
        assert not rows['WATCH_LISTEN_BESTTAKE_GUARD_AUTHORITY_ENABLED']['enabled']
        assert rows['WATCH_LISTEN_BESTTAKE_GUARD_AUTHORITY_ENABLED']['blocked_dependencies']
    run()


def test_master_rollback_preserves_explicit_flags(monkeypatch):
    monkeypatch.setenv(MASTER,'0')
    @automatic_watch_listen
    def run():
        assert not capability_enabled('CUTSELL_WATCH_LISTEN_BESTTAKE_EVIDENCE_ENABLED',{})
        assert capability_enabled('CUTSELL_WATCH_LISTEN_BESTTAKE_EVIDENCE_ENABLED',{'CUTSELL_WATCH_LISTEN_BESTTAKE_EVIDENCE_ENABLED':'1'})
    run()


def region(source):
    return SimpleNamespace(source_asset_id=source,source_start=0,source_end=4,
                           dominant_process_status='TAKE_SERIES_REGION',audience_delivery_status='UNCERTAIN',
                           confidence='SUPPORTED',conflict_flags=('conflicting_evidence',))


def test_global_context_reaches_editorial_prompt_without_cross_source_leak():
    context=WholeVideoContext((SourceVideoContext('a','original','raw','record'),
                               SourceVideoContext('b','other','raw','record')),ProviderStatus('test','applied',True,True))
    understanding=SimpleNamespace(regions=(region('a'),))
    result=SimpleNamespace(understanding=understanding,capability_status='PARTIAL')
    updated, audit=with_global_editorial_evidence(context,result)
    assert context.sources[0].editorial_evidence == ''
    assert updated.sources[0].summary == 'original'
    evidence=json.loads(dict(_source_context(updated,'a'))['global_editorial_evidence'])
    assert evidence['regions'][0]['conflicts'] == ['conflicting_evidence']
    assert dict(_source_context(updated,'b'))['global_editorial_evidence'] == ''
    assert audit['stage'] == 'before_composite_resolution'
    assert audit['capability_status'] == 'PARTIAL'


def test_missing_understanding_never_invents_context():
    assert with_global_editorial_evidence(None,None)[1]['status'] == 'not_evaluable'


def test_global_construction_precedes_cleanup_and_is_not_recomputed():
    import inspect
    from pathlib import Path
    source=Path('cutsell_worker/pipeline.py').read_text().split('def build_flow_b_draft(',1)[1]
    assert source.index('whole_video_editorial_reasoning_result = build_whole_video_editorial_reasoning(') < source.index('kept, deterministic_discarded, decisions = apply_clean_cut(')
    assert source.count('whole_video_editorial_reasoning_result = build_whole_video_editorial_reasoning(') == 1
    assert source.index('with_global_editorial_evidence(') < source.index('hybrid_cleanup, composite_split_ids = apply_composite_resolution(')


def test_real_ingest_activates_layers_without_asr_repetition(tmp_path, monkeypatch):
    from cutsell_worker.flow_b import process_local_sources
    from cutsell_worker.contracts import ProcessingRequest, SourceAsset, TranscriptSegment, Word
    from cutsell_worker.media_probe import MediaProbe
    source=SourceAsset('src','p','u','raw.mp4',0,5,'s3://bucket/raw.mp4')
    path=tmp_path/'raw.mp4'; path.write_bytes(b'fixture')
    calls=[]
    class ASR:
        def transcribe(self,path,*,source_asset_id,language_hint=None):
            calls.append(1)
            return (TranscriptSegment('src',0,3,'This has two useful settings.',
                    tuple(Word(t,i*.4,i*.4+.3) for i,t in enumerate('This has two useful settings.'.split()))),)
    monkeypatch.setattr('cutsell_worker.flow_b.probe_media',lambda _:MediaProbe(5,1080,1920,30,True))
    for name in DEPENDENCIES: monkeypatch.delenv('CUTSELL_'+name,raising=False)
    monkeypatch.delenv(MASTER,raising=False)
    result=process_local_sources(ProcessingRequest(project_id='p',user_id='u',sources=(source,)),
                                 {'src':str(path)},asr_provider=ASR(),editorial_mode='clean_cut')
    diag=result.draft.diagnostics
    assert len(calls)==1
    assert diag['watch_listen_runtime']['mode']=='automatic'
    assert all(r['enabled'] for r in diag['watch_listen_runtime']['capabilities'].values())
    assert diag['editorial_moment_sequence']['status']=='evaluated'
    assert diag['live_language_spine']['status']=='evaluated'
    assert diag['whole_video_editorial_reasoning'].get('status') != 'disabled'
    assert diag['global_editorial_handoff']['stage']=='before_composite_resolution'
    assert result.draft.selected


def test_provider_budget_keeps_global_evidence_parseable():
    from cutsell_worker.hybrid_payload import _compact_source_context, _shrink_context_once
    payload=json.dumps({'kind':'global_editorial_hypotheses', 'rule':'Corroborate; do not delete from context alone.',
                        'regions':[{'start':i,'end':i+1,'conflicts':['uncertain'],'process':'RETRY_CANDIDATE'} for i in range(30)]})
    value=_compact_source_context((('global_editorial_evidence',payload),))
    assert len(value['global_editorial_evidence'])<=1800
    assert json.loads(value['global_editorial_evidence'])['regions']
    for _ in range(12):
        value=_shrink_context_once(value)
        if value['global_editorial_evidence']:
            assert isinstance(json.loads(value['global_editorial_evidence'])['regions'],list)


def test_automatic_scope_survives_nested_ingest_until_post_selection_authority():
    @automatic_watch_listen
    def ingest():
        return None
    @automatic_watch_listen
    def universal():
        ingest()
        assert capability_enabled('CUTSELL_WATCH_LISTEN_BESTTAKE_GUARD_AUTHORITY_ENABLED',{})
    universal()
    assert not capability_enabled('CUTSELL_WATCH_LISTEN_BESTTAKE_GUARD_AUTHORITY_ENABLED',{})
