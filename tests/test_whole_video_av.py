import json
from pathlib import Path
from dataclasses import replace
import pytest
import requests
from cutsell_worker.whole_video_av import GeminiWholeVideoAVProvider, build_av_provider
from cutsell_worker.hybrid_google_transport import DollarBudgetLedger
from cutsell_worker.contracts import SourceAsset
from cutsell_worker.whole_video_analysis import safe_whole_video_analyze


def source():
    return SourceAsset('source','p','u','video.mp4',0,10,'local')


def result():
    return {'summary':'A demonstration with a recording reset.', 'creator_intent':'demonstrate the product',
            'story_logic':'demonstration then invitation', 'regions':[{'start':1,'end':2,'role':'mixed',
            'confidence':.9,'audio_observation':'hesitation then delivery',
            'visual_observation':'turns away then returns','reason':'possible recording reset'}]}


class Session:
    def __init__(self,data=None): self.calls=[]; self.data=data if data is not None else result()
    def post(self,url,headers,json,timeout):
        self.calls.append((url,json))
        payload=({'totalTokens':100} if url.endswith('countTokens') else
                 {'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':__import__('json').dumps(self.data)}]}}]})
        class Response:
            def raise_for_status(self): pass
            def json(self): return payload
        return Response()


class TimeoutThenSuccessSession(Session):
    def post(self,url,headers,json,timeout):
        self.calls.append((url,json,timeout))
        if url.endswith('countTokens'):
            payload={'totalTokens':100}
        elif sum(call[0].endswith('generateContent') for call in self.calls) == 1:
            raise requests.exceptions.ReadTimeout('first generation timed out')
        else:
            payload={'candidates':[{'finishReason':'STOP','content':{
                'parts':[{'text':__import__('json').dumps(self.data)}]}}]}
        class Response:
            def raise_for_status(self): pass
            def json(self): return payload
        return Response()


class InvalidThenSuccessSession(Session):
    def post(self,url,headers,json,timeout):
        self.calls.append((url,json,timeout))
        if url.endswith('countTokens'):
            payload={'totalTokens':100}
        else:
            generation_count=sum(call[0].endswith('generateContent') for call in self.calls)
            data=(
                {**result(), 'regions': [{**result()['regions'][0], 'start': 8, 'end': 2}]}
                if generation_count == 1 else self.data
            )
            payload={'candidates':[{'finishReason':'STOP','content':{
                'parts':[{'text':__import__('json').dumps(data)}]}}]}
        class Response:
            def raise_for_status(self): pass
            def json(self): return payload
        return Response()


def provider(tmp_path,data=None,budget=.1):
    raw=tmp_path/'raw.mp4';raw.write_bytes(b'original media fixture')
    def prepare(path,target): target.write_bytes(b'AV input fixture');return 10
    session=Session(data)
    av=GeminiWholeVideoAVProvider('test-key','configured-model',DollarBudgetLedger(budget),1,2,
                                 session=session,media_preparer=prepare)
    return av,raw,session


def test_actual_media_handoff_and_evidence_reaches_classifier(tmp_path):
    from cutsell_worker.hybrid_session_cleanup import _editorial_session
    from cutsell_worker.hybrid_payload import build_compact_editorial_payload
    from cutsell_worker.contracts import CandidateTake
    av,raw,session=provider(tmp_path)
    context=safe_whole_video_analyze(av,(source(),),(),(),local_paths={'source':str(raw)})
    assert context.status.available
    assert len(session.calls)==2
    parts=session.calls[1][1]['contents'][0]['parts']
    assert parts[0]['inline_data']['mime_type']=='video/mp4'
    assert parts[0]['inline_data']['data']
    assert session.calls[1][1]['generationConfig']['temperature']==0.0
    evidence=json.loads(context.sources[0].audiovisual_evidence)
    assert evidence['input_modalities']==['video','audio']
    assert len(evidence['source_sha256'])==64
    take=CandidateTake('clip','source',0,0,3,'A demonstration with a reset')
    payload=build_compact_editorial_payload(_editorial_session((take,),context,partition_index=0,chunk_index=0))
    assert payload['candidates'][0]['evidence']['audiovisual']['observations'][0]['audio']
    assert json.loads(payload['source_context']['audiovisual_evidence'])['regions']


@pytest.mark.parametrize('mutation',['time','nan','missing_audio','missing_visual','role'])
def test_invalid_model_observations_fail_closed(tmp_path,mutation):
    data=result(); region=data['regions'][0]
    if mutation=='time': region['end']=11
    if mutation=='nan': region['confidence']=float('nan')
    if mutation=='missing_audio': del region['audio_observation']
    if mutation=='missing_visual': del region['visual_observation']
    if mutation=='role': region['role']='delete'
    av,raw,session=provider(tmp_path,data)
    context=safe_whole_video_analyze(av,(source(),),(),(),local_paths={'source':str(raw)})
    assert not context.status.available and context.status.status=='provider_error'
    assert not context.sources
    assert len(session.calls)==2  # no paid retry


def test_no_budget_means_no_generation(tmp_path):
    av,raw,session=provider(tmp_path,budget=.00001)
    context=safe_whole_video_analyze(av,(source(),),(),(),local_paths={'source':str(raw)})
    assert not context.status.available
    assert len(session.calls)==1 and session.calls[0][0].endswith('countTokens')


def test_timeout_retry_requires_explicit_flag_and_reserves_second_attempt(tmp_path):
    av,raw,_=provider(tmp_path)
    session=TimeoutThenSuccessSession()
    av.session=session
    av.retry_generation_timeout=True
    context=safe_whole_video_analyze(
        av,(source(),),(),(),local_paths={'source':str(raw)},
    )
    assert context.status.available
    assert [call[2] for call in session.calls]==[60,60,120]
    record=context.diagnostics['native_av'][0]
    assert record['status']=='validated'
    assert record['generation_attempts']==2
    assert record['retry_reason']=='read_timeout'
    assert av.ledger.reserved_usd==record['reserved_usd']


def test_timeout_does_not_retry_without_explicit_flag(tmp_path):
    av,raw,_=provider(tmp_path)
    session=TimeoutThenSuccessSession()
    av.session=session
    context=safe_whole_video_analyze(
        av,(source(),),(),(),local_paths={'source':str(raw)},
    )
    assert not context.status.available
    assert [call[2] for call in session.calls]==[60,60]
    assert context.diagnostics['native_av'][0]['generation_attempts']==1


def test_invalid_response_retry_requires_flag_and_second_budget(tmp_path):
    av,raw,_=provider(tmp_path)
    session=InvalidThenSuccessSession()
    av.session=session
    av.retry_generation_timeout=True
    context=safe_whole_video_analyze(
        av,(source(),),(),(),local_paths={'source':str(raw)},
    )
    assert context.status.available
    assert [call[2] for call in session.calls]==[60,60,120]
    record=context.diagnostics['native_av'][0]
    assert record['status']=='validated'
    assert record['generation_attempts']==2
    assert record['retry_reason']=='invalid_response'
    assert record['retry_initial_rejection'].startswith('AV_TIME_RANGE')
    assert av.ledger.reserved_usd==record['reserved_usd']


def test_invalid_response_retry_budget_exhaustion_sends_no_second_generation(tmp_path):
    av,raw,_=provider(tmp_path,budget=.006)
    session=InvalidThenSuccessSession()
    av.session=session
    av.retry_generation_timeout=True
    context=safe_whole_video_analyze(
        av,(source(),),(),(),local_paths={'source':str(raw)},
    )
    assert not context.status.available
    assert len([call for call in session.calls if call[0].endswith('generateContent')])==1
    record=context.diagnostics['native_av'][0]
    assert record['status']=='retry_budget_exhausted'
    assert record['retry_reason']=='invalid_response'


def test_second_invalid_response_fails_closed_after_two_generations(tmp_path):
    invalid={**result(), 'regions': [{**result()['regions'][0], 'start': 8, 'end': 2}]}
    av,raw,_=provider(tmp_path,data=invalid)
    session=InvalidThenSuccessSession(data=invalid)
    av.session=session
    av.retry_generation_timeout=True
    context=safe_whole_video_analyze(
        av,(source(),),(),(),local_paths={'source':str(raw)},
    )
    assert not context.status.available
    assert [call[2] for call in session.calls]==[60,60,120]
    record=context.diagnostics['native_av'][0]
    assert record['generation_attempts']==2
    assert record['status']=='generation_retry_failed'
    assert record['retry_failure_type']=='ValueError'


def test_timeout_then_invalid_response_never_sends_third_generation(tmp_path):
    invalid={**result(), 'regions': [{**result()['regions'][0], 'start': 8, 'end': 2}]}
    av,raw,_=provider(tmp_path)
    session=TimeoutThenSuccessSession(data=invalid)
    av.session=session
    av.retry_generation_timeout=True
    context=safe_whole_video_analyze(
        av,(source(),),(),(),local_paths={'source':str(raw)},
    )
    assert not context.status.available
    assert [call[2] for call in session.calls]==[60,60,120]
    record=context.diagnostics['native_av'][0]
    assert record['generation_attempts']==2
    assert record['status']=='rejected'
    assert record['retry_reason']=='read_timeout'


def test_second_timeout_fails_closed_with_terminal_audit_status(tmp_path):
    av,raw,_=provider(tmp_path)
    session=TimeoutThenSuccessSession()
    session.post=lambda url,headers,json,timeout: (
        Session.post(session,url,headers,json,timeout)
        if url.endswith('countTokens')
        else (_ for _ in ()).throw(requests.exceptions.ReadTimeout('timeout'))
    )
    av.session=session
    av.retry_generation_timeout=True
    context=safe_whole_video_analyze(
        av,(source(),),(),(),local_paths={'source':str(raw)},
    )
    assert not context.status.available
    record=context.diagnostics['native_av'][0]
    assert record['status']=='generation_retry_failed'
    assert record['generation_attempts']==2
    assert record['retry_failure_type']=='ReadTimeout'


def test_missing_actual_source_does_not_fall_back_to_text(tmp_path):
    av,raw,session=provider(tmp_path)
    context=safe_whole_video_analyze(av,(source(),),(),())
    assert not context.status.available and not session.calls


def test_explicit_budget_required_and_no_automatic_spending():
    from cutsell_worker.hybrid_provider_settings import HybridProviderSettings
    settings=HybridProviderSettings(enabled=True)
    assert build_av_provider(settings,{'GEMINI_API_KEY':'key'}) is None
    with pytest.raises(ValueError):
        build_av_provider(settings,{'GEMINI_API_KEY':'key','CUTSELL_WATCH_LISTEN_AV_ENABLED':'1'})


def test_native_media_preparation_retains_audio_video_and_duration(tmp_path):
    import subprocess
    from cutsell_worker.whole_video_av import prepare_av
    raw=tmp_path/'raw.mp4'; out=tmp_path/'prepared.mp4'
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y',
        '-f','lavfi','-i','testsrc=size=160x120:rate=12',
        '-f','lavfi','-i','sine=frequency=600:sample_rate=16000',
        '-t','2','-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac',str(raw)],check=True,capture_output=True)
    duration=prepare_av(raw,out)
    assert abs(duration-2)<.15 and out.stat().st_size>0


def test_runtime_selects_av_only_with_explicit_complete_configuration():
    from cutsell_worker.brain_runtime import build_brain_runtime
    from cutsell_worker.config import load_runtime_config
    values={'CUTSELL_BRAIN_BACKEND':'runpod_local','CUTSELL_HYBRID_LLM_ENABLED':'1',
            'CUTSELL_HYBRID_PROVIDER':'google','GEMINI_API_KEY':'test',
            'CUTSELL_WATCH_LISTEN_AV_ENABLED':'1','CUTSELL_WATCH_LISTEN_AV_MAX_EDIT_USD':'.1',
            'CUTSELL_WATCH_LISTEN_AV_INPUT_USD_PER_MILLION':'1',
            'CUTSELL_WATCH_LISTEN_AV_OUTPUT_USD_PER_MILLION':'2'}
    brain=build_brain_runtime(load_runtime_config(values),values)
    assert isinstance(brain.whole_video_provider,GeminiWholeVideoAVProvider)


def test_local_global_hypotheses_do_not_overwrite_audiovisual_evidence():
    from types import SimpleNamespace
    from cutsell_worker.global_editorial_context import with_global_editorial_evidence
    from cutsell_worker.whole_video_analysis import WholeVideoContext,SourceVideoContext
    from cutsell_worker.providers import ProviderStatus
    evidence=json.dumps({'regions':result()['regions'],'kind':'audiovisual_observations_v1'})
    source_context=SourceVideoContext('source','summary','raw','record',audiovisual_evidence=evidence)
    context=WholeVideoContext((source_context,),ProviderStatus('test',True,True,'applied'))
    region=SimpleNamespace(source_asset_id='source',source_start=0,source_end=3,
        dominant_process_status='uncertain',audience_delivery_status='mixed',confidence='SUPPORTED',conflict_flags=())
    result_context,_=with_global_editorial_evidence(context,SimpleNamespace(
        understanding=SimpleNamespace(regions=(region,)),capability_status='ok'))
    assert result_context.sources[0].audiovisual_evidence==evidence
    assert result_context.sources[0].editorial_evidence!=evidence


def test_encoder_padding_is_not_a_source_region(tmp_path):
    data=result();data['regions'][0].update(start=9,end=10)
    av,raw,session=provider(tmp_path,data)
    original=replace(source(),duration_sec=9.9)
    context=safe_whole_video_analyze(av,(original,),(),(),local_paths={'source':str(raw)})
    assert context.status.available
    region=json.loads(context.sources[0].audiovisual_evidence)['regions'][0]
    assert region['end']==9.9 and region['encoder_padding_trimmed_sec']>0
    assert '9.900000' in session.calls[0][1]['contents'][0]['parts'][1]['text']


def test_region_entirely_in_padding_is_rejected(tmp_path):
    data=result();data['regions'][0].update(start=9.95,end=10)
    av,raw,session=provider(tmp_path,data)
    context=safe_whole_video_analyze(av,(replace(source(),duration_sec=9.9),),(),(),local_paths={'source':str(raw)})
    assert not context.status.available
    assert 'source_end=9.9' in context.status.reason


def test_rejected_response_is_replayable_without_network(tmp_path):
    from cutsell_worker.av_response_contract import replay
    data=result();data['regions'][0]['confidence']=90
    av,raw,session=provider(tmp_path,data)
    context=safe_whole_video_analyze(av,(source(),),(),(),local_paths={'source':str(raw)})
    record=context.diagnostics['native_av'][0]
    assert record['status']=='rejected' and 'AV_CONFIDENCE_RANGE' in record['rejection']
    assert json.loads(record['response']['candidates'][0]['content']['parts'][0]['text'])['regions'][0]['confidence']==90
    calls=len(session.calls)
    with pytest.raises(ValueError,match='AV_CONFIDENCE_RANGE'):
        replay(json.loads(json.dumps(record)))
    assert len(session.calls)==calls
    assert 'test-key' not in json.dumps(record) and 'inline_data' not in json.dumps(record)
    schema=session.calls[1][1]['generationConfig']['responseJsonSchema']
    assert schema['properties']['regions']['items']['properties']['confidence']['maximum']==1


def test_replay_preserves_original_padding_response(tmp_path):
    from cutsell_worker.av_response_contract import replay
    data=result();data['regions'][0].update(start=9,end=10)
    av,raw,session=provider(tmp_path,data)
    context=safe_whole_video_analyze(av,(replace(source(),duration_sec=9.9),),(),(),local_paths={'source':str(raw)})
    record=context.diagnostics['native_av'][0]
    assert replay(record)['regions'][0]['end']==9.9
    assert json.loads(record['response']['candidates'][0]['content']['parts'][0]['text'])['regions'][0]['end']==10
