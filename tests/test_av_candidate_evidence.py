import json
from dataclasses import replace
from cutsell_worker.av_candidate_evidence import candidate_observations
from cutsell_worker.contracts import CandidateTake,Word
from cutsell_worker.whole_video_analysis import WholeVideoContext,SourceVideoContext
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.hybrid_session_cleanup import _editorial_session
from cutsell_worker.hybrid_payload import build_compact_editorial_payload,_bounded_av_evidence


def context():
    regions=[dict(observation_id=f'av_{i}',start=i,end=i+8,role='mixed',confidence=.9,
                  audio_observation='a'*240,visual_observation='v'*240,reason='r'*240) for i in range(12)]
    return WholeVideoContext((SourceVideoContext('s','Global summary','raw','record',
        audiovisual_evidence=json.dumps({'kind':'audiovisual_observations_v1','source_sha256':'a'*64,'regions':regions})),),
        ProviderStatus('test',True,True,'ok'))


def take(cid='c',source='s',start=5):
    words=tuple(Word(t,start+i*.4,start+(i+1)*.4) for i,t in enumerate('This is valid audience content'.split()))
    return CandidateTake(cid,source,0,start,start+2,'This is valid audience content',words)


def test_links_are_source_time_and_word_bound_and_advisory():
    linked=candidate_observations(take(),context())
    assert linked['status']=='overlap' and linked['observations'][0]['word_range']==[0,4]
    assert linked['omitted_count']>0 and 'not_independent' in linked['authority']
    assert candidate_observations(take(source='other'),context())['status']=='unavailable'
    assert candidate_observations(take(start=100),context())['status']=='no_observation_for_span'


def test_global_region_coverage_survives_verbose_observations():
    compact=json.loads(_bounded_av_evidence(context().sources[0].audiovisual_evidence,1800))
    assert len(compact['regions'])==12


def test_dense_av_window_fits_existing_budget_without_dropping_local_links():
    session=_editorial_session(tuple(take(f'c{i}') for i in range(10)),context(),partition_index=0,chunk_index=0)
    payload=build_compact_editorial_payload(session)
    assert len(repr(payload))<=12000 and len(payload['candidates'])==10
    assert all(len(c['evidence']['audiovisual']['observations'])==3 for c in payload['candidates'])
