from dataclasses import replace
import pytest
from cutsell_worker.contracts import CandidateTake, Word
from cutsell_worker.recording_process_evidence import identity, recording_process_proofs, proof_for_clip
from cutsell_worker.recording_process_trim import apply_recording_process_trims


def sample(text='Let me restart this bag holds two laptops', prefix=3, suffix=0, **changes):
    words=tuple(Word(t, i*.4, (i+1)*.4) for i,t in enumerate(text.split()))
    take=CandidateTake('parent','source',0,0,len(words)*.4,text,words)
    row=dict(clip_id=take.clip_id, source_identity=identity(take), content_role='mixed',
             label='failed', confidence=.8, recording_confidence=.98,
             recording_prefix_words=prefix, recording_suffix_words=suffix)
    row.update(changes)
    return take, row


def run(take, rows, monkeypatch, corroborated=True):
    from cutsell_worker import hybrid_session_cleanup
    def evidence(child, context):
        assert child.signals is None
        return corroborated, ('localized_event',) if corroborated else ()
    monkeypatch.setattr(hybrid_session_cleanup,'_failed_local_evidence', evidence)
    return apply_recording_process_trims((take,), ({'decisions': rows},), None)


@pytest.mark.parametrize('text,prefix,suffix,expected',[
    ('Let me restart this bag holds two laptops',3,0,'this bag holds two laptops'),
    ('Otra vez esta mochila no cuesta veinte dólares',2,0,'esta mochila no cuesta veinte dólares'),
    ('this bag holds two laptops stop the camera',0,3,'this bag holds two laptops'),
    ('try again this bag holds two laptops stop recording',2,2,'this bag holds two laptops'),
])
def test_only_aligned_recording_edges_removed(text,prefix,suffix,expected,monkeypatch):
    take,row=sample(text,prefix,suffix)
    kept,deleted,proof_rows,diag=run(take,[row],monkeypatch)
    assert [t.text for t in kept]==[expected]
    assert diag[0]['applied_mixed_trim']
    assert sum(len(t.words) for t in (*kept,*deleted)) == len(take.words)
    proofs=recording_process_proofs([{'decisions':proof_rows}])
    assert all(proof_for_clip(t,proofs) for t in deleted)
    assert not proof_for_clip(kept[0],proofs)
    assert not proof_for_clip(take,proofs)


@pytest.mark.parametrize('changes',[
    {'content_role':'audience'}, {'content_role':'uncertain'},
    {'recording_confidence':.96}, {'recording_confidence':float('nan')},
    {'recording_confidence':True}, {'recording_confidence':None},
    {'recording_prefix_words':-1}, {'recording_prefix_words':True},
    {'recording_prefix_words':99}, {'recording_suffix_words':99},
    {'source_identity':{}},
])
def test_unsafe_or_unbound_evidence_preserves(changes,monkeypatch):
    take,row=sample(**changes)
    kept,deleted,proofs,diag=run(take,[row],monkeypatch)
    assert kept==(take,) and deleted==proofs==()
    assert diag and not diag[0]["applied_mixed_trim"]


def test_window_disagreement_preserves(monkeypatch):
    take,row=sample()
    kept,deleted,*_=run(take,[row,{**row,'recording_prefix_words':2}],monkeypatch)
    assert kept==(take,) and not deleted


def test_parent_failure_does_not_replace_fragment_corroboration(monkeypatch):
    take,row=sample()
    row['local_failure_corroborated']=True
    kept,deleted,*_=run(take,[row],monkeypatch,corroborated=False)
    assert kept==(take,) and not deleted


@pytest.mark.parametrize('mutation', ['no_words','mismatch','overlap','outside'])
def test_bad_alignment_preserves(mutation,monkeypatch):
    take,row=sample()
    if mutation=='no_words': take=replace(take,words=())
    if mutation=='mismatch': take=replace(take,words=take.words[1:])
    if mutation=='overlap': take=replace(take,words=(replace(take.words[0],end=.5),*take.words[1:]))
    if mutation=='outside': take=replace(take,words=(*take.words[:-1],replace(take.words[-1],end=100)))
    kept,deleted,*_=run(take,[row],monkeypatch)
    assert kept==(take,) and not deleted


def test_recording_certainty_is_not_failed_take_certainty():
    take,row=sample(content_role='recording_only',recording_confidence=.98)
    row['local_failure_corroborated']=True
    assert recording_process_proofs([{'decisions':[row]}])
    assert not recording_process_proofs([{'decisions':[{**row,'confidence':.99,'recording_confidence':.7}]}])


def test_real_local_event_must_overlap_removed_edge():
    from cutsell_worker.whole_video_analysis import WholeVideoContext, SourceVideoContext, TemporalEvent
    from cutsell_worker.providers import ProviderStatus
    take,row=sample()
    def context(start,end):
        event=TemporalEvent('source',start,end,'retry_setup',.98,'confirmed reset')
        return WholeVideoContext((SourceVideoContext('source','','','',(event,)),),ProviderStatus('test',True,True,'ok'))
    kept,deleted,_,_=apply_recording_process_trims((take,),({'decisions':[row]},),context(0,1.2))
    assert deleted and kept[0].text=='this bag holds two laptops'
    kept,deleted,_,_=apply_recording_process_trims((take,),({'decisions':[row]},),context(2,3))
    assert kept==(take,) and not deleted


def test_pipeline_consumes_trim_and_validator_uses_only_fragment_proof(monkeypatch):
    from cutsell_worker import pipeline, hybrid_session_cleanup
    from cutsell_worker.contracts import ProcessingRequest,SourceAsset
    from cutsell_worker.hybrid_editorial import EditorialDecision,EditorialJudgeResult
    from cutsell_worker.final_story_coherence_validation import _lost_semantic_atoms
    take,row=sample('Please reset the camera this bag holds two laptops',prefix=4)
    class Judge:
        def judge(self,session):
            return EditorialJudgeResult(tuple(EditorialDecision(c.clip_id,'failed',.8,'test','mixed',.98,4,0) for c in session.candidates),'test','test',True,True)
    monkeypatch.setattr(hybrid_session_cleanup,'_failed_local_evidence',lambda *args:(True,('reset',)))
    request=ProcessingRequest('p','u',(SourceAsset('source','p','u','raw.mp4',0,20,'local'),))
    result=pipeline.build_flow_b_draft(request,(take,),editorial_judge=Judge(),boundary_owner='post_freeze')
    assert any('this bag holds two laptops' in c.text for c in result.draft.selected)
    assert all('Please reset' not in c.text for c in result.draft.selected)
    assert any(c.text=='Please reset the camera' for c in result.draft.discarded)
    findings=_lost_semantic_atoms(result.draft)
    assert any(f.get('classification')=='RECORDING_PROCESS_ONLY_REMOVED' for f in findings)


def test_truncated_provider_text_cannot_authorize_word_trim():
    from cutsell_worker.hybrid_editorial import EditorialCandidate,EditorialSession
    from cutsell_worker.hybrid_provider import TransportEditorialJudge
    from cutsell_worker.hybrid_payload import HybridCostPolicy
    text='Please restart the camera this bag holds two laptops'
    candidate=EditorialCandidate('parent',text,0,4,'keep',.5,word_texts=tuple(text.split()))
    session=EditorialSession('s','source',(candidate,),.5,.5)
    def transport(payload,limit):
        assert 'word_texts' not in payload['candidates'][0]
        return {'decisions':[{'clip_id':'parent','label':'failed','confidence':.8,'content_role':'mixed',
                             'recording_confidence':.99,'recording_prefix_words':4}]}
    judge=TransportEditorialJudge('test','test',transport,cost_policy=HybridCostPolicy(max_chars_per_candidate=20))
    with pytest.raises(ValueError,match='untruncated aligned words'):
        judge.judge(session)


def test_optional_word_evidence_reservation_stays_inside_existing_ceiling():
    from cutsell_worker.hybrid_google_transport import _compact_output_token_ceiling
    payload={'candidates':[{'word_texts':['hello','world']}] * 10}
    assert _compact_output_token_ceiling(payload,500)==500
    assert _compact_output_token_ceiling(payload,320)==320


@pytest.mark.parametrize('text',[
    'This backpack fits my laptop. Please restart the camera. Shipping arrives in three days.',
    'Esta mochila protege mi computadora. Necesito revisar mis notas. El envío tarda tres días.',
])
def test_interior_preparation_keeps_separate_source_sentences(text,monkeypatch):
    take,row=sample(text,prefix=0,recording_word_ranges=((5,8),))
    kept,deleted,proofs,diag=run(take,[row],monkeypatch)
    assert len(kept)==2 and len(deleted)==1
    assert tuple(w for t in sorted((*kept,*deleted),key=lambda t:t.start) for w in t.words)==take.words
    assert kept[0].end<=deleted[0].start and deleted[0].end<=kept[1].start
    assert diag[0]['kept_clip_ids']==[t.clip_id for t in kept]


def test_interior_self_correction_is_not_silently_spliced(monkeypatch):
    take,row=sample('It costs twenty wait no I mean fifty dollars today',prefix=0,recording_word_ranges=((4,6),))
    kept,deleted,proofs,diag=run(take,[row],monkeypatch)
    assert kept==(take,) and not deleted
    assert diag[0]['reason']=='interior_sentence_boundary_unproven'


def test_full_pipeline_keeps_both_unique_sentences_after_interior_cleanup():
    from cutsell_worker import pipeline
    from cutsell_worker.contracts import ProcessingRequest,SourceAsset
    from cutsell_worker.hybrid_editorial import EditorialDecision,EditorialJudgeResult
    from cutsell_worker.whole_video_analysis import WholeVideoContext,SourceVideoContext,TemporalEvent
    from cutsell_worker.providers import ProviderStatus
    take,row=sample('This backpack fits my laptop. Please restart the camera. Shipping arrives in three days.',prefix=0)
    class Judge:
        def judge(self,session):
            return EditorialJudgeResult(tuple(EditorialDecision(c.clip_id,'failed',.8,'test','mixed',.99,0,0,((5,8),)) for c in session.candidates),'test','test',True,True)
    context=WholeVideoContext((SourceVideoContext('source','Product and delivery facts.','raw','record',
        (TemporalEvent('source',2,3.6,'retry_setup',.99,'confirmed physical recording restart'),)),),ProviderStatus('test',True,True,'ok'))
    request=ProcessingRequest('p','u',(SourceAsset('source','p','u','raw.mp4',0,20,'local'),))
    result=pipeline.build_flow_b_draft(request,(take,),editorial_judge=Judge(),whole_video_context=context,boundary_owner='post_freeze')
    text=' '.join(c.text for c in result.draft.selected)
    assert 'backpack fits my laptop' in text and 'Shipping arrives in three days' in text
    assert 'restart the camera' not in text
