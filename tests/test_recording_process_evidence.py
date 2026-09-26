from dataclasses import replace
from types import SimpleNamespace
import pytest
from cutsell_worker.contracts import CandidateTake, DraftClip, DraftTimeline, EditStrategy, RankedTake
from cutsell_worker.recording_process_evidence import identity, recording_process_proofs, proof_for_clip
from cutsell_worker.pipeline import _semantic_best_take
from cutsell_worker.final_story_coherence_validation import _lost_semantic_atoms


def clip(text='Please restart the camera before my next attempt.'):
    return DraftClip('x','src',0,0,3,text,text,selected=False)


def row(c, **changes):
    r=dict(clip_id=c.clip_id,label='failed',confidence=.98,content_role='recording_only',
           local_failure_corroborated=True,source_identity=identity(c))
    r.update(changes)
    return r


def draft(c, rows):
    good=DraftClip('good','src',0,10,15,'This product fits both sizes.','This product fits both sizes.')
    return DraftTimeline('cutsell.v1','p',EditStrategy.STORYTELLING,(good,),(),(c,),
                         {'hybrid_editorial_chunks':[{'decisions':rows}]})

@pytest.mark.parametrize('text', [
    'Please restart the camera before my next attempt.',
    'Necesito consultar mis notas antes de volver a grabar.',
    'Could you move the microphone before we record again?',
])
def test_same_source_bound_evidence_controls_singleton_and_validator(text):
    c=clip(text);rows=[row(c)];proofs=recording_process_proofs([{'decisions':rows}])
    take=CandidateTake(c.clip_id,c.source_asset_id,0,c.start,c.end,c.text)
    selected,_,reason=_semantic_best_take((take,),{'x':('failed',.98)},'x',(RankedTake('x',.5,'baseline'),),recording_process_proofs=proofs)
    assert selected is None and reason=='single_bts_unusable'
    findings=_lost_semantic_atoms(draft(c,rows))
    assert findings[0]['classification']=='RECORDING_PROCESS_ONLY_REMOVED'
    assert findings[0]['blocking'] is False

@pytest.mark.parametrize('changes',[
    {'content_role':'mixed'}, {'content_role':'audience'}, {'content_role':'uncertain'},
    {'confidence':.94}, {'confidence':float('nan')}, {'confidence':True},
    {'local_failure_corroborated':False}, {'label':'keep'}, {'label':'winner'}])
def test_incomplete_conflicting_or_weak_evidence_preserves(changes):
    c=clip();assert not recording_process_proofs([{'decisions':[row(c),row(c,**changes)]}])

@pytest.mark.parametrize('changes',[
    {'source_asset_id':'other'}, {'text':'A different audience-facing statement.'},
    {'start':.01}, {'end':3.01}, {'clip_id':'child'},
])
def test_proof_cannot_leak_to_other_source_text_or_derived_clip(changes):
    c=clip();proofs=recording_process_proofs([{'decisions':[row(c)]}])
    assert proof_for_clip(replace(c,**changes),proofs) is None

@pytest.mark.parametrize('role,text',[
    ('audience','The comedian jokes about restarting the camera before every scene.'),
    ('mixed','Restart the camera. This device costs 25 dollars and never needs batteries.'),
    ('uncertain','No, it does not fit the larger model and costs 25 dollars.'),
])
def test_audience_humor_mixed_and_unique_facts_still_block_loss(role,text):
    c=clip(text);findings=_lost_semantic_atoms(draft(c,[row(c,content_role=role)]))
    assert any(f['blocking'] for f in findings)


def test_absent_new_role_does_not_retroactively_approve_old_diagnostics():
    c=clip();r=row(c);r.pop('content_role')
    assert not recording_process_proofs([{'decisions':[r]}])
    assert any(f['blocking'] for f in _lost_semantic_atoms(draft(c,[r])))


def test_source_bound_whole_video_recording_observation_removes_failed_false_start():
    c = clip('Tuve problemas de estómago, no.')
    av = {'status': 'overlap', 'observations': [{
        'role': 'recording_only', 'confidence': .85,
        'source_start': 0, 'source_end': 8,
        'audio': 'Speech interrupted while fixing hair.',
    }]}
    rows = [
        row(c, label='bts', confidence=.90, local_failure_corroborated=False, audiovisual=av),
        row(c, label='failed', confidence=.85, local_failure_corroborated=False, audiovisual=av),
    ]
    proofs = recording_process_proofs([{'decisions': rows}])
    assert proof_for_clip(c, proofs)['basis'] == 'whole_video_recording_only'


def test_whole_video_recording_observation_must_cover_exact_clip_span():
    c = clip('A unique audience statement.')
    av = {'status': 'overlap', 'observations': [{
        'role': 'recording_only', 'confidence': .90,
        'source_start': 1, 'source_end': 2,
    }]}
    assert not recording_process_proofs([{'decisions': [
        row(c, confidence=.90, local_failure_corroborated=False, audiovisual=av)
    ]}])


def test_pipeline_classifies_before_cleanup_and_records_source_identity(monkeypatch):
    from cutsell_worker import pipeline, hybrid_session_cleanup
    from cutsell_worker.contracts import ProcessingRequest, SourceAsset, CleanCutDecision
    from cutsell_worker.hybrid_editorial import EditorialDecision, EditorialJudgeResult
    take=CandidateTake('setup','src',0,0,3,'Please adjust the microphone before the next attempt.')
    good=CandidateTake('good','src',0,8,13,'This backpack keeps your laptop dry when it rains.')
    seen=[]
    class Judge:
        def judge(self,session):
            seen.extend(c.clip_id for c in session.candidates)
            return EditorialJudgeResult(tuple(EditorialDecision(c.clip_id,'bts' if c.clip_id=='setup' else 'keep',.99,'test','recording_only' if c.clip_id=='setup' else 'audience') for c in session.candidates),'test','test',True,True)
    def cleanup(takes,context):
        takes=tuple(takes)
        assert 'setup' in seen  # fails on the old destructive-before-classification order
        return tuple(t for t in takes if t.clip_id!='setup'), tuple(t for t in takes if t.clip_id=='setup'), (CleanCutDecision('setup',False,'test_physical_reset',.99),)
    monkeypatch.setattr(pipeline,'apply_clean_cut',cleanup)
    monkeypatch.setattr(hybrid_session_cleanup,'_failed_local_evidence',lambda *a:(True,('physical_reset',)))
    request=ProcessingRequest('p','u',(SourceAsset('src','p','u','raw.mp4',0,20,'local'),))
    result=pipeline.build_flow_b_draft(request,(take,good),editorial_judge=Judge(),boundary_owner='post_freeze')
    assert 'setup' not in {c.clip_id for c in result.draft.selected}
    assert result.draft.diagnostics['recording_process_evidence']['proof_count'] >= 1
    findings=_lost_semantic_atoms(result.draft)
    assert any(f.get('classification')=='RECORDING_PROCESS_ONLY_REMOVED' for f in findings)
