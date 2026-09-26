import copy
import json
from pathlib import Path

import pytest

from cutsell_worker.contextual_bts_evidence import contextual_bts_ids
from cutsell_worker.contracts import CandidateTake, RankedTake
from cutsell_worker.pipeline import _semantic_best_take


def row(**changes):
    value = dict(clip_id='x', label='bts', confidence=.9,
                 semantic_delete_recommended=True, dense_semantic_failure_cluster=True,
                 delete_basis='semantic_bts_inside_corroborated_failure_cluster',
                 local_failure_corroborated=False)
    value.update(changes)
    return value


def decide(rows, text='Can we adjust the camera before the next take?', label='bts', confidence=.9):
    take = CandidateTake('x','source',0,0,3,text)
    return _semantic_best_take((take,), {'x':(label,confidence)}, 'x',
                              (RankedTake('x',.5,'baseline'),),
                              contextual_bts_evidence_ids=contextual_bts_ids([{'decisions':rows}]))


@pytest.mark.parametrize('text', [
    'Can we adjust the camera before the next take?',
    'Necesito volver a grabar porque me equivoqué.',
    'Please wait until the microphone is ready.',
])
def test_general_contextual_evidence_reaches_singleton_authority(text):
    assert decide([row()],text)[0] is None
    assert decide([row()],text)[2] == 'single_bts_unusable'


@pytest.mark.parametrize('changes', [
    {'label':'winner'}, {'label':'keep'}, {'label':'alternate'}, {'label':'failed'},
    {'confidence':.89}, {'confidence':True}, {'confidence':float('nan')},
    {'confidence':1.1}, {'dense_semantic_failure_cluster':False},
    {'semantic_delete_recommended':False},
])
def test_label_or_weak_context_alone_never_deletes(changes):
    assert decide([row(**changes)])[0] == 'x'


@pytest.mark.parametrize('label', ['winner','keep','alternate','failed'])
def test_any_conflicting_window_preserves_content(label):
    assert decide([row(),row(label=label)])[0] == 'x'


def test_family_label_must_also_agree():
    assert decide([row()],label='failed')[0] == 'x'
    assert decide([row()],label='winner')[0] == 'x'
    assert decide([row()],confidence=.8)[0] == 'x'


def test_dense_context_survives_generic_high_confidence_basis_precedence():
    selected, _preferred, reason = decide([row(delete_basis='high_confidence_semantic', confidence=.95)])
    assert selected is None
    assert reason == 'single_bts_unusable'


def test_recorded_mov_evidence_reaches_decision_without_phrase_matching():
    data=json.loads((Path(__file__).parent/'fixtures'/'cutsell_mov_selected_editorial_evidence.json').read_text())
    before=copy.deepcopy(data)
    ids=contextual_bts_ids([data])
    assert ids == {'clip_9dd11261ade9c29970a7'}
    assert data == before
    rows=[dict(r,clip_id='x') for r in data['decisions'] if r['clip_id'] in ids]
    assert decide(rows, text='Arbitrary transcript; selection uses evidence, not a phrase blacklist.')[0] is None


def test_full_draft_wiring_retains_audience_content(monkeypatch):
    from cutsell_worker import pipeline
    from cutsell_worker.contracts import ProcessingRequest, SourceAsset
    from cutsell_worker.hybrid_session_cleanup import HybridSessionCleanupResult
    source=SourceAsset('src','p','u','raw.mp4',0,20,'s3://bucket/raw.mp4')
    request=ProcessingRequest(project_id='p',user_id='u',sources=(source,))
    takes=(CandidateTake('x','src',0,0,2,'Let me reset the camera.'),
           CandidateTake('good','src',0,5,9,'This bottle keeps water cold for twelve hours.'))
    def resolution(kept, *args):
        return HybridSessionCleanupResult(tuple(kept),(),1,1,({'decisions':[row()]},),
                                          (('x','bts',.9),('good','keep',.95))), set()
    monkeypatch.setattr(pipeline,'apply_composite_resolution',resolution)
    result=pipeline.build_flow_b_draft(request,takes)
    assert 'x' not in {c.clip_id for c in result.draft.selected}
    assert 'x' in {c.clip_id for c in result.draft.discarded}
    assert 'good' in {c.clip_id for c in result.draft.selected}
    assert any(r.get('contextual_bts_evidence_ids') == ['x'] for r in result.draft.diagnostics['take_judge_groups'])