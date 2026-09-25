"""General EN/ES regressions; no provider calls or media renders."""
from dataclasses import replace
import pytest
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.hybrid_editorial import EditorialCandidate, EditorialSession, EditorialDecision, EditorialJudgeResult, validate_editorial_result
from cutsell_worker.hybrid_session_cleanup import apply_hybrid_session_cleanup
from cutsell_worker.hybrid_retry_winner_authority import enforce_proven_retry_winners
from cutsell_worker.retry_replacement_coverage import replacement_semantics
from cutsell_worker.whole_video_analysis import WholeVideoContext, SourceVideoContext, TemporalEvent
from cutsell_worker.providers import ProviderStatus


def take(cid, text, start=0, complete=True):
    return CandidateTake(cid, 'source', 0, start, start+2, text, complete_idea=complete)


def context():
    return WholeVideoContext((SourceVideoContext('source', '', '', '',
        (TemporalEvent('source', 0, 2, 'retry_setup', .99, 'confirmed'),)),),
        ProviderStatus('test', True, True, 'ok'))


@pytest.mark.parametrize('role', ['mixed', 'recording_only'])
def test_recording_winner_loses_authority_without_losing_trim_evidence(role):
    candidate = EditorialCandidate('a', 'Wait. This backpack holds laptops.', 0, 4, 'keep', .5)
    session = EditorialSession('s', 'source', (candidate,), .5)
    decision = EditorialDecision('a', 'winner', .99, 'test', role,
                                 .99, 1, 0)
    result = validate_editorial_result(session, EditorialJudgeResult((decision,), 'stub', 'stub', True, True))
    actual = result.decisions[0]
    assert actual.label == 'uncertain' and actual.proposed_label == 'winner'
    assert actual.content_role == role and actual.recording_prefix_words == 1
    assert actual.recording_confidence == .99
    assert validate_editorial_result(session, result) == result


def test_audience_humor_is_not_demoted():
    session = EditorialSession('s', 'source', (EditorialCandidate('a', 'I laughed at my mistake.', 0, 3, 'keep', .5),), .5)
    d = EditorialDecision('a', 'winner', .99, 'intentional_humor', 'audience')
    assert validate_editorial_result(session, EditorialJudgeResult((d,), 'stub', 'stub', True, True)).decisions[0].label == 'winner'


class Judge:
    def __init__(self, decisions):
        self.decisions = decisions
        self.calls = []
    def judge(self, session):
        self.calls.append(session)
        return EditorialJudgeResult(tuple(self.decisions[c.clip_id] for c in session.candidates),
                                    'stub', 'stub', True, True)


@pytest.mark.parametrize('text', ['This backpack holds two laptops.', 'Esta mochila guarda dos computadoras.'])
def test_clean_keep_across_windows_reaches_existing_retry_authority(text):
    a, b = take('a', text, complete=False), take('b', text, 5)
    judge = Judge({'a': EditorialDecision('a', 'failed', .95, 'abandoned', 'audience'),
                   'b': EditorialDecision('b', 'keep', .95, 'independent_delivery', 'audience')})
    cleanup = apply_hybrid_session_cleanup((a, b), context(), judge, chunk_size=1, chunk_stride=1)
    row = next(r for w in cleanup.diagnostics for r in w.get('decisions', ()) if r['clip_id'] == 'a')
    assert row['later_retry_replacement_id'] == 'b'
    assert row['replacement_scope'] == 'complete_classified_session'
    assert len(judge.calls) == 2  # no additional provider request
    kept, removed, rows = enforce_proven_retry_winners(cleanup.kept, cleanup.semantic_decisions,
        context(), session_diagnostics=cleanup.diagnostics)
    assert kept == (b,) and removed == (a,)
    assert rows[-1]['replacement_coverage']['coverage_verified']


@pytest.mark.parametrize('prefix,full', [
    ('This backpack can hold', 'This backpack can hold two laptops.'),
    ('Esta mochila puede guardar', 'Esta mochila puede guardar dos computadoras.'),
])
def test_short_abandoned_opening_can_compete_only_with_exact_completion(prefix, full):
    a, b = take('a', prefix, complete=False), take('b', full, 5)
    kept, removed, rows = enforce_proven_retry_winners((a,b), (('a','failed',.95),('b','winner',.95)), context())
    assert removed == (a,) and kept == (b,)
    assert rows[-1]['exact_abandoned_prefix']
    assert enforce_proven_retry_winners((a,b), (('a','failed',.95),('b','winner',.95)), None)[1] == ()


@pytest.mark.parametrize('role,label', [('mixed','keep'), ('audience','failed'), ('recording_only','winner'), ('uncertain','uncertain')])
def test_any_conflicting_window_blocks_keep_replacement(role, label):
    text = 'This sturdy backpack holds two laptops.'
    a,b = take('a',text,complete=False),take('b',text,5)
    windows = ({'decisions':[{'clip_id':'b','label':'keep','confidence':.99,'content_role':'audience'}]},
               {'decisions':[{'clip_id':'b','label':label,'confidence':.9,'content_role':role}]})
    assert 'b' not in replacement_semantics(windows)
    assert enforce_proven_retry_winners((a,b),(('a','failed',.95),('b','keep',.99)),
        context(),session_diagnostics=windows)[1] == ()


@pytest.mark.parametrize('text,replacement', [
    ('This backpack holds 2 laptops.', 'This backpack holds 3 laptops.'),
    ('Esta mochila no es impermeable.', 'Esta mochila es impermeable.'),
    ('This backpack holds laptops. Shipping arrives tomorrow.', 'This backpack holds laptops.'),
])
def test_global_replacement_proposal_preserves_facts(text,replacement):
    a,b = take('a',text,complete=False),take('b',replacement,5)
    judge = Judge({'a':EditorialDecision('a','failed',.95,'test','audience'),
                   'b':EditorialDecision('b','keep',.99,'test','audience')})
    result = apply_hybrid_session_cleanup((a,b),context(),judge,chunk_size=1)
    row = next(r for w in result.diagnostics for r in w.get('decisions',()) if r['clip_id']=='a')
    assert row['later_retry_replacement_id'] is None
    assert not enforce_proven_retry_winners((a,b), result.semantic_decisions, context(),
                                           session_diagnostics=result.diagnostics)[1]


@pytest.mark.parametrize('label,confidence', [('uncertain', .9), ('winner', .5)])
def test_winner_also_needs_cross_window_consistency(label, confidence):
    text='This sturdy backpack holds two laptops.'
    a,b=take('a',text,complete=False),take('b',text,5)
    windows=({'decisions':[{'clip_id':'b','label':'winner','confidence':.99,'content_role':'audience'}]},
             {'decisions':[{'clip_id':'b','label':label,'confidence':confidence,'content_role':'audience'}]})
    kept,removed,diag=enforce_proven_retry_winners((a,b),(('a','failed',.95),('b','winner',.99)),
                                                 context(),session_diagnostics=windows)
    assert not removed and kept==(a,b)
    assert diag[-1]['reason']=='replacement_not_consistently_usable'


def test_abbreviated_retake_reports_coverage_block_instead_of_no_candidate():
    from cutsell_worker.hybrid_session_cleanup import _later_semantic_retry_replacement
    from cutsell_worker.complete_retry_identity_guard import _consume_replacement_guard_diagnostic, COVERAGE_NOT_VERIFIED
    a=take('a','ahi fue cuando me mandaron a hacer sonografias de tiroides y otros')
    b=take('b','a hacer sonografia de tiroides y otras sonografias',5)
    assert _later_semantic_retry_replacement(a,(a,b),{'b':('winner',.99)})[0] is None
    assert _consume_replacement_guard_diagnostic().replacement_rejection_reason==COVERAGE_NOT_VERIFIED


def test_mixed_winner_stays_available_and_is_not_a_replacement():
    text='This sturdy backpack holds two laptops.'
    a,b=take('a',text,complete=False),take('b',text+' Wait let me try again.',5)
    judge=Judge({'a':EditorialDecision('a','failed',.95,'test','audience'),
                 'b':EditorialDecision('b','winner',.99,'test','mixed',.99)})
    result=apply_hybrid_session_cleanup((a,b),context(),judge,chunk_size=1)
    rows=[r for w in result.diagnostics for r in w.get('decisions',())]
    assert rows[0]['later_retry_replacement_id'] is None
    mixed=next(r for r in rows if r['clip_id']=='b')
    assert mixed['proposed_label']=='winner' and mixed['winner_authority_withheld']
    assert mixed['label']=='uncertain' and mixed['content_role']=='mixed'
    assert b in result.kept


@pytest.mark.parametrize('peer_label', ['winner', 'keep'])
def test_cross_window_search_never_crosses_creator_session_boundary(peer_label):
    text='This sturdy backpack holds two laptops.'
    a,b=take('a',text,complete=False),take('b',text,5)
    ctx=context()
    boundary=tuple(TemporalEvent('source',3.5,3.6,kind,.99,'scene boundary') for kind in
                   ('camera_disengagement_candidate','facial_expression_shift_candidate','body_reset_candidate'))
    ctx=replace(ctx,sources=(replace(ctx.sources[0],events=ctx.sources[0].events+boundary),))
    judge=Judge({'a':EditorialDecision('a','failed',.95,'test','audience'),
                 'b':EditorialDecision('b',peer_label,.99,'test','audience')})
    result=apply_hybrid_session_cleanup((a,b),ctx,judge,chunk_size=1)
    row=next(r for w in result.diagnostics for r in w.get('decisions',()) if r['clip_id']=='a')
    assert row['later_retry_replacement_id'] is None
    assert not enforce_proven_retry_winners((a,b), result.semantic_decisions, ctx,
                                           session_diagnostics=result.diagnostics)[1]


def test_original_partition_identity_survives_pruned_boundary_neighbors():
    text='This sturdy backpack holds two laptops.'
    a,b=take('a',text,complete=False),take('b',text,12)
    windows=({'partition_index':0,'member_ids':['a','removed-left'],'decisions':[]},
             {'partition_index':1,'member_ids':['removed-right','b'],
              'decisions':[{'clip_id':'b','label':'keep','confidence':.99,'content_role':'audience'}]})
    assert not enforce_proven_retry_winners((a,b),(('a','failed',.95),('b','keep',.99)),
        context(),session_diagnostics=windows)[1]


def test_existing_composite_chain_consumes_clean_keep_replacement():
    from cutsell_worker.composite_resolver import apply_composite_resolution
    a=take('a','This backpack can hold',complete=False)
    b=take('b','This backpack can hold two laptops.',5)
    judge=Judge({'a':EditorialDecision('a','failed',.95,'abandoned','audience'),
                 'b':EditorialDecision('b','keep',.99,'complete_delivery','audience')})
    result,_=apply_composite_resolution((a,b),context(),judge)
    assert a not in result.kept and b in result.kept
