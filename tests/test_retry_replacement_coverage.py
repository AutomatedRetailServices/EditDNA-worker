from dataclasses import replace
import pytest
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.retry_replacement_coverage import replacement_coverage, review_retry_pool
from cutsell_worker.hybrid_retry_winner_authority import enforce_proven_retry_winners
from cutsell_worker.whole_video_analysis import WholeVideoContext, SourceVideoContext, TemporalEvent
from cutsell_worker.providers import ProviderStatus


def take(cid, text, start=0, complete=True):
    return CandidateTake(cid, 'source', 0, start, start+3, text, complete_idea=complete)


def context():
    return WholeVideoContext((SourceVideoContext('source','','','',
        (TemporalEvent('source',0,3,'retry_setup',.99,'confirmed'),)),),ProviderStatus('test',True,True,'ok'))


@pytest.mark.parametrize('text', ['This sturdy backpack holds two laptops.', 'Esta mochila resistente guarda dos computadoras.'])
def test_failed_attempt_yields_only_after_full_coverage(text):
    a=take('a',text,complete=False);b=take('b',text,5)
    kept,removed,diag=enforce_proven_retry_winners((a,b),(('a','failed',.95),('b','winner',.95)),context())
    assert removed==(a,) and kept==(b,)
    assert diag[-1]['replacement_coverage']['coverage_verified']


@pytest.mark.parametrize('original,replacement', [
    ('This backpack holds 2 laptops.', 'This backpack holds laptops.'),
    ('This backpack holds 2 laptops.', 'This backpack holds 3 laptops.'),
    ('This backpack is not waterproof.', 'This backpack is waterproof.'),
    ('Esta mochila no es impermeable.', 'Esta mochila es impermeable.'),
    ('This backpack holds laptops. Shipping arrives tomorrow.', 'This backpack holds laptops.'),
])
def test_missing_or_changed_information_never_certified(original,replacement):
    verdict=replacement_coverage(take('a',original),take('b',replacement,5))
    assert not verdict['coverage_verified']


@pytest.mark.parametrize('mutation', ['incomplete','mixed','conflicting','source'])
def test_replacement_must_itself_be_usable(mutation):
    text='This sturdy backpack holds two laptops.'
    a,b=take('a',text),take('b',text,5)
    rows=[{'clip_id':'b','label':'winner','content_role':'audience'}]
    if mutation=='incomplete': b=replace(b,complete_idea=False)
    if mutation=='mixed': rows[0]['content_role']='mixed'
    if mutation=='conflicting': rows.append({'clip_id':'b','label':'failed'})
    if mutation=='source': b=replace(b,source_asset_id='different')
    assert not replacement_coverage(a,b,({'decisions':rows},))['coverage_verified']


def test_rejected_first_retake_does_not_hide_a_valid_later_retake():
    text='This sturdy backpack holds two laptops.'
    a,b,c=take('a',text,complete=False),take('b',text,5),take('c',text,10)
    windows=({'decisions':[{'clip_id':'b','label':'winner','content_role':'mixed'}]},)
    kept,removed,diag=enforce_proven_retry_winners((a,b,c),
        (('a','failed',.95),('b','winner',.95),('c','winner',.95)),context(),session_diagnostics=windows)
    assert removed==(a,) and kept==(b,c)
    assert diag[0]['reason']=='replacement_contains_recording_process'
    assert diag[-1]['winner_clip_id']=='c'


def test_pool_review_crosses_classification_windows_without_deleting():
    text='This sturdy backpack holds two laptops.'
    a,b=take('a',text,complete=False),take('b',text,5)
    windows=({'decisions':[{'clip_id':'a','label':'failed','confidence':.95}]},
             {'decisions':[{'clip_id':'b','label':'winner','confidence':.95,'content_role':'audience'}]})
    review=review_retry_pool((a,b),windows)
    assert review[0]['comparisons'][0]['coverage_verified']
    assert review[0]['authority']=='comparison_only'
    assert review[0]['status']=='compared'  # retain the existing v1 reader value
    assert review[0]['comparison_status']=='coverage_checked'


def test_retry_comparison_crosses_long_same_session_gap_when_identity_and_coverage_hold():
    text='This sturdy backpack holds two laptops.'
    a=take('a',text,start=0,complete=False)
    b=take('b',text,start=85,complete=True)
    windows=({'partition_index':0,'member_ids':['a'],
              'decisions':[{'clip_id':'a','label':'failed','confidence':.95}]},
             {'partition_index':0,'member_ids':['b'],
              'decisions':[{'clip_id':'b','label':'winner','confidence':.95,'content_role':'audience'}]})
    review=review_retry_pool((a,b),windows)
    comparison=review[0]['comparisons'][0]
    assert comparison['gap_sec'] > 24
    assert comparison['comparison_status']=='coverage_checked'
    assert comparison['coverage_verified'] is True


def test_retry_authority_crosses_long_same_session_gap_but_keeps_safety_gates():
    text='This sturdy backpack holds two laptops.'
    a=take('a',text,start=0,complete=False)
    b=take('b',text,start=85,complete=True)
    kept,removed,diag=enforce_proven_retry_winners(
        (a,b),(('a','failed',.95),('b','winner',.95)),context())
    assert kept==(b,) and removed==(a,)
    assert diag[-1]['gap_sec'] > 20


def test_authoritative_audience_winner_survives_an_earlier_uncertain_mixed_window():
    text='This sturdy backpack holds two laptops.'
    failed=take('failed',text,start=0,complete=False)
    winner=take('winner',text,start=40,complete=True)
    windows=({'decisions':[
        {'clip_id':'failed','label':'failed','confidence':.95},
        {'clip_id':'winner','label':'uncertain','confidence':.9,'content_role':'mixed'},
    ]},{'decisions':[
        {'clip_id':'winner','label':'winner','confidence':.95,'content_role':'audience'},
    ]})
    assert replacement_coverage(failed,winner,windows)['coverage_verified'] is True


def test_uncertain_mixed_alone_never_certifies_a_replacement():
    text='This sturdy backpack holds two laptops.'
    failed=take('failed',text,start=0,complete=False)
    peer=take('peer',text,start=40,complete=True)
    windows=({'decisions':[
        {'clip_id':'peer','label':'uncertain','confidence':.99,'content_role':'mixed'},
    ]},)
    verdict=replacement_coverage(failed,peer,windows)
    assert verdict['coverage_verified'] is False
    assert verdict['reason']=='replacement_contains_recording_process'


@pytest.mark.parametrize('shadow_label,shadow_role', [
    ('keep','mixed'), ('keep','recording_only'),
    ('alternate','mixed'), ('alternate','recording_only'),
])
def test_authoritative_winner_never_overrides_positive_recording_role_conflict(
    shadow_label, shadow_role,
):
    text='This sturdy backpack holds two laptops.'
    failed=take('failed',text,start=0,complete=False)
    winner=take('winner',text,start=40,complete=True)
    windows=({'decisions':[
        {'clip_id':'winner','label':shadow_label,'confidence':.95,'content_role':shadow_role},
    ]},{'decisions':[
        {'clip_id':'winner','label':'winner','confidence':.99,'content_role':'audience'},
    ]})
    verdict=replacement_coverage(failed,winner,windows)
    assert verdict['coverage_verified'] is False
    assert verdict['reason']=='replacement_not_consistently_usable'


def test_similar_topic_does_not_prove_coverage_of_all_claims():
    a=take('a','I noticed spots that felt like a rash and an allergy.')
    b=take('b','I noticed marks behind my ear that were caused by hormones.',5)
    assert not replacement_coverage(a,b)['coverage_verified']


def test_prior_identity_rejection_does_not_hide_another_covered_retake():
    from cutsell_worker.complete_retry_identity_guard import SEQUENCE_IDENTITY_BELOW_THRESHOLD
    text='This sturdy backpack holds two laptops.'
    a,b,c=take('a',text),take('b',text,5),take('c',text,10)
    windows=({'decisions':[{'clip_id':'a','replacement_candidate_clip_id_before_guard':'b',
        'replacement_rejection_reason':SEQUENCE_IDENTITY_BELOW_THRESHOLD}]},)
    kept,removed,diag=enforce_proven_retry_winners((a,b,c),
        (('a','failed',.95),('b','winner',.95),('c','winner',.95)),context(),session_diagnostics=windows)
    assert removed==(a,) and kept==(b,c)
    assert diag[0]['reason']=='prior_replacement_rejection_respected'
    assert diag[-1]['winner_clip_id']=='c'
    assert not replacement_coverage(a,b,windows)['coverage_verified']
