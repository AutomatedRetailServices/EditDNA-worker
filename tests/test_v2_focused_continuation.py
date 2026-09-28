"""Source-level confirmation remains optional and bound to adjacent spoken candidates."""
from dataclasses import replace
import json
import subprocess

from cutsell_worker.contracts import Word
from cutsell_worker.hybrid_google_transport import DollarBudgetLedger
from cutsell_worker.hybrid_provider_settings import HybridProviderSettings
from cutsell_worker.unified_selection_google import GoogleUnifiedSelectionReasoner
from cutsell_worker.unified_selection_reasoner import UnifiedSelectionDecision
from test_cutsell_unified_selection_reasoner import clip, draft


class Response:
    def __init__(self, data):
        self.data = data
    def raise_for_status(self):
        pass
    def json(self):
        return self.data


class Session:
    def __init__(self, observation=None, error=None):
        self.observation = observation
        self.error = error
        self.calls = []
    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url.endswith('countTokens'):
            return Response({'totalTokens': 2200})
        if self.error:
            raise self.error
        return Response({'candidates': [{'finishReason': 'STOP', 'content': {'parts': [
            {'text': json.dumps(self.observation)}]}}],
            'usageMetadata': {'promptTokenCount': 2200, 'candidatesTokenCount': 120}})


def fixtures(monkeypatch, observation, error=None):
    import cutsell_worker.whole_video_av as av
    monkeypatch.setattr(av, 'slice_prepared_av', lambda path, dest, start, length: dest.write_bytes(b'video'))
    a = replace(clip('a', 73.8, 77.07, 'in your', selected=False),
                words=(Word('your', 76.8, 77.07),))
    b = replace(clip('b', 77.07, 86.5, 'first week', selected=True),
                words=(Word('first', 77.07, 77.4),))
    d = replace(draft(), selected=(b,), alternates=(), discarded=(a,),
                diagnostics={'editorial_engine_v2_request': True})
    choices = [UnifiedSelectionDecision('a', 'discard', 'retry_alternate', .9, 1,
                                         'redundant_retry', sequence_index=0),
               UnifiedSelectionDecision('b', 'select', 'independent', .9, 2,
                                         'independent_story_coverage', sequence_index=1)]
    session = Session(observation, error)
    reasoner = GoogleUnifiedSelectionReasoner('test', 'gemini-3.5-flash-lite',
                 HybridProviderSettings(enabled=True), DollarBudgetLedger(.03),
                 session=session, source_paths=(('src', '/fake/source.mp4'),))
    return reasoner, d, choices, session


def test_focused_continuation_requires_source_observation_and_audits_budget(monkeypatch):
    observed = dict(linked=True, restart_observed=False, uncertainty='low',
                    audio_evidence='Uninterrupted sentence, no pause or reset.',
                    visual_evidence='Consistent posture, gaze and gestures throughout.')
    reasoner, d, choices, session = fixtures(monkeypatch, observed)
    links, evidence = reasoner._verify_adjacent_continuations(d, choices)
    assert links == (('a', 'b'),)
    assert evidence[0]['actual_cost_usd'] > 0
    assert evidence[0]['source_window'][0] < 77.07 < evidence[0]['source_window'][1]
    assert reasoner.ledger.reserved_usd > 0
    assert len(session.calls) == 2
    observed['restart_observed'] = True
    assert reasoner._verify_adjacent_continuations(d, choices)[0] == ()
    assert reasoner._verify_adjacent_continuations(replace(d, selected=(replace(d.selected[0],
        source_asset_id='other'),)), choices)[0] == ()


def test_focused_continuation_provider_and_ffmpeg_fail_closed(monkeypatch):
    reasoner, d, choices, _ = fixtures(monkeypatch, {}, TimeoutError('provider timed out'))
    assert reasoner._verify_adjacent_continuations(d, choices)[0] == ()
    assert reasoner.ledger.reserved_usd > 0  # provider may have billed
    import cutsell_worker.whole_video_av as av
    def failed(*_args):
        raise subprocess.CalledProcessError(1, 'ffmpeg')
    monkeypatch.setattr(av, 'slice_prepared_av', failed)
    assert reasoner._verify_adjacent_continuations(d, choices)[0] == ()


def test_verified_source_continuation_repairs_model_story_order_only_when_confirmed(monkeypatch):
    from cutsell_worker.unified_selection_google import _reconcile_verified_continuation_order
    choices = [UnifiedSelectionDecision('a','discard','retry_alternate',.9,1,
                                         'redundant_retry',sequence_index=9),
               UnifiedSelectionDecision('b','select','independent',.95,2,
                                         'independent_story_coverage',sequence_index=5)]
    observed = dict(linked=True, restart_observed=False, uncertainty='low',
                    audio_evidence='Continuous speech and one utterance with no reset.',
                    visual_evidence='Same pose and gestures without any visible restart.')
    reasoner, draft_with_pair, _, _ = fixtures(monkeypatch, observed)
    links, audit = reasoner._verify_adjacent_continuations(draft_with_pair, choices)
    assert links == (('a','b'),), audit
    fixed = _reconcile_verified_continuation_order(choices, links)
    assert fixed[0].sequence_index < fixed[1].sequence_index
    assert _reconcile_verified_continuation_order(choices, ()) == choices
