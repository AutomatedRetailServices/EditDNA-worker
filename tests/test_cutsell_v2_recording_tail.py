from dataclasses import replace
import json
import pytest

from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, Word
from cutsell_worker.unified_selection_reasoner import (
    UnifiedSelectionDecision, UnifiedSelectionPlan, apply_unified_selection_reasoner,
    _preserve_continuous_demonstration, _high_confidence_audience_spans,
)
from cutsell_worker.v2_recording_tail import trim_recording_tail


def sample():
    words = (Word("Find", 0, .2), Word("it", .2, .4), Word("in", .4, .6),
             Word("cart", .6, 2.3), Word("recording", 2.3, 2.7), Word("finished", 2.7, 3))
    text = " ".join(w.text for w in words)
    clip = DraftClip("c", "src", 0, 0, 3, text, text, words=words)
    decision = UnifiedSelectionDecision("c", "select", "independent", .99, 0,
                                        "independent_story_coverage", 0, 2)
    event = {"kind": "audio_silence_interval", "evidence_source": "audio_silence",
             "confidence": 1.0, "start": .8, "end": 2.2}
    diagnostics = {"editorial_engine_v2_request": {"require_audiovisual_evidence": True},
                   "attempt_reconstruction": {"positioned_performance_evidence": [
                       {"source_asset_id": "src", "positioned_events": [event]}]}}
    return clip, decision, diagnostics


@pytest.mark.parametrize("change", [
    {"confidence": .96}, {"action": "discard"}, {"trailing_recording_word_count": 0},
    {"trailing_recording_word_count": 8}, {"trailing_recording_word_count": True},
])
def test_uncertain_or_invalid_proposal_preserves(change):
    clip, decision, diagnostics = sample()
    assert trim_recording_tail(clip, replace(decision, **change), diagnostics)[0] == clip


@pytest.mark.parametrize("change", [
    {"confidence": .9}, {"evidence_source": "model"}, {"kind": "gesture"},
    {"start": 2.2}, {"end": 9}, {"start": float("nan")},
])
def test_independent_physical_pause_required(change):
    clip, decision, diagnostics = sample()
    diagnostics["attempt_reconstruction"]["positioned_performance_evidence"][0]["positioned_events"][0].update(change)
    assert trim_recording_tail(clip, decision, diagnostics)[0] == clip


def test_other_source_pause_does_not_authorize_trim():
    clip, decision, diagnostics = sample()
    diagnostics["attempt_reconstruction"]["positioned_performance_evidence"][0]["source_asset_id"] = "other"
    assert trim_recording_tail(clip, decision, diagnostics)[0] == clip


@pytest.mark.parametrize("tail", ["not free", "no gratis", "only 18", "never mix"])
def test_protected_tail_fact_preserved_even_with_proposal(tail):
    clip, decision, diagnostics = sample()
    words = clip.words[:-2] + tuple(replace(w, text=t) for w, t in zip(clip.words[-2:], tail.split()))
    clip = replace(clip, words=words, text=" ".join(w.text for w in words))
    assert trim_recording_tail(clip, decision, diagnostics)[0] == clip


def test_word_text_mismatch_preserved():
    clip, decision, diagnostics = sample()
    clip = replace(clip, text=clip.text + " important")
    assert trim_recording_tail(clip, decision, diagnostics)[0] == clip


@pytest.mark.parametrize("bad_word", [Word("it", .1, .4), Word("it", .4, .3), Word("it", float("nan"), .4)])
def test_invalid_word_alignment_preserved(bad_word):
    clip, decision, diagnostics = sample()
    clip = replace(clip, words=(clip.words[0], bad_word, *clip.words[2:]))
    assert trim_recording_tail(clip, decision, diagnostics)[0] == clip


def test_provider_plan_tail_proposal_survives_validation_and_is_applied():
    clip, decision, diagnostics = sample()
    draft = DraftTimeline("cutsell.v1", "p", EditStrategy.STORYTELLING, (clip,), (), (), diagnostics)
    class Reasoner:
        def reason(self, _):
            return UnifiedSelectionPlan((decision,), "fake", "synthetic")
    out = apply_unified_selection_reasoner(draft, Reasoner())
    assert out.selected[0].text == "Find it in cart"
    assert out.diagnostics["v2_recording_tail"][0]["action"] == "trim"
    legacy = apply_unified_selection_reasoner(replace(draft, diagnostics={}), Reasoner())
    assert legacy.selected[0].text == clip.text


def test_real_repeated_instruction_is_not_restored_as_demo():
    left = DraftClip("a", "src", 0, 0, 2, "Mix the powder in water", "")
    right = DraftClip("b", "src", 0, 5, 8, "Mix the powder in water until dissolved", "")
    decisions = {
        "a": UnifiedSelectionDecision("a", "discard", "retry_alternate", .89, 0, "redundant_retry", 0),
        "b": UnifiedSelectionDecision("b", "select", "retry_winner", .95, 0, "best_complete_take", 1),
    }
    av = json.dumps({"regions": [{"start": 0, "end": 9, "role": "audience", "confidence": .95,
                                  "visual_observation": "Creator demonstrating mixing powder in water"}]})
    draft = DraftTimeline("cutsell.v1", "p", EditStrategy.STORYTELLING, (left, right), (), (),
                          {"whole_video_context": {"sources": [{"source_asset_id": "src", "audiovisual_evidence": av}]}})
    actions, overrides = ["discard", "select"], [None, None]
    _preserve_continuous_demonstration(draft, (left, right), decisions, actions, overrides)
    assert actions == ["discard", "select"]


def test_av_overlap_is_union_and_invalid_ranges_are_ignored():
    regions = [{"start": a, "end": b, "role": "audience", "confidence": .95}
               for a, b in [(0, 2), (1, 3), (5, 4), (float("nan"), 7)]]
    draft = DraftTimeline("cutsell.v1", "p", EditStrategy.STORYTELLING, (), (), (), {
        "whole_video_context": {"sources": [{"source_asset_id": "src", "audiovisual_evidence": json.dumps({"regions": regions})}]}})
    assert _high_confidence_audience_spans(draft) == {"src": ((0., 3.),)}
