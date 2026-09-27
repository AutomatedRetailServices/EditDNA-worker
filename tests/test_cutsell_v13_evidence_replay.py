"""Recorded v13 component replay, not a new inference or full engine replay."""
import json
from pathlib import Path
from types import SimpleNamespace

from cutsell_worker.contracts import (
    DraftClip, DraftTimeline, EditStrategy, JobState, ProcessingResult, Word,
)
from cutsell_worker.final_boundary_authority import enforce_complete_idea_boundaries
from cutsell_worker.unified_selection_reasoner import (
    UnifiedSelectionDecision, UnifiedSelectionPlan, _preserve_unique_content_when_av_contradicts_failed,
)
from cutsell_worker.editorial_engine_v2 import run_editorial_engine_v2


def evidence():
    return json.loads((Path(__file__).parent / "fixtures/yaskira09_v13_selection_evidence.json").read_text())


def test_recorded_failed_opening_preserved_despite_shared_story_vocabulary():
    data = evidence()
    source = data["av"][0]["source_asset_id"]
    rows = {row["clip_id"]: row for row in data["decisions"]}
    clips = tuple(DraftClip(r["clip_id"], source, 0, r["start"], r["end"], r["text"], r["text"])
                  for r in data["clips"] if r["clip_id"] in rows)
    decisions = {key: UnifiedSelectionDecision(
        key, r["model_action"], r["relation"], r["confidence"], r["family_index"],
        r["reason_code"], r["sequence_index"],
    ) for key, r in rows.items()}
    actions = [rows[c.clip_id]["effective_action"] for c in clips]
    overrides = [None] * len(clips)
    draft = DraftTimeline("cutsell.v1", "p", EditStrategy.STORYTELLING, (), (), (),
                          {"whole_video_context": {"sources": data["av"]}})
    _preserve_unique_content_when_av_contradicts_failed(draft, clips, decisions, actions, overrides)
    opening = next(i for i, c in enumerate(clips) if c.start == 5.404)
    assert actions[opening] == "select"
    assert overrides[opening] == "av_audience_unique_content_overrides_failed_label"
    for i, c in enumerate(clips):
        if 28 <= c.start <= 44:
            assert actions[i] == "discard"


def test_v13_without_explicit_proposal_cannot_delete_tail_from_pause_alone():
    data = evidence()
    source = data["av"][0]["source_asset_id"]
    words = tuple(Word(*w) for w in data["words"])
    tail_words = tuple(w for w in words if w.end > 106.764)
    text = " ".join(w.text for w in tail_words)
    clip = DraftClip("tail", source, 0, 106.764, 121.521, text, text, words=tail_words)
    draft = DraftTimeline("cutsell.v1", "p", EditStrategy.STORYTELLING, (clip,), (), (), {
        "editorial_engine_v2_request": {"require_audiovisual_evidence": True},
        "attempt_reconstruction": {"positioned_performance_evidence": data["events"]},
    })
    result = ProcessingResult("cutsell.v1", "p", JobState.DRAFT_READY, draft, {})
    class ASR:
        def transcribe(self, *args, **kwargs):
            return (SimpleNamespace(words=words),)
    out = enforce_complete_idea_boundaries(result, {source: "unused"}, asr_provider=ASR())
    assert out.draft.selected[-1].text.endswith("ya se acabó ese")
    assert out.draft.diagnostics["final_boundary_post_cta_aside_trim_count"] == 0


def test_recorded_demo_selection_and_bridge_through_v2_freeze():
    data = evidence()
    source = data["av"][0]["source_asset_id"]
    words = tuple(Word(*w) for w in data["words"])
    left_row = next(r for r in data["clips"] if r["start"] == 96.61)
    left = DraftClip("demo", source, 0, 96.61, 98.435, left_row["text"], left_row["text"],
                     words=tuple(w for w in words if 96.61 <= w.start < 98.435))
    right_words = tuple(w for w in words if w.end > 106.764)
    right_text = " ".join(w.text for w in right_words)
    right = DraftClip("explain", source, 0, 106.764, 121.521, right_text, right_text, words=right_words)
    diagnostics = {
        "whole_video_context": {"status": {"status": "ok"},
                                "audiovisual_input_status": "received_and_parsed", "sources": data["av"]},
        "attempt_reconstruction": {"positioned_performance_evidence": data["events"]},
    }
    draft = DraftTimeline("cutsell.v1", "p", EditStrategy.STORYTELLING, (left, right), (), (), diagnostics)
    result = ProcessingResult("cutsell.v1", "p", JobState.DRAFT_READY, draft, {})
    class Reasoner:
        def reason(self, _draft):
            return UnifiedSelectionPlan((
                UnifiedSelectionDecision("demo", "discard", "retry_alternate", .8, 3, "redundant_retry", 0),
                UnifiedSelectionDecision("explain", "select", "retry_winner", .95, 3, "best_complete_take", 1),
            ), "replay", "recorded-decisions")
    class ASR:
        def transcribe(self, *args, **kwargs):
            return (SimpleNamespace(words=words),)
    out = run_editorial_engine_v2(result, selection_reasoner=Reasoner(),
        recover_complete_boundaries=lambda r: enforce_complete_idea_boundaries(r, {source: "unused"}, asr_provider=ASR()),
        execute_boundaries=lambda r: r)
    assert [c.clip_id for c in out.draft.selected] == ["demo", "explain"]
    assert out.draft.selected[0].end == out.draft.selected[1].start
    assert out.draft.diagnostics["editorial_engine_v2"]["selection_contract_status"] == "verified"


def test_simulated_explicit_tail_proposal_on_recorded_words_survives_boundary_refresh():
    # The historical run has NO such proposal; this validates execution only.
    from cutsell_worker.v2_recording_tail import trim_recording_tail
    data = evidence()
    source = data["av"][0]["source_asset_id"]
    words = tuple(Word(*w) for w in data["words"])
    tail_words = tuple(w for w in words if w.end > 106.764)
    text = " ".join(w.text for w in tail_words)
    clip = DraftClip("tail", source, 0, 106.66, 121.78, text, text, words=tail_words)
    diagnostics = {"editorial_engine_v2_request": {"require_audiovisual_evidence": True},
                   "attempt_reconstruction": {"positioned_performance_evidence": data["events"]}}
    decision = UnifiedSelectionDecision("tail", "select", "independent", .99, 0,
                                        "independent_story_coverage", 0, 4)
    trimmed, row = trim_recording_tail(clip, decision, diagnostics)
    assert trimmed.text.endswith("carrito")
    assert row["action"] == "trim"
    diagnostics["v2_recording_tail"] = [row]
    draft = DraftTimeline("cutsell.v1", "p", EditStrategy.STORYTELLING, (trimmed,), (), (), diagnostics)
    result = ProcessingResult("cutsell.v1", "p", JobState.DRAFT_READY, draft, {})
    class ASR:
        def transcribe(self, *args, **kwargs):
            return (SimpleNamespace(words=words),)
    out = enforce_complete_idea_boundaries(result, {source: "unused"}, asr_provider=ASR())
    assert out.draft.selected[0].text.endswith("carrito")
    assert out.draft.selected[0].end == trimmed.end


def test_recorded_opening_accent_only_evidence_does_not_authorize_deletion():
    data = evidence()
    source = data["av"][0]["source_asset_id"]
    words = tuple(Word(*w) for w in data["words"])
    clips = []
    for name, start, end in (("open", 4.98, 16.98), ("next", 19, 28.14)):
        aligned = tuple(w for w in words if start <= w.start and w.end <= end)
        text = " ".join(w.text for w in aligned)
        clips.append(DraftClip(name, source, 0, start, end, text, text, words=aligned))
    draft = DraftTimeline("cutsell.v1", "p", EditStrategy.STORYTELLING, tuple(clips), (), (), {
        "editorial_engine_v2_request": {"require_audiovisual_evidence": True},
    })
    result = ProcessingResult("cutsell.v1", "p", JobState.DRAFT_READY, draft, {})
    class ASR:
        def transcribe(self, *args, **kwargs):
            return (SimpleNamespace(words=words),)
    out = enforce_complete_idea_boundaries(result, {source: "unused"}, asr_provider=ASR())
    assert out.draft.selected[0].text.endswith("tus mus")
    assert out.draft.selected[0].end == 16.98
    assert out.draft.selected[1].text.startswith("y aparte tus músculos")
