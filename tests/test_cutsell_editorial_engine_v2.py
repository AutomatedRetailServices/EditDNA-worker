from dataclasses import replace

import pytest

from cutsell_worker.contracts import (
    DraftClip,
    DraftTimeline,
    EditStrategy,
    JobState,
    ProcessingResult,
    SCHEMA_VERSION,
    Word,
)
from cutsell_worker.editorial_engine_v2 import run_editorial_engine_v2, _coalesce_overlapping_selected_speech, _restore_safe_audience_continuity, _add_focused_visual_action_candidates, _run_speech_boundary_preserving_visual_actions


def test_focused_visual_action_becomes_wordless_candidate_before_selection():
    import json
    base = source_result()
    whole = base.draft.diagnostics['whole_video_context']
    digest = 'a' * 64
    evidence = {
        'source_sha256': digest,
        'focused_silent_visual_actions': [{
            'start': 98.4, 'end': 106.0,
            'observed_start': 96.0, 'observed_end': 106.0,
            'measured_silence_start': 98.4, 'measured_silence_end': 106.8,
            'confidence': .92, 'visual_observation': 'Mixing the product',
            'source_sha256': digest,
            'basis': 'focused_av_action_intersect_source_measured_silence',
        }],
    }
    source = {**whole['sources'][0], 'audiovisual_evidence': json.dumps(evidence)}
    updated = _add_focused_visual_action_candidates(base.draft, {**whole, 'sources': [source]})
    action = updated.discarded[-1]
    assert (action.start, action.end, action.text, action.words, action.audio_muted) == (
        98.4, 106.0, '', (), True)
    assert not action.selected
    # A forged digest or an unverified wide AV summary cannot add a candidate.
    evidence['focused_silent_visual_actions'][0]['source_sha256'] = 'other'
    source['audiovisual_evidence'] = json.dumps(evidence)
    rejected = _add_focused_visual_action_candidates(base.draft, {**whole, 'sources': [source]})
    assert len(rejected.discarded) == len(base.draft.discarded)


def test_boundary_stage_never_passes_silent_visual_action_to_asr_rebuilder():
    original = source_result()
    before, after = clip('before', 0, selected=True), clip('after', 10, selected=True)
    visual = replace(clip('visual', 3, selected=True), end=8, text='',
                     caption_text='', words=(), audio_muted=True)
    result = replace(original, draft=replace(original.draft,
                                            selected=(before, visual, after)))

    def simulated_asr_boundary(current):
        assert [clip.clip_id for clip in current.draft.selected] == ['before', 'after']
        processed = tuple(replace(clip, start=clip.start + .03)
                          for clip in current.draft.selected)
        return replace(current, draft=replace(current.draft, selected=processed))

    output = _run_speech_boundary_preserving_visual_actions(result, simulated_asr_boundary)
    assert [clip.clip_id for clip in output.draft.selected] == ['before', 'visual', 'after']
    assert output.draft.selected[1] == visual
    assert output.draft.selected[0].start == .03


def test_v2_measured_silence_blocks_broad_av_continuity_bridge():
    base = source_result()
    a = replace(clip("left", 100, selected=True), end=116.57, text="demonstration workout")
    b = replace(clip("right", 120.33, selected=True), end=140, text="workout demonstration")
    whole = {"sources": [{
        "source_asset_id": "source",
        "audiovisual_evidence": '{"regions":[{"start":90,"end":145,"role":"audience","confidence":0.95,"visual_observation":"Demonstrates the product"}]}',
        "events": [{"kind": "audio_silence_interval", "start": 117.128, "end": 119.584}],
    }]}
    result = replace(base, draft=replace(base.draft, selected=(a, b), alternates=(), discarded=()))
    output = _restore_safe_audience_continuity(result, whole)
    assert output.draft.selected[0].end == 116.57
    assert not output.draft.diagnostics.get("editorial_engine_v2_continuity_restoration")
    # Explicit product manipulation can justify a long silent action bridge.
    a = replace(a, start=95, end=100)
    b = replace(b, start=106, end=110)
    whole["sources"][0]["audiovisual_evidence"] = (
        '{"regions":[{"start":94,"end":111,"role":"audience","confidence":0.95,'
        '"visual_observation":"Demonstrates the product"}]}'
    )
    whole["sources"][0]["events"] = [{"kind": "audio_silence_interval", "start": 101, "end": 103}]
    # A generic presentation description is insufficient.
    output = _restore_safe_audience_continuity(
        replace(base, draft=replace(base.draft, selected=(a, b), alternates=(), discarded=())), whole)
    assert output.draft.selected[0].end == 100
    whole["sources"][0]["audiovisual_evidence"] = whole["sources"][0]["audiovisual_evidence"].replace(
        'Demonstrates the product', 'Mixes powder into a bottle')
    output = _restore_safe_audience_continuity(
        replace(base, draft=replace(base.draft, selected=(a, b), alternates=(), discarded=())), whole)
    assert output.draft.selected[0].end == 106


def test_v2_coalesces_identical_source_word_overlap_preserving_unique_edges():
    words = tuple(Word(word, i, i + 1) for i, word in enumerate(("unique", "prefix", "shared", "shared2", "tail")))
    a = replace(clip("a", 0, selected=True), end=4, words=words[:4], text="unique prefix shared shared2")
    b = replace(clip("b", 2, selected=True), end=5, words=words[2:], text="shared shared2 tail")
    source = source_result().draft
    output = _coalesce_overlapping_selected_speech(replace(source, selected=(a, b)))
    assert len(output.selected) == 1
    assert output.selected[0].text == "unique prefix shared shared2 tail"
    assert (output.selected[0].start, output.selected[0].end) == (0, 5)
    assert output.diagnostics["v2_overlapping_selection_word_union"][0]["action"] == "source_word_union"
    output = _coalesce_overlapping_selected_speech(replace(source, selected=(b, a)))
    assert output.selected[0].text == "unique prefix shared shared2 tail"
    bad = replace(b, words=(Word("different", 2, 3), *words[3:]))
    output = _coalesce_overlapping_selected_speech(replace(source, selected=(a, bad)))
    assert len(output.selected) == 2
    crossing = replace(b, words=(words[2], Word("crossing", 3.8, 4.2), words[4]),
                       text="shared crossing tail")
    output = _coalesce_overlapping_selected_speech(replace(source, selected=(a, crossing)))
    assert len(output.selected) == 2
from cutsell_worker.unified_selection_reasoner import (
    UnifiedSelectionDecision,
    UnifiedSelectionPlan,
)
from cutsell_worker.unified_selection_google import (
    build_unified_selection_payload,
    unified_selection_response_schema,
)


def clip(clip_id, start, *, selected=False):
    return DraftClip(
        clip_id=clip_id,
        source_asset_id="source",
        source_order=0,
        start=float(start),
        end=float(start + 1),
        text=f"spoken {clip_id}",
        caption_text=f"spoken {clip_id}",
        selected=selected,
    )


def source_result(*, audiovisual=True):
    a, b, c = clip("a", 0, selected=True), clip("b", 1), clip("c", 2)
    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION,
        project_id="project",
        strategy=EditStrategy.STORYTELLING,
        selected=(a,),
        alternates=(b,),
        discarded=(c,),
        diagnostics={
            "whole_video_context": {
                "status": {"status": "applied", "available": True},
                "audiovisual_input_status": (
                    "received_and_parsed" if audiovisual else "not_verified"
                ),
                "sources": [{
                    "source_asset_id": "source",
                    "audiovisual_evidence": '{"regions":[{"audio":"clean","visual":"steady"}]}',
                }],
            }
        },
    )
    return ProcessingResult(
        schema_version=SCHEMA_VERSION,
        project_id="project",
        state=JobState.DRAFT_READY,
        draft=draft,
        stage_status={},
    )


class Plan:
    def reason(self, draft):
        return UnifiedSelectionPlan(
            decisions=(
                UnifiedSelectionDecision("a", "discard", "retry_alternate", .95, 0, "redundant_retry", 1),
                UnifiedSelectionDecision("b", "select", "retry_winner", .98, 0, "best_complete_take", 0),
                UnifiedSelectionDecision("c", "discard", "failed", .99, 1, "failed_delivery", 2),
            ),
            provider="test",
            model="test",
        )


def identity(result):
    return result


def test_v2_resolves_once_folds_swap_freezes_and_verifies_boundary():
    result = run_editorial_engine_v2(
        source_result(),
        selection_reasoner=Plan(),
        recover_complete_boundaries=identity,
        execute_boundaries=identity,
    )

    assert [item.clip_id for item in result.draft.selected] == ["b"]
    assert result.draft.alternates == ()
    assert [item.clip_id for item in result.draft.discarded] == ["a", "c"]
    assert result.draft.diagnostics["selection_boundary_contract"]["status"] == "verified"
    assert result.draft.diagnostics["editorial_engine_v2"]["status"].endswith(
        "pending_post_render_review"
    )
    assert result.stage_status["brain_mode"] == "editorial_engine_v2_whole_video"


def test_v2_fails_closed_without_verified_audiovisual_input():
    with pytest.raises(RuntimeError, match="verified audiovisual"):
        run_editorial_engine_v2(
            source_result(audiovisual=False),
            selection_reasoner=Plan(),
            recover_complete_boundaries=identity,
            execute_boundaries=identity,
        )


def test_v2_fails_closed_when_post_freeze_stage_resurrects_discard():
    def resurrect(result):
        restored = result.draft.discarded[0]
        return replace(result, draft=replace(
            result.draft,
            selected=(*result.draft.selected, replace(restored, selected=True)),
            discarded=result.draft.discarded[1:],
        ))

    with pytest.raises(RuntimeError, match="Boundary changed frozen Selection semantic content") as caught:
        run_editorial_engine_v2(
            source_result(),
            selection_reasoner=Plan(),
            recover_complete_boundaries=identity,
            execute_boundaries=resurrect,
        )
    import json
    evidence = caught.value.boundary_failure_evidence
    assert [c["clip_id"] for c in evidence["before"]] == ["b"]
    assert [c["clip_id"] for c in evidence["after"]] == ["b", "a"]
    assert evidence["diagnostics"]["selection_boundary_contract"]["status"] == "frozen"
    assert "whole_video_context" not in evidence["diagnostics"]
    json.dumps(evidence)


def test_v2_fails_closed_on_reasoner_failure():
    class Broken:
        def reason(self, draft):
            raise TimeoutError("provider timeout")

    with pytest.raises(RuntimeError, match="whole-video plan failed"):
        run_editorial_engine_v2(
            source_result(),
            selection_reasoner=Broken(),
            recover_complete_boundaries=identity,
            execute_boundaries=identity,
        )


def test_v2_fails_closed_when_boundary_changes_story_order():
    class TwoSelected:
        def reason(self, draft):
            return UnifiedSelectionPlan(
                decisions=(
                    UnifiedSelectionDecision("a", "select", "independent", .99, 0, "independent_story_coverage", 1),
                    UnifiedSelectionDecision("b", "select", "independent", .99, 1, "independent_story_coverage", 0),
                    UnifiedSelectionDecision("c", "discard", "failed", .99, 2, "failed_delivery", 2),
                ),
                provider="test",
                model="test",
            )

    def reverse(result):
        return replace(result, draft=replace(result.draft, selected=tuple(reversed(result.draft.selected))))

    with pytest.raises(RuntimeError, match="changed story order"):
        run_editorial_engine_v2(
            source_result(),
            selection_reasoner=TwoSelected(),
            recover_complete_boundaries=identity,
            execute_boundaries=reverse,
        )


def test_v2_provider_payload_includes_av_evidence_and_requires_story_order():
    draft = source_result().draft
    draft = replace(draft, diagnostics={
        **draft.diagnostics,
        "editorial_engine_v2_request": {"require_audiovisual_evidence": True},
    })

    payload = build_unified_selection_payload(draft)
    item_schema = unified_selection_response_schema(3, v2=True)["properties"]["decisions"]["items"]

    assert payload["engine_version"] == "v2"
    assert "audio" in payload["source_context"]["sources"][0]["audiovisual_evidence"]
    assert "sequence_index" in item_schema["required"]


def test_v2_payload_requires_global_retry_comparison_and_unique_story_preservation():
    draft = source_result().draft
    draft = replace(draft, diagnostics={
        **draft.diagnostics,
        "editorial_engine_v2_request": {"require_audiovisual_evidence": True},
    })
    contract = " ".join(build_unified_selection_payload(draft)["editorial_contract"])

    assert "UNION of earlier fragments" in contract
    assert "no selected winner" in contract
    assert "unique hook" in contract
    assert "audience region contradicts failed_delivery" in contract


def test_v2_restores_short_verified_audience_gap_without_changing_membership():
    result = source_result()
    a = replace(result.draft.selected[0], start=0.0, end=1.0)
    b = replace(result.draft.alternates[0], start=2.0, end=3.0)
    result = replace(result, draft=replace(
        result.draft,
        selected=(a,),
        alternates=(b,),
        discarded=(),
        diagnostics={
            "whole_video_context": {
                "status": {"status": "applied", "available": True},
                "audiovisual_input_status": "received_and_parsed",
                "sources": [{
                    "source_asset_id": "source",
                    "audiovisual_evidence": '{"regions":[{"start":0,"end":4,"role":"audience","confidence":0.95}]}',
                }],
            }
        },
    ))

    class KeepBoth:
        def reason(self, draft):
            return UnifiedSelectionPlan(
                decisions=(
                    UnifiedSelectionDecision("a", "select", "composite_piece", .99, 0, "composite_best_take_piece", 0),
                    UnifiedSelectionDecision("b", "select", "continuation", .99, 0, "necessary_continuation", 1),
                ),
                provider="test",
                model="test",
            )

    output = run_editorial_engine_v2(
        result,
        selection_reasoner=KeepBoth(),
        recover_complete_boundaries=identity,
        execute_boundaries=identity,
    )

    assert [item.clip_id for item in output.draft.selected] == ["a", "b"]
    assert output.draft.selected[0].end == 2.0
    assert output.draft.diagnostics["editorial_engine_v2_continuity_restoration"][0]["restored_sec"] == 1.0


def test_v2_does_not_restore_gap_with_explicitly_discarded_material():
    result = source_result()
    a = replace(result.draft.selected[0], start=0.0, end=1.0)
    b = replace(result.draft.alternates[0], start=2.0, end=3.0)
    blocked = replace(result.draft.discarded[0], start=1.25, end=1.75)
    result = replace(result, draft=replace(
        result.draft,
        selected=(a,), alternates=(b,), discarded=(blocked,),
        diagnostics={
            "whole_video_context": {
                "status": {"status": "applied", "available": True},
                "audiovisual_input_status": "received_and_parsed",
                "sources": [{
                    "source_asset_id": "source",
                    "audiovisual_evidence": '{"regions":[{"start":0,"end":4,"role":"audience","confidence":0.95}]}',
                }],
            }
        },
    ))

    class KeepEnds:
        def reason(self, draft):
            return UnifiedSelectionPlan(
                decisions=(
                    UnifiedSelectionDecision("a", "select", "composite_piece", .99, 0, "composite_best_take_piece", 0),
                    UnifiedSelectionDecision("b", "select", "continuation", .99, 0, "necessary_continuation", 1),
                    UnifiedSelectionDecision("c", "discard", "failed", .99, 1, "failed_delivery", 2),
                ), provider="test", model="test",
            )

    output = run_editorial_engine_v2(
        result, selection_reasoner=KeepEnds(),
        recover_complete_boundaries=identity, execute_boundaries=identity,
    )
    assert output.draft.selected[0].end == 1.0


def test_v2_restores_longer_action_gap_inside_verified_product_demonstration():
    result = source_result()
    a = replace(result.draft.selected[0], start=90.0, end=99.5)
    b = replace(result.draft.alternates[0], start=106.5, end=118.0)
    a = replace(a, text="Put one scoop into the water bottle")
    b = replace(b, text="Mix the water bottle and continue")
    result = replace(result, draft=replace(
        result.draft, selected=(a,), alternates=(b,), discarded=(),
        diagnostics={"whole_video_context": {
            "status": {"status": "applied", "available": True},
            "audiovisual_input_status": "received_and_parsed",
            "sources": [{
                "source_asset_id": "source",
                "audiovisual_evidence": '{"regions":[{"start":80,"end":120,"role":"audience","confidence":0.95,"visual_observation":"Demonstrates scooping powder into water"}]}',
            }],
        }},
    ))

    class KeepDemo:
        def reason(self, draft):
            return UnifiedSelectionPlan(decisions=(
                UnifiedSelectionDecision("a", "select", "composite_piece", .99, 0, "composite_best_take_piece", 0),
                UnifiedSelectionDecision("b", "select", "continuation", .99, 0, "necessary_continuation", 1),
            ), provider="test", model="test")

    output = run_editorial_engine_v2(
        result, selection_reasoner=KeepDemo(),
        recover_complete_boundaries=identity, execute_boundaries=identity,
    )
    assert output.draft.selected[0].end == 106.5
    row = output.draft.diagnostics["editorial_engine_v2_continuity_restoration"][0]
    assert row["basis"].endswith("audience_demonstration")


def test_v2_does_not_restore_long_non_demo_audience_gap():
    result = source_result()
    a = replace(result.draft.selected[0], start=0.0, end=1.0)
    b = replace(result.draft.alternates[0], start=7.0, end=8.0)
    result = replace(result, draft=replace(
        result.draft, selected=(a,), alternates=(b,), discarded=(),
        diagnostics={"whole_video_context": {
            "status": {"status": "applied", "available": True},
            "audiovisual_input_status": "received_and_parsed",
            "sources": [{
                "source_asset_id": "source",
                "audiovisual_evidence": '{"regions":[{"start":0,"end":9,"role":"audience","confidence":0.95,"reason":"Direct speech to camera"}]}',
            }],
        }},
    ))

    class KeepSpeech:
        def reason(self, draft):
            return UnifiedSelectionPlan(decisions=(
                UnifiedSelectionDecision("a", "select", "independent", .99, 0, "independent_story_coverage", 0),
                UnifiedSelectionDecision("b", "select", "continuation", .99, 0, "necessary_continuation", 1),
            ), provider="test", model="test")

    output = run_editorial_engine_v2(
        result, selection_reasoner=KeepSpeech(),
        recover_complete_boundaries=identity, execute_boundaries=identity,
    )
    assert output.draft.selected[0].end == 1.0


def test_v2_reasserts_demo_bridge_after_boundary_rebuild_erases_it():
    result = source_result()
    a = replace(result.draft.selected[0], start=90.0, end=99.5)
    b = replace(result.draft.alternates[0], start=106.5, end=118.0)
    a = replace(a, text="Put one scoop into the water bottle")
    b = replace(b, text="Mix the water bottle and continue")
    result = replace(result, draft=replace(
        result.draft, selected=(a,), alternates=(b,), discarded=(),
        diagnostics={"whole_video_context": {
            "status": {"status": "applied", "available": True},
            "audiovisual_input_status": "received_and_parsed",
            "sources": [{
                "source_asset_id": "source",
                "audiovisual_evidence": '{"regions":[{"start":80,"end":120,"role":"audience","confidence":0.95,"reason":"Preparation instructions with product bottle"}]}',
            }],
        }},
    ))

    class KeepDemo:
        def reason(self, draft):
            return UnifiedSelectionPlan(decisions=(
                UnifiedSelectionDecision("a", "select", "composite_piece", .99, 0, "composite_best_take_piece", 0),
                UnifiedSelectionDecision("b", "select", "continuation", .99, 0, "necessary_continuation", 1),
            ), provider="test", model="test")

    def erase_bridge(output):
        first, second = output.draft.selected
        return replace(output, draft=replace(output.draft, selected=(replace(first, end=99.5), second)))

    output = run_editorial_engine_v2(
        result, selection_reasoner=KeepDemo(),
        recover_complete_boundaries=identity, execute_boundaries=erase_bridge,
    )
    assert output.draft.selected[0].end == 106.5
