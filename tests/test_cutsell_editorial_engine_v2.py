from dataclasses import replace

import pytest

from cutsell_worker.contracts import (
    DraftClip,
    DraftTimeline,
    EditStrategy,
    JobState,
    ProcessingResult,
    SCHEMA_VERSION,
)
from cutsell_worker.editorial_engine_v2 import run_editorial_engine_v2
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

    with pytest.raises(RuntimeError, match="Boundary changed frozen Selection semantic content"):
        run_editorial_engine_v2(
            source_result(),
            selection_reasoner=Plan(),
            recover_complete_boundaries=identity,
            execute_boundaries=resurrect,
        )


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
