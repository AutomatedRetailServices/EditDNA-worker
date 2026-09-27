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
                "sources": [{"source_asset_id": "source"}],
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
                UnifiedSelectionDecision("a", "discard", "retry_alternate", .95, 0, "redundant_retry"),
                UnifiedSelectionDecision("b", "select", "retry_winner", .98, 0, "best_complete_take"),
                UnifiedSelectionDecision("c", "discard", "failed", .99, 1, "failed_delivery"),
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
