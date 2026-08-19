from cutsell_worker.clean_cut import evaluate_take
from cutsell_worker.contracts import (
    CandidateTake,
    MediaSignals,
    ProcessingRequest,
    SemanticLabel,
    SemanticRole,
    SourceAsset,
)
from cutsell_worker.pipeline import build_flow_b_draft
from cutsell_worker.source_identity import stable_clip_id, stable_source_id


def _source(order=0, name="raw.mp4"):
    project_id = "project-1"
    return SourceAsset(
        source_asset_id=stable_source_id(project_id, order, name),
        project_id=project_id,
        user_id="user-1",
        original_name=name,
        source_order=order,
        duration_sec=30.0,
        uri=f"s3://bucket/{name}",
    )


def _take(source, start, end, text, **signal_overrides):
    signals = MediaSignals(
        source_asset_id=source.source_asset_id,
        start=start,
        end=end,
        **signal_overrides,
    )
    return CandidateTake(
        clip_id=stable_clip_id(source.source_asset_id, start, end, text),
        source_asset_id=source.source_asset_id,
        source_order=source.source_order,
        start=start,
        end=end,
        text=text,
        signals=signals,
    )


def test_source_and_clip_ids_are_deterministic():
    first = _source()
    second = _source()
    assert first.source_asset_id == second.source_asset_id
    assert stable_clip_id(first.source_asset_id, 1.0, 2.0, "hello") == stable_clip_id(
        first.source_asset_id, 1.0, 2.0, "hello"
    )


def test_clean_cut_keeps_uncertain_sales_speech_regardless_of_role():
    source = _source()
    take = _take(source, 1.0, 3.0, "I bought this because my skin was always dry")
    decision = evaluate_take(take)
    assert decision.keep is True
    assert decision.reason == "valid_or_uncertain_speech"


def test_clean_cut_removes_explicit_restart_direction():
    source = _source()
    take = _take(source, 1.0, 2.0, "one more time")
    decision = evaluate_take(take)
    assert decision.keep is False
    assert decision.reason == "explicit_restart_direction"


def test_flow_b_pipeline_selects_best_take_and_keeps_alternate():
    source = _source()
    weak = _take(source, 1.0, 3.0, "this serum changed my skin", audio_quality=0.3, eye_contact=0.2)
    strong = _take(source, 4.0, 6.0, "this serum changed my skin", audio_quality=0.95, eye_contact=0.95)
    request = ProcessingRequest(project_id="project-1", user_id="user-1", sources=(source,))
    labels = (
        SemanticLabel(weak.clip_id, SemanticRole.PROOF, 0.9),
        SemanticLabel(strong.clip_id, SemanticRole.PROOF, 0.9),
    )
    result = build_flow_b_draft(request, (weak, strong), labels)
    assert result.state.value == "draft_ready"
    assert [clip.clip_id for clip in result.draft.selected] == [strong.clip_id]
    assert [clip.clip_id for clip in result.draft.alternates] == [weak.clip_id]


def test_composer_preserves_source_order_across_multiple_sources():
    first = _source(0, "one.mp4")
    second = _source(1, "two.mp4")
    later_source_take = _take(second, 0.0, 2.0, "second source line")
    first_source_take = _take(first, 10.0, 12.0, "first source line")
    request = ProcessingRequest(project_id="project-1", user_id="user-1", sources=(first, second))
    result = build_flow_b_draft(request, (later_source_take, first_source_take))
    assert [clip.source_order for clip in result.draft.selected] == [0, 1]


def test_pipeline_keeps_valid_ungrouped_story_material_selected():
    source = _source()
    first = _take(source, 0.0, 5.0, "My skin was dry for months and nothing seemed to help.")
    second = _take(source, 6.0, 11.0, "Then I changed one part of my routine and the difference was obvious.")
    request = ProcessingRequest(project_id="project-1", user_id="user-1", sources=(source,))

    result = build_flow_b_draft(request, (first, second))

    assert [clip.clip_id for clip in result.draft.selected] == [first.clip_id, second.clip_id]
    assert result.draft.alternates == ()


def test_pipeline_collapses_retry_envelope_and_selects_full_delivery_only():
    source = _source()
    short = _take(
        source, 0.0, 2.4, "the popular croc",
        audio_quality=0.45, eye_contact=0.35, continuity=0.40,
    )
    repeat = _take(
        source, 3.0, 5.8, "crop popular crop popular crop popular",
        audio_quality=0.40, eye_contact=0.30, continuity=0.30,
    )
    partial = _take(
        source, 6.5, 10.5, "the popular crop black jeans okay now whole sentence okay",
        audio_quality=0.55, eye_contact=0.45, continuity=0.50,
    )
    full = _take(
        source, 11.5, 17.0,
        "the popular crop black denim jeans are back in stock anything with pockets is a win for me",
        audio_quality=0.95, eye_contact=0.95, continuity=0.95,
    )
    request = ProcessingRequest(project_id="project-1", user_id="user-1", sources=(source,))

    result = build_flow_b_draft(request, (short, repeat, partial, full))

    assert [clip.clip_id for clip in result.draft.selected] == [full.clip_id]
    # Failed/restart material may be surfaced as a swap alternate or moved to the
    # discarded-takes lane by later deterministic cleanup. The product invariant is
    # that none of those attempts re-enters the final timeline and all remain accounted
    # for in the editable draft.
    non_selected_ids = {
        clip.clip_id for clip in (*result.draft.alternates, *result.draft.discarded)
    }
    assert non_selected_ids == {short.clip_id, repeat.clip_id, partial.clip_id}
