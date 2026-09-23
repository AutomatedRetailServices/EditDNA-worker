"""RAW #122 audit root cause (classification A): local_performance.py computes
real per-take face/pose/motion evidence (MediaSignals) onto CandidateTake.signals,
but DraftClip -- the type that flows through Boundary and into the Unified
Selection payload (unified_selection_google._candidate_universe) -- had no
field to carry it at all, so pipeline.py's take->DraftClip conversion silently
dropped 100% of it. This test pins the fix at its exact origin: a CandidateTake
built with signals produces a DraftClip (in either bucket) carrying that same
MediaSignals object, not None.
"""
from cutsell_worker.contracts import CandidateTake, MediaSignals, ProcessingRequest, SourceAsset
from cutsell_worker.pipeline import build_flow_b_draft


def request() -> ProcessingRequest:
    source = SourceAsset(
        source_asset_id="src",
        project_id="project",
        user_id="user",
        original_name="raw.mp4",
        source_order=0,
        duration_sec=20.0,
        uri="s3://bucket/raw.mp4",
    )
    return ProcessingRequest(project_id="project", user_id="user", sources=(source,))


def test_candidate_take_signals_survive_into_the_draft_clip():
    signals = MediaSignals(
        source_asset_id="src", start=0.0, end=2.5,
        visual_fumble=0.73, eye_contact=0.2, distraction_risk=0.61,
    )
    take = CandidateTake(
        clip_id="a", source_asset_id="src", source_order=0,
        start=0.0, end=2.5, text="This cardigan is soft and comes in three colors",
        signals=signals,
    )

    result = build_flow_b_draft(request(), (take,), editorial_judge=None)

    clips = (*result.draft.selected, *result.draft.alternates, *result.draft.discarded)
    by_id = {clip.clip_id: clip for clip in clips}
    assert "a" in by_id
    assert by_id["a"].signals == signals


def test_candidate_take_without_signals_leaves_draft_clip_signals_none():
    take = CandidateTake(
        clip_id="a", source_asset_id="src", source_order=0,
        start=0.0, end=2.5, text="This cardigan is soft and comes in three colors",
    )

    result = build_flow_b_draft(request(), (take,), editorial_judge=None)

    clips = (*result.draft.selected, *result.draft.alternates, *result.draft.discarded)
    by_id = {clip.clip_id: clip for clip in clips}
    assert by_id["a"].signals is None
