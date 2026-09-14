from cutsell_worker.contracts import CandidateTake
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.session_boundaries import (
    infer_session_boundaries,
    partition_takes_by_sessions,
    safe_group_takes_by_sessions,
)
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def take(clip_id, text, start, end):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
    )


def event(kind, start, end, confidence):
    return TemporalEvent("src", start, end, kind, confidence, kind)


def context(events):
    return WholeVideoContext(
        sources=(SourceVideoContext(
            source_asset_id="src",
            summary="compilation",
            dominant_style="mixed",
            creator_intent="recording",
            events=tuple(events),
        ),),
        status=ProviderStatus("test", True, True, "applied"),
    )


def hard_cut_context():
    return context((
        event("camera_disengagement_candidate", 2.00, 2.08, 0.92),
        event("facial_expression_shift_candidate", 2.02, 2.10, 0.90),
        event("body_reset_candidate", 2.01, 2.09, 0.96),
    ))


def test_dense_multi_family_discontinuity_between_takes_creates_session_boundary():
    boundaries = infer_session_boundaries(hard_cut_context(), "src")
    assert len(boundaries) == 1
    assert boundaries[0].confidence >= 0.90
    assert set(boundaries[0].evidence_kinds) == {
        "body_reset_candidate",
        "camera_disengagement_candidate",
        "facial_expression_shift_candidate",
    }


def test_identical_speech_on_two_creators_does_not_become_one_retry_group_across_boundary():
    first_creator = take("a", "this product is actually amazing", 0.0, 2.0)
    second_creator = take("b", "this product is actually amazing", 2.10, 4.20)

    partitions = partition_takes_by_sessions((first_creator, second_creator), hard_cut_context())
    grouped = safe_group_takes_by_sessions(
        None,
        (first_creator, second_creator),
        hard_cut_context(),
    )

    assert partitions == ((first_creator,), (second_creator,))
    assert grouped.groups == (("a",), ("b",))
    assert "session_boundary_scoped:2" in grouped.reason


def test_single_body_or_hand_reset_does_not_split_normal_retry_session():
    first = take("a", "the popular crop black jeans are back", 0.0, 2.0)
    retry = take("b", "the popular crop black jeans are back", 2.10, 4.20)
    ctx = context((event("body_reset_candidate", 2.01, 2.09, 0.99),))

    assert infer_session_boundaries(ctx, "src") == ()
    assert partition_takes_by_sessions((first, retry), ctx) == ((first, retry),)

    grouped = safe_group_takes_by_sessions(None, (first, retry), ctx)
    assert grouped.groups == (("a", "b"),)


def test_visual_cluster_inside_a_take_never_splits_that_take():
    long_take = take("a", "continuous intentional creator delivery", 0.0, 4.0)
    later = take("b", "different line after a pause", 4.5, 6.5)
    ctx = context((
        event("camera_disengagement_candidate", 2.00, 2.08, 0.94),
        event("facial_expression_shift_candidate", 2.01, 2.09, 0.91),
        event("hand_motion_reset_candidate", 2.00, 2.10, 0.96),
    ))

    partitions = partition_takes_by_sessions((long_take, later), ctx)
    assert partitions == ((long_take, later),)
