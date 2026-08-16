from cutsell_worker.contracts import CandidateTake
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.word_search_attempts import apply_word_search_attempt_cleanup
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


def event(kind, start, end, confidence=1.0):
    return TemporalEvent("src", start, end, kind, confidence, kind)


def context(events):
    return WholeVideoContext(
        sources=(SourceVideoContext(
            source_asset_id="src",
            summary="",
            dominant_style="talking_head",
            creator_intent="recording",
            events=tuple(events),
        ),),
        status=ProviderStatus("test", True, True, "applied"),
    )


def dense_context():
    return context((
        event("hand_motion_reset_candidate", 1.0, 1.1),
        event("body_reset_candidate", 2.0, 2.1),
        event("hand_motion_reset_candidate", 3.0, 3.1),
        event("hand_motion_reset_candidate", 4.0, 4.1),
        event("body_reset_candidate", 5.0, 5.1),
        event("hand_motion_reset_candidate", 6.0, 6.1),
        event("facial_expression_shift_candidate", 7.0, 7.1, 0.82),
    ))


def test_longer_word_search_attempt_cluster_is_removed_under_dense_reset():
    takes = (
        take("a", "As a content creator if you do not have this election", 0.0, 3.1),
        take("b", "election electric suction phone holder", 3.8, 7.7),
        take("c", "election suction phone holder election oh my god", 8.4, 12.5),
        take("next", "hands down cutest Christmas card ever", 13.1, 16.0),
    )
    ctx = context((
        event("hand_motion_reset_candidate", 1.0, 1.1),
        event("body_reset_candidate", 2.0, 2.1),
        event("hand_motion_reset_candidate", 3.0, 3.1),
        event("hand_motion_reset_candidate", 4.0, 4.1),
        event("body_reset_candidate", 5.0, 5.1),
        event("hand_motion_reset_candidate", 6.0, 6.1),
        event("hand_motion_reset_candidate", 8.8, 8.9),
        event("facial_expression_shift_candidate", 9.0, 9.1, 0.82),
    ))

    kept, removed, diagnostics = apply_word_search_attempt_cleanup(takes, ctx)

    assert kept == (takes[-1],)
    assert removed == takes[:3]
    assert {item["reason"] for item in diagnostics} == {
        "multi_segment_word_search_cluster_with_dense_reset"
    }


def test_discarded_micro_fragments_still_prove_surviving_word_search_attempt():
    survivor = take("a", "As a content creator if you do not have this election", 0.0, 3.1)
    fragments = (
        take("b", "election electric suction phone holder", 3.8, 7.7),
        take("c", "election suction phone holder election oh my god", 8.4, 12.5),
    )
    evidence = (survivor, *fragments)
    ctx = context((
        event("hand_motion_reset_candidate", 1.0, 1.1),
        event("body_reset_candidate", 2.0, 2.1),
        event("hand_motion_reset_candidate", 3.0, 3.1),
        event("hand_motion_reset_candidate", 4.0, 4.1),
        event("body_reset_candidate", 5.0, 5.1),
        event("hand_motion_reset_candidate", 6.0, 6.1),
        event("hand_motion_reset_candidate", 8.8, 8.9),
        event("facial_expression_shift_candidate", 9.0, 9.1, 0.82),
    ))

    kept, removed, diagnostics = apply_word_search_attempt_cleanup(
        (survivor,),
        ctx,
        evidence_takes=evidence,
    )

    assert kept == ()
    assert removed == (survivor,)
    assert diagnostics[0]["reason"] == "multi_segment_word_search_cluster_with_dense_reset"


def test_same_lexical_pattern_without_dense_reset_fails_open():
    takes = (
        take("a", "As a content creator if you do not have this election", 0.0, 3.1),
        take("b", "election electric suction phone holder", 3.8, 7.7),
        take("c", "election suction phone holder election oh my god", 8.4, 12.5),
    )
    kept, removed, diagnostics = apply_word_search_attempt_cleanup(takes, context(()))
    assert kept == takes
    assert removed == ()
    assert diagnostics == ()


def test_intentional_repeated_brand_phrase_without_near_stem_variant_survives():
    takes = (
        take("a", "the sunshine bottle is finally back sunshine bottle", 0.0, 3.0),
        take("b", "sunshine bottle has a leakproof lid sunshine bottle", 3.5, 7.0),
    )
    kept, removed, diagnostics = apply_word_search_attempt_cleanup(takes, dense_context())
    assert kept == takes
    assert removed == ()
    assert diagnostics == ()
