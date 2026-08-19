from cutsell_worker.attempt_reconstruction import reconstruct_delivery_attempts
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def take(clip_id, text, start, end, *, source="src", complete=True):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id=source,
        source_order=0,
        start=start,
        end=end,
        text=text,
        complete_idea=complete,
    )


def event(kind, start, end, confidence):
    return TemporalEvent("src", start, end, kind, confidence, kind)


def context(events=()):
    return WholeVideoContext(
        sources=(SourceVideoContext(
            source_asset_id="src",
            summary="creator recording",
            dominant_style="talking_head",
            creator_intent="natural delivery",
            events=tuple(events),
        ),),
        status=ProviderStatus("test", True, True, "applied"),
    )


def test_contiguous_complete_sentences_become_one_delivery_attempt():
    first = take("a", "I started using this in January.", 0.0, 4.0)
    second = take("b", "Then I noticed my skin felt calmer.", 4.08, 8.2)
    third = take("c", "That is why I kept using it.", 8.28, 12.0)

    attempts, diagnostics = reconstruct_delivery_attempts((first, second, third), context())

    assert len(attempts) == 1
    assert attempts[0].start == 0.0
    assert attempts[0].end == 12.0
    assert "January" in attempts[0].text
    assert "kept using it" in attempts[0].text
    assert diagnostics["merged_fragment_count"] == 2


def test_talking_head_reset_creates_delivery_attempt_boundary():
    first = take("a", "Al terminar mi contrato hablé con mi ginecóloga.", 0.0, 4.0)
    retry = take("b", "Al terminar mi contrato hablé con mi ginecóloga y pedí los análisis.", 4.25, 9.0)
    ctx = context((
        event("body_reset_candidate", 4.03, 4.15, 0.96),
        event("camera_disengagement_candidate", 4.04, 4.16, 0.91),
    ))

    attempts, diagnostics = reconstruct_delivery_attempts((first, retry), ctx)

    assert len(attempts) == 2
    assert diagnostics["boundaries"][0]["reason"] in {
        "lexical_restart",
        "multi_family_delivery_reset",
    }


def test_single_body_reset_without_pause_or_break_does_not_destroy_continuous_story():
    first = take("a", "This happened after my appointment.", 0.0, 3.0)
    second = take("b", "The doctor explained the results to me.", 3.08, 6.5)
    ctx = context((event("body_reset_candidate", 3.00, 3.10, 0.93),))

    attempts, _ = reconstruct_delivery_attempts((first, second), ctx)

    assert len(attempts) == 1


def test_clear_lexical_restart_stays_separate_even_without_visual_context():
    first = take("a", "The popular crop black jeans are finally back.", 0.0, 3.0)
    retry = take("b", "The popular crop black jeans are finally back in stock today.", 3.15, 6.5)

    attempts, diagnostics = reconstruct_delivery_attempts((first, retry), None)

    assert len(attempts) == 2
    assert diagnostics["boundaries"][0]["reason"] == "lexical_restart"


def test_real_speech_pause_separates_attempts_without_deleting_either():
    first = take("a", "This is the first part of the story.", 0.0, 3.0)
    second = take("b", "This is a different thought later.", 5.0, 8.0)

    attempts, diagnostics = reconstruct_delivery_attempts((first, second), None)

    assert len(attempts) == 2
    assert diagnostics["boundaries"][0]["reason"] == "real_speech_pause"


def test_source_change_never_merges_attempts():
    first = take("a", "Creator one sentence.", 0.0, 2.0, source="src-a")
    second = take("b", "Creator two sentence.", 2.05, 4.0, source="src-b")

    attempts, diagnostics = reconstruct_delivery_attempts((first, second), None)

    assert len(attempts) == 2
    assert diagnostics["boundaries"][0]["reason"] == "source_change"


def test_tail_completeness_controls_reconstructed_attempt():
    first = take("a", "This sentence is complete.", 0.0, 2.0, complete=True)
    dangling = take("b", "because the next thing that", 2.05, 4.0, complete=False)

    attempts, _ = reconstruct_delivery_attempts((first, dangling), None)

    assert len(attempts) == 1
    assert attempts[0].complete_idea is False
