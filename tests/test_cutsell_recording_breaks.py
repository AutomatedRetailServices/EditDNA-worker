from cutsell_worker.contracts import CandidateTake
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.recording_breaks import apply_recording_break_cleanup
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def take(clip_id, text, start=0.0, end=3.0):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
    )


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


def event(kind, start, end, confidence=1.0):
    return TemporalEvent("src", start, end, kind, confidence, kind)


def removed_reason(text):
    kept, removed, diagnostics = apply_recording_break_cleanup((take("x", text),))
    return kept, removed, diagnostics


def test_i_cant_talk_is_explicit_recording_failure():
    kept, removed, diagnostics = removed_reason("one fuck these wipes right I can't talk these wipes")
    assert kept == ()
    assert len(removed) == 1
    assert diagnostics[0]["reason"] == "explicit_recording_failure"


def test_why_cant_i_talk_is_explicit_recording_failure():
    kept, removed, diagnostics = removed_reason("why can't I talk?")
    assert kept == ()
    assert diagnostics[0]["reason"] == "explicit_recording_failure"


def test_i_dont_know_how_to_end_is_explicit_recording_failure():
    kept, removed, diagnostics = removed_reason("oh I don't know how to end TikTok shop")
    assert kept == ()
    assert diagnostics[0]["reason"] == "explicit_recording_failure"


def test_lets_do_that_again_is_explicit_restart_failure():
    kept, removed, diagnostics = removed_reason("okay let's do that again I'm always on the hunt")
    assert kept == ()
    assert diagnostics[0]["reason"] == "explicit_recording_failure"


def test_frustrated_hand_self_direction_is_removed():
    kept, removed, diagnostics = removed_reason("what are you doing with your hands fuck")
    assert kept == ()
    assert diagnostics[0]["reason"] == "frustrated_self_direction"


def test_frustrated_repeated_multiword_restart_is_removed():
    kept, removed, diagnostics = removed_reason(
        "fuck I know y'all seen the viral I know y'all seen the viral I know y'all seen the viral"
    )
    assert kept == ()
    assert diagnostics[0]["reason"] == "frustrated_internal_restart_repetition"


def test_partial_repeated_restart_with_frustration_is_removed():
    kept, removed, diagnostics = removed_reason(
        "there's literally 80 little fuck there's literally 80 little fuck there's literally"
    )
    assert kept == ()
    assert diagnostics[0]["reason"] == "frustrated_internal_restart_repetition"


def test_multimodal_reaction_cluster_is_removed_as_recording_break():
    takes = (
        take("a", "Fuck is happening", 77.57, 79.87),
        take("b", "Okay, what the frig okay?", 81.31, 83.83),
        take("c", "What just happened?", 85.53, 86.39),
        take("d", "Okay, anyways", 87.51, 88.73),
    )
    ctx = context((
        event("hand_motion_reset_candidate", 77.70, 77.80),
        event("hand_motion_reset_candidate", 81.20, 81.30),
        event("body_reset_candidate", 81.60, 81.70),
        event("hand_motion_reset_candidate", 82.30, 82.40),
        event("facial_expression_shift_candidate", 84.60, 84.70, 0.78),
        event("facial_expression_shift_candidate", 85.80, 85.90, 0.81),
    ))

    kept, removed, diagnostics = apply_recording_break_cleanup(takes, ctx)

    assert kept == ()
    assert removed == takes
    assert {item["reason"] for item in diagnostics} == {
        "multimodal_recording_break_reaction_cluster"
    }


def self_review_context():
    return context((
        event("hand_motion_reset_candidate", 307.55, 307.65),
        event("hand_motion_reset_candidate", 308.00, 308.10),
        event("body_reset_candidate", 308.30, 308.40),
        event("hand_motion_reset_candidate", 308.90, 309.00),
        event("hand_motion_reset_candidate", 309.20, 309.30),
        event("body_reset_candidate", 309.80, 309.90),
        event("facial_expression_shift_candidate", 308.75, 308.85, 0.80),
    ))


def test_speech_self_review_plus_confused_echo_is_removed_with_dense_reset():
    review = take("review", "What did I just say?", 307.49, 308.19)
    echo = take("echo", "What?", 309.65, 310.15)

    kept, removed, diagnostics = apply_recording_break_cleanup((review, echo), self_review_context())

    assert kept == ()
    assert removed == (review, echo)
    assert {item["reason"] for item in diagnostics} == {
        "speech_self_review_confusion_pair_with_physical_reset"
    }


def test_discarded_confused_echo_still_proves_surviving_self_review_failure():
    review = take("review", "What did I just say?", 307.49, 308.19)
    echo = take("echo", "What?", 309.65, 310.15)

    kept, removed, diagnostics = apply_recording_break_cleanup(
        (review,),
        self_review_context(),
        evidence_takes=(review, echo),
    )

    assert kept == ()
    assert removed == (review,)
    assert diagnostics[0]["reason"] == "speech_self_review_confusion_pair_with_physical_reset"


def test_isolated_what_did_i_just_say_survives_without_confused_echo():
    review = take("review", "What did I just say?", 10.0, 11.0)
    next_take = take("next", "I said this serum is my favorite", 12.0, 14.5)
    ctx = context((
        event("hand_motion_reset_candidate", 10.1, 10.2),
        event("hand_motion_reset_candidate", 10.3, 10.4),
        event("body_reset_candidate", 10.5, 10.6),
        event("hand_motion_reset_candidate", 10.7, 10.8),
        event("facial_expression_shift_candidate", 10.8, 10.9, 0.90),
    ))

    kept, removed, diagnostics = apply_recording_break_cleanup((review, next_take), ctx)

    assert kept == (review, next_take)
    assert removed == ()
    assert diagnostics == ()


def test_isolated_what_just_happened_survives_even_with_visual_motion():
    reaction = take("reaction", "What just happened?", 10.0, 11.2)
    ctx = context((
        event("hand_motion_reset_candidate", 10.1, 10.2),
        event("hand_motion_reset_candidate", 10.3, 10.4),
        event("body_reset_candidate", 10.5, 10.6),
        event("hand_motion_reset_candidate", 10.7, 10.8),
        event("facial_expression_shift_candidate", 10.8, 10.9, 0.90),
        event("facial_expression_shift_candidate", 11.0, 11.1, 0.90),
    ))

    kept, removed, diagnostics = apply_recording_break_cleanup((reaction,), ctx)

    assert kept == (reaction,)
    assert removed == ()
    assert diagnostics == ()


def test_reaction_cluster_without_multimodal_break_evidence_survives():
    takes = (
        take("a", "What is happening?", 10.0, 11.0),
        take("b", "Okay, what the frig?", 12.0, 13.0),
        take("c", "What just happened?", 14.0, 15.0),
    )

    kept, removed, diagnostics = apply_recording_break_cleanup(takes, context(()))

    assert kept == takes
    assert removed == ()
    assert diagnostics == ()


def test_self_critique_before_explicit_failure_is_removed_with_visual_break():
    critique = take(
        "critique",
        "no why do i say don't miss out on this hop on the cozy cardigan train why do i keep saying",
        435.99,
        442.11,
    )
    failure = take(
        "failure",
        "that it's stupid it's stupid yeah yeah it's stupid oh i don't know how to end tiktok shop",
        442.11,
        447.65,
    )
    ctx = context((
        event("hand_motion_reset_candidate", 438.0, 438.1),
        event("body_reset_candidate", 438.8, 438.9),
        event("hand_motion_reset_candidate", 440.3, 440.4),
        event("hand_motion_reset_candidate", 442.5, 442.6),
        event("facial_expression_shift_candidate", 443.0, 443.1, 0.80),
        event("facial_expression_shift_candidate", 443.8, 443.9, 0.82),
    ))

    kept, removed, diagnostics = apply_recording_break_cleanup((critique, failure), ctx)

    assert kept == ()
    assert removed == (critique, failure)
    assert {item["reason"] for item in diagnostics} == {
        "self_critique_before_explicit_recording_failure",
        "explicit_recording_failure",
    }


def test_rhetorical_why_do_i_say_survives_without_following_recording_failure():
    valid = take("valid", "why do I say this every morning because it actually works", 10.0, 14.0)
    following = take("next", "and the bottle lasts me all month", 14.1, 17.0)
    ctx = context((
        event("hand_motion_reset_candidate", 10.1, 10.2),
        event("hand_motion_reset_candidate", 10.4, 10.5),
        event("body_reset_candidate", 11.0, 11.1),
        event("hand_motion_reset_candidate", 11.5, 11.6),
        event("facial_expression_shift_candidate", 12.0, 12.1, 0.90),
        event("facial_expression_shift_candidate", 13.0, 13.1, 0.90),
    ))

    kept, removed, diagnostics = apply_recording_break_cleanup((valid, following), ctx)

    assert kept == (valid, following)
    assert removed == ()
    assert diagnostics == ()


def test_profanity_alone_is_not_a_recording_break():
    valid = take("valid", "this shit actually works really well")
    kept, removed, diagnostics = apply_recording_break_cleanup((valid,))
    assert kept == (valid,)
    assert removed == ()
    assert diagnostics == ()


def test_intentional_single_word_emphasis_survives():
    valid = take("valid", "they are so so super cute")
    kept, removed, diagnostics = apply_recording_break_cleanup((valid,))
    assert kept == (valid,)
    assert removed == ()
    assert diagnostics == ()


def test_normal_repeated_sales_phrase_without_frustration_survives():
    valid = take("valid", "shop now shop now because the deal ends tonight")
    kept, removed, diagnostics = apply_recording_break_cleanup((valid,))
    assert kept == (valid,)
    assert removed == ()
    assert diagnostics == ()
