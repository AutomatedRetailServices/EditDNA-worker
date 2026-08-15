from cutsell_worker.contracts import CandidateTake
from cutsell_worker.merged_self_review import apply_merged_self_review_cleanup
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def take(text, start=0.0, end=3.0):
    return CandidateTake("x", "src", 0, start, end, text)


def event(kind, start, end, confidence=1.0):
    return TemporalEvent("src", start, end, kind, confidence, kind)


def context(events):
    return WholeVideoContext(
        sources=(SourceVideoContext("src", "", "talking_head", "recording", tuple(events)),),
        status=ProviderStatus("test", True, True, "applied"),
    )


def dense_context():
    return context((
        event("hand_motion_reset_candidate", 0.4, 0.5),
        event("body_reset_candidate", 0.8, 0.9),
        event("hand_motion_reset_candidate", 1.2, 1.3),
        event("hand_motion_reset_candidate", 1.8, 1.9),
        event("facial_expression_shift_candidate", 2.1, 2.2, 0.82),
    ))


def test_merged_self_review_and_confusion_is_removed_with_dense_reset():
    item = take("What did I just say? And then they have black, what?", 0.0, 3.2)
    kept, removed, diagnostics = apply_merged_self_review_cleanup((item,), dense_context())
    assert kept == ()
    assert removed == (item,)
    assert diagnostics[0]["reason"] == "merged_speech_self_review_confusion_with_physical_reset"


def test_same_words_without_dense_reset_fail_open():
    item = take("What did I just say? And then they have black, what?", 0.0, 3.2)
    kept, removed, diagnostics = apply_merged_self_review_cleanup((item,), context(()))
    assert kept == (item,)
    assert removed == ()
    assert diagnostics == ()


def test_rhetorical_question_is_not_merged_self_review_shape():
    item = take("What did I just say about the fabric? Let me show you the pocket", 0.0, 3.5)
    kept, removed, diagnostics = apply_merged_self_review_cleanup((item,), dense_context())
    assert kept == (item,)
    assert removed == ()
    assert diagnostics == ()
