from cutsell_worker.contracts import CandidateTake
from cutsell_worker.recording_meta_continuation import apply_recording_meta_continuation_cleanup


def take(clip_id, text, start, end):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
    )


def test_contiguous_recording_process_continuation_is_removed_after_proven_chain():
    discarded = (
        take("a", "it's stupid I don't know how to end TikTok shop videos like I hate", 443.0, 449.3),
        take("b", "saying the link below I hate saying don't miss out on this deal I hate being like", 449.3, 455.15),
    )
    continuation = take(
        "c",
        "salesy some people don't end them they just stop saying it they're like I love it",
        455.15,
        459.2,
    )
    kept, removed, diagnostics = apply_recording_meta_continuation_cleanup((continuation,), discarded)
    assert kept == ()
    assert removed == (continuation,)
    assert diagnostics[0]["reason"] == "recording_process_continuation_after_discarded_meta_chain"


def test_short_syntactic_tail_after_direct_recording_meta_is_removed():
    discarded = (take("a", "this is so hard to make a video", 299.68, 302.0),)
    tail = take("tail", "with kids", 302.92, 304.26)
    kept, removed, diagnostics = apply_recording_meta_continuation_cleanup((tail,), discarded)
    assert kept == ()
    assert removed == (tail,)
    assert diagnostics[0]["reason"] == "short_continuation_after_direct_recording_meta"


def test_short_take_after_unrelated_discard_is_preserved():
    discarded = (take("a", "random failed product phrase", 0.0, 2.0),)
    valid = take("v", "with pockets", 2.5, 3.8)
    kept, removed, diagnostics = apply_recording_meta_continuation_cleanup((valid,), discarded)
    assert kept == (valid,)
    assert removed == ()
    assert diagnostics == ()


def test_long_viewer_facing_sentence_after_direct_meta_is_preserved():
    discarded = (take("a", "this is so hard to make a video", 0.0, 2.0),)
    valid = take("v", "with kids you can still keep this bottle completely spill proof", 2.4, 5.8)
    kept, removed, diagnostics = apply_recording_meta_continuation_cleanup((valid,), discarded)
    assert kept == (valid,)
    assert removed == ()
    assert diagnostics == ()


def test_single_discarded_blooper_does_not_delete_following_process_words():
    discarded = (take("a", "I don't know how to end this", 0.0, 2.0),)
    valid = take("v", "stop saying yes to weak batteries in your videos", 2.0, 5.0)
    kept, removed, diagnostics = apply_recording_meta_continuation_cleanup((valid,), discarded)
    assert kept == (valid,)
    assert removed == ()
    assert diagnostics == ()


def test_noncontiguous_following_take_survives():
    discarded = (
        take("a", "I don't know how to end this video", 0.0, 2.0),
        take("b", "I hate saying the call to action", 2.0, 4.0),
    )
    valid = take("v", "this video shows how to stop frizz and end split ends", 5.0, 8.0)
    kept, removed, diagnostics = apply_recording_meta_continuation_cleanup((valid,), discarded)
    assert kept == (valid,)
    assert removed == ()
    assert diagnostics == ()
