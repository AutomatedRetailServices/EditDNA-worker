from cutsell_worker.contracts import CandidateTake
from cutsell_worker.recording_breaks import apply_recording_break_cleanup


def take(clip_id, text):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=0.0,
        end=3.0,
        text=text,
    )


def removed_reason(text):
    kept, removed, diagnostics = apply_recording_break_cleanup((take("x", text),))
    return kept, removed, diagnostics


def test_i_cant_talk_is_explicit_recording_failure():
    kept, removed, diagnostics = removed_reason("one fuck these wipes right I can't talk these wipes")
    assert kept == ()
    assert len(removed) == 1
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
