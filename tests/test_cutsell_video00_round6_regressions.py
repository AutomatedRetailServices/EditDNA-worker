from cutsell_worker.contracts import CandidateTake, MediaSignals, Word
from cutsell_worker.final_delivery_integrity import collapse_proven_retry_transitions
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def _words(text, start, end):
    tokens = text.split()
    step = max(0.05, (float(end) - float(start)) / max(1, len(tokens)))
    cursor = float(start)
    out = []
    for token in tokens:
        out.append(Word(token, cursor, min(float(end), cursor + step), 0.97))
        cursor += step
    return tuple(out)


def _take(clip_id, start, end, text, *, complete=True, visual_fumble=0.0):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=float(start),
        end=float(end),
        text=text,
        words=_words(text, start, end),
        signals=MediaSignals(
            "src",
            float(start),
            float(end),
            visual_fumble=float(visual_fumble),
            expression_naturalness=0.35 if visual_fumble >= 0.5 else 0.78,
            gesture_naturalness=0.35 if visual_fumble >= 0.5 else 0.75,
        ),
        complete_idea=complete,
    )


def _context(*events):
    return WholeVideoContext(
        sources=(SourceVideoContext(
            source_asset_id="src",
            summary="raw talking head with retries",
            dominant_style="talking_head",
            creator_intent="clean recording",
            events=tuple(events),
        ),),
        status=ProviderStatus("test", True, True, "applied"),
    )


def test_round6_both_selected_sonography_attempts_collapse_to_clean_later_delivery():
    """Exact Round 6 failure: both 108-112 and 121-124 survived selection."""
    broken = _take(
        "broken",
        108.90,
        112.34,
        "Ahí fue cuando me mandaron a hacer sonografías de tiroides y otros",
        complete=True,
        visual_fumble=0.72,
    )
    clean = _take(
        "clean",
        121.27,
        124.57,
        "a hacer sonografía de tiroides y otras sonografías",
        complete=True,
        visual_fumble=0.05,
    )
    context = _context(
        TemporalEvent("src", 112.05, 112.45, "retry_setup", 0.90, "creator abandons and retries"),
    )

    kept, removed, diagnostics = collapse_proven_retry_transitions(
        (broken, clean),
        (("broken", "winner", 0.92), ("clean", "alternate", 0.78)),
        context,
    )

    assert tuple(t.clip_id for t in kept) == ("clean",)
    assert tuple(t.clip_id for t in removed) == ("broken",)
    assert diagnostics[0]["reason"] == "proven_retry_transition_later_clean_delivery_wins"
    assert diagnostics[0]["retry_setup_confidence"] == 0.90
    assert diagnostics[0]["visual_prefers_later"] is True


def test_retry_transition_does_not_delete_when_later_delivery_is_not_proven_better():
    first = _take("first", 10.0, 15.0, "esta es mi historia completa sobre la tiroides", complete=True)
    second = _take("second", 18.0, 23.0, "esta es mi historia completa sobre la tiroides", complete=True)
    context = _context(
        TemporalEvent("src", 15.1, 15.4, "retry_setup", 0.90, "another take"),
    )

    kept, removed, diagnostics = collapse_proven_retry_transitions(
        (first, second),
        (("first", "winner", 0.95), ("second", "alternate", 0.75)),
        context,
    )

    assert tuple(t.clip_id for t in kept) == ("first", "second")
    assert removed == ()
    assert diagnostics == ()
