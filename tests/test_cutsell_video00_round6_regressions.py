from cutsell_worker.contracts import CandidateTake, MediaSignals, Word
from cutsell_worker.final_delivery_integrity import collapse_proven_retry_transitions
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.terminal_delivery_reconciliation import (
    restore_tiny_completion_suffixes,
    suppress_open_failed_retry_prefixes,
)
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


def test_round6_open_losing_close_de_los_yields_to_earlier_complete_close():
    """Round 6 kept 319-334 even though it ends at 'de los' and its tail fails."""
    prior = _take(
        "prior",
        295.33,
        314.68,
        (
            "Esta es mi experiencia soy la única en mi familia que tiene este tipo de cáncer "
            "por eso no creo que los cánceres sean hereditarios más bien solo un 5-10% son "
            "de carácter hereditario mayormente son nuestras elecciones de vida así que cuídate."
        ),
        complete=True,
    )
    open_prefix = _take(
        "open_prefix",
        319.37,
        334.25,
        (
            "Soy la primera en mi familia con este tipo de cáncer nadie en mi familia tiene un "
            "carcinoma papilar en la tiroides ni sufre de la tiroides así que estoy convencida "
            "y la ciencia lo avala que solo un 5 -10 % de los"
        ),
        complete=False,
    )
    failed_a = _take(
        "failed_a",
        338.18,
        341.22,
        "cánceres son",
        complete=False,
    )
    failed_b = _take(
        "failed_b",
        342.06,
        346.50,
        "hereditarios soy la única en mi familia que tiene este tipo de cáncer",
        complete=False,
    )

    kept, removed, diagnostics = suppress_open_failed_retry_prefixes(
        (prior, open_prefix),
        (failed_a, failed_b),
        (
            ("prior", "alternate", 0.75),
            ("open_prefix", "winner", 0.95),
            ("failed_a", "failed", 0.85),
            ("failed_b", "failed", 0.85),
        ),
    )

    assert tuple(t.clip_id for t in kept) == ("prior",)
    assert tuple(t.clip_id for t in removed) == ("open_prefix",)
    assert diagnostics[0]["reason"] == "open_failed_retry_prefix_yields_to_complete_prior_delivery"
    assert diagnostics[0]["prior_winner_clip_id"] == "prior"


def test_round6_numeric_format_drift_5_dash_10_does_not_block_same_close_detection():
    prior = _take(
        "prior",
        10.0,
        20.0,
        "soy la única de mi familia solo un 5-10% de los cánceres son hereditarios cuídate.",
        complete=True,
    )
    open_prefix = _take(
        "open",
        25.0,
        35.0,
        "soy la primera de mi familia y la ciencia dice solo un 5 -10 % de los",
        complete=False,
    )
    failed = _take("tail", 38.5, 42.0, "cánceres son hereditarios soy la única de mi familia", complete=False)

    kept, removed, _ = suppress_open_failed_retry_prefixes(
        (prior, open_prefix),
        (failed,),
        (("prior", "keep", 0.9), ("open", "winner", 0.95), ("tail", "failed", 0.85)),
    )

    assert tuple(t.clip_id for t in kept) == ("prior",)
    assert tuple(t.clip_id for t in removed) == ("open",)


def test_round6_tiny_deleted_word_can_complete_incomplete_selected_delivery_without_fabrication():
    incomplete = _take(
        "incomplete",
        356.22,
        360.80,
        "Por eso cuídate aliméntate bien hidrátate y haz",
        complete=False,
    )
    completion = _take(
        "completion",
        361.02,
        361.55,
        "ejercicio.",
        complete=True,
    )

    kept, deleted, diagnostics = restore_tiny_completion_suffixes(
        (incomplete,),
        (completion,),
        (("incomplete", "keep", 0.90), ("completion", "failed", 0.85)),
        _context(),
    )

    # D-094.2: the completion is restored as its OWN take (identity
    # preserved) rather than merged into the incomplete candidate -- the
    # CompositeResolver chain rebuilds kept/deleted from the source takes by
    # clip id, so a merged child never survived it live (run 33983880111).
    assert tuple(t.clip_id for t in kept) == ("incomplete", "completion")
    assert kept[0].text.endswith("hidrátate y haz") and kept[0].end == 360.80
    assert kept[1].text == "ejercicio." and kept[1].end == 361.55
    assert deleted == ()
    assert diagnostics[0]["reason"] == "restore_tiny_completion_suffix"
    assert diagnostics[0]["restored_as"] == "separate_take"
    assert diagnostics[0]["restored_text"].endswith("haz ejercicio.")


def test_tiny_completion_is_not_restored_across_retry_setup():
    incomplete = _take("incomplete", 10.0, 14.0, "quiero terminar esta frase y", complete=False)
    fragment = _take("fragment", 14.2, 14.8, "otra.", complete=True)
    context = _context(
        TemporalEvent("src", 14.02, 14.18, "retry_setup", 0.95, "new attempt begins"),
    )

    kept, deleted, diagnostics = restore_tiny_completion_suffixes(
        (incomplete,),
        (fragment,),
        (("incomplete", "keep", 0.9), ("fragment", "failed", 0.8)),
        context,
    )

    assert tuple(t.clip_id for t in kept) == ("incomplete",)
    assert tuple(t.clip_id for t in deleted) == ("fragment",)
    assert diagnostics == ()
