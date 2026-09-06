from cutsell_worker.contracts import CandidateTake, MediaSignals, Word
from cutsell_worker.delivery_edge_trim import trim_delivery_edge_slack
from cutsell_worker.hybrid_alternate_integrity import suppress_stranded_hybrid_alternates
from cutsell_worker.hybrid_retry_winner_authority import enforce_proven_retry_winners
from cutsell_worker.internal_retake_winner import prefer_internal_clean_retakes
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def _words(tokens, start=0.0, step=0.22):
    cursor = float(start)
    out = []
    for token in tokens:
        out.append(Word(token, cursor, cursor + step, 0.97))
        cursor += step
    return tuple(out)


def _take(clip_id, start, end, text, *, complete=True):
    words = _words(tuple(text.split()), start=start, step=max(0.08, (end - start) / max(1, len(text.split()))))
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=float(start),
        end=float(end),
        text=text,
        words=words,
        signals=MediaSignals("src", float(start), float(end)),
        complete_idea=complete,
    )


def _context(*events):
    return WholeVideoContext(
        sources=(
            SourceVideoContext(
                source_asset_id="src",
                summary="raw talking head with retries",
                dominant_style="talking_head",
                creator_intent="tell personal story naturally",
                events=tuple(events),
            ),
        ),
        status=ProviderStatus("test", True, True, "applied"),
    )


def test_video00_finished_sentence_drops_visible_postroll_reset():
    """Human 0:12 note means timestamp 00:12, not a 120 ms trim target.

    The invariant is: once spoken delivery is complete, proven dead-air/body-reset
    post-roll should not remain attached to the usable sentence.
    """
    words = _words(("esta", "frase", "ya", "termino"), start=10.6, step=0.3)
    take = CandidateTake(
        clip_id="sentence",
        source_asset_id="src",
        source_order=0,
        start=10.6,
        end=13.2,
        text="esta frase ya termino",
        words=words,
        signals=MediaSignals("src", 10.6, 13.2),
        complete_idea=True,
    )
    # Speech ends at 11.8 and the creator visibly exits the delivery until 13.2.
    context = _context(
        TemporalEvent("src", 11.82, 13.10, "unintentional_dead_air", 0.95, "sentence is over; creator pauses and resets"),
        TemporalEvent("src", 11.90, 12.80, "body_reset_candidate", 0.96, "visible post-delivery mueca/body reset"),
    )

    trimmed, diagnostics = trim_delivery_edge_slack((take,), context)

    assert len(trimmed) == 1
    assert trimmed[0].end == words[-1].end
    assert diagnostics
    assert diagnostics[0]["actions"][0]["action"] == "trim_trailing_non_speech_cut_signal"


def test_video00_broken_internal_attempt_yields_to_later_clean_retake():
    tokens = (
        "nunca", "se", "nos", "ocurrio", "hacer", "chequeo", "tiroides", "sonografia",
        "porque", "siempre", "eh", "no",  # broken first attempt/fumble
        "nunca", "se", "nos", "ocurrio", "hacer", "chequeo", "tiroides", "sonografia",
        "porque", "siempre", "salio", "perfectamente",
    )
    words = _words(tokens, start=10.0, step=0.24)
    take = CandidateTake(
        clip_id="merged_retry",
        source_asset_id="src",
        source_order=0,
        start=10.0,
        end=words[-1].end + 0.1,
        text=" ".join(tokens),
        words=words,
        signals=MediaSignals("src", 10.0, words[-1].end + 0.1),
        complete_idea=True,
    )
    retry_start = words[12].start - 0.08
    context = _context(
        TemporalEvent("src", retry_start, words[12].start + 0.06, "retry_setup", 0.94, "creator restarts same idea"),
    )

    resolved, diagnostics = prefer_internal_clean_retakes((take,), context)

    assert len(resolved) == 1
    assert resolved[0].start == words[12].start
    assert resolved[0].text.startswith("nunca se nos ocurrio hacer chequeo")
    assert "eh no nunca" not in resolved[0].text
    assert diagnostics[0]["reason"] == "internal_broken_attempt_yields_to_clean_retake"


def test_video00_internal_retake_guard_fails_open_without_retry_evidence():
    tokens = (
        "esta", "es", "una", "historia", "sobre", "mi", "familia", "y", "mi", "salud",
        "despues", "cuento", "otra", "parte", "distinta", "de", "la", "historia",
    )
    words = _words(tokens, start=30.0, step=0.22)
    take = CandidateTake(
        clip_id="story",
        source_asset_id="src",
        source_order=0,
        start=30.0,
        end=words[-1].end,
        text=" ".join(tokens),
        words=words,
        signals=MediaSignals("src", 30.0, words[-1].end),
        complete_idea=True,
    )

    resolved, diagnostics = prefer_internal_clean_retakes((take,), _context())

    assert resolved == (take,)
    assert diagnostics == ()


def test_video00_round2_failed_sonography_attempt_yields_to_clean_later_winner():
    failed = _take(
        "clip_aa7c",
        25.60,
        32.42,
        "Nunca se nos ocurrió hacer un chequeo de sonografía de la tiroides pues porque cada año que me hacía mínimo dos",
        complete=False,
    )
    winner = _take(
        "clip_8e43",
        35.46,
        45.54,
        "Nunca se nos ocurrió hacer un chequeo de la tiroides por sonografía porque siempre en mis exámenes la tiroides salía completamente normal",
        complete=True,
    )
    context = _context(
        TemporalEvent("src", 32.60, 33.40, "retry_setup", 0.86, "creator stops broken first attempt and retries"),
    )

    kept, removed, diagnostics = enforce_proven_retry_winners(
        (failed, winner),
        ((failed.clip_id, "failed", 0.80), (winner.clip_id, "winner", 0.95)),
        context,
    )

    assert kept == (winner,)
    assert removed == (failed,)
    assert diagnostics[0]["reason"] == "failed_attempt_yields_to_proven_later_retry_winner"


def test_video00_round2_failed_partial_sonography_restart_yields_to_later_winner():
    failed = _take(
        "clip_9aad",
        108.56,
        111.86,
        "Ahí fue cuando me mandaron a hacer sonografías de tiroides",
        complete=False,
    )
    winner = _take(
        "clip_8b3e",
        120.11,
        124.15,
        "a hacer sonografías de tiroides y otras sonografías",
        complete=True,
    )
    context = _context(
        TemporalEvent("src", 112.01, 112.34, "retry_setup", 0.86, "creator resets and restarts delivery"),
    )

    kept, removed, diagnostics = enforce_proven_retry_winners(
        (failed, winner),
        ((failed.clip_id, "failed", 0.80), (winner.clip_id, "winner", 0.95)),
        context,
    )

    assert kept == (winner,)
    assert removed == (failed,)
    assert diagnostics[0]["winner_clip_id"] == winner.clip_id


def test_retry_winner_authority_fails_open_without_retry_setup():
    failed = _take("failed", 10.0, 16.0, "mi historia de salud y mi experiencia personal completa", complete=True)
    winner = _take("winner", 18.0, 24.0, "mi historia de salud y otra experiencia personal completa", complete=True)

    kept, removed, diagnostics = enforce_proven_retry_winners(
        (failed, winner),
        (("failed", "failed", 0.90), ("winner", "winner", 0.96)),
        _context(),
    )

    assert kept == (failed, winner)
    assert removed == ()
    assert diagnostics == ()


def test_video00_clean_final_winner_suppresses_later_open_alternate():
    winner = _take(
        "winner",
        295.52,
        313.82,
        "Esta es mi experiencia soy la única en mi familia que tiene este tipo de cáncer por eso no creo y está comprobado científicamente que los cánceres son hereditarios más bien solo un 5-10% son de carácter hereditario mayormente son nuestras elecciones de vida así que cuídate",
        complete=True,
    )
    later_open_alternate = _take(
        "alternate",
        319.38,
        334.24,
        "Soy la primera en mi familia con este tipo de cáncer nadie en mi familia tiene un carcinoma papilar en la tiroides ni sufre de la tiroides así que estoy convencida y la ciencia lo avala que solo un 5-10% de los",
        complete=False,
    )

    kept, removed, diagnostics = suppress_stranded_hybrid_alternates(
        (winner, later_open_alternate),
        (
            ("winner", "winner", 0.93),
            ("alternate", "alternate", 0.78),
        ),
    )

    assert "winner" in {take.clip_id for take in kept}
    assert "alternate" not in {take.clip_id for take in kept}
    assert "alternate" in {take.clip_id for take in removed}
    assert diagnostics[0]["temporal_relation"] == "after"
    assert diagnostics[0]["reason"] == "semantic_alternate_incomplete_retry_after_winner"
