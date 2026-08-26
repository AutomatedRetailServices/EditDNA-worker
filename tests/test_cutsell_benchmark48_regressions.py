from cutsell_worker.contracts import CandidateTake, MediaSignals, Word
from cutsell_worker.hybrid_retry_completion_integrity import apply_hybrid_retry_completion_integrity
from cutsell_worker.hybrid_session_cleanup import HybridSessionCleanupResult


def _words(text, start=0.0, step=0.25):
    output = []
    cursor = float(start)
    for token in text.split():
        output.append(Word(token, cursor, cursor + step, 0.95))
        cursor += step
    return tuple(output)


def _take(clip_id, start, end, text, *, complete=True, fumble=0.0):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=float(start),
        end=float(end),
        text=text,
        words=_words(text, start),
        signals=MediaSignals("src", float(start), float(end), visual_fumble=fumble),
        complete_idea=complete,
    )


def _result(kept, deleted=(), decisions=()):
    return HybridSessionCleanupResult(
        kept=tuple(kept),
        deleted=tuple(deleted),
        requested_chunk_count=1,
        available_chunk_count=1,
        diagnostics=(),
        semantic_decisions=tuple(decisions),
    )


def test_video00_reset_backed_failed_reformulation_yields_to_clean_peer():
    failed = _take(
        "failed",
        25.60,
        32.42,
        "Nunca se nos ocurrió hacer un chequeo de sonografía de la tiroides pues porque cada año que me hacía mínimo dos",
        complete=False,
        fumble=0.82,
    )
    clean = _take(
        "clean",
        35.46,
        45.54,
        "Nunca se nos ocurrió hacer un chequeo de la tiroides por sonografía porque siempre en mis exámenes la tiroides salía funcionando perfectamente",
    )
    result = _result(
        (failed, clean),
        decisions=(("failed", "failed", 0.80), ("clean", "keep", 0.90)),
    )

    repaired = apply_hybrid_retry_completion_integrity(result, (failed, clean))

    assert tuple(t.clip_id for t in repaired.kept) == ("clean",)
    assert "failed" in {t.clip_id for t in repaired.deleted}


def test_video00_short_sonography_incomplete_debris_is_removed_but_complete_retry_survives():
    previous = _take(
        "previous",
        95.58,
        107.48,
        "Al terminar mi contrato cambié de ginecóloga y me mandó a hacer sonografías",
    )
    retry_a = _take(
        "retry-a",
        108.56,
        111.86,
        "Ahí fue cuando me mandaron a hacer sonografías de tiroides",
        complete=False,
    )
    retry_b = _take(
        "retry-b",
        120.11,
        124.15,
        "a hacer sonografías de tiroides y otras sonografías",
        complete=True,
    )
    following = _take(
        "following",
        128.16,
        134.22,
        "En la sonografía de tiroides apareció un nódulo sospechoso que se mandó a biopsia",
    )
    result = _result(
        (previous, retry_a, retry_b, following),
        decisions=(
            ("previous", "winner", 0.95),
            ("retry-a", "alternate", 0.75),
            ("retry-b", "alternate", 0.80),
            ("following", "keep", 0.90),
        ),
    )

    repaired = apply_hybrid_retry_completion_integrity(
        result, (previous, retry_a, retry_b, following)
    )

    assert tuple(t.clip_id for t in repaired.kept) == ("previous", "retry-b", "following")
    assert "retry-a" in {t.clip_id for t in repaired.deleted}
    assert "retry-b" not in {t.clip_id for t in repaired.deleted}


def test_video00_full_reset_backed_alternate_can_yield_to_winner_plus_open_continuation():
    alternate = _take(
        "alternate",
        295.36,
        314.60,
        "Esta es mi experiencia soy la única en mi familia que tiene este tipo de cáncer está comprobado científicamente que los cánceres son hereditarios solo un porcentaje son de carácter hereditario mayormente son nuestras elecciones de vida",
        fumble=0.80,
    )
    winner = _take(
        "winner",
        319.38,
        334.24,
        "Soy la primera en mi familia con este tipo de cáncer nadie en mi familia tiene un carcinoma papilar en la tiroides y la ciencia avala que solo un porcentaje de los",
        complete=False,
    )
    continuation = _take(
        "continuation",
        335.88,
        346.54,
        "cánceres son hereditarios soy la única que tiene este tipo de cáncer",
    )
    result = _result(
        (alternate, winner, continuation),
        decisions=(
            ("alternate", "alternate", 0.75),
            ("winner", "winner", 0.95),
            ("continuation", "keep", 0.85),
        ),
    )

    repaired = apply_hybrid_retry_completion_integrity(
        result, (alternate, winner, continuation)
    )

    assert "alternate" not in {t.clip_id for t in repaired.kept}
    assert {"winner", "continuation"}.issubset({t.clip_id for t in repaired.kept})


def test_unique_long_alternate_without_recording_failure_remains_fail_open():
    alternate = _take(
        "unique",
        0,
        12,
        "This paragraph adds a separate breakfast lunch dinner and hidden hunger story that matters to the audience",
    )
    winner = _take(
        "winner",
        15,
        27,
        "This winner explains the product ingredients and why the formula works",
    )
    result = _result(
        (alternate, winner),
        decisions=(("unique", "alternate", 0.90), ("winner", "winner", 0.96)),
    )

    repaired = apply_hybrid_retry_completion_integrity(result, (alternate, winner))

    assert tuple(t.clip_id for t in repaired.kept) == ("unique", "winner")
    assert repaired.deleted == ()


def test_video03_complete_delivery_is_preserved_before_failed_tail():
    text = "esta crema es mágica tiene unos componentes que de verdad te protegen te reparan la barrera"
    winner = _take("winner", 7.85, 45.35, text)
    winner = CandidateTake(
        clip_id=winner.clip_id,
        source_asset_id=winner.source_asset_id,
        source_order=winner.source_order,
        start=winner.start,
        end=winner.end,
        text=winner.text,
        words=_words(text, 41.10, 0.25),
        signals=winner.signals,
        complete_idea=True,
    )
    failed_tail = _take(
        "tail",
        45.35,
        47.65,
        "de la de hace como",
        complete=False,
        fumble=0.90,
    )
    result = _result(
        (winner,),
        deleted=(failed_tail,),
        decisions=(("winner", "winner", 0.95), ("tail", "failed", 0.90)),
    )

    repaired = apply_hybrid_retry_completion_integrity(result, (winner, failed_tail))

    assert len(repaired.kept) == 1
    repaired_winner = repaired.kept[0]
    assert repaired_winner.text == winner.text
    assert repaired_winner.text.endswith("te reparan la barrera")
    assert repaired_winner.end == winner.end
    assert {t.clip_id for t in repaired.deleted} == {"tail"}