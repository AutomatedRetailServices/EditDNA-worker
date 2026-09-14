from cutsell_worker.contracts import CandidateTake, MediaSignals, Word
from cutsell_worker.hybrid_cross_group_retry_integrity import collapse_cross_group_semantic_retries


def _words(text, start=0.0, step=0.25):
    output = []
    cursor = float(start)
    for token in text.split():
        output.append(Word(token, cursor, cursor + step, 0.95))
        cursor += step
    return tuple(output)


def _take(clip_id, start, end, text, *, complete=True):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=float(start),
        end=float(end),
        text=text,
        words=_words(text, start),
        signals=MediaSignals("src", float(start), float(end)),
        complete_idea=complete,
    )


def test_video00_failed_sonography_delivery_yields_to_clean_reformulation():
    failed = _take(
        "failed",
        25.60,
        32.42,
        "Nunca se nos ocurrió hacer un chequeo de sonografía de la tiroides pues porque cada año que me hacía mínimo dos",
    )
    winner = _take(
        "winner",
        35.46,
        45.54,
        "Nunca se nos ocurrió hacer un chequeo de la tiroides por sonografía porque siempre en mis exámenes la tiroides salía funcionando perfectamente",
    )

    kept, removed, diagnostics = collapse_cross_group_semantic_retries(
        (failed, winner),
        (("failed", "failed", 0.88), ("winner", "winner", 0.95)),
    )

    assert tuple(t.clip_id for t in kept) == ("winner",)
    assert tuple(t.clip_id for t in removed) == ("failed",)
    assert diagnostics[0]["reason"] == "cross_group_semantic_retry_covered_by_authoritative_delivery"


def test_video00_short_sonography_retries_collapse_across_groups():
    prior = _take(
        "prior",
        95.58,
        107.48,
        "Al terminar mi contrato cambié de ginecóloga y le pedí que me hiciera un test de todo y ahí me mandó a hacer sonografías",
    )
    alternate = _take(
        "alternate",
        108.56,
        111.86,
        "Ahí fue cuando me mandaron a hacer sonografías de tiroides",
    )
    failed = _take(
        "failed",
        120.11,
        124.15,
        "a hacer sonografías de tiroides y otras sonografías",
    )
    winner = _take(
        "winner",
        128.16,
        134.22,
        "En la sonografía de tiroides apareció un nódulo sospechoso que se mandó a biopsia",
    )

    kept, removed, _ = collapse_cross_group_semantic_retries(
        (prior, alternate, failed, winner),
        (
            ("prior", "winner", 0.93),
            ("alternate", "alternate", 0.80),
            ("failed", "failed", 0.85),
            ("winner", "winner", 0.98),
        ),
    )

    assert tuple(t.clip_id for t in kept) == ("prior", "winner")
    assert {t.clip_id for t in removed} == {"alternate", "failed"}


def test_video00_hereditary_restatement_yields_to_winner_plus_continuation():
    alternate = _take(
        "alternate",
        295.36,
        314.60,
        "Esta es mi experiencia soy la única en mi familia que tiene este tipo de cáncer solo un 5-10% son de carácter hereditario mayormente son nuestras elecciones de vida",
    )
    winner = _take(
        "winner",
        319.38,
        334.24,
        "Soy la primera en mi familia con este tipo de cáncer nadie en mi familia tiene un carcinoma papilar en la tiroides y la ciencia avala que solo un 5-10% de los",
        complete=False,
    )
    continuation = _take(
        "continuation",
        335.88,
        346.54,
        "cánceres son hereditarios soy la única que tiene este tipo de cáncer",
    )

    kept, removed, _ = collapse_cross_group_semantic_retries(
        (alternate, winner, continuation),
        (
            ("alternate", "alternate", 0.85),
            ("winner", "winner", 0.95),
            ("continuation", "keep", 0.90),
        ),
    )

    assert tuple(t.clip_id for t in kept) == ("winner", "continuation")
    assert tuple(t.clip_id for t in removed) == ("alternate",)


def test_video02_unique_story_is_not_removed_by_topic_similarity():
    unique = _take(
        "unique",
        0.62,
        35.44,
        "You know those videos where creators are actually having fun this one was one of them my favorite second video was with Blaine we had instant chemistry and an amazing friendship",
    )
    later = _take(
        "later",
        177.83,
        205.01,
        "People ask me all the time do you actually have fun in your job and the answer is yes sometimes you are hanging out with one of your friends and people can tell when there is real chemistry",
    )

    kept, removed, _ = collapse_cross_group_semantic_retries(
        (unique, later),
        (("unique", "alternate", 0.82), ("later", "winner", 0.95)),
    )

    assert {t.clip_id for t in kept} == {"unique", "later"}
    assert removed == ()


def test_video05_unique_hidden_hunger_story_remains_fail_open():
    story = _take(
        "story",
        30.39,
        74.09,
        "frozen waffles cereal lunch chicken nuggets french fries fruit dinner mac and cheese broccoli snacks hidden hunger multivitamin nutrients",
    )
    product = _take(
        "product",
        80.0,
        92.0,
        "two vitamins a day no artificial flavors cane sugar non gmo gluten free vegan",
    )

    kept, removed, _ = collapse_cross_group_semantic_retries(
        (story, product),
        (("story", "alternate", 0.80), ("product", "winner", 0.95)),
    )

    assert {t.clip_id for t in kept} == {"story", "product"}
    assert removed == ()
