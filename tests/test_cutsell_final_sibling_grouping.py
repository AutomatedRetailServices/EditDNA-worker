from cutsell_worker.contracts import CandidateTake, MediaSignals, Word
from cutsell_worker.final_sibling_grouping import reconcile_final_sibling_groups
from cutsell_worker import session_boundaries


def _words(text, start=0.0, step=0.20):
    out = []
    cursor = float(start)
    for token in text.split():
        out.append(Word(token, cursor, cursor + step, 0.95))
        cursor += step
    return tuple(out)


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


def _video00_retry_triplet():
    complete = _take(
        "complete",
        295.36,
        314.60,
        "Esta es mi experiencia soy la única en mi familia que tiene este tipo de cáncer por eso no creo y está comprobado científicamente que los cánceres son hereditarios más bien solo un 5-10% son de carácter hereditario mayormente son nuestras elecciones de vida así que cuídate.",
    )
    prefix = _take(
        "prefix",
        319.38,
        334.24,
        "Soy la primera en mi familia con este tipo de cáncer nadie en mi familia tiene un carcinoma papilar en la tiroides ni sufre de la tiroides así que estoy convencida y la ciencia lo avala que solo un 5-10% de los",
        complete=False,
    )
    continuation = _take(
        "continuation",
        335.88,
        346.54,
        "cánceres son hereditarios soy la única que tiene este tipo de cáncer.",
    )
    return complete, prefix, continuation


def test_video00_complete_delivery_and_split_retry_become_one_sibling_family():
    complete, prefix, continuation = _video00_retry_triplet()

    groups, changed = reconcile_final_sibling_groups(
        (("complete",), ("prefix",), ("continuation",)),
        (complete, prefix, continuation),
    )

    assert changed is True
    assert len(groups) == 1
    assert set(groups[0]) == {"complete", "prefix", "continuation"}


def test_video00_reconciles_globally_even_when_session_partitioning_separates_all_three(monkeypatch):
    complete, prefix, continuation = _video00_retry_triplet()
    takes = (complete, prefix, continuation)

    monkeypatch.setattr(
        session_boundaries,
        "partition_takes_by_sessions",
        lambda *_args, **_kwargs: ((complete,), (prefix,), (continuation,)),
    )

    result = session_boundaries.safe_group_takes_by_sessions(
        None,
        takes,
        None,
        context_text="",
    )

    assert len(result.groups) == 1
    assert set(result.groups[0]) == {"complete", "prefix", "continuation"}
    assert "global_post_session_sibling_reconciled" in result.reason


def test_full_reformulations_of_same_delivery_merge_for_best_take():
    first = _take(
        "first",
        10,
        20,
        "Nunca se nos ocurrió hacer un chequeo de sonografía de la tiroides porque cada año mis resultados salían normales.",
    )
    second = _take(
        "second",
        24,
        35,
        "Nunca se nos ocurrió hacer un chequeo de la tiroides por sonografía porque siempre mis exámenes de tiroides salían normales.",
    )

    groups, changed = reconcile_final_sibling_groups((("first",), ("second",)), (first, second))

    assert changed is True
    assert groups == (("first", "second"),)


def test_related_story_paragraphs_remain_separate():
    symptoms = _take(
        "symptoms",
        100,
        114,
        "Comencé a notar hinchazón en mi cara aumento de peso pérdida de pelo y cambios en mi metabolismo.",
    )
    diagnosis = _take(
        "diagnosis",
        120,
        134,
        "En la sonografía de tiroides apareció un nódulo sospechoso de tres centímetros que luego se mandó a biopsia.",
    )

    groups, changed = reconcile_final_sibling_groups(
        (("symptoms",), ("diagnosis",)),
        (symptoms, diagnosis),
    )

    assert changed is False
    assert groups == (("symptoms",), ("diagnosis",))
