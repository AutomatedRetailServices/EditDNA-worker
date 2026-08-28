from cutsell_worker.contracts import CandidateTake
from cutsell_worker.hybrid_alternate_integrity import suppress_stranded_hybrid_alternates


def _take(clip_id, start, end, text, *, complete=True):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
        complete_idea=complete,
    )


def test_closed_adjacent_alternate_with_same_opening_yields_to_high_confidence_keep():
    alternate = _take(
        "alternate",
        10.0,
        18.0,
        "Nunca pensamos revisar la tiroides con una sonografía porque cada año hacíamos estudios.",
    )
    clean = _take(
        "clean",
        18.8,
        28.0,
        "Nunca pensamos revisar la tiroides con una sonografía porque los análisis siempre salían normales.",
    )

    kept, removed, diagnostics = suppress_stranded_hybrid_alternates(
        (alternate, clean),
        (
            ("alternate", "alternate", 0.70),
            ("clean", "keep", 0.90),
        ),
    )

    assert [take.clip_id for take in kept] == ["clean"]
    assert [take.clip_id for take in removed] == ["alternate"]
    assert any(row.get("reason") == "adjacent_same_opening_alternate_yields_to_clean_delivery" for row in diagnostics)


def test_adjacent_alternate_with_different_opening_remains_available():
    alternate = _take(
        "alternate",
        10.0,
        18.0,
        "También tuve un detalle diferente que solo ocurrió durante el viaje.",
    )
    clean = _take(
        "clean",
        18.7,
        28.0,
        "Nunca pensamos revisar la tiroides con una sonografía porque los análisis siempre salían normales.",
    )

    kept, removed, diagnostics = suppress_stranded_hybrid_alternates(
        (alternate, clean),
        (
            ("alternate", "alternate", 0.70),
            ("clean", "keep", 0.90),
        ),
    )

    assert [take.clip_id for take in kept] == ["alternate", "clean"]
    assert removed == ()
    assert not any(row.get("reason") == "adjacent_same_opening_alternate_yields_to_clean_delivery" for row in diagnostics)
