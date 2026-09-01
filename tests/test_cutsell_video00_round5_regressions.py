from cutsell_worker.contracts import CandidateTake, MediaSignals, Word
from cutsell_worker.hybrid_failed_continuation_integrity import (
    suppress_selected_prefixes_with_failed_suffixes,
)
from cutsell_worker.internal_retake_winner import prefer_internal_clean_retakes


def _take(clip_id, start, end, text, *, complete=True, visual_fumble=0.0):
    tokens = text.split()
    step = max(0.08, (float(end) - float(start)) / max(1, len(tokens)))
    words = []
    cursor = float(start)
    for token in tokens:
        words.append(Word(token, cursor, min(float(end), cursor + step), 0.97))
        cursor += step
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=float(start),
        end=float(end),
        text=text,
        words=tuple(words),
        signals=MediaSignals(
            "src",
            float(start),
            float(end),
            visual_fumble=float(visual_fumble),
        ),
        complete_idea=complete,
    )


def test_round5_merged_broken_attempt_is_cut_at_repeated_clean_opening():
    """Exact structural failure observed in Video 00 Round 5.

    Attempt reconstruction merged the bad 0:14-0:22 delivery and the clean 0:23-0:32
    retake into one CandidateTake. Hybrid then judged the merged object as a winner. The
    second verbatim opening plus proven visual fumble must split away the first attempt.
    """
    take = _take(
        "merged",
        25.64,
        45.92,
        (
            "Nunca se nos ocurrió hacer un chequeo de sonografía de la tiroides pues porque "
            "cada año que me hacía mínimo dos estados Nunca se nos ocurrió hacer un chequeo "
            "de la tiroides por sonografía porque siempre en mis exámenes la tiroides salía "
            "como que estaba funcionando perfectamente"
        ),
        visual_fumble=0.69,
    )

    resolved, diagnostics = prefer_internal_clean_retakes((take,), None)

    assert len(resolved) == 1
    assert resolved[0].start > 25.64
    assert resolved[0].text.startswith("Nunca se nos ocurrió hacer un chequeo")
    assert "mínimo dos estados Nunca" not in resolved[0].text
    assert diagnostics
    assert diagnostics[0]["reason"] == "internal_broken_attempt_yields_to_clean_retake"
    assert diagnostics[0]["evidence_type"] == "repeated_opening_plus_visual_fumble"
    assert diagnostics[0]["repeated_phrase"].startswith("nunca se nos ocurrió hacer")


def test_round5_late_closing_prefix_loses_when_its_immediate_suffix_fails():
    """Exact closing structure observed in Video 00 Round 5.

    The later retry prefix 319.82-336.44 looked clean in isolation and Gemini called it a
    winner, but its immediate continuation fragments were failed. The earlier complete
    295.42-314.68 close therefore remains authoritative and the later prefix must go.
    """
    earlier = _take(
        "earlier",
        295.42,
        314.68,
        (
            "Esta es mi experiencia soy la única en mi familia que tiene este tipo de cáncer "
            "por eso no creo y está comprobado científicamente que los cánceres son hereditarios "
            "más bien sólo un 5 o 10 por ciento son de carácter hereditario mayormente son "
            "nuestras elecciones de vida así que cuídate"
        ),
        complete=True,
    )
    later_prefix = _take(
        "later_prefix",
        319.82,
        336.44,
        (
            "Soy la primera en mi familia con este tipo de cáncer Nadie en mi familia tiene un "
            "carcinoma papilar ni sufre de la tiroides así que estoy convencida y la ciencia lo "
            "avala que sólo un 5 o 10 por ciento de los cánceres son hereditarios"
        ),
        complete=True,
    )
    failed_tail_a = _take(
        "failed_tail_a",
        338.18,
        341.22,
        "Sólo un 5 a un 10 por ciento de los cánceres son",
        complete=False,
    )
    failed_tail_b = _take(
        "failed_tail_b",
        342.06,
        346.50,
        "hereditarios Soy la única en mi familia que tiene este tipo de cáncer",
        complete=False,
    )

    kept, removed, diagnostics = suppress_selected_prefixes_with_failed_suffixes(
        (earlier, later_prefix),
        (failed_tail_a, failed_tail_b),
        (
            ("earlier", "alternate", 0.75),
            ("later_prefix", "winner", 0.95),
            ("failed_tail_a", "failed", 0.85),
            ("failed_tail_b", "failed", 0.85),
        ),
    )

    assert tuple(take.clip_id for take in kept) == ("earlier",)
    assert tuple(take.clip_id for take in removed) == ("later_prefix",)
    assert diagnostics
    assert diagnostics[0]["reason"] == "selected_prefix_yields_when_immediate_suffix_fails_same_retry"
    assert diagnostics[0]["prior_winner_clip_id"] == "earlier"
    assert diagnostics[0]["failed_suffix_clip_ids"] == ["failed_tail_a", "failed_tail_b"]
