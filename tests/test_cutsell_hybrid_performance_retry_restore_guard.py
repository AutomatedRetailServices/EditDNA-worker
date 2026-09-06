from cutsell_worker.contracts import CandidateTake
from cutsell_worker.hybrid_performance_retry_restore_guard import (
    conflicting_winner_fail_open_ids,
    same_strong_opening,
)


def take(clip_id, start, end, text, complete=True):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
        complete_idea=complete,
    )


def test_same_strong_opening_detects_earlier_failed_retry_of_later_winner():
    earlier = "Nunca se nos ocurrió hacer un chequeo de sonografía de la tiroides porque cada año hacía estudios."
    later = "Nunca se nos ocurrió hacer un chequeo de la tiroides por sonografía porque siempre salía funcionando perfectamente."
    assert same_strong_opening(earlier, later)


def test_same_strong_opening_rejects_complementary_delivery_with_different_opening():
    earlier = "También me salían espinillas y era como un rush, una alergia."
    later = "Otro síntoma era que me salían espinillas detrás de la oreja y en el cuello por temporadas."
    assert same_strong_opening(earlier, later) is False


def test_same_strong_opening_requires_enough_content():
    assert same_strong_opening("Nunca ocurrió", "Nunca ocurrió otra vez") is False


def test_conflicting_semantics_restores_complete_clip_when_stronger_window_says_winner():
    candidate = take(
        "story",
        10.0,
        20.0,
        "La biopsia confirmó el diagnóstico y luego expliqué los síntomas que había tenido.",
    )
    restore = conflicting_winner_fail_open_ids(
        (candidate,),
        {"story"},
        (
            ("story", "winner", 0.95),
            ("story", "failed", 0.90),
        ),
    )
    assert restore == {"story"}


def test_conflicting_semantics_does_not_restore_when_failed_is_stronger():
    candidate = take("story", 10.0, 20.0, "Una entrega completa con información para la audiencia.")
    restore = conflicting_winner_fail_open_ids(
        (candidate,),
        {"story"},
        (
            ("story", "winner", 0.90),
            ("story", "failed", 0.95),
        ),
    )
    assert restore == set()


def test_conflicting_semantics_does_not_restore_without_any_winner_evidence():
    retry = take(
        "retry",
        10.0,
        20.0,
        "Nunca se nos ocurrió hacer un chequeo de sonografía de la tiroides porque cada año hacía estudios.",
    )
    restore = conflicting_winner_fail_open_ids(
        (retry,),
        {"retry"},
        (("retry", "failed", 0.90),),
    )
    assert restore == set()


def test_conflicting_semantics_requires_complete_delivery():
    fragment = take("fragment", 10.0, 14.0, "La biopsia confirmó que era", complete=False)
    restore = conflicting_winner_fail_open_ids(
        (fragment,),
        {"fragment"},
        (
            ("fragment", "winner", 0.96),
            ("fragment", "failed", 0.85),
        ),
    )
    assert restore == set()
