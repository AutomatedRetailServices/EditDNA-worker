from dataclasses import dataclass

from cutsell_worker.contracts import CandidateTake
from cutsell_worker.hybrid_performance_retry_restore_guard import same_strong_opening


def take(clip_id, start, end, text):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
        complete_idea=True,
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
