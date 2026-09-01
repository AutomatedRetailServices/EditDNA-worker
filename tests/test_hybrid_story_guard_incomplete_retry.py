from cutsell_worker.contracts import CandidateTake
from cutsell_worker.hybrid_story_guard import _covered_by_kept_delivery


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


def test_incomplete_failed_retry_is_covered_when_prior_delivery_preserves_numbers_and_negation():
    kept = _take(
        "kept",
        10.0,
        20.0,
        "Soy la única en mi familia con este tipo de cáncer. Solo un 5-10% son hereditarios.",
    )
    retry = _take(
        "retry",
        21.0,
        28.0,
        "Soy la primera en mi familia con este tipo de cáncer. Nadie en mi familia tiene este cáncer. Solo un 5-10% de los",
        complete=False,
    )
    assert _covered_by_kept_delivery(retry, (kept,)) is True


def test_incomplete_retry_with_new_number_is_not_treated_as_covered():
    kept = _take(
        "kept",
        10.0,
        20.0,
        "Soy la única en mi familia con este tipo de cáncer. Solo un 5-10% son hereditarios.",
    )
    retry = _take(
        "retry",
        21.0,
        28.0,
        "Soy la primera en mi familia con este tipo de cáncer. El estudio nuevo mostró 37% de riesgo.",
        complete=False,
    )
    assert _covered_by_kept_delivery(retry, (kept,)) is False
