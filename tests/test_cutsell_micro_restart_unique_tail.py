from cutsell_worker.contracts import CandidateTake
from cutsell_worker.micro_restart_cleanup import _destructive_repeated_phrase_restart


def test_repeated_opening_with_unique_tail_is_not_whole_take_debris():
    text = (
        "Otro sintoma era que me salian espinillas otro sintoma era que me salian "
        "espinillas detras de la oreja y en el cuello me salia por temporadas"
    )
    destructive, diagnostic = _destructive_repeated_phrase_restart(text)
    assert destructive is False
    assert diagnostic["unique_tail_count"] >= 3


def test_repeated_restart_dominated_by_duplicate_phrase_remains_destructive():
    text = "me salian espinillas me salian espinillas"
    destructive, diagnostic = _destructive_repeated_phrase_restart(text)
    assert destructive is True
    assert diagnostic["unique_tail_count"] <= 2
    assert diagnostic["repeated_share"] >= 0.55
