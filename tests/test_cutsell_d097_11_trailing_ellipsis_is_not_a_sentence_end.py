"""D-097.11 (R15) -- a trailing ellipsis is a trailing-off marker, not a full stop.

RAW 34048444463 (head 3c8ec4a): the abandoned stomach attempt "Tuve problemas
estomacales a un tiempo en donde se me hizo una endoscopía y me diagnosticaron
con..." was marked `complete_idea=True` because `_ends_sentence` matched the
ellipsis as a period before the open-tail check could see the bridge word
"con". The marker feeds the pair ranking bonus, clean-cut's incomplete rules,
the Resolver's usability/claim protections and the guards that exist only to
work around this error (`cross_group_truncated_winner_authority`). 2 of the
run's 39 takes ended in an ellipsis; both were genuinely cut off.
"""
from __future__ import annotations

from cutsell_worker.contracts import CandidateTake
from cutsell_worker.take_segmentation import (
    _ends_sentence,
    _grammatically_open_tail,
    _looks_complete_idea,
    _repair_boundary_fragments,
)


def test_a_trailing_ellipsis_is_an_open_tail_not_a_sentence_end():
    text = "Tuve problemas estomacales a un tiempo en donde se me hizo una endoscopía y me diagnosticaron con..."
    assert _ends_sentence(text) is False
    assert _grammatically_open_tail(text) is True
    assert _looks_complete_idea(text, 7.96) is False
    assert _looks_complete_idea("Por temporada me salía acné en la espalda la cual yo resor...", 6.0) is False
    assert _looks_complete_idea("Por temporada me salía acné en la espalda la cual yo resor…", 6.0) is False


def test_real_sentence_ends_are_unchanged():
    assert _looks_complete_idea("Tuve problemas de digestión y dijeron que tenía gastritis.", 9.0) is True
    assert _looks_complete_idea("¿Y saben qué pasó?", 2.0) is True
    assert _looks_complete_idea("Fue increíble!", 1.5) is True
    assert _looks_complete_idea('Me dijo "tranquila".', 2.0) is True
    # An ellipsis in the MIDDLE of a delivery is not a trailing-off marker.
    assert _looks_complete_idea("Pensé... que era el estrés y nada más.", 4.0) is True


def _take(clip_id, start, end, text):
    return CandidateTake(clip_id=clip_id, source_asset_id="src", source_order=0, start=start, end=end, text=text,
                         complete_idea=_looks_complete_idea(text, end - start))


def test_an_ellipsis_tail_still_never_joins_across_a_real_pause():
    abandoned = _take("a", 236.23, 244.19, "Tuve problemas estomacales en donde se me hizo una endoscopía y me diagnosticaron con...")
    aside = _take("b", 245.39, 251.61, "Tuve problemas de estómago en una temporada, en 2023, no hay que preguntar.")
    repaired = _repair_boundary_fragments((abandoned, aside))
    assert [t.clip_id for t in repaired] == ["a", "b"]  # 1.2 s gap: no open-tail join
    assert repaired[0].complete_idea is False and repaired[1].complete_idea is True


def test_an_ellipsis_tail_joins_its_contiguous_continuation_like_any_open_tail():
    left = _take("a", 10.0, 13.0, "Comencé a notar un aumento de...")
    right = _take("b", 13.1, 14.2, "peso en la cara y en el cuello.")
    repaired = _repair_boundary_fragments((left, right))
    assert len(repaired) == 1 and repaired[0].complete_idea is True
