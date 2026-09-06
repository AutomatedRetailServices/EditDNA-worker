from cutsell_worker.contracts import CandidateTake, Word
from cutsell_worker.take_segmentation import _looks_complete_idea, _repair_boundary_fragments


def _take(clip_id, start, end, text):
    words = tuple(
        Word(token, start + index * 0.2, min(end, start + index * 0.2 + 0.15))
        for index, token in enumerate(text.split())
    )
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src-1",
        source_order=0,
        start=start,
        end=end,
        text=text,
        words=words,
        complete_idea=_looks_complete_idea(text, end - start),
    )


def test_spanish_long_fragment_ending_que_is_not_complete():
    assert _looks_complete_idea(
        "Al terminar mi contrato hablé con mi ginecóloga y le pedí todos los test que",
        4.1,
    ) is False


def test_english_long_fragment_ending_because_is_not_complete():
    assert _looks_complete_idea(
        "I kept using this every morning because",
        3.4,
    ) is False


def test_long_spanish_open_tail_stitches_to_contiguous_continuation():
    left = _take(
        "left",
        82.82,
        86.90,
        "Al terminar mi contrato hablé con mi ginecóloga y le pedí todos los test que",
    )
    right = _take(
        "right",
        86.90,
        90.60,
        "ella pudiera imaginarse o que me pudiera indicar.",
    )
    repaired = _repair_boundary_fragments((left, right))
    assert len(repaired) == 1
    assert repaired[0].start == 82.82
    assert repaired[0].end == 90.60
    assert repaired[0].text.endswith("ella pudiera imaginarse o que me pudiera indicar.")
    assert repaired[0].complete_idea is True


def test_spanish_article_tail_stitches_to_next_chunk():
    left = _take(
        "left",
        15.46,
        19.96,
        "Tenía como costumbre cada vez que terminaba un",
    )
    right = _take("right", 19.96, 23.28, "contrato hacerme un chequeo de rutina.")
    repaired = _repair_boundary_fragments((left, right))
    assert len(repaired) == 1
    assert "terminaba un contrato" in repaired[0].text


def test_open_tail_does_not_cross_real_pause():
    left = _take("left", 10.0, 13.0, "I kept using it because")
    right = _take("right", 14.0, 17.0, "it helped my skin.")
    repaired = _repair_boundary_fragments((left, right))
    assert len(repaired) == 2


def test_complete_sentence_remains_separate_even_when_contiguous():
    left = _take("left", 10.0, 13.0, "This helped my skin.")
    right = _take("right", 13.05, 16.0, "Then I tried it at night.")
    repaired = _repair_boundary_fragments((left, right))
    assert len(repaired) == 2
