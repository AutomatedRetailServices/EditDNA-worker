from pathlib import Path

from cutsell_worker.contracts import (
    DraftClip,
    DraftTimeline,
    EditStrategy,
    JobState,
    ProcessingResult,
    SCHEMA_VERSION,
    SemanticRole,
    Word,
)
from cutsell_worker.human_boundary_polish import _bridge_short_completion, _dedupe_repeated_tail
from cutsell_worker.hybrid_failed_soft_restore import install_hybrid_failed_soft_restore


def _clip(cid, start, end, text, words=()):
    return DraftClip(
        clip_id=cid,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
        caption_text=text,
        words=tuple(words),
        semantic_role=SemanticRole.OTHER,
        selected=True,
    )


def test_short_sentence_completion_prefers_continuous_source_span_over_hard_cut():
    left = _clip("left", 10.0, 14.8, "la tiroides salía como que estaba funcionando")
    right = _clip("right", 15.8, 16.4, "perfectamente.")
    merged, audit = _bridge_short_completion([left, right])
    assert len(merged) == 1
    assert merged[0].start == 10.0
    assert merged[0].end == 16.4
    assert merged[0].text.endswith("funcionando perfectamente.")
    assert audit[0]["action"] == "bridge_sentence_completion"


def test_short_completion_does_not_bridge_a_finished_sentence():
    left = _clip("left", 10.0, 14.8, "La idea termina aquí.")
    right = _clip("right", 15.2, 15.8, "Perfectamente.")
    merged, audit = _bridge_short_completion([left, right])
    assert len(merged) == 2
    assert audit == []


def test_repeated_tail_is_removed_when_next_short_delivery_reopens_with_same_word():
    words = (
        Word("Mayormente", 20.0, 20.5),
        Word("son", 20.6, 20.8),
        Word("nuestras", 20.9, 21.3),
        Word("elecciones", 21.4, 22.0),
        Word("de", 22.1, 22.2),
        Word("vida", 22.3, 22.6),
        Word("cuídate", 22.8, 23.3),
    )
    left = _clip("left", 20.0, 23.5, "Mayormente son nuestras elecciones de vida cuídate", words)
    right = _clip("right", 30.0, 34.0, "cuídate alimentate bien hidrátate y haz ejercicio")
    polished, audit = _dedupe_repeated_tail([left, right])
    assert polished[0].end == 22.8
    assert polished[0].text.endswith("vida")
    assert audit[0]["action"] == "remove_repeated_trailing_phrase"


def test_repeated_tail_fails_open_without_word_timestamps():
    left = _clip("left", 20.0, 23.5, "Mayormente son nuestras elecciones de vida cuídate")
    right = _clip("right", 30.0, 34.0, "cuídate alimentate bien hidrátate y haz ejercicio")
    polished, audit = _dedupe_repeated_tail([left, right])
    assert polished[0] == left
    assert audit == []
