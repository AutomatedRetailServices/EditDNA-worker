from cutsell_worker.audio_boundary_completion import (
    _candidate_pairs,
    reconcile_audio_confirmed_completion,
)
from cutsell_worker.contracts import (
    DraftClip,
    DraftTimeline,
    EditStrategy,
    SemanticRole,
    TranscriptSegment,
    Word,
)


def _clip(clip_id, start, end, text, *, selected=True):
    return DraftClip(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=float(start),
        end=float(end),
        text=text,
        caption_text=text,
        semantic_role=SemanticRole.STORY,
        selected=selected,
    )


def _segment(words):
    return TranscriptSegment(
        source_asset_id="src",
        start=min(word.start for word in words),
        end=max(word.end for word in words),
        text=" ".join(word.text for word in words),
        words=tuple(words),
    )


def _word(text, start, end):
    return Word(text=text, start=float(start), end=float(end), confidence=0.98)


def test_round10_video03_relisten_restores_cutanea_from_audio_before_failed_collision():
    selected = _clip(
        "winner",
        7.85,
        45.35,
        "esta crema es mágica tiene unos componentes que de verdad te protegen te reparan la barrera",
    )
    failed = _clip("failed", 45.35, 47.65, "de la de hace como", selected=False)
    decoded = _segment((
        _word("te", 0.10, 0.20),
        _word("reparan", 0.22, 0.55),
        _word("la", 0.56, 0.65),
        _word("barrera", 0.66, 1.05),
        _word("cutánea", 1.06, 1.50),
        _word("te", 1.55, 1.65),
        _word("la", 1.66, 1.74),
        _word("te", 1.75, 1.84),
        _word("hace", 1.85, 2.05),
        _word("como", 2.06, 2.25),
    ))

    repaired, audit = reconcile_audio_confirmed_completion(
        selected,
        failed,
        (decoded,),
        window_start=43.80,
    )

    assert audit is not None
    assert audit["reason"] == "audio_confirmed_missing_completion_before_failed_collision"
    assert audit["completion_text"].casefold() == "cutánea"
    assert repaired.text.endswith("la barrera cutánea")
    # The confirmed word is already physically inside the existing selected boundary.
    assert repaired.end == 45.35


def test_audio_completion_fails_open_without_repeated_function_word_collision():
    selected = _clip("winner", 7.85, 45.35, "te protegen te reparan la barrera")
    failed = _clip("failed", 45.35, 47.65, "de la de hace como", selected=False)
    decoded = _segment((
        _word("te", 0.10, 0.20),
        _word("reparan", 0.22, 0.55),
        _word("la", 0.56, 0.65),
        _word("barrera", 0.66, 1.05),
        _word("cutánea", 1.06, 1.50),
        _word("y", 1.55, 1.65),
        _word("queda", 1.66, 1.90),
        _word("bien", 1.91, 2.10),
    ))

    repaired, audit = reconcile_audio_confirmed_completion(selected, failed, (decoded,), window_start=43.80)

    assert repaired == selected
    assert audit is None


def test_audio_completion_fails_open_when_second_decode_does_not_anchor_to_selected_tail():
    selected = _clip("winner", 7.85, 45.35, "te protegen te reparan la barrera")
    failed = _clip("failed", 45.35, 47.65, "de la de hace como", selected=False)
    decoded = _segment((
        _word("otra", 0.10, 0.30),
        _word("frase", 0.31, 0.55),
        _word("totalmente", 0.56, 0.85),
        _word("distinta", 0.86, 1.10),
        _word("te", 1.20, 1.30),
        _word("la", 1.31, 1.40),
        _word("te", 1.41, 1.50),
        _word("hace", 1.51, 1.70),
    ))

    repaired, audit = reconcile_audio_confirmed_completion(selected, failed, (decoded,), window_start=43.80)

    assert repaired == selected
    assert audit is None


def test_candidate_pair_requires_high_confidence_failed_low_content_tail_at_boundary():
    selected = _clip("winner", 7.85, 45.35, "te protegen te reparan la barrera")
    failed = _clip("failed", 45.35, 47.65, "de la de hace como", selected=False)
    draft = DraftTimeline(
        schema_version="cutsell.v1",
        project_id="p",
        strategy=EditStrategy.STORYTELLING,
        selected=(selected,),
        alternates=(),
        discarded=(failed,),
        diagnostics={
            "hybrid_editorial_chunks": [
                {"decisions": [{"clip_id": "failed", "label": "failed", "confidence": 0.90}]}
            ]
        },
    )

    assert _candidate_pairs(draft) == ((selected, failed),)
