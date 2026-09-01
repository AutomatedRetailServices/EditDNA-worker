from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy
from cutsell_worker.final_selection_retry_arbiter import losing_retry_ids, same_strong_opening
from cutsell_worker.selection_phase_authority import apply_selection_phase_authority


def clip(clip_id, start, end, text):
    return DraftClip(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
        caption_text=text,
    )


def diagnostics(*decisions):
    return {"hybrid_editorial_chunks": [{"decisions": list(decisions)}]}


def test_same_opening_failed_retry_yields_only_at_final_selection():
    earlier = clip("bad", 10, 19, "Nunca se nos ocurrió hacer un chequeo de sonografía de la tiroides porque cada año hacía estudios.")
    later = clip("good", 20, 31, "Nunca se nos ocurrió hacer un chequeo de sonografía de la tiroides porque siempre salía funcionando perfectamente.")
    diag = diagnostics(
        {"clip_id": "bad", "label": "failed", "confidence": 0.85},
        {"clip_id": "good", "label": "winner", "confidence": 0.95},
    )
    remove, audit = losing_retry_ids((earlier, later), diag)
    assert ("bad", 10.0, 19.0) in remove
    assert audit[0]["later_winner_clip_id"] == "good"


def test_physical_alternate_same_opening_yields_to_immediate_high_confidence_keep():
    earlier = clip(
        "alternate",
        10,
        19,
        "Nunca se nos ocurrió hacer un chequeo de sonografía de la tiroides pues cada año hacía estudios.",
    )
    later = clip(
        "good",
        19.8,
        31,
        "Nunca se nos ocurrió hacer un chequeo de la tiroides por sonografía porque siempre salía funcionando perfectamente.",
    )
    diag = diagnostics(
        {
            "clip_id": "alternate",
            "label": "alternate",
            "confidence": 0.70,
            "local_failure_reasons": ["dense_physical_reset:5"],
        },
        {"clip_id": "good", "label": "keep", "confidence": 0.90},
    )
    remove, audit = losing_retry_ids((earlier, later), diag)
    assert ("alternate", 10.0, 19.0) in remove
    assert audit[0]["later_winner_clip_id"] == "good"


def test_removed_semantic_alternate_moves_to_swap_alternates_not_discarded():
    earlier = clip(
        "alternate",
        10,
        19,
        "Nunca se nos ocurrió hacer un chequeo de sonografía de la tiroides pues cada año hacía estudios.",
    )
    later = clip(
        "good",
        19.8,
        31,
        "Nunca se nos ocurrió hacer un chequeo de la tiroides por sonografía porque siempre salía funcionando perfectamente.",
    )
    diag = diagnostics(
        {
            "clip_id": "alternate",
            "label": "alternate",
            "confidence": 0.70,
            "local_failure_reasons": ["dense_physical_reset:5"],
        },
        {"clip_id": "good", "label": "keep", "confidence": 0.90},
    )
    draft = DraftTimeline(
        schema_version="cutsell.v1",
        project_id="p",
        strategy=EditStrategy.STORYTELLING,
        selected=(earlier, later),
        alternates=(),
        discarded=(),
        diagnostics=diag,
    )

    repaired = apply_selection_phase_authority(draft)

    assert [item.clip_id for item in repaired.selected] == ["good"]
    assert [item.clip_id for item in repaired.alternates] == ["alternate"]
    assert repaired.discarded == ()
    assert repaired.diagnostics["selection_phase_authority"]["status"] == "executed"


def test_conflicted_clip_with_strong_winner_is_not_removed():
    earlier = clip("keep", 10, 19, "Nunca se nos ocurrió hacer un chequeo de sonografía de la tiroides porque cada año hacía estudios.")
    later = clip("later", 20, 31, "Nunca se nos ocurrió hacer un chequeo de sonografía de la tiroides porque siempre salía funcionando perfectamente.")
    diag = diagnostics(
        {"clip_id": "keep", "label": "failed", "confidence": 0.90},
        {"clip_id": "keep", "label": "winner", "confidence": 0.95},
        {"clip_id": "later", "label": "winner", "confidence": 0.95},
    )
    remove, _ = losing_retry_ids((earlier, later), diag)
    assert not remove


def test_complementary_different_opening_is_preserved():
    earlier = clip("prefix", 10, 18, "También me salían espinillas y era como un rush, una alergia.")
    later = clip("later", 20, 30, "Otro síntoma era que me salían espinillas detrás de la oreja y en el cuello por temporadas.")
    diag = diagnostics(
        {"clip_id": "prefix", "label": "failed", "confidence": 0.90},
        {"clip_id": "later", "label": "winner", "confidence": 0.95},
    )
    remove, _ = losing_retry_ids((earlier, later), diag)
    assert not remove
    assert same_strong_opening(earlier.text, later.text) is False
