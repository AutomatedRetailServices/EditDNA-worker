from cutsell_worker.contracts import CandidateTake
from cutsell_worker.hybrid_semantic_composite_bridge import (
    reconcile_semantic_rescues,
    strong_same_opening,
)


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


def test_strong_same_opening_detects_retry_restart():
    assert strong_same_opening(
        "Nunca se nos ocurrió hacer un chequeo de sonografía de la tiroides porque...",
        "Nunca se nos ocurrió hacer un chequeo de la tiroides por sonografía porque...",
    )


def test_different_opening_is_not_same_opening_retry():
    assert not strong_same_opening(
        "Otro síntoma era que me salían espinillas detrás de la oreja y el cuello.",
        "También me salían espinillas detrás de la oreja y todo el cuello.",
    )


def test_same_opening_semantic_rescue_is_revoked():
    alt = take(
        "alt", 10, 19,
        "Nunca se nos ocurrió hacer un chequeo de sonografía de la tiroides porque cada año hacía dos estudios.",
    )
    winner = take(
        "winner", 20, 31,
        "Nunca se nos ocurrió hacer un chequeo de la tiroides por sonografía porque siempre los exámenes salían bien.",
    )
    diagnostics = ({
        "hybrid_semantic_complementary_rescue": [{
            "clip_id": "alt",
            "winner_clip_id": "winner",
            "semantic_confidence": 0.8,
            "alternate_coverage_by_winner": 0.5,
            "unique_content_tokens": ["cada", "año", "dos", "estudios"],
            "unique_fraction": 0.4,
        }]
    },)
    kept, revoked, normalized = reconcile_semantic_rescues(
        (alt, winner), (alt, winner), diagnostics
    )
    assert [t.clip_id for t in kept] == ["winner"]
    assert revoked and revoked[0]["clip_id"] == "alt"
    assert normalized == ()


def test_complementary_semantic_rescue_is_normalized_for_composite():
    alt = take(
        "alt", 30, 39,
        "Otro síntoma era que me salían espinillas detrás de la oreja y el cuello y me salía por temporadas.",
    )
    winner = take(
        "winner", 15, 28,
        "También me salían espinillas detrás de la oreja y todo el cuello y pensaba que era alergia hormonal.",
    )
    diagnostics = ({
        "hybrid_semantic_complementary_rescue": [{
            "clip_id": "alt",
            "winner_clip_id": "winner",
            "semantic_confidence": 0.8,
            "alternate_coverage_by_winner": 0.64,
            "unique_content_tokens": ["otro", "síntoma", "temporadas"],
            "unique_fraction": 0.35,
        }]
    },)
    kept, revoked, normalized = reconcile_semantic_rescues(
        (winner, alt), (winner, alt), diagnostics
    )
    assert [t.clip_id for t in kept] == ["winner", "alt"]
    assert revoked == ()
    assert normalized and normalized[0]["peer_clip_id"] == "winner"
