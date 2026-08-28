from cutsell_worker.contracts import DraftClip
from cutsell_worker.selection_conflicted_bridge_guard import conflicted_redundant_bridge_ids


def _clip(clip_id, start, end, text):
    return DraftClip(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
        caption_text=text,
    )


def test_conflicted_redundant_bridge_moves_to_swap_candidate():
    left = _clip("left", 100.0, 104.3, "Me mandaron a hacer sonografías de tiroides.")
    bridge = _clip(
        "bridge",
        108.70,
        112.42,
        "Ahí fue cuando me mandaron a hacer sonografías de tiroides y otros estudios.",
    )
    right = _clip(
        "right",
        119.95,
        124.51,
        "Me mandaron sonografías de tiroides y otros estudios.",
    )
    diagnostics = {
        "hybrid_editorial_chunks": [
            {"decisions": [
                {"clip_id": "left", "label": "winner", "confidence": 0.96},
                {"clip_id": "bridge", "label": "keep", "confidence": 0.85},
            ]},
            {"decisions": [
                {"clip_id": "bridge", "label": "alternate", "confidence": 0.80},
                {"clip_id": "right", "label": "keep", "confidence": 0.90},
            ]},
        ]
    }

    move, audit = conflicted_redundant_bridge_ids((left, bridge, right), diagnostics)

    assert move == {"bridge"}
    assert audit[0]["keep_confidence"] == 0.85
    assert audit[0]["alternate_confidence"] == 0.80
    assert audit[0]["thematic_union_coverage"] >= 0.80


def test_conflicted_bridge_with_unique_critical_fact_fails_open():
    left = _clip("left", 100.0, 104.3, "Me mandaron a hacer sonografías de tiroides.")
    bridge = _clip(
        "bridge",
        108.70,
        112.42,
        "No encontraron un nódulo de 3 centímetros en la tiroides.",
    )
    right = _clip("right", 119.95, 124.51, "Me hicieron otros estudios de tiroides.")
    diagnostics = {
        "hybrid_editorial_chunks": [
            {"decisions": [
                {"clip_id": "left", "label": "winner", "confidence": 0.96},
                {"clip_id": "bridge", "label": "keep", "confidence": 0.85},
            ]},
            {"decisions": [
                {"clip_id": "bridge", "label": "alternate", "confidence": 0.80},
                {"clip_id": "right", "label": "keep", "confidence": 0.90},
            ]},
        ]
    }

    move, audit = conflicted_redundant_bridge_ids((left, bridge, right), diagnostics)

    assert move == set()
    assert audit == []


def test_very_strong_keep_wins_conflict_and_fails_open():
    left = _clip("left", 100.0, 104.3, "Me mandaron a hacer sonografías de tiroides.")
    bridge = _clip(
        "bridge",
        108.70,
        112.42,
        "Me mandaron a hacer sonografías de tiroides y otros estudios.",
    )
    right = _clip("right", 119.95, 124.51, "Me mandaron sonografías de tiroides y otros estudios.")
    diagnostics = {
        "hybrid_editorial_chunks": [
            {"decisions": [
                {"clip_id": "left", "label": "winner", "confidence": 0.96},
                {"clip_id": "bridge", "label": "keep", "confidence": 0.93},
            ]},
            {"decisions": [
                {"clip_id": "bridge", "label": "alternate", "confidence": 0.80},
                {"clip_id": "right", "label": "keep", "confidence": 0.90},
            ]},
        ]
    }

    move, audit = conflicted_redundant_bridge_ids((left, bridge, right), diagnostics)

    assert move == set()
    assert audit == []
