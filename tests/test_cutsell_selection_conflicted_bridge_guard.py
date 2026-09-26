from cutsell_worker.contracts import DraftClip
from cutsell_worker.selection_conflicted_bridge_guard import (
    confirmed_selected_duplicate_ids,
    conflicted_redundant_bridge_ids,
    terminally_incomplete_selected_ids,
)


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


def test_raw105_near_tied_winner_and_alternate_bridge_moves_to_swap():
    """Regression for the exact structural evidence observed in RAW #105."""
    left = _clip(
        "left",
        95.68,
        107.50,
        "Al terminar mi contrato cambié de ginecóloga y le pedí que me hiciera un test de todo lo que ella se pudiera imaginar y me pudiese indicar. Ahí me mandó a hacer sonografías.",
    )
    bridge = _clip(
        "bridge",
        108.70,
        112.42,
        "Ahí fue cuando me mandaron a hacer sonografías de tiroides y otros.",
    )
    right = _clip(
        "right",
        119.95,
        124.51,
        "a hacer sonografía de tiroides y otras sonografías.",
    )
    diagnostics = {
        "hybrid_editorial_chunks": [
            {"decisions": [
                {"clip_id": "left", "label": "keep", "confidence": 0.92},
                {"clip_id": "bridge", "label": "alternate", "confidence": 0.88},
            ]},
            {"decisions": [
                {"clip_id": "left", "label": "winner", "confidence": 0.95},
                {"clip_id": "bridge", "label": "winner", "confidence": 0.90},
                {"clip_id": "right", "label": "failed", "confidence": 0.85},
            ]},
            {"decisions": [
                {"clip_id": "right", "label": "alternate", "confidence": 0.85},
            ]},
        ]
    }

    move, audit = conflicted_redundant_bridge_ids((left, bridge, right), diagnostics)

    assert move == {"bridge"}
    assert audit[0]["alternate_confidence"] == 0.88
    assert audit[0]["keep_confidence"] == 0.90
    assert audit[0]["keep_margin"] == 0.02
    assert audit[0]["thematic_union_coverage"] == 1.0
    assert audit[0]["left_gap_sec"] == 1.2
    assert audit[0]["right_gap_sec"] == 7.53


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


def test_very_strong_keep_with_clear_margin_wins_conflict_and_fails_open():
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


def test_directly_confirmed_selected_duplicate_keeps_semantic_winner():
    earlier = _clip("earlier", 198.8, 210.3, "También me salían espinillas detrás de la oreja y el cuello.")
    later = _clip("later", 213.6, 221.7, "Otro síntoma eran espinillas detrás de la oreja y en el cuello.")
    diagnostics = {
        "semantic_idea_equivalence": {"merges": [{
            "left_clip_id": "earlier", "right_clip_id": "later",
            "confidence": 0.90, "reason": "same symptom restated",
        }]},
        "hybrid_editorial_chunks": [
            {"decisions": [{"clip_id": "earlier", "label": "alternate", "confidence": 0.85}]},
            {"decisions": [{"clip_id": "later", "label": "winner", "confidence": 0.95}]},
        ],
    }
    move, audit = confirmed_selected_duplicate_ids((earlier, later), diagnostics)
    assert move == {"earlier"}
    assert audit[0]["winner_clip_id"] == "later"


def test_confirmed_duplicate_with_unique_negation_fails_open():
    earlier = _clip("earlier", 1.0, 4.0, "No tuve ese síntoma.")
    later = _clip("later", 5.0, 8.0, "Tuve ese síntoma.")
    diagnostics = {
        "semantic_idea_equivalence": {"merges": [{
            "left_clip_id": "earlier", "right_clip_id": "later", "confidence": 0.90,
        }]},
        "hybrid_editorial_chunks": [{"decisions": [
            {"clip_id": "earlier", "label": "alternate", "confidence": 0.85},
            {"clip_id": "later", "label": "winner", "confidence": 0.95},
        ]}],
    }
    move, audit = confirmed_selected_duplicate_ids((earlier, later), diagnostics)
    assert move == set()
    assert audit == []


def test_short_terminally_incomplete_singleton_is_removed():
    fragment = _clip("fragment", 241.75, 243.254, "me diagnosticaron con...")
    diagnostics = {"attempt_reconstruction": {"attempts": [{
        "clip_id": "fragment", "complete_idea": False, "duration_sec": 1.504,
    }]}}
    move, audit = terminally_incomplete_selected_ids((fragment,), diagnostics)
    assert move == {"fragment"}
    assert audit[0]["reason"] == "short_terminally_incomplete_attempt"


def test_complete_or_long_open_delivery_fails_open():
    complete = _clip("complete", 1.0, 2.5, "Esta idea está completa.")
    long_open = _clip("long", 3.0, 8.0, "Esta explicación todavía continúa...")
    diagnostics = {"attempt_reconstruction": {"attempts": [
        {"clip_id": "complete", "complete_idea": True, "duration_sec": 1.5},
        {"clip_id": "long", "complete_idea": False, "duration_sec": 5.0},
    ]}}
    move, audit = terminally_incomplete_selected_ids((complete, long_open), diagnostics)
    assert move == set()
    assert audit == []
