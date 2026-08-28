from cutsell_worker.contracts import DraftClip
from cutsell_worker.selection_phase_authority import (
    authoritative_failed_retry_ids,
    redundant_alternate_bridge_ids,
    short_alternate_before_fuller_delivery_ids,
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


def test_authoritatively_covered_failed_short_retry_yields_to_later_winner():
    failed = _clip("failed", 10.0, 12.0, "Tuve problemas de estómago.")
    winner = _clip(
        "winner",
        17.0,
        27.0,
        "Tuve problemas de digestión y me hicieron una endoscopía; tenía gastritis.",
    )
    diagnostics = {
        "hybrid_editorial_chunks": [
            {
                "decisions": [
                    {"clip_id": "failed", "label": "failed", "confidence": 0.85},
                    {"clip_id": "winner", "label": "winner", "confidence": 0.95},
                ],
                "hybrid_cross_group_retry_integrity": [
                    {
                        "clip_id": "failed",
                        "reason": "cross_group_semantic_retry_covered_by_authoritative_delivery",
                        "semantic_label": "failed",
                        "semantic_confidence": 0.85,
                        "coverage": 0.6667,
                        "content_token_count": 3,
                        "shared_union": 2,
                        "critical_preserved": True,
                        "strongest_peer_clip_id": "winner",
                    }
                ],
            }
        ]
    }

    remove, audit = authoritative_failed_retry_ids((failed, winner), diagnostics)

    assert "failed" in remove
    assert audit[0]["winner_clip_id"] == "winner"


def test_failed_retry_is_preserved_when_authoritative_coverage_is_not_proven():
    failed = _clip("failed", 10.0, 12.0, "Tuve un síntoma único llamado vértigo.")
    winner = _clip("winner", 17.0, 27.0, "Tuve problemas de digestión y gastritis.")
    diagnostics = {
        "hybrid_editorial_chunks": [{
            "decisions": [
                {"clip_id": "failed", "label": "failed", "confidence": 0.85},
                {"clip_id": "winner", "label": "winner", "confidence": 0.95},
            ]
        }]
    }

    remove, audit = authoritative_failed_retry_ids((failed, winner), diagnostics)

    assert remove == set()
    assert audit == []


def test_consensus_alternate_bridge_yields_to_neighboring_selected_deliveries():
    left = _clip(
        "left",
        90.0,
        100.0,
        "Cambié de doctora y ahí me mandó a hacer sonografías.",
    )
    bridge = _clip(
        "bridge",
        101.0,
        104.5,
        "Ahí fue cuando me mandaron a hacer sonografías de tiroides y otras.",
    )
    right = _clip(
        "right",
        111.0,
        115.0,
        "a hacer sonografía de tiroides y otras sonografías.",
    )
    diagnostics = {
        "hybrid_editorial_chunks": [
            {"decisions": [
                {"clip_id": "left", "label": "winner", "confidence": 0.96},
                {"clip_id": "bridge", "label": "alternate", "confidence": 0.80},
            ]},
            {"decisions": [
                {"clip_id": "bridge", "label": "alternate", "confidence": 0.75},
                {"clip_id": "right", "label": "alternate", "confidence": 0.80},
                {"clip_id": "right", "label": "failed", "confidence": 0.88},
            ]},
        ]
    }

    remove, audit = redundant_alternate_bridge_ids((left, bridge, right), diagnostics)

    assert "bridge" in remove
    assert audit[0]["left_clip_id"] == "left"
    assert audit[0]["right_clip_id"] == "right"


def test_transition_language_does_not_make_consensus_bridge_unique():
    left = _clip("left", 95.68, 104.30, "Ahí me mandó a hacer sonografías.")
    bridge = _clip(
        "bridge",
        108.70,
        112.42,
        "Ahí fue cuando me mandaron a hacer sonografías de tiroides y otros.",
    )
    right = _clip("right", 119.95, 124.51, "Me mandaron sonografías de tiroides y otros estudios.")
    diagnostics = {
        "hybrid_editorial_chunks": [
            {"decisions": [
                {"clip_id": "left", "label": "winner", "confidence": 0.96},
                {"clip_id": "bridge", "label": "alternate", "confidence": 0.80},
            ]},
            {"decisions": [
                {"clip_id": "bridge", "label": "alternate", "confidence": 0.75},
                {"clip_id": "right", "label": "keep", "confidence": 0.90},
            ]},
        ]
    }

    remove, audit = redundant_alternate_bridge_ids((left, bridge, right), diagnostics)

    assert remove == {"bridge"}
    assert audit[0]["thematic_union_coverage"] >= 0.8


def test_single_vote_or_unique_alternate_bridge_fails_open():
    left = _clip("left", 90.0, 100.0, "Ahí me mandó a hacer sonografías.")
    bridge = _clip("bridge", 101.0, 104.5, "También apareció un nódulo de tres centímetros.")
    right = _clip("right", 111.0, 115.0, "a hacer sonografía de tiroides.")
    diagnostics = {
        "hybrid_editorial_chunks": [{"decisions": [
            {"clip_id": "left", "label": "winner", "confidence": 0.96},
            {"clip_id": "bridge", "label": "alternate", "confidence": 0.80},
            {"clip_id": "right", "label": "keep", "confidence": 0.90},
        ]}]
    }

    remove, audit = redundant_alternate_bridge_ids((left, bridge, right), diagnostics)

    assert remove == set()
    assert audit == []


def test_short_alternate_before_much_fuller_delivery_moves_to_swap():
    short = _clip("short", 251.39, 253.37, "Tuve problemas de estómago.")
    full = _clip(
        "full",
        258.77,
        268.51,
        "Tuve problemas de digestión; me hicieron una endoscopía, tenía gastritis y me mandaron tratamiento.",
    )
    diagnostics = {
        "hybrid_editorial_chunks": [{"decisions": [
            {"clip_id": "short", "label": "alternate", "confidence": 0.70},
            {"clip_id": "full", "label": "keep", "confidence": 0.90},
        ]}]
    }

    remove, audit = short_alternate_before_fuller_delivery_ids((short, full), diagnostics)

    assert remove == {"short"}
    assert audit[0]["fuller_clip_id"] == "full"


def test_short_alternate_with_protected_or_unique_fact_fails_open():
    short = _clip("short", 10.0, 12.0, "No tuve fiebre de 40 grados.")
    full = _clip("full", 16.0, 25.0, "Tuve dolor de cabeza y cansancio durante varios días.")
    diagnostics = {
        "hybrid_editorial_chunks": [{"decisions": [
            {"clip_id": "short", "label": "alternate", "confidence": 0.75},
            {"clip_id": "full", "label": "keep", "confidence": 0.95},
        ]}]
    }

    remove, audit = short_alternate_before_fuller_delivery_ids((short, full), diagnostics)

    assert remove == set()
    assert audit == []
