from cutsell_worker.contracts import DraftClip, SemanticRole
from cutsell_worker.round9_orphan_prefix_integrity import (
    suppress_orphan_failed_open_prefix_v2,
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


def _diagnostics(*, cross_group=True):
    chunk = {
        "decisions": [
            {"clip_id": "open", "label": "failed", "confidence": 0.90},
            {"clip_id": "continuation", "label": "alternate", "confidence": 0.80},
            {"clip_id": "prior", "label": "winner", "confidence": 0.95},
        ]
    }
    if cross_group:
        chunk["hybrid_cross_group_retry_integrity"] = [
            {
                "clip_id": "continuation",
                "content_token_count": 9,
                "coverage": 1.0,
                "coverage_mode": "single_authoritative_peer",
                "critical_preserved": True,
                "critical_tokens": ["num:5", "num:10"],
                "peer_clip_ids": ["prior"],
                "reason": "cross_group_semantic_retry_covered_by_authoritative_delivery",
                "semantic_confidence": 0.80,
                "semantic_label": "alternate",
                "shared_union": 9,
                "strongest_peer_clip_id": "prior",
                "strongest_peer_coverage": 1.0,
                "strongest_shared": 9,
            }
        ]
    return {"hybrid_editorial_chunks": [chunk]}


def test_round9_orphan_319_prefix_uses_proven_cross_group_continuation_beyond_short_gap():
    """Exact Round 9 regression: continuation moved to 340.38, 6.14 s after open prefix."""
    prior = _clip(
        "prior",
        295.52,
        313.82,
        (
            "Esta es mi experiencia. Soy la única en mi familia que tiene este tipo de cáncer. "
            "Por eso no creo y está comprobado científicamente que los cánceres son hereditarios. "
            "Más bien solo un 5-10% son de carácter hereditario. Mayormente son nuestras elecciones de vida. "
            "Así que cuídate."
        ),
    )
    open_prefix = _clip(
        "open",
        319.38,
        334.24,
        (
            "Soy la primera en mi familia con este tipo de cáncer. Nadie en mi familia tiene un carcinoma papilar "
            "en la tiroides ni sufre de la tiroides. Así que estoy convencida y la ciencia lo avala que solo un 5-10% de los"
        ),
    )
    continuation = _clip(
        "continuation",
        340.38,
        346.76,
        "cánceres son hereditarios. Soy la única en mi familia que tiene este tipo de cáncer.",
        selected=False,
    )

    selected, discarded, audit = suppress_orphan_failed_open_prefix_v2(
        (prior, open_prefix),
        (continuation,),
        _diagnostics(cross_group=True),
    )

    assert tuple(clip.clip_id for clip in selected) == ("prior",)
    assert tuple(clip.clip_id for clip in discarded) == ("continuation", "open")
    assert audit[0]["reason"] == "orphan_failed_open_prefix_yields_via_proven_cross_group_continuation"
    assert audit[0]["prior_winner_clip_id"] == "prior"
    assert audit[0]["discarded_continuation_clip_id"] == "continuation"
    assert audit[0]["continuation_gap_sec"] == 6.14


def test_round9_longer_gap_without_cross_group_authority_fails_open():
    prior = _clip(
        "prior",
        10.0,
        20.0,
        "Soy la única en mi familia y solo un 5-10% de los cánceres son hereditarios. Así que cuídate.",
    )
    prefix = _clip(
        "open",
        25.0,
        35.0,
        "Soy la primera en mi familia y la ciencia lo avala que solo un 5-10% de los",
    )
    continuation = _clip(
        "continuation",
        41.0,
        46.0,
        "cánceres son hereditarios y soy la única en mi familia.",
        selected=False,
    )

    selected, discarded, audit = suppress_orphan_failed_open_prefix_v2(
        (prior, prefix),
        (continuation,),
        _diagnostics(cross_group=False),
    )

    assert tuple(clip.clip_id for clip in selected) == ("prior", "open")
    assert tuple(clip.clip_id for clip in discarded) == ("continuation",)
    assert audit == ()


def test_round9_cross_group_authority_for_different_peer_does_not_delete_prefix():
    prior = _clip(
        "prior",
        10.0,
        20.0,
        "Soy la única en mi familia y solo un 5-10% de los cánceres son hereditarios. Así que cuídate.",
    )
    prefix = _clip(
        "open",
        25.0,
        35.0,
        "Soy la primera en mi familia y la ciencia lo avala que solo un 5-10% de los",
    )
    continuation = _clip(
        "continuation",
        41.0,
        46.0,
        "cánceres son hereditarios y soy la única en mi familia.",
        selected=False,
    )
    diagnostics = _diagnostics(cross_group=True)
    diagnostics["hybrid_editorial_chunks"][0]["hybrid_cross_group_retry_integrity"][0]["strongest_peer_clip_id"] = "other"
    diagnostics["hybrid_editorial_chunks"][0]["hybrid_cross_group_retry_integrity"][0]["peer_clip_ids"] = ["other"]

    selected, discarded, audit = suppress_orphan_failed_open_prefix_v2(
        (prior, prefix),
        (continuation,),
        diagnostics,
    )

    assert tuple(clip.clip_id for clip in selected) == ("prior", "open")
    assert tuple(clip.clip_id for clip in discarded) == ("continuation",)
    assert audit == ()
