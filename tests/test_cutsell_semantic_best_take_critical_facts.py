from cutsell_worker.contracts import CandidateTake
from cutsell_worker.semantic_best_take_integrity import (
    _prefer_complete_peer_with_preserved_critical_facts,
)


def _take(clip_id, text, *, complete=True):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=0.0,
        end=5.0,
        text=text,
        complete_idea=complete,
    )


def test_complete_peer_with_unique_numeric_fact_beats_shorter_semantic_winner():
    complete_peer = _take(
        "long",
        "Soy la única en mi familia con este cáncer. Solo un 5-10% son hereditarios y nuestras elecciones de vida importan.",
    )
    shorter_winner = _take(
        "short",
        "Los cánceres son hereditarios. Soy la única en mi familia con este tipo de cáncer.",
    )
    semantic = {
        "long": ("alternate", 0.80),
        "short": ("winner", 0.97),
    }

    assert _prefer_complete_peer_with_preserved_critical_facts(
        (complete_peer, shorter_winner), semantic, "short"
    ) == "long"


def test_no_override_when_selected_winner_preserves_numeric_fact():
    peer = _take(
        "peer",
        "Solo un 5-10% son hereditarios y nuestras elecciones de vida importan.",
    )
    winner = _take(
        "winner",
        "Solo un 5-10% son hereditarios; nuestras elecciones de vida importan.",
    )
    semantic = {
        "peer": ("alternate", 0.80),
        "winner": ("winner", 0.97),
    }

    assert _prefer_complete_peer_with_preserved_critical_facts(
        (peer, winner), semantic, "winner"
    ) is None


def test_incomplete_peer_cannot_override_winner_even_with_numeric_fact():
    incomplete = _take("partial", "Solo un 5-10% de los", complete=False)
    winner = _take("winner", "Soy la única en mi familia con este tipo de cáncer.")
    semantic = {
        "partial": ("alternate", 0.90),
        "winner": ("winner", 0.97),
    }

    assert _prefer_complete_peer_with_preserved_critical_facts(
        (incomplete, winner), semantic, "winner"
    ) is None
