from cutsell_worker.contracts import CandidateTake, MediaSignals, Word
from cutsell_worker.hybrid_cross_group_retry_integrity import collapse_cross_group_semantic_retries


def _words(text, start=0.0, step=0.2):
    cursor = float(start)
    out = []
    for token in text.split():
        out.append(Word(token, cursor, cursor + step, 0.95))
        cursor += step
    return tuple(out)


def _take(clip_id, start, end, text, *, complete=True):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=float(start),
        end=float(end),
        text=text,
        words=_words(text, start),
        signals=MediaSignals("src", float(start), float(end)),
        complete_idea=complete,
    )


def test_video02_unique_story_is_not_deleted_by_diffuse_peers_on_both_sides():
    before = _take(
        "before",
        55.38,
        65.02,
        "Like this was such a fun video to make because I'm bringing the pizza and give me the money",
    )
    story = _take(
        "story",
        66.16,
        76.40,
        "You know the thing is that this video is like so sexy so funny So I think it was just as fun to make and to do and the action and everything that",
        complete=False,
    )
    after = _take(
        "after",
        78.12,
        83.84,
        "It really really I look back to and I just laugh about it because it's such a fun video that we made",
    )
    distant = _take(
        "distant",
        10.0,
        35.0,
        "We made a lot of videos and tried different skits because we did not want to make the same video over and over",
    )

    kept, removed, diagnostics = collapse_cross_group_semantic_retries(
        (before, story, after, distant),
        (
            ("before", "winner", 0.92),
            ("story", "failed", 0.78),
            ("after", "keep", 0.90),
            ("distant", "keep", 0.90),
        ),
    )

    assert "story" in {take.clip_id for take in kept}
    assert "story" not in {take.clip_id for take in removed}
    assert not any(item.get("clip_id") == "story" for item in diagnostics)


def test_video00_loser_can_be_replaced_by_same_side_winner_plus_continuation_chain():
    loser = _take(
        "loser",
        295.36,
        314.60,
        "Esta es mi experiencia soy la única en mi familia que tiene este tipo de cáncer por eso no creo y está comprobado científicamente que los cánceres son hereditarios más bien solo un 5-10% son de carácter hereditario mayormente son nuestras elecciones de vida así que cuídate",
    )
    winner = _take(
        "winner",
        319.38,
        334.24,
        "Soy la primera en mi familia con este tipo de cáncer nadie en mi familia tiene un carcinoma papilar en la tiroides ni sufre de la tiroides así que estoy convencida y la ciencia lo avala que solo un 5-10% de los",
        complete=False,
    )
    continuation = _take(
        "continuation",
        335.88,
        346.54,
        "cánceres son hereditarios soy la única que tiene este tipo de cáncer",
    )

    kept, removed, diagnostics = collapse_cross_group_semantic_retries(
        (loser, winner, continuation),
        (
            ("loser", "alternate", 0.85),
            ("winner", "winner", 0.98),
            ("continuation", "keep", 0.92),
        ),
    )

    assert "loser" not in {take.clip_id for take in kept}
    assert "loser" in {take.clip_id for take in removed}
    assert any(
        item.get("clip_id") == "loser" and item.get("coverage_mode") == "same_side_contiguous_chain"
        for item in diagnostics
    )
