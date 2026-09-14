from cutsell_worker.contracts import CandidateTake
from cutsell_worker.incomplete_unique_bridge_completion_rescue import bridge_completion_relation


def take(clip_id, start, end, text, *, complete):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
        complete_idea=complete,
    )


def test_unique_incomplete_bridge_can_join_short_following_completion():
    incomplete = take(
        "incomplete", 10.0, 15.0,
        "Por temporada me salía acné en la espalda y yo lo resolvía con",
        complete=False,
    )
    following = take(
        "following", 16.0, 25.0,
        "resorcina. También me salían espinillas detrás de la oreja y en el cuello.",
        complete=True,
    )
    relation = bridge_completion_relation(incomplete, following)
    assert relation is not None
    assert relation["completion_tokens"] == ["resorcina"]
    assert "espalda" in relation["unique_content_tokens"]


def test_does_not_rescue_when_following_starts_with_full_sentence_not_short_completion():
    incomplete = take("incomplete", 10, 15, "Yo resolvía el problema con", complete=False)
    following = take(
        "following", 16, 25,
        "Después comencé otro tratamiento completamente diferente para el problema.",
        complete=True,
    )
    assert bridge_completion_relation(incomplete, following) is None


def test_does_not_rescue_when_incomplete_does_not_end_on_bridge_word():
    incomplete = take("incomplete", 10, 15, "Me salía acné en la espalda por temporadas", complete=False)
    following = take("following", 16, 25, "resorcina. Después tuve otro síntoma.", complete=True)
    assert bridge_completion_relation(incomplete, following) is None


def test_does_not_rescue_redundant_incomplete_fragment():
    incomplete = take("incomplete", 10, 15, "Tenía espinillas en el cuello con", complete=False)
    following = take(
        "following", 16, 25,
        "espinillas. Tenía espinillas en el cuello con una alergia por temporadas.",
        complete=True,
    )
    assert bridge_completion_relation(incomplete, following) is None
