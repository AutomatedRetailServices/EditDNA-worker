from cutsell_worker.contracts import DraftClip
from cutsell_worker.post_selection_complementary_family_stabilizer import complementary_family_swaps


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


def test_concise_complement_plus_later_winner_replaces_redundant_monolith():
    concise = clip("concise", 10.0, 15.0, "También me salían espinillas. Era como un rash, una alergia.")
    monolith = clip(
        "monolith",
        16.0,
        28.0,
        "También me salían espinillas detrás de la oreja y por todo el cuello; pensaba que era alergia y parecían problemas hormonales.",
    )
    later = clip(
        "later",
        29.0,
        37.0,
        "Otro síntoma era que me salían espinillas como una alergia detrás de la oreja y en el cuello por temporadas.",
    )
    swaps = complementary_family_swaps((monolith, later), (concise,))
    assert len(swaps) == 1
    assert swaps[0]["restore"].clip_id == "concise"
    assert swaps[0]["suppress"].clip_id == "monolith"
    assert swaps[0]["later"].clip_id == "later"


def test_no_swap_when_discarded_clip_has_no_unique_information():
    concise = clip("concise", 10.0, 15.0, "Me salían espinillas detrás de la oreja y en el cuello.")
    monolith = clip(
        "monolith",
        16.0,
        28.0,
        "También me salían espinillas detrás de la oreja y por todo el cuello; pensaba que era alergia.",
    )
    later = clip(
        "later",
        29.0,
        37.0,
        "Otro síntoma era que me salían espinillas como una alergia detrás de la oreja y en el cuello.",
    )
    assert complementary_family_swaps((monolith, later), (concise,)) == []


def test_no_swap_when_selected_deliveries_are_not_redundant():
    concise = clip("concise", 10.0, 15.0, "Me mareaba al levantarme y sentía presión en la cabeza.")
    monolith = clip("monolith", 16.0, 28.0, "Me mareaba al levantarme y sentía presión en la cabeza casi todos los días.")
    later = clip("later", 29.0, 37.0, "Después tuve dolor de estómago y me hicieron una endoscopía.")
    assert complementary_family_swaps((monolith, later), (concise,)) == []


def test_no_swap_when_concise_candidate_is_too_far_away():
    concise = clip("concise", 1.0, 5.0, "También me salían espinillas. Era como un rash, una alergia.")
    monolith = clip(
        "monolith",
        20.0,
        32.0,
        "También me salían espinillas detrás de la oreja y por todo el cuello; pensaba que era alergia y parecían problemas hormonales.",
    )
    later = clip(
        "later",
        33.0,
        41.0,
        "Otro síntoma era que me salían espinillas como una alergia detrás de la oreja y en el cuello por temporadas.",
    )
    assert complementary_family_swaps((monolith, later), (concise,)) == []
