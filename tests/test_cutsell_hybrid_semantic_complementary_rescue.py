from cutsell_worker.contracts import CandidateTake
from cutsell_worker.hybrid_semantic_complementary_rescue import complementary_relation


def take(clip_id, start, end, text, *, complete=True):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
        complete_idea=complete,
    )


def test_preserves_full_alternate_with_material_unique_information():
    winner = take(
        "winner", 10.0, 30.0,
        "También me salían espinillas en esta parte detrás de la oreja y todo el cuello que yo pensaba que era una alergia, pero eran espinillas de personas con problemas hormonales.",
    )
    alternate = take(
        "alternate", 31.0, 40.5,
        "Otro síntoma era que me salían espinillas como si fuera una alergia detrás de la oreja y en el cuello. Me salía por temporadas.",
    )
    semantic = {"winner": ("winner", 0.95), "alternate": ("alternate", 0.90)}
    relation = complementary_relation(alternate, winner, semantic)
    assert relation is not None
    assert relation["winner_clip_id"] == "winner"
    assert "temporadas" in relation["unique_content_tokens"]


def test_does_not_restore_nearly_redundant_alternate():
    winner = take(
        "winner", 10.0, 20.0,
        "Me salían espinillas detrás de la oreja y en el cuello y parecía una alergia por temporadas.",
    )
    alternate = take(
        "alternate", 21.0, 29.0,
        "Me salían espinillas detrás de la oreja y en el cuello y parecía una alergia por temporadas también.",
    )
    semantic = {"winner": ("winner", 0.95), "alternate": ("alternate", 0.90)}
    assert complementary_relation(alternate, winner, semantic) is None


def test_does_not_restore_low_confidence_alternate():
    winner = take("winner", 10.0, 20.0, "Me salían espinillas detrás de la oreja y en el cuello por temporadas.")
    alternate = take("alternate", 21.0, 30.0, "Otro síntoma eran espinillas detrás de la oreja y cuello con una alergia recurrente por temporadas.")
    semantic = {"winner": ("winner", 0.95), "alternate": ("alternate", 0.60)}
    assert complementary_relation(alternate, winner, semantic) is None


def test_does_not_restore_incomplete_alternate():
    winner = take("winner", 10.0, 20.0, "Me salían espinillas detrás de la oreja y en el cuello por temporadas.")
    alternate = take("alternate", 21.0, 30.0, "Otro síntoma eran espinillas detrás de la oreja y cuello con una alergia recurrente por temporadas.", complete=False)
    semantic = {"winner": ("winner", 0.95), "alternate": ("alternate", 0.90)}
    assert complementary_relation(alternate, winner, semantic) is None
