from cutsell_worker.contracts import DraftClip
from cutsell_worker.post_selection_composite_handoff_trim import suffix_handoff_relation


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


def test_last_fragment_can_yield_to_later_selected_complementary_delivery():
    prefix = [
        clip("long", 10.0, 11.0, "resorcina."),
        clip("long", 11.5, 14.5, "También me salían espinillas. Era como un rush, una alergia."),
    ]
    suffix = clip(
        "long", 15.0, 27.0,
        "También me salían espinillas detrás de la oreja y todo el cuello y pensaba que era una alergia por problemas hormonales.",
    )
    later = clip(
        "later", 29.0, 38.0,
        "Otro síntoma era que me salían espinillas como una alergia detrás de la oreja y en el cuello. Me salía por temporadas.",
    )
    relation = suffix_handoff_relation(suffix, later, prefix)
    assert relation is not None
    assert relation["later_clip_id"] == "later"
    assert "resorcina" in relation["preserved_prefix_unique_content_tokens"]


def test_does_not_remove_only_fragment_of_a_take():
    suffix = clip("long", 15.0, 27.0, "Me salían espinillas detrás de la oreja y el cuello como alergia.")
    later = clip("later", 29.0, 38.0, "Me salían espinillas detrás de la oreja y el cuello como alergia por temporadas.")
    assert suffix_handoff_relation(suffix, later, ()) is None


def test_does_not_remove_suffix_when_it_has_critical_number_missing_later():
    prefix = [clip("long", 10, 14, "Primero tuve otro síntoma único.")]
    suffix = clip("long", 15, 27, "El riesgo de esta parte era 5 a 10% y parecía una alergia detrás de la oreja.")
    later = clip("later", 29, 38, "Parecía una alergia detrás de la oreja y el cuello por temporadas con otros síntomas.")
    assert suffix_handoff_relation(suffix, later, prefix) is None


def test_does_not_remove_suffix_for_weak_topic_overlap():
    prefix = [clip("long", 10, 14, "Primero tuve otro síntoma único.")]
    suffix = clip("long", 15, 27, "Me salían espinillas detrás de la oreja y el cuello como alergia.")
    later = clip("later", 29, 38, "Tuve problemas de digestión y gastritis y me dieron pastillas por tres meses.")
    assert suffix_handoff_relation(suffix, later, prefix) is None
