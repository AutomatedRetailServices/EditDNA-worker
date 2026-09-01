from cutsell_worker.contracts import CandidateTake
from cutsell_worker.cross_group_truncated_winner_authority import truncated_retry_relation


def take(clip_id, start, end, text):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
    )


def test_later_complete_retry_can_replace_visibly_truncated_delivery():
    earlier = take(
        "earlier", 10.0, 20.0,
        "Tuve problemas estomacales en donde me hicieron una endoscopía y me diagnosticaron con...",
    )
    later = take(
        "later", 32.0, 43.0,
        "Tuve problemas de digestión en donde me hicieron una endoscopía y dijeron que tenía gastritis. Nada severo, pero tenía gastritis y me mandaron tres meses con pastillas.",
    )
    relation = truncated_retry_relation(earlier, later)
    assert relation is not None
    assert relation["later_retry_clip_id"] == "later"
    assert "gastritis" in relation["later_unique_content_tokens"]


def test_complete_sentence_does_not_yield_merely_because_later_take_is_richer():
    earlier = take(
        "earlier", 10.0, 20.0,
        "Tuve problemas estomacales y me hicieron una endoscopía.",
    )
    later = take(
        "later", 32.0, 43.0,
        "Tuve problemas estomacales y me hicieron una endoscopía donde diagnosticaron gastritis y me dieron pastillas.",
    )
    assert truncated_retry_relation(earlier, later) is None


def test_truncated_delivery_keeps_unique_critical_number():
    earlier = take(
        "earlier", 10.0, 20.0,
        "El estudio mostró un riesgo de 5 a 10% y me diagnosticaron con...",
    )
    later = take(
        "later", 32.0, 43.0,
        "El estudio mostró riesgo y luego me diagnosticaron gastritis con una evaluación completa y pastillas.",
    )
    assert truncated_retry_relation(earlier, later) is None


def test_topic_similarity_without_strong_coverage_does_not_delete():
    earlier = take("earlier", 10.0, 20.0, "Tuve problemas de estómago y me diagnosticaron con...")
    later = take("later", 32.0, 43.0, "Después perdía mucho pelo por el estrés y tenía síntomas hormonales por temporadas.")
    assert truncated_retry_relation(earlier, later) is None
