from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SemanticRole
from cutsell_worker.final_selection_retry_arbiter import apply_final_selection_retry_arbiter


def clip(cid, start, end, text):
    return DraftClip(
        clip_id=cid,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
        caption_text=text,
        semantic_role=SemanticRole.OTHER,
    )


def test_physical_failed_alternate_yields_to_later_same_opening_winner():
    bad = clip("bad", 25.6, 34.6, "Nunca se nos ocurrió hacer un chequeo de sonografía de la tiroides, pues porque cada año que me hacía mínimo dos estados.")
    good = clip("good", 35.46, 46.42, "Nunca se nos ocurrió hacer un chequeo de la tiroides por sonografía porque siempre en mis exámenes la tiroides salía como que estaba funcionando perfectamente.")
    draft = DraftTimeline(
        schema_version="cutsell.v1",
        project_id="p",
        strategy=EditStrategy.STORYTELLING,
        selected=(bad, good),
        alternates=(),
        discarded=(),
        diagnostics={
            "hybrid_editorial_chunks": [
                {"decisions": [
                    {"clip_id": "bad", "label": "alternate", "confidence": 0.70, "local_failure_corroborated": True, "local_failure_reasons": ["dense_physical_reset:5"]},
                    {"clip_id": "good", "label": "winner", "confidence": 0.95, "local_failure_corroborated": False, "local_failure_reasons": []},
                ]}
            ]
        },
    )

    repaired = apply_final_selection_retry_arbiter(draft)
    assert [c.clip_id for c in repaired.selected] == ["good"]
    assert any(c.clip_id == "bad" for c in repaired.discarded)


def test_complementary_delivery_without_same_opening_is_not_removed():
    concise = clip("concise", 192.4, 197.98, "También me salían espinillas. Era como un rush, una alergia.")
    later = clip("later", 213.38, 222.74, "Otro síntoma era que me salían espinillas como si fuera una alergia de esta parte aquí detrás de la oreja y en el cuello. Me salía por temporadas.")
    draft = DraftTimeline(
        schema_version="cutsell.v1",
        project_id="p",
        strategy=EditStrategy.STORYTELLING,
        selected=(concise, later),
        alternates=(),
        discarded=(),
        diagnostics={
            "hybrid_editorial_chunks": [
                {"decisions": [
                    {"clip_id": "concise", "label": "alternate", "confidence": 0.85, "local_failure_corroborated": True, "local_failure_reasons": ["dense_physical_reset:7", "visual_fumble:0.85"]},
                    {"clip_id": "later", "label": "winner", "confidence": 0.98, "local_failure_corroborated": True, "local_failure_reasons": ["dense_physical_reset:6"]},
                ]}
            ]
        },
    )

    repaired = apply_final_selection_retry_arbiter(draft)
    assert [c.clip_id for c in repaired.selected] == ["concise", "later"]
