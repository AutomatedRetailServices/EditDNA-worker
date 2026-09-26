from cutsell_worker.contracts import CandidateTake, DraftClip, DraftTimeline, EditStrategy, MediaSignals, Word
from cutsell_worker.final_delivery_integrity import (
    collapse_overlapping_contained_deliveries,
)
from cutsell_worker.selection_conflicted_bridge_guard import apply_selection_conflicted_bridge_guard


def _take(clip_id, start, end, text, *, complete=True, source="src"):
    tokens = text.split()
    step = max(0.01, (end - start) / max(1, len(tokens)))
    words = tuple(
        Word(token, start + index * step, min(end, start + (index + 1) * step), 0.95)
        for index, token in enumerate(tokens)
    )
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id=source,
        source_order=0,
        start=start,
        end=end,
        text=text,
        words=words,
        signals=MediaSignals(source, start, end),
        complete_idea=complete,
    )


def test_run47_contained_intro_suffix_is_not_rendered_twice():
    full = _take(
        "full", 13.818, 22.488,
        "No es secreto para nadie que llevo años trabajando para cruceros. "
        "Tenía como costumbre terminar un contrato y hacerme un chequeo con mi ginecóloga.",
    )
    suffix = _take(
        "suffix", 17.750, 22.488,
        "Tenía como costumbre terminar un contrato y hacerme un chequeo con mi ginecóloga.",
    )
    kept, removed, rows = collapse_overlapping_contained_deliveries((full, suffix))
    assert [take.clip_id for take in kept] == ["full"]
    assert [take.clip_id for take in removed] == ["suffix"]
    assert rows[0]["reason"] == "overlapping_contained_delivery_yields_to_full_span"


def test_overlapping_distinct_claim_is_preserved():
    left = _take("left", 10.0, 18.0, "El producto tiene hierro y vitamina C para energía.")
    right = _take("right", 15.0, 20.0, "La vitamina C también ayuda al sistema inmune.")
    kept, removed, _ = collapse_overlapping_contained_deliveries((left, right))
    assert {take.clip_id for take in kept} == {"left", "right"}
    assert removed == ()


def _draft_clip(clip_id, start, end, text, *, selected=True, complete=True):
    return DraftClip(
        clip_id, "src", 0, start, end, text, text,
        selected=selected, complete_idea=complete,
    )


def test_run47_final_guard_resolves_cross_family_debris_without_video_specific_rules():
    selected = (
        _draft_clip("bad_sonography", 25.597, 32.226,
                    "Nunca se nos ocurrió hacer un chequeo de sonografía de la tiroides, pues porque cada año hacía mínimo dos estados."),
        _draft_clip("pimples_tail", 209.293, 210.739, "de personas con problemas hormonales."),
        _draft_clip("pimples_winner", 213.621, 221.896,
                    "Otro síntoma era que me salían espinillas como una alergia detrás de la oreja y en el cuello."),
        _draft_clip("stomach_open", 236.167, 243.227,
                    "Tuve problemas estomacales y me diagnosticaron con...", complete=False),
        _draft_clip("stomach_complete", 258.885, 268.385,
                    "Tuve problemas de digestión y dijeron que tenía gastritis."),
        _draft_clip("conclusion", 295.520, 305.166,
                    "Soy la única en mi familia con este cáncer. No creo que los cánceres sean hereditarios."),
        _draft_clip("conclusion_tail", 308.540, 313.078,
                    "carácter hereditario. Mayormente son nuestras elecciones de vida. Así que cuídate."),
        _draft_clip("family_aside", 319.782, 326.911,
                    "Nadie en mi familia tiene carcinoma papilar ni sufre de la tiroides."),
        _draft_clip("repeat_head", 327.654, 334.059,
                    "La ciencia avala que solo un 5-10% de los", complete=False),
        _draft_clip("repeat_tail", 340.462, 341.834, "cánceres son hereditarios."),
    )
    discarded = (
        _draft_clip("good_sonography", 35.641, 45.214,
                    "Nunca se nos ocurrió hacer un chequeo de la tiroides por sonografía porque siempre en mis exámenes salía perfectamente.",
                    selected=False),
        _draft_clip("pimples_proxy", 198.832, 210.315,
                    "También me salían espinillas detrás de la oreja y el cuello, de personas con problemas hormonales.",
                    selected=False),
        _draft_clip("percentage_bridge", 305.427, 306.993,
                    "Más bien solo un 5-10% son de", selected=False, complete=False),
    )
    attempts = [
        {"clip_id": clip.clip_id, "complete_idea": clip.complete_idea}
        for clip in (*selected, *discarded)
    ]
    decisions = [
        {"clip_id": "bad_sonography", "label": "winner", "confidence": 0.90},
        {"clip_id": "good_sonography", "label": "keep", "confidence": 0.95},
        {"clip_id": "stomach_open", "label": "winner", "confidence": 0.90},
        {"clip_id": "stomach_complete", "label": "winner", "confidence": 0.95},
        {"clip_id": "percentage_bridge", "label": "failed", "confidence": 0.80},
        {"clip_id": "percentage_bridge", "label": "keep", "confidence": 0.80},
    ]
    diagnostics = {
        "attempt_reconstruction": {"attempts": attempts},
        "hybrid_editorial_chunks": [{"decisions": decisions}],
        "semantic_idea_equivalence": {
            "merges": [
                {"left_clip_id": "bad_sonography", "right_clip_id": "good_sonography",
                 "accepted_by": "same_opening_restart", "confidence": 1.0},
                {"left_clip_id": "stomach_open", "right_clip_id": "stomach_complete",
                 "accepted_by": "incomplete_attempt_completed_by_retry", "confidence": 1.0},
                {"left_clip_id": "pimples_proxy", "right_clip_id": "pimples_winner", "confidence": 0.90},
            ],
            "continuation_merges": [
                {"left_clip_id": "repeat_head", "right_clip_id": "repeat_tail",
                 "accepted_by": "sentence_continuation", "confidence": 1.0},
            ],
        },
    }
    draft = DraftTimeline(
        "v1", "project", EditStrategy.STORYTELLING,
        selected, (), discarded, diagnostics,
    )
    repaired = apply_selection_conflicted_bridge_guard(draft)
    selected_ids = {clip.clip_id for clip in repaired.selected}
    assert {"good_sonography", "stomach_complete", "pimples_winner", "percentage_bridge"} <= selected_ids
    assert not ({"bad_sonography", "stomach_open", "pimples_tail", "repeat_head", "repeat_tail"} & selected_ids)
    reasons = {row["reason"] for row in repaired.diagnostics["selection_conflicted_bridge_guard"]}
    assert {
        "deterministic_retry_final_membership_resolution",
        "contained_fragment_of_confirmed_duplicate",
        "missing_positive_continuation_bridge_restored",
        "later_continuation_chain_repeats_nearby_critical_claim",
    } <= reasons
