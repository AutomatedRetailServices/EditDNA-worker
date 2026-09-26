from dataclasses import replace

from cutsell_worker.contracts import CandidateTake, DraftClip, DraftTimeline, EditStrategy, MediaSignals, Word
from cutsell_worker.final_delivery_integrity import (
    collapse_overlapping_contained_deliveries,
)
from cutsell_worker.selection_conflicted_bridge_guard import apply_selection_conflicted_bridge_guard
from cutsell_worker.semantic_ledger import build_semantic_ledger_shadow
from cutsell_worker.realization_resolver import resolve_realizations_shadow


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


def test_final_membership_winner_is_not_resurrected_by_authoritative_resolver():
    bad = _draft_clip("bad", 10.0, 15.0, "Nunca hicimos el chequeo porque cada año hacía dos estados.")
    good = _draft_clip(
        "good", 18.0, 26.0,
        "Nunca hicimos el chequeo porque los exámenes indicaban que funcionaba perfectamente.",
        selected=False,
    )
    bad = replace(bad, realization_id="real_bad", semantic_idea_id="idea_retry")
    good = replace(good, realization_id="real_good", semantic_idea_id="idea_retry")
    draft = DraftTimeline(
        "v1", "project", EditStrategy.STORYTELLING,
        (good,), (), (bad,),
        {
            "selection_conflicted_bridge_guard": [{
                "clip_id": "bad",
                "winner_clip_id": "good",
                "reason": "deterministic_retry_final_membership_resolution",
            }],
            "take_group_members": [["bad", "good"]],
            "take_judge_groups": [{
                "ranked": [{"clip_id": "bad", "score": 0.9}, {"clip_id": "good", "score": 0.8}],
                "local_selected_clip_id": "bad",
                "selected_clip_id": "bad",
                "semantic_override_applied": True,
                "semantic_candidates": [{"clip_id": "bad", "confidence": 0.95}],
            }],
        },
    )
    ledger = build_semantic_ledger_shadow(draft)
    resolution = resolve_realizations_shadow(ledger).idea_resolutions["idea_retry"]
    assert resolution.winner_realization_id == "real_good"


def test_ledger_ignores_stale_reciprocal_final_membership_history():
    loser = replace(
        _draft_clip("loser", 10.0, 16.0, "A complete earlier version."),
        realization_id="real_loser", semantic_idea_id="idea_retry",
    )
    winner = replace(
        _draft_clip("winner", 20.0, 30.0, "A complete fuller retry."),
        realization_id="real_winner", semantic_idea_id="idea_retry",
    )
    draft = DraftTimeline(
        "v1", "project", EditStrategy.STORYTELLING,
        (winner,), (), (loser,),
        {
            "selection_conflicted_bridge_guard": [
                {
                    "clip_id": "loser", "winner_clip_id": "winner",
                    "reason": "deterministic_retry_final_membership_resolution",
                },
                {
                    "clip_id": "winner", "winner_clip_id": "loser",
                    "reason": "deterministic_retry_final_membership_resolution",
                },
            ],
            "take_group_members": [["loser", "winner"]],
        },
    )

    ledger = build_semantic_ledger_shadow(draft)
    resolution = resolve_realizations_shadow(ledger).idea_resolutions["idea_retry"]

    assert resolution.winner_realization_id == "real_winner"
    assert resolution.decision_status == "RESOLVED_WINNER"


def test_post_authority_replays_prior_removal_only_proof():
    fragment = _draft_clip("fragment", 10.0, 12.0, "It was like an allergy.")
    winner = _draft_clip("winner", 20.0, 28.0, "The complete later allergy delivery.")
    draft = DraftTimeline(
        "v1", "project", EditStrategy.STORYTELLING,
        (fragment, winner), (), (),
        {"selection_conflicted_bridge_guard": [{
            "clip_id": "fragment",
            "winner_clip_id": "winner",
            "reason": "orphaned_anaphoric_fragment_of_confirmed_retry",
        }]},
    )

    repaired = apply_selection_conflicted_bridge_guard(
        draft, allow_membership_additions=False,
    )

    assert [clip.clip_id for clip in repaired.selected] == ["winner"]
    assert repaired.diagnostics["selection_conflicted_bridge_guard_post_authority"] == {
        "membership_additions_allowed": False,
        "suppressed_add_clip_ids": [],
    }


def test_chain_coverage_audit_excludes_concurrently_removed_duplicate_witness():
    prior = _draft_clip(
        "prior", 10.0, 18.0,
        "Science says only 5-10% of cancers are hereditary.",
    )
    duplicate = _draft_clip(
        "duplicate", 19.0, 24.0,
        "Science says only 5-10% of cancers are hereditary.",
    )
    bridge = _draft_clip("bridge", 25.0, 27.0, "Only 5-10% of")
    tail = _draft_clip("tail", 28.0, 31.0, "cancers are hereditary.")
    repeat_head = _draft_clip("repeat_head", 34.0, 38.0, "Science says only 5-10% of")
    repeat_tail = _draft_clip("repeat_tail", 39.0, 42.0, "cancers are hereditary.")
    draft = DraftTimeline(
        "v1", "project", EditStrategy.STORYTELLING,
        (prior, duplicate, bridge, tail, repeat_head, repeat_tail), (), (),
        {
            "semantic_idea_equivalence": {
                "merges": [{
                    "left_clip_id": "duplicate", "right_clip_id": "prior",
                    "confidence": 0.95,
                }],
                "continuation_merges": [{
                    "left_clip_id": "repeat_head", "right_clip_id": "repeat_tail",
                    "accepted_by": "sentence_continuation", "confidence": 1.0,
                }],
            },
            "hybrid_editorial_chunks": [{"decisions": [
                {"clip_id": "prior", "label": "winner", "confidence": 0.95},
                {"clip_id": "duplicate", "label": "alternate", "confidence": 0.85},
            ]}],
        },
    )

    repaired = apply_selection_conflicted_bridge_guard(draft)
    chain = next(
        row for row in repaired.diagnostics["selection_conflicted_bridge_guard"]
        if row["reason"] == "later_continuation_chain_repeats_nearby_critical_claim"
    )
    final_ids = {clip.clip_id for clip in repaired.selected}

    assert "duplicate" not in final_ids
    assert {"repeat_head", "repeat_tail"}.isdisjoint(final_ids)
    assert set(chain["prior_clip_ids"]).issubset(final_ids)


def test_orphan_dependent_opening_yields_to_complete_peer_in_same_family():
    full = _draft_clip(
        "full", 10.0, 14.0,
        "Ahí fue cuando me mandaron a hacer los estudios completos.",
        selected=False,
    )
    orphan = _draft_clip(
        "orphan", 18.0, 21.0,
        "cuando me mandaron a hacer los estudios completos.",
    )
    draft = DraftTimeline(
        "v1", "project", EditStrategy.STORYTELLING,
        (orphan,), (), (full,),
        {
            "attempt_reconstruction": {"attempts": [
                {"clip_id": "full", "complete_idea": True},
                {"clip_id": "orphan", "complete_idea": True},
            ]},
            "take_judge_groups": [{
                "ranked": [{"clip_id": "orphan"}, {"clip_id": "full"}],
                "member_usability": {
                    "orphan": {"deterministic_unusable": False, "delete_recommended": False},
                    "full": {"deterministic_unusable": False, "delete_recommended": False},
                },
            }],
        },
    )
    repaired = apply_selection_conflicted_bridge_guard(draft)
    assert [clip.clip_id for clip in repaired.selected] == ["full"]


def test_provider_rejected_numeric_restatement_is_removed_after_full_delivery():
    prior = _draft_clip(
        "prior", 10.0, 20.0,
        "Está comprobado que solo un 5-10% de los cánceres son hereditarios y el resto depende de elecciones de vida.",
    )
    repeated = _draft_clip(
        "repeated", 30.0, 36.0,
        "Estoy convencida y la ciencia avala que solo un 5-10% de los cánceres son hereditarios.",
    )
    draft = DraftTimeline(
        "v1", "project", EditStrategy.STORYTELLING,
        (prior, repeated), (), (),
        {"hybrid_editorial_chunks": [{"decisions": [
            {"clip_id": "prior", "label": "winner", "confidence": 0.95},
            {"clip_id": "repeated", "label": "alternate", "confidence": 0.85},
        ]}]},
    )
    repaired = apply_selection_conflicted_bridge_guard(draft)
    assert [clip.clip_id for clip in repaired.selected] == ["prior"]


def test_selected_borderline_suffix_restores_safe_complete_parent_prefix():
    prefix = _draft_clip(
        "prefix", 10.0, 14.0,
        "Esta es mi experiencia y soy la única persona de mi familia con este cáncer.",
        selected=False,
    )
    suffix = _draft_clip(
        "suffix", 14.5, 23.0,
        "Por eso comparto esta conclusión y las elecciones que aprendí a cuidar.",
    )
    parent = _draft_clip(
        "parent", 10.0, 23.0,
        prefix.text + " " + suffix.text,
        selected=False,
    )
    draft = DraftTimeline(
        "v1", "project", EditStrategy.STORYTELLING,
        (suffix,), (), (prefix, parent),
        {
            "attempt_reconstruction": {
                "attempts": [{"clip_id": "parent", "complete_idea": True}],
                "preserved_borderline_subspans": [{
                    "parent_clip_id": "parent",
                    "prefix_clip_id": "prefix",
                    "suffix_clip_id": "suffix",
                }],
            },
            "hybrid_editorial_chunks": [{"decisions": [{
                "clip_id": "prefix", "label": "alternate", "confidence": 0.8,
                "content_role": "audience",
            }]}],
            "take_judge_groups": [{
                "ranked": [{"clip_id": "prefix"}, {"clip_id": "suffix"}],
                "member_usability": {
                    "prefix": {"deterministic_unusable": False, "delete_recommended": False},
                },
            }],
        },
    )
    repaired = apply_selection_conflicted_bridge_guard(draft)
    assert [clip.clip_id for clip in repaired.selected] == ["prefix", "suffix"]


def test_post_authority_guard_cannot_restore_discarded_prefix():
    prefix = _draft_clip(
        "prefix", 10.0, 14.0,
        "Esta es mi experiencia y soy la única persona de mi familia con este cáncer.",
        selected=False,
    )
    suffix = _draft_clip(
        "suffix", 14.5, 23.0,
        "Por eso comparto esta conclusión y las elecciones que aprendí a cuidar.",
    )
    parent = _draft_clip(
        "parent", 10.0, 23.0,
        prefix.text + " " + suffix.text,
        selected=False,
    )
    draft = DraftTimeline(
        "v1", "project", EditStrategy.STORYTELLING,
        (suffix,), (), (prefix, parent),
        {
            "attempt_reconstruction": {
                "attempts": [{"clip_id": "parent", "complete_idea": True}],
                "preserved_borderline_subspans": [{
                    "parent_clip_id": "parent",
                    "prefix_clip_id": "prefix",
                    "suffix_clip_id": "suffix",
                }],
            },
            "hybrid_editorial_chunks": [{"decisions": [{
                "clip_id": "prefix", "label": "alternate", "confidence": 0.8,
                "content_role": "audience",
            }]}],
            "take_judge_groups": [{
                "ranked": [{"clip_id": "prefix"}, {"clip_id": "suffix"}],
                "member_usability": {
                    "prefix": {"deterministic_unusable": False, "delete_recommended": False},
                },
            }],
        },
    )

    repaired = apply_selection_conflicted_bridge_guard(
        draft, allow_membership_additions=False,
    )

    assert [clip.clip_id for clip in repaired.selected] == ["suffix"]
    assert {clip.clip_id for clip in repaired.discarded} == {"prefix", "parent"}
    assert repaired.diagnostics["selection_conflicted_bridge_guard_post_authority"] == {
        "membership_additions_allowed": False,
        "suppressed_add_clip_ids": ["prefix"],
    }
