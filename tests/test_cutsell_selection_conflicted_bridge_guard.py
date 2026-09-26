from cutsell_worker.contracts import DraftClip
from cutsell_worker.selection_conflicted_bridge_guard import (
    contained_proxy_duplicate_ids,
    confirmed_selected_duplicate_ids,
    conflicted_redundant_bridge_ids,
    deterministic_retry_resolution,
    missing_continuation_bridge_ids,
    redundant_continuation_chain_ids,
    terminally_incomplete_selected_ids,
)


def _clip(clip_id, start, end, text):
    return DraftClip(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
        caption_text=text,
    )


def test_conflicted_redundant_bridge_moves_to_swap_candidate():
    left = _clip("left", 100.0, 104.3, "Me mandaron a hacer sonografías de tiroides.")
    bridge = _clip(
        "bridge",
        108.70,
        112.42,
        "Ahí fue cuando me mandaron a hacer sonografías de tiroides y otros estudios.",
    )
    right = _clip(
        "right",
        119.95,
        124.51,
        "Me mandaron sonografías de tiroides y otros estudios.",
    )
    diagnostics = {
        "hybrid_editorial_chunks": [
            {"decisions": [
                {"clip_id": "left", "label": "winner", "confidence": 0.96},
                {"clip_id": "bridge", "label": "keep", "confidence": 0.85},
            ]},
            {"decisions": [
                {"clip_id": "bridge", "label": "alternate", "confidence": 0.80},
                {"clip_id": "right", "label": "keep", "confidence": 0.90},
            ]},
        ]
    }

    move, audit = conflicted_redundant_bridge_ids((left, bridge, right), diagnostics)

    assert move == {"bridge"}
    assert audit[0]["keep_confidence"] == 0.85
    assert audit[0]["alternate_confidence"] == 0.80
    assert audit[0]["thematic_union_coverage"] >= 0.80


def test_raw105_near_tied_winner_and_alternate_bridge_moves_to_swap():
    """Regression for the exact structural evidence observed in RAW #105."""
    left = _clip(
        "left",
        95.68,
        107.50,
        "Al terminar mi contrato cambié de ginecóloga y le pedí que me hiciera un test de todo lo que ella se pudiera imaginar y me pudiese indicar. Ahí me mandó a hacer sonografías.",
    )
    bridge = _clip(
        "bridge",
        108.70,
        112.42,
        "Ahí fue cuando me mandaron a hacer sonografías de tiroides y otros.",
    )
    right = _clip(
        "right",
        119.95,
        124.51,
        "a hacer sonografía de tiroides y otras sonografías.",
    )
    diagnostics = {
        "hybrid_editorial_chunks": [
            {"decisions": [
                {"clip_id": "left", "label": "keep", "confidence": 0.92},
                {"clip_id": "bridge", "label": "alternate", "confidence": 0.88},
            ]},
            {"decisions": [
                {"clip_id": "left", "label": "winner", "confidence": 0.95},
                {"clip_id": "bridge", "label": "winner", "confidence": 0.90},
                {"clip_id": "right", "label": "failed", "confidence": 0.85},
            ]},
            {"decisions": [
                {"clip_id": "right", "label": "alternate", "confidence": 0.85},
            ]},
        ]
    }

    move, audit = conflicted_redundant_bridge_ids((left, bridge, right), diagnostics)

    assert move == {"bridge"}
    assert audit[0]["alternate_confidence"] == 0.88
    assert audit[0]["keep_confidence"] == 0.90
    assert audit[0]["keep_margin"] == 0.02
    assert audit[0]["thematic_union_coverage"] == 1.0
    assert audit[0]["left_gap_sec"] == 1.2
    assert audit[0]["right_gap_sec"] == 7.53


def test_conflicted_bridge_with_unique_critical_fact_fails_open():
    left = _clip("left", 100.0, 104.3, "Me mandaron a hacer sonografías de tiroides.")
    bridge = _clip(
        "bridge",
        108.70,
        112.42,
        "No encontraron un nódulo de 3 centímetros en la tiroides.",
    )
    right = _clip("right", 119.95, 124.51, "Me hicieron otros estudios de tiroides.")
    diagnostics = {
        "hybrid_editorial_chunks": [
            {"decisions": [
                {"clip_id": "left", "label": "winner", "confidence": 0.96},
                {"clip_id": "bridge", "label": "keep", "confidence": 0.85},
            ]},
            {"decisions": [
                {"clip_id": "bridge", "label": "alternate", "confidence": 0.80},
                {"clip_id": "right", "label": "keep", "confidence": 0.90},
            ]},
        ]
    }

    move, audit = conflicted_redundant_bridge_ids((left, bridge, right), diagnostics)

    assert move == set()
    assert audit == []


def test_very_strong_keep_with_clear_margin_wins_conflict_and_fails_open():
    left = _clip("left", 100.0, 104.3, "Me mandaron a hacer sonografías de tiroides.")
    bridge = _clip(
        "bridge",
        108.70,
        112.42,
        "Me mandaron a hacer sonografías de tiroides y otros estudios.",
    )
    right = _clip("right", 119.95, 124.51, "Me mandaron sonografías de tiroides y otros estudios.")
    diagnostics = {
        "hybrid_editorial_chunks": [
            {"decisions": [
                {"clip_id": "left", "label": "winner", "confidence": 0.96},
                {"clip_id": "bridge", "label": "keep", "confidence": 0.93},
            ]},
            {"decisions": [
                {"clip_id": "bridge", "label": "alternate", "confidence": 0.80},
                {"clip_id": "right", "label": "keep", "confidence": 0.90},
            ]},
        ]
    }

    move, audit = conflicted_redundant_bridge_ids((left, bridge, right), diagnostics)

    assert move == set()
    assert audit == []


def test_directly_confirmed_selected_duplicate_keeps_semantic_winner():
    earlier = _clip("earlier", 198.8, 210.3, "También me salían espinillas detrás de la oreja y el cuello.")
    later = _clip("later", 213.6, 221.7, "Otro síntoma eran espinillas detrás de la oreja y en el cuello.")
    diagnostics = {
        "semantic_idea_equivalence": {"merges": [{
            "left_clip_id": "earlier", "right_clip_id": "later",
            "confidence": 0.90, "reason": "same symptom restated",
        }]},
        "hybrid_editorial_chunks": [
            {"decisions": [{"clip_id": "earlier", "label": "alternate", "confidence": 0.85}]},
            {"decisions": [{"clip_id": "later", "label": "winner", "confidence": 0.95}]},
        ],
    }
    move, audit = confirmed_selected_duplicate_ids((earlier, later), diagnostics)
    assert move == {"earlier"}
    assert audit[0]["winner_clip_id"] == "later"


def test_equal_safe_confirmed_duplicates_keep_only_later_delivery():
    earlier = _clip("earlier", 10.0, 18.0, "A rash appeared behind my ear and neck.")
    later = _clip("later", 21.0, 28.0, "The rash appeared behind my ear and on my neck.")
    diagnostics = {
        "semantic_idea_equivalence": {"merges": [{
            "left_clip_id": "earlier", "right_clip_id": "later", "confidence": 0.95,
        }]},
        "hybrid_editorial_chunks": [{"decisions": [
            {"clip_id": "earlier", "label": "keep", "confidence": 0.90},
            {"clip_id": "later", "label": "keep", "confidence": 0.90},
        ]}],
    }
    move, audit = confirmed_selected_duplicate_ids((earlier, later), diagnostics)
    assert move == {"earlier"}
    assert audit[0]["winner_clip_id"] == "later"


def test_confirmed_duplicate_with_unique_negation_fails_open():
    earlier = _clip("earlier", 1.0, 4.0, "No tuve ese síntoma.")
    later = _clip("later", 5.0, 8.0, "Tuve ese síntoma.")
    diagnostics = {
        "semantic_idea_equivalence": {"merges": [{
            "left_clip_id": "earlier", "right_clip_id": "later", "confidence": 0.90,
        }]},
        "hybrid_editorial_chunks": [{"decisions": [
            {"clip_id": "earlier", "label": "alternate", "confidence": 0.85},
            {"clip_id": "later", "label": "winner", "confidence": 0.95},
        ]}],
    }
    move, audit = confirmed_selected_duplicate_ids((earlier, later), diagnostics)
    assert move == set()
    assert audit == []


def test_short_terminally_incomplete_singleton_is_removed():
    fragment = _clip("fragment", 241.75, 243.254, "me diagnosticaron con...")
    diagnostics = {"attempt_reconstruction": {"attempts": [{
        "clip_id": "fragment", "complete_idea": False, "duration_sec": 1.504,
    }]}}
    move, audit = terminally_incomplete_selected_ids((fragment,), diagnostics)
    assert move == {"fragment"}
    assert audit[0]["reason"] == "short_terminally_incomplete_attempt"


def test_complete_or_long_open_delivery_fails_open():
    complete = _clip("complete", 1.0, 2.5, "Esta idea está completa.")
    long_open = _clip("long", 3.0, 8.0, "Esta explicación todavía continúa...")
    diagnostics = {"attempt_reconstruction": {"attempts": [
        {"clip_id": "complete", "complete_idea": True, "duration_sec": 1.5},
        {"clip_id": "long", "complete_idea": False, "duration_sec": 5.0},
    ]}}
    move, audit = terminally_incomplete_selected_ids((complete, long_open), diagnostics)
    assert move == set()
    assert audit == []


def test_deterministic_restart_can_swap_to_stronger_positive_peer():
    first = _clip("first", 10.0, 16.0, "The machine worked perfectly last year.")
    retry = _clip("retry", 20.0, 27.0, "The machine worked perfectly throughout last year.")
    diagnostics = {
        "semantic_idea_equivalence": {"merges": [{
            "left_clip_id": "first", "right_clip_id": "retry",
            "confidence": 1.0, "accepted_by": "same_opening_restart",
        }]},
        "attempt_reconstruction": {"attempts": [
            {"clip_id": "first", "complete_idea": True},
            {"clip_id": "retry", "complete_idea": True},
        ]},
        "hybrid_editorial_chunks": [
            {"decisions": [{"clip_id": "first", "label": "winner", "confidence": 0.90}]},
            {"decisions": [{"clip_id": "retry", "label": "keep", "confidence": 0.95}]},
        ],
    }
    move, add, audit = deterministic_retry_resolution((first,), (), (retry,), diagnostics)
    assert move == {"first"} and add == {"retry"}
    assert audit[0]["accepted_by"] == "same_opening_restart"


def test_equal_positive_restart_prefers_substantially_fuller_later_delivery():
    first = _clip("first", 10.0, 16.0, "We checked the thyroid every year and stopped.")
    retry = _clip(
        "retry", 20.0, 29.0,
        "We checked the thyroid every year and the results always worked perfectly.",
    )
    diagnostics = {
        "semantic_idea_equivalence": {"merges": [{
            "left_clip_id": "first", "right_clip_id": "retry",
            "confidence": 1.0, "accepted_by": "same_opening_restart",
        }]},
        "attempt_reconstruction": {"attempts": [
            {"clip_id": "first", "complete_idea": True},
            {"clip_id": "retry", "complete_idea": True},
        ]},
        "hybrid_editorial_chunks": [{"decisions": [
            {"clip_id": "first", "label": "winner", "confidence": 0.95},
            {"clip_id": "retry", "label": "winner", "confidence": 0.95},
        ]}],
    }
    move, add, audit = deterministic_retry_resolution((first,), (), (retry,), diagnostics)
    assert move == {"first"} and add == {"retry"}
    assert audit[0]["winner_clip_id"] == "retry"


def test_failed_complete_peer_is_not_resurrected_over_incomplete_selection():
    incomplete = _clip("incomplete", 10.0, 16.0, "I had trouble with...")
    failed = _clip("failed", 20.0, 22.0, "I had trouble, no.")
    diagnostics = {
        "semantic_idea_equivalence": {"merges": [{
            "left_clip_id": "incomplete", "right_clip_id": "failed",
            "confidence": 1.0, "accepted_by": "multimodal_corroborated_retry",
        }]},
        "attempt_reconstruction": {"attempts": [
            {"clip_id": "incomplete", "complete_idea": False},
            {"clip_id": "failed", "complete_idea": True},
        ]},
        "hybrid_editorial_chunks": [{"decisions": [
            {"clip_id": "incomplete", "label": "winner", "confidence": 0.90},
            {"clip_id": "failed", "label": "failed", "confidence": 0.90},
        ]}],
    }
    move, add, audit = deterministic_retry_resolution((incomplete,), (), (failed,), diagnostics)
    assert move == set() and add == set() and audit == []


def test_selected_suffix_of_confirmed_duplicate_is_removed():
    suffix = _clip("suffix", 18.0, 20.5, "for customers with annual plans.")
    full = _clip("full", 10.0, 20.0, "This offer is for customers with annual plans.")
    winner = _clip("winner", 30.0, 38.0, "The annual-plan customer offer is available now.")
    diagnostics = {"semantic_idea_equivalence": {"merges": [{
        "left_clip_id": "full", "right_clip_id": "winner", "confidence": 0.90,
    }]}}
    move, audit = contained_proxy_duplicate_ids((suffix, winner), (), (full,), diagnostics)
    assert move == {"suffix"}
    assert audit[0]["proxy_clip_id"] == "full"


def test_positive_incomplete_bridge_between_selected_neighbors_is_restored():
    left = _clip("left", 10.0, 15.0, "The first explanation is complete.")
    bridge = _clip("bridge", 15.2, 16.5, "Only 7% are from")
    right = _clip("right", 17.5, 20.0, "that category; choices matter.")
    diagnostics = {
        "attempt_reconstruction": {"attempts": [{"clip_id": "bridge", "complete_idea": False}]},
        "hybrid_editorial_chunks": [{"decisions": [
            {"clip_id": "bridge", "label": "keep", "confidence": 0.80},
            {"clip_id": "bridge", "label": "failed", "confidence": 0.80},
        ]}],
    }
    add, audit = missing_continuation_bridge_ids((left, right), (), (bridge,), diagnostics)
    assert add == {"bridge"}
    assert audit[0]["terminal_token"] == "from"


def test_audience_bridge_with_negative_fragment_vote_uses_later_duplicate_witness():
    left = _clip("left", 10.0, 15.0, "Most outcomes reflect our choices.")
    bridge = _clip("bridge", 15.2, 16.5, "Only 7% are from")
    right = _clip("right", 17.5, 20.0, "hereditary cases, so take care.")
    repeat_a = _clip("repeat_a", 24.0, 27.0, "Science confirms only 7% of")
    repeat_b = _clip("repeat_b", 28.0, 30.0, "cases are hereditary.")
    diagnostics = {
        "attempt_reconstruction": {"attempts": [{"clip_id": "bridge", "complete_idea": False}]},
        "hybrid_editorial_chunks": [{"decisions": [{
            "clip_id": "bridge", "label": "failed", "confidence": 0.90,
            "content_role": "audience",
        }]}],
        "semantic_idea_equivalence": {"continuation_merges": [{
            "left_clip_id": "repeat_a", "right_clip_id": "repeat_b",
            "accepted_by": "sentence_continuation", "confidence": 1.0,
        }]},
    }
    add, audit = missing_continuation_bridge_ids(
        (left, right, repeat_a, repeat_b), (), (bridge,), diagnostics,
    )
    assert add == {"bridge"}
    assert audit[0]["reason"] == "audience_continuation_bridge_restored_from_duplicate_witness"
    assert audit[0]["duplicate_witness_clip_ids"] == ["repeat_a", "repeat_b"]


def test_incomplete_bridge_with_stronger_negative_evidence_fails_open():
    left = _clip("left", 10.0, 15.0, "The first explanation is complete.")
    bridge = _clip("bridge", 15.2, 16.5, "Only 7% are from")
    right = _clip("right", 17.5, 20.0, "that category; choices matter.")
    diagnostics = {
        "attempt_reconstruction": {"attempts": [{"clip_id": "bridge", "complete_idea": False}]},
        "hybrid_editorial_chunks": [{"decisions": [
            {"clip_id": "bridge", "label": "keep", "confidence": 0.80},
            {"clip_id": "bridge", "label": "failed", "confidence": 0.90},
        ]}],
    }
    add, audit = missing_continuation_bridge_ids((left, right), (), (bridge,), diagnostics)
    assert add == set() and audit == []


def test_later_numeric_continuation_chain_is_removed_when_facts_already_covered():
    earlier = _clip("earlier", 10.0, 17.0, "Research shows only 7% belong to this category.")
    aside = _clip("aside", 18.0, 21.0, "My own case is unusual.")
    repeat_a = _clip("repeat_a", 22.0, 25.0, "Science confirms only 7% of")
    repeat_b = _clip("repeat_b", 26.0, 28.0, "cases belong to this category.")
    diagnostics = {"semantic_idea_equivalence": {"continuation_merges": [{
        "left_clip_id": "repeat_a", "right_clip_id": "repeat_b",
        "accepted_by": "sentence_continuation", "confidence": 1.0,
    }]}}
    move, audit = redundant_continuation_chain_ids((earlier, aside, repeat_a, repeat_b), diagnostics)
    assert move == {"repeat_a", "repeat_b"}
    assert audit[0]["critical_markers"] == ["7%"]


def test_numeric_chain_with_a_new_number_fails_open():
    earlier = _clip("earlier", 10.0, 17.0, "Research shows only 7% belong to this category.")
    repeat_a = _clip("repeat_a", 22.0, 25.0, "Science confirms only 12% of")
    repeat_b = _clip("repeat_b", 26.0, 28.0, "cases belong to this category.")
    diagnostics = {"semantic_idea_equivalence": {"continuation_merges": [{
        "left_clip_id": "repeat_a", "right_clip_id": "repeat_b",
        "accepted_by": "sentence_continuation", "confidence": 1.0,
    }]}}
    move, audit = redundant_continuation_chain_ids((earlier, repeat_a, repeat_b), diagnostics)
    assert move == set() and audit == []
