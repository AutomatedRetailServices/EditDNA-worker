from cutsell_worker.contracts import DraftClip
from cutsell_worker.selection_conflicted_bridge_guard import (
    abandoned_negated_restart_ids,
    contained_proxy_duplicate_ids,
    confirmed_selected_duplicate_ids,
    conflicted_redundant_bridge_ids,
    deterministic_retry_resolution,
    failed_retry_component_ids,
    missing_continuation_bridge_ids,
    redundant_continuation_chain_ids,
    terminally_incomplete_selected_ids,
    unmerged_same_opening_retry_resolution,
    nearby_contained_selected_realization_ids,
    orphaned_anaphoric_retry_fragment_ids,
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


def test_unusable_equivalent_take_yields_to_later_global_winner():
    earlier = _clip(
        "earlier", 10.0, 20.0,
        "A rash appeared behind my ear and neck and looked hormonal.",
    )
    later = _clip(
        "later", 23.0, 31.0,
        "The rash appeared behind my ear and on my neck in seasons.",
    )
    diagnostics = {
        "semantic_idea_equivalence": {"merges": [{
            "left_clip_id": "earlier", "right_clip_id": "later", "confidence": 0.90,
        }]},
        "hybrid_editorial_chunks": [{"decisions": [
            {"clip_id": "earlier", "label": "winner", "confidence": 0.90},
            {"clip_id": "later", "label": "winner", "confidence": 0.95},
        ]}],
        "take_judge_groups": [{
            "candidate_usability_summary": {"earlier": "UNUSABLE", "later": "USABLE"},
        }],
    }

    move, audit = confirmed_selected_duplicate_ids((earlier, later), diagnostics)

    assert move == {"earlier"}
    assert audit[0]["winner_clip_id"] == "later"


def test_failed_wrong_take_yields_through_transitive_retry_component():
    abandoned = _clip("abandoned", 10.0, 16.0, "I had stomach problems and was diagnosed with...")
    false_start = _clip("false_start", 20.0, 21.6, "I had stomach problems, no.")
    complete = _clip("complete", 27.0, 36.0, "I had digestion problems and was diagnosed with gastritis.")
    diagnostics = {
        "semantic_idea_equivalence": {"merges": [
            {"left_clip_id": "abandoned", "right_clip_id": "false_start",
             "accepted_by": "multimodal_corroborated_retry", "confidence": 1.0},
            {"left_clip_id": "abandoned", "right_clip_id": "complete",
             "accepted_by": "incomplete_attempt_completed_by_retry", "confidence": 1.0},
        ]},
        "attempt_reconstruction": {"attempts": [
            {"clip_id": "abandoned", "complete_idea": False},
            {"clip_id": "false_start", "complete_idea": True},
            {"clip_id": "complete", "complete_idea": True},
        ]},
        "hybrid_editorial_chunks": [{"decisions": [
            {"clip_id": "false_start", "label": "failed", "confidence": 0.90},
            {"clip_id": "complete", "label": "winner", "confidence": 0.95},
        ]}],
        "take_judge_groups": [{
            "candidate_usability_summary": {"false_start": "UNUSABLE", "complete": "USABLE"},
            "member_usability": {
                "false_start": {"deterministic_unusable": True, "delete_recommended": True},
                "complete": {"deterministic_unusable": False, "delete_recommended": False},
            },
        }],
    }

    move, audit = failed_retry_component_ids(
        (false_start, complete), (), (abandoned,), diagnostics,
    )

    assert move == {"false_start"}
    assert audit[0]["winner_clip_id"] == "complete"


def test_multimodal_wrong_take_can_settle_component_without_hybrid_failure_vote():
    abandoned = _clip("abandoned", 10.0, 16.0, "I had stomach problems and was diagnosed with...")
    false_start = _clip("false_start", 20.0, 21.6, "I had stomach problems, no.")
    complete = _clip("complete", 27.0, 36.0, "I had digestion problems and was diagnosed with gastritis.")
    diagnostics = {
        "semantic_idea_equivalence": {"merges": [
            {"left_clip_id": "abandoned", "right_clip_id": "false_start",
             "accepted_by": "multimodal_corroborated_retry", "confidence": 1.0,
             "corroborating_event_kind": "wrong_take"},
            {"left_clip_id": "abandoned", "right_clip_id": "complete",
             "accepted_by": "incomplete_attempt_completed_by_retry", "confidence": 1.0},
        ]},
        "attempt_reconstruction": {"attempts": [
            {"clip_id": "abandoned", "complete_idea": False},
            {"clip_id": "false_start", "complete_idea": True},
            {"clip_id": "complete", "complete_idea": True},
        ]},
        "hybrid_editorial_chunks": [{"decisions": [
            {"clip_id": "complete", "label": "keep", "confidence": 0.95},
        ]}],
        "take_judge_groups": [{
            "candidate_usability_summary": {"false_start": "UNUSABLE"},
            "member_usability": {"false_start": {
                "deterministic_unusable": False, "delete_recommended": False,
            }},
        }],
    }

    move, audit = failed_retry_component_ids((false_start, complete), (), (abandoned,), diagnostics)

    assert move == {"false_start"}
    assert audit[0]["multimodal_wrong_take_corroborated"] is True


def test_short_multimodal_wrong_take_tail_uses_native_av_audience_winner():
    abandoned = _clip("abandoned", 10.0, 16.0, "Tuve problemas de estómago y me diagnosticaron con...")
    false_start = _clip("false_start", 20.0, 21.6, "Tuve problemas de estómago, no.")
    complete = _clip("complete", 27.0, 36.0, "Tuve problemas de digestión y me diagnosticaron gastritis.")
    diagnostics = {
        "semantic_idea_equivalence": {"merges": [
            {"left_clip_id": "abandoned", "right_clip_id": "false_start",
             "accepted_by": "multimodal_corroborated_retry", "confidence": 1.0,
             "corroborating_event_kind": "wrong_take"},
            {"left_clip_id": "abandoned", "right_clip_id": "complete",
             "accepted_by": "incomplete_attempt_completed_by_retry", "confidence": 1.0},
        ]},
        "attempt_reconstruction": {"attempts": [
            {"clip_id": "abandoned", "complete_idea": False},
            {"clip_id": "false_start", "complete_idea": True},
            {"clip_id": "complete", "complete_idea": True},
        ]},
        "hybrid_editorial_chunks": [{"decisions": [
            {"clip_id": "complete", "label": "keep", "confidence": 0.80,
             "content_role": "audience", "audiovisual": {"observations": [
                 {"role": "audience", "confidence": 0.90},
             ]}},
        ]}],
        "take_judge_groups": [{
            "candidate_usability_summary": {"complete": "UNUSABLE"},
            "member_usability": {"complete": {
                "deterministic_unusable": False, "delete_recommended": False,
            }},
        }],
    }

    move, audit = failed_retry_component_ids((false_start, complete), (), (abandoned,), diagnostics)

    assert move == {"false_start"}
    assert audit[0]["short_wrong_take_tail"] is True
    assert audit[0]["winner_positive_confidence"] == 0.80


def test_failed_retry_component_preserves_negation_without_full_failure_evidence():
    abandoned = _clip("abandoned", 10.0, 16.0, "I had symptoms and...")
    negated = _clip("negated", 20.0, 22.0, "I did not have symptoms.")
    complete = _clip("complete", 27.0, 32.0, "I had symptoms later.")
    diagnostics = {
        "semantic_idea_equivalence": {"merges": [
            {"left_clip_id": "abandoned", "right_clip_id": "negated",
             "accepted_by": "multimodal_corroborated_retry", "confidence": 1.0},
            {"left_clip_id": "abandoned", "right_clip_id": "complete",
             "accepted_by": "incomplete_attempt_completed_by_retry", "confidence": 1.0},
        ]},
        "hybrid_editorial_chunks": [{"decisions": [
            {"clip_id": "negated", "label": "failed", "confidence": 0.89},
            {"clip_id": "complete", "label": "winner", "confidence": 0.95},
        ]}],
        "take_judge_groups": [{
            "candidate_usability_summary": {"negated": "UNUSABLE", "complete": "USABLE"},
            "member_usability": {
                "negated": {"deterministic_unusable": True, "delete_recommended": True},
            },
        }],
    }

    move, audit = failed_retry_component_ids((negated, complete), (), (abandoned,), diagnostics)

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


def test_deterministic_restart_accepts_clear_positive_over_conflicting_alternate():
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
        "hybrid_editorial_chunks": [{"decisions": [
            {"clip_id": "first", "label": "keep", "confidence": 0.90},
            {"clip_id": "retry", "label": "keep", "confidence": 0.95},
            {"clip_id": "retry", "label": "alternate", "confidence": 0.85},
        ]}],
    }

    move, add, _audit = deterministic_retry_resolution((first,), (), (retry,), diagnostics)

    assert move == {"first"}
    assert add == {"retry"}


def test_stronger_vote_cannot_replace_full_delivery_with_short_open_restart():
    full = _clip(
        "full", 95.942, 102.714,
        "Al terminar mi contrato cambié de ginecóloga y le pedí que me hiciera todos los estudios.",
    )
    short = _clip("short", 91.24, 93.515, "Al terminar mi contrato le pedía a mi ginecóloga")
    diagnostics = {
        "semantic_idea_equivalence": {"merges": [{
            "left_clip_id": "full", "right_clip_id": "short",
            "confidence": 1.0, "accepted_by": "same_opening_abandoned_start",
        }]},
        "attempt_reconstruction": {"attempts": [
            {"clip_id": "full", "complete_idea": True},
            {"clip_id": "short", "complete_idea": True},
        ]},
        "hybrid_editorial_chunks": [{"decisions": [
            {"clip_id": "full", "label": "winner", "confidence": 0.90},
            {"clip_id": "short", "label": "winner", "confidence": 0.95},
        ]}],
    }

    move, add, audit = deterministic_retry_resolution((full,), (), (short,), diagnostics)

    assert move == set()
    assert add == set()
    assert audit == []


def test_same_opening_near_tie_prefers_substantially_fuller_later_delivery():
    first = _clip(
        "first", 10.0, 16.5,
        "We never considered a thyroid scan because every year gave two states.",
    )
    retry = _clip(
        "retry", 20.0, 30.0,
        "We never considered a thyroid scan because every examination showed it worked perfectly.",
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
        "take_judge_groups": [{"candidate_usability_summary": {
            "first": "USABLE", "retry": "USABLE",
        }}],
        "hybrid_editorial_chunks": [{"decisions": [
            {"clip_id": "first", "label": "winner", "confidence": 0.95},
            {"clip_id": "retry", "label": "keep", "confidence": 0.85},
            {"clip_id": "retry", "label": "alternate", "confidence": 0.80},
        ]}],
    }

    move, add, _audit = deterministic_retry_resolution((first,), (), (retry,), diagnostics)

    assert move == {"first"}
    assert add == {"retry"}


def test_short_positive_restart_with_standard_negative_vote_yields_to_safe_full_peer():
    full = _clip(
        "full", 10.0, 17.0,
        "After my contract I spoke with my doctor and requested every available test.",
    )
    short = _clip("short", 20.0, 22.3, "After my contract I asked my doctor")
    diagnostics = {
        "semantic_idea_equivalence": {"merges": [{
            "left_clip_id": "full", "right_clip_id": "short",
            "confidence": 1.0, "accepted_by": "same_opening_abandoned_start",
        }]},
        "attempt_reconstruction": {"attempts": [
            {"clip_id": "full", "complete_idea": True},
            {"clip_id": "short", "complete_idea": True},
        ]},
        "take_judge_groups": [{"member_usability": {
            "full": {"deterministic_unusable": False, "delete_recommended": False},
            "short": {"deterministic_unusable": False, "delete_recommended": False},
        }}],
        "hybrid_editorial_chunks": [{"decisions": [
            {"clip_id": "full", "label": "keep", "confidence": 0.95},
            {"clip_id": "full", "label": "alternate", "confidence": 0.70},
            {"clip_id": "short", "label": "keep", "confidence": 0.95},
            {"clip_id": "short", "label": "failed", "confidence": 0.80},
        ]}],
    }

    move, add, audit = deterministic_retry_resolution((short,), (), (full,), diagnostics)

    assert move == {"short"}
    assert add == {"full"}
    assert audit[0]["reason"] == "deterministic_retry_semantic_superset_dominance"


def test_fuller_restart_accepts_standard_positive_alternate_window_tie():
    first = _clip(
        "first", 10.0, 16.5,
        "We never considered a thyroid scan because every year gave two states.",
    )
    retry = _clip(
        "retry", 20.0, 30.0,
        "We never considered a thyroid scan because every examination showed it worked perfectly.",
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
        "take_judge_groups": [{
            "candidate_usability_summary": {"first": "UNUSABLE", "retry": "UNUSABLE"},
            "member_usability": {
                "first": {"delete_recommended": False, "deterministic_unusable": False},
                "retry": {"delete_recommended": False, "deterministic_unusable": False},
            },
        }],
        "hybrid_editorial_chunks": [{"decisions": [
            {"clip_id": "first", "label": "winner", "confidence": 0.95},
            {"clip_id": "retry", "label": "keep", "confidence": 0.85},
            {"clip_id": "retry", "label": "alternate", "confidence": 0.85},
        ]}],
    }

    move, add, _audit = deterministic_retry_resolution((first,), (), (retry,), diagnostics)

    assert move == {"first"}
    assert add == {"retry"}


def test_unmerged_same_opening_restart_selects_rich_full_later_delivery():
    first = _clip(
        "first", 10.0, 16.5,
        "After my contract I spoke with my doctor and requested every test available.",
    )
    short_restart = _clip(
        "short", 18.0, 20.2,
        "After my contract I requested my doctor",
    )
    full_restart = _clip(
        "full", 23.0, 30.0,
        "After my contract I changed my doctor and requested every test she could provide.",
    )
    diagnostics = {
        "attempt_reconstruction": {"attempts": [
            {"clip_id": "first", "complete_idea": True},
            {"clip_id": "short", "complete_idea": True},
            {"clip_id": "full", "complete_idea": True},
        ]},
        "hybrid_editorial_chunks": [{"decisions": [
            {"clip_id": "first", "label": "winner", "confidence": 0.95},
            {"clip_id": "first", "label": "alternate", "confidence": 0.80},
            {"clip_id": "full", "label": "failed", "confidence": 0.90},
        ]}],
    }

    move, add, audit = unmerged_same_opening_retry_resolution(
        (first,), (), (short_restart, full_restart), diagnostics,
    )

    assert move == {"first"}
    assert add == {"full"}
    assert audit[0]["reason"] == "ungrouped_same_opening_full_retry_resolution"


def test_unmerged_same_opening_retry_accepts_positive_non_deleting_fuller_peer():
    first = _clip("first", 10.0, 12.2, "After my contract I asked my doctor.")
    retry = _clip(
        "retry", 15.0, 22.0,
        "After my contract I changed my doctor and asked for every available test.",
    )
    diagnostics = {
        "attempt_reconstruction": {"attempts": [
            {"clip_id": "first", "complete_idea": True},
            {"clip_id": "retry", "complete_idea": True},
        ]},
        "take_judge_groups": [{
            "candidate_usability_summary": {"retry": "UNUSABLE"},
            "member_usability": {"retry": {"delete_recommended": False}},
        }],
        "hybrid_editorial_chunks": [{"decisions": [
            {"clip_id": "first", "label": "failed", "confidence": 0.75},
            {"clip_id": "retry", "label": "keep", "confidence": 0.95},
            {"clip_id": "retry", "label": "alternate", "confidence": 0.80},
        ]}],
    }

    move, add, _audit = unmerged_same_opening_retry_resolution(
        (first,), (), (retry,), diagnostics,
    )

    assert move == {"first"}
    assert add == {"retry"}


def test_unmerged_same_opening_restart_preserves_changed_number():
    first = _clip("first", 10.0, 17.0, "After my contract I requested 5 medical tests from my doctor.")
    retry = _clip("retry", 20.0, 27.0, "After my contract I requested 10 medical tests from my doctor.")
    diagnostics = {
        "attempt_reconstruction": {"attempts": [
            {"clip_id": "first", "complete_idea": True},
            {"clip_id": "retry", "complete_idea": True},
        ]},
        "hybrid_editorial_chunks": [{"decisions": [
            {"clip_id": "first", "label": "alternate", "confidence": 0.80},
        ]}],
    }

    move, add, audit = unmerged_same_opening_retry_resolution((first,), (), (retry,), diagnostics)

    assert move == set()
    assert add == set()
    assert audit == []


def test_later_selected_statement_fully_contained_in_richer_nearby_delivery_is_removed():
    earlier = _clip(
        "earlier", 10.0, 22.0,
        "Only five percent of cancers are hereditary and most outcomes reflect our daily choices and care.",
    )
    later = _clip(
        "later", 28.0, 33.0,
        "Only five percent of cancers are hereditary.",
    )

    move, audit = nearby_contained_selected_realization_ids((earlier, later))

    assert move == {"later"}
    assert audit[0]["winner_clip_id"] == "earlier"


def test_nearby_statement_with_new_fact_is_preserved():
    earlier = _clip(
        "earlier", 10.0, 22.0,
        "Only five percent of cancers are hereditary and most outcomes reflect our daily choices and care.",
    )
    later = _clip(
        "later", 28.0, 34.0,
        "Only ten percent are hereditary according to a new clinical study.",
    )

    move, audit = nearby_contained_selected_realization_ids((earlier, later))

    assert move == set()
    assert audit == []


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


def test_shared_abandoned_attempt_resolves_short_failed_component_debris():
    abandoned = _clip("abandoned", 10.0, 16.0, "I had stomach trouble and was diagnosed with...")
    correction = _clip("correction", 18.0, 19.6, "I had stomach trouble, no.")
    complete = _clip(
        "complete", 24.0, 33.5,
        "I had digestive trouble and the endoscopy showed gastritis.",
    )
    diagnostics = {
        "semantic_idea_equivalence": {"merges": [
            {"left_clip_id": "abandoned", "right_clip_id": "correction",
             "confidence": 1.0, "accepted_by": "multimodal_corroborated_retry"},
            {"left_clip_id": "abandoned", "right_clip_id": "complete",
             "confidence": 1.0, "accepted_by": "incomplete_attempt_completed_by_retry"},
        ]},
        "attempt_reconstruction": {"attempts": [
            {"clip_id": "abandoned", "complete_idea": False},
            {"clip_id": "correction", "complete_idea": True},
            {"clip_id": "complete", "complete_idea": True},
        ]},
        "hybrid_editorial_chunks": [{"decisions": [
            {"clip_id": "correction", "label": "failed", "confidence": 0.90,
             "content_role": "mixed"},
            {"clip_id": "complete", "label": "winner", "confidence": 0.95,
             "content_role": "audience"},
        ]}],
    }
    move, add, audit = deterministic_retry_resolution(
        (correction, complete), (), (abandoned,), diagnostics,
    )
    assert move == {"correction"} and add == set()
    assert audit[0]["reason"] == "deterministic_retry_component_failed_debris"


def test_deterministic_retry_prefers_semantically_full_delivery_over_clean_prefix():
    first_full = _clip(
        "first_full", 10.0, 16.5,
        "After my contract I spoke with my doctor and requested every available test.",
    )
    clean_prefix = _clip(
        "clean_prefix", 18.0, 20.3,
        "After my contract I asked my doctor",
    )
    final_full = _clip(
        "final_full", 23.0, 30.0,
        "After my contract I changed my doctor and requested every test she could provide.",
    )
    diagnostics = {
        "semantic_idea_equivalence": {"merges": [
            {"left_clip_id": "first_full", "right_clip_id": "clean_prefix",
             "accepted_by": "same_opening_abandoned_start", "confidence": 1.0},
            {"left_clip_id": "final_full", "right_clip_id": "clean_prefix",
             "accepted_by": "same_opening_abandoned_start", "confidence": 1.0},
        ]},
        "attempt_reconstruction": {"attempts": [
            {"clip_id": "first_full", "complete_idea": True},
            {"clip_id": "clean_prefix", "complete_idea": True},
            {"clip_id": "final_full", "complete_idea": True},
        ]},
        "hybrid_editorial_chunks": [{"decisions": [
            {"clip_id": "clean_prefix", "label": "keep", "confidence": 0.95},
        ]}],
        "take_judge_groups": [{
            "member_usability": {
                "first_full": {"deterministic_unusable": False, "delete_recommended": False},
                "clean_prefix": {"deterministic_unusable": False, "delete_recommended": False},
                "final_full": {"deterministic_unusable": False, "delete_recommended": False},
            },
        }],
    }

    move, add, audit = deterministic_retry_resolution(
        (clean_prefix,), (), (first_full, final_full), diagnostics,
    )

    assert move == {"clean_prefix"}
    assert add == {"final_full"}
    assert audit[0]["reason"] == "deterministic_retry_semantic_superset_dominance"


def test_semantic_superset_accepts_stronger_keep_over_standard_failed_shadow():
    """Overlapping windows may call a complete delivery KEEP .95 and FAILED
    .80.  The weaker shadow is not enough to make an otherwise safe, fuller
    deterministic retry ineligible when no terminal evidence recommends
    deletion.
    """
    prefix = _clip(
        "prefix", 10.0, 12.3,
        "After my contract I asked my doctor",
    )
    fuller = _clip(
        "fuller", 15.0, 21.6,
        "After my contract I spoke with my doctor and requested every available test.",
    )
    diagnostics = {
        "semantic_idea_equivalence": {"merges": [{
            "left_clip_id": "fuller", "right_clip_id": "prefix",
            "accepted_by": "same_opening_abandoned_start", "confidence": 1.0,
        }]},
        "attempt_reconstruction": {"attempts": [
            {"clip_id": "prefix", "complete_idea": True},
            {"clip_id": "fuller", "complete_idea": True},
        ]},
        "hybrid_editorial_chunks": [{"decisions": [
            {"clip_id": "prefix", "label": "keep", "confidence": 0.95},
            {"clip_id": "prefix", "label": "alternate", "confidence": 0.85},
            {"clip_id": "fuller", "label": "keep", "confidence": 0.95},
            {"clip_id": "fuller", "label": "failed", "confidence": 0.80},
        ]}],
        "take_judge_groups": [{
            "member_usability": {
                "prefix": {"deterministic_unusable": False, "delete_recommended": False},
                "fuller": {"deterministic_unusable": False, "delete_recommended": False},
            },
        }],
    }

    move, add, audit = deterministic_retry_resolution(
        (prefix,), (), (fuller,), diagnostics,
    )

    assert move == {"prefix"}
    assert add == {"fuller"}
    assert audit[0]["reason"] == "deterministic_retry_semantic_superset_dominance"
    assert audit[0]["winner_positive_confidence"] == 0.95
    assert audit[0]["winner_negative_confidence"] == 0.8


def test_semantic_superset_retry_preserves_changed_number():
    prefix = _clip("prefix", 10.0, 12.0, "After my contract I requested 5 tests")
    fuller = _clip(
        "fuller", 15.0, 22.0,
        "After my contract I requested 10 tests from every available specialist.",
    )
    diagnostics = {
        "semantic_idea_equivalence": {"merges": [{
            "left_clip_id": "prefix", "right_clip_id": "fuller",
            "accepted_by": "same_opening_restart", "confidence": 1.0,
        }]},
        "attempt_reconstruction": {"attempts": [
            {"clip_id": "prefix", "complete_idea": True},
            {"clip_id": "fuller", "complete_idea": True},
        ]},
        "hybrid_editorial_chunks": [{"decisions": [
            {"clip_id": "prefix", "label": "keep", "confidence": 0.95},
        ]}],
        "take_judge_groups": [{"member_usability": {
            "fuller": {"deterministic_unusable": False, "delete_recommended": False},
        }}],
    }

    move, add, audit = deterministic_retry_resolution((prefix,), (), (fuller,), diagnostics)

    assert move == set()
    assert add == set()
    assert audit == []


def test_semantic_superset_retry_requires_explicit_nonfailure_evidence():
    prefix = _clip("prefix", 10.0, 12.0, "After my contract I asked my doctor")
    fuller = _clip(
        "fuller", 15.0, 22.0,
        "After my contract I asked my doctor for every available test.",
    )
    diagnostics = {
        "semantic_idea_equivalence": {"merges": [{
            "left_clip_id": "prefix", "right_clip_id": "fuller",
            "accepted_by": "same_opening_restart", "confidence": 1.0,
        }]},
        "attempt_reconstruction": {"attempts": [
            {"clip_id": "prefix", "complete_idea": True},
            {"clip_id": "fuller", "complete_idea": True},
        ]},
        "hybrid_editorial_chunks": [{"decisions": [
            {"clip_id": "prefix", "label": "keep", "confidence": 0.95},
        ]}],
    }

    move, add, audit = deterministic_retry_resolution((prefix,), (), (fuller,), diagnostics)

    assert move == set()
    assert add == set()
    assert audit == []


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


def test_terminal_negation_attempt_yields_to_full_audience_retry():
    abandoned = _clip("abandoned", 10.0, 11.6, "Tuve problemas de estómago, no.")
    retry = _clip(
        "retry", 16.8, 26.3,
        "Tuve problemas de digestión en donde me hicieron una endoscopía y dijeron que tenía gastritis.",
    )
    diagnostics = {
        "attempt_reconstruction": {"attempts": [
            {"clip_id": "abandoned", "complete_idea": True},
            {"clip_id": "retry", "complete_idea": True},
        ]},
        "clean_cut_judge": [
            {"clip_id": "abandoned", "audiovisual": {"observations": [
                {"role": "mixed", "confidence": 0.85},
            ]}},
            {"clip_id": "retry", "audiovisual": {"observations": [
                {"role": "audience", "confidence": 0.95},
            ]}},
        ],
    }

    move, audit = abandoned_negated_restart_ids((abandoned, retry), diagnostics)
    assert move == {"abandoned"}
    assert audit[0]["reason"] == "terminal_negation_abandoned_restart"


def test_terminal_negation_without_independent_roles_fails_open():
    abandoned = _clip("abandoned", 10.0, 11.6, "I had stomach trouble, no.")
    retry = _clip(
        "retry", 16.8, 26.3,
        "I had digestive trouble and the endoscopy confirmed a mild gastritis diagnosis.",
    )
    move, audit = abandoned_negated_restart_ids((abandoned, retry), {})
    assert move == set() and audit == []


def test_orphaned_anaphoric_fragment_uses_confirmed_retry_proxy():
    fragment = _clip("fragment", 10.0, 12.5, "Era como un rash, una alergia.")
    proxy = _clip(
        "proxy", 13.8, 22.0,
        "También aparecía una alergia detrás de la oreja y en el cuello.",
    )
    winner = _clip(
        "winner", 25.0, 34.0,
        "Otro síntoma era una alergia detrás de la oreja y en todo el cuello por temporadas.",
    )
    diagnostics = {"semantic_idea_equivalence": {"merges": [{
        "left_clip_id": "proxy", "right_clip_id": "winner", "confidence": 0.90,
    }]}}

    move, audit = orphaned_anaphoric_retry_fragment_ids(
        (fragment, winner), (), (proxy,), diagnostics,
    )
    assert move == {"fragment"}
    assert audit[0]["reason"] == "orphaned_anaphoric_fragment_of_confirmed_retry"


def test_anaphoric_fragment_without_confirmed_proxy_fails_open():
    fragment = _clip("fragment", 10.0, 12.5, "It was like an unusual rash.")
    winner = _clip("winner", 25.0, 34.0, "A later explanation of a different event.")
    move, audit = orphaned_anaphoric_retry_fragment_ids((fragment, winner), (), (), {})
    assert move == set() and audit == []
