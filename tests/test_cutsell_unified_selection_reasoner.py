from dataclasses import replace
import json

from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.unified_selection_google import (
    build_unified_selection_payload,
    build_unified_selection_request,
)
from cutsell_worker.unified_selection_reasoner import (
    UnifiedSelectionDecision,
    UnifiedSelectionPlan,
    UnifiedTakeCompetition,
    apply_unified_selection_reasoner,
)


def clip(clip_id, start, end, text, *, selected):
    return DraftClip(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
        caption_text=text,
        selected=selected,
    )


def draft():
    selected = clip("selected_old", 10.0, 15.0, "First attempt of the idea.", selected=True)
    swap = clip("swap_old", 16.0, 22.0, "Clean continuation with unique information.", selected=False)
    discarded = clip("discarded_old", 23.0, 30.0, "A clean later take that local rules removed.", selected=False)
    return DraftTimeline(
        schema_version=SCHEMA_VERSION,
        project_id="p",
        strategy=EditStrategy.STORYTELLING,
        selected=(selected,),
        alternates=(swap,),
        discarded=(discarded,),
        diagnostics={
            "whole_video_context": {
                "dominant_edit_mode": "natural",
                "sources": [{
                    "source_asset_id": "src",
                    "summary": "Creator explains one experience with multiple takes.",
                    "creator_intent": "tell the story cleanly",
                    "main_topic": "experience",
                    "story_logic": "chronological",
                    "dominant_style": "talking head",
                    "edit_mode": "natural",
                }],
            },
            "hybrid_editorial_chunks": [{
                "decisions": [
                    {"clip_id": "selected_old", "label": "alternate", "confidence": 0.82},
                    {"clip_id": "discarded_old", "label": "winner", "confidence": 0.91},
                ]
            }],
        },
    )


class FakeReasoner:
    def __init__(self, decisions):
        self.decisions = decisions

    def reason(self, _draft):
        return UnifiedSelectionPlan(
            decisions=tuple(self.decisions),
            provider="fake",
            model="human-style-test",
            estimated_input_tokens=100,
            estimated_output_tokens=40,
        )


def decision(clip_id, action, relation, confidence, family, reason):
    return UnifiedSelectionDecision(
        clip_id=clip_id,
        action=action,
        relation=relation,
        confidence=confidence,
        family_index=family,
        reason_code=reason,
    )


def v2_decision(clip_id, action, relation, confidence, family, reason, sequence):
    return UnifiedSelectionDecision(
        clip_id=clip_id, action=action, relation=relation, confidence=confidence,
        family_index=family, reason_code=reason, sequence_index=sequence,
    )


def test_unified_reasoner_can_overturn_legacy_buckets_and_preserve_natural_order():
    reasoner = FakeReasoner([
        decision("selected_old", "swap", "retry_alternate", 0.91, 0, "usable_alternate"),
        decision("swap_old", "select", "continuation", 0.94, 1, "necessary_continuation"),
        decision("discarded_old", "select", "retry_winner", 0.96, 0, "best_complete_take"),
    ])

    out = apply_unified_selection_reasoner(draft(), reasoner)

    assert [item.clip_id for item in out.selected] == ["swap_old", "discarded_old"]
    assert [item.clip_id for item in out.alternates] == ["selected_old"]
    assert out.discarded == ()
    diag = out.diagnostics["unified_selection_reasoner"]
    assert diag["status"] == "applied"
    assert diag["selected_count"] == 2


def test_uncertain_never_destructively_deletes_content():
    reasoner = FakeReasoner([
        decision("selected_old", "discard", "uncertain", 0.50, 0, "uncertain_preserve"),
        decision("swap_old", "discard", "uncertain", 0.60, 1, "uncertain_preserve"),
        decision("discarded_old", "discard", "uncertain", 0.60, 2, "uncertain_preserve"),
    ])

    out = apply_unified_selection_reasoner(draft(), reasoner)

    assert [item.clip_id for item in out.selected] == ["selected_old"]
    assert [item.clip_id for item in out.alternates] == ["swap_old", "discarded_old"]
    assert out.discarded == ()


def test_incomplete_provider_plan_fails_open_to_previous_draft():
    original = draft()
    reasoner = FakeReasoner([
        decision("selected_old", "discard", "failed", 0.99, 0, "failed_delivery"),
    ])

    out = apply_unified_selection_reasoner(original, reasoner)

    assert out.selected == original.selected
    assert out.alternates == original.alternates
    assert out.discarded == original.discarded
    assert out.diagnostics["unified_selection_reasoner"]["status"] == "provider_error_fail_open"


def test_payload_contains_complete_candidate_universe_and_global_context():
    payload = build_unified_selection_payload(draft())

    assert [row["clip_id"] for row in payload["candidates"]] == [
        "selected_old", "swap_old", "discarded_old"
    ]
    assert [row["current_bucket"] for row in payload["candidates"]] == [
        "select", "swap", "discard"
    ]
    assert payload["source_context"]["sources"][0]["story_logic"] == "chronological"
    assert payload["candidates"][0]["hybrid_votes"][0]["label"] == "alternate"


def draft_with_three_clips_in_one_family():
    a = clip("a", 0.0, 5.0, "First attempt, cut off mid", selected=False)
    b = clip("b", 5.0, 10.0, "Second attempt, also incomplete", selected=False)
    c = clip("c", 10.0, 15.0, "Third attempt, clean and complete.", selected=False)
    return DraftTimeline(
        schema_version=SCHEMA_VERSION,
        project_id="p",
        strategy=EditStrategy.STORYTELLING,
        selected=(),
        alternates=(a, b, c),
        discarded=(),
    )


# --- RAW #122 audit: a retry family must produce exactly one SELECT ------
#
# The reasoner itself was found selecting multiple takes from the same
# retry family, including a clip whose own reason_code said it was merely a
# "usable_alternate" (SWAP-tier by the editorial contract's own definition)
# or a "failed_delivery" (DISCARD-tier). Nothing in _effective_action or the
# family-application loop caught either contradiction, nor did anything cap
# how many retry_winner/retry_alternate decisions in one family could reach
# SELECT. These tests pin the general (non-Video00-specific) fix.

def test_select_action_contradicting_failed_delivery_reason_is_forced_to_discard():
    reasoner = FakeReasoner([
        decision("a", "select", "failed", 0.95, 0, "failed_delivery"),
    ])
    draft_obj = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(), alternates=(clip("a", 0.0, 5.0, "cut off mid", selected=False),), discarded=(),
    )

    out = apply_unified_selection_reasoner(draft_obj, reasoner)

    assert out.discarded[0].clip_id == "a"
    assert out.selected == ()
    diag = out.diagnostics["unified_selection_reasoner"]["decisions"][0]
    assert diag["safety_override"] == "failed_delivery_reason_overrides_select_action"


def test_select_action_contradicting_usable_alternate_reason_is_forced_to_swap():
    reasoner = FakeReasoner([
        decision("a", "select", "retry_alternate", 0.9, 0, "usable_alternate"),
    ])
    draft_obj = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(), alternates=(clip("a", 0.0, 5.0, "a usable but non-winning take", selected=False),), discarded=(),
    )

    out = apply_unified_selection_reasoner(draft_obj, reasoner)

    assert out.alternates[0].clip_id == "a"
    assert out.selected == ()
    diag = out.diagnostics["unified_selection_reasoner"]["decisions"][0]
    assert diag["safety_override"] == "usable_alternate_reason_overrides_select_action"


def test_retry_family_with_multiple_selects_keeps_only_the_highest_confidence_winner():
    reasoner = FakeReasoner([
        decision("a", "select", "retry_alternate", 0.80, 0, "best_complete_take"),
        decision("b", "select", "retry_winner", 0.99, 0, "best_complete_take"),
        decision("c", "select", "retry_alternate", 0.85, 0, "best_complete_take"),
    ])

    out = apply_unified_selection_reasoner(draft_with_three_clips_in_one_family(), reasoner)

    assert [item.clip_id for item in out.selected] == ["b"]
    assert sorted(item.clip_id for item in out.alternates) == ["a", "c"]
    diag_by_id = {row["clip_id"]: row for row in out.diagnostics["unified_selection_reasoner"]["decisions"]}
    assert diag_by_id["a"]["safety_override"] == "retry_family_single_winner_enforced"
    assert diag_by_id["c"]["safety_override"] == "retry_family_single_winner_enforced"
    assert diag_by_id["b"]["safety_override"] is None


def test_retry_family_demoted_losers_go_to_swap_never_discard():
    # A candidate good enough to reach SELECT before the single-winner rule
    # applies is not thrown away -- it stays available for manual
    # replacement, exactly like any other SWAP.
    reasoner = FakeReasoner([
        decision("a", "select", "retry_winner", 0.60, 0, "best_complete_take"),
        decision("b", "select", "retry_alternate", 0.99, 0, "best_complete_take"),
    ])
    draft_obj = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(), alternates=(
            clip("a", 0.0, 5.0, "a weaker complete take", selected=False),
            clip("b", 5.0, 10.0, "the clean winning take", selected=False),
        ), discarded=(),
    )

    out = apply_unified_selection_reasoner(draft_obj, reasoner)

    assert [item.clip_id for item in out.selected] == ["b"]
    assert [item.clip_id for item in out.discarded] == []
    assert [item.clip_id for item in out.alternates] == ["a"]


def test_independent_relation_family_allows_multiple_selects_untouched():
    # The single-winner rule only applies to relation retry_winner/
    # retry_alternate -- independent story beats sharing a family_index (or
    # composite/continuation pieces) must not be capped to one SELECT.
    reasoner = FakeReasoner([
        decision("a", "select", "independent", 0.9, 0, "independent_story_coverage"),
        decision("b", "select", "independent", 0.9, 0, "independent_story_coverage"),
        decision("c", "select", "continuation", 0.9, 0, "necessary_continuation"),
    ])

    out = apply_unified_selection_reasoner(draft_with_three_clips_in_one_family(), reasoner)

    assert sorted(item.clip_id for item in out.selected) == ["a", "b", "c"]
    assert out.alternates == ()


def test_v2_preserves_usable_retry_alternate_with_materially_unique_information():
    hook = clip("hook", 0, 5, "If you use GLP this gives repetitions energy and stronger muscles", selected=False)
    winner = clip("winner", 10, 15, "Creatine watermelon flavor mixes into one bottle daily", selected=False)
    d = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(), alternates=(hook, winner), discarded=(),
        diagnostics={"editorial_engine_v2_request": {"require_audiovisual_evidence": True}},
    )
    reasoner = FakeReasoner([
        v2_decision("hook", "swap", "retry_alternate", .9, 0, "usable_alternate", 0),
        v2_decision("winner", "select", "retry_winner", .98, 0, "best_complete_take", 1),
    ])

    out = apply_unified_selection_reasoner(d, reasoner)

    assert [item.clip_id for item in out.selected] == ["hook", "winner"]
    row = next(row for row in out.diagnostics["unified_selection_reasoner"]["decisions"]
               if row["clip_id"] == "hook")
    assert row["safety_override"] == "material_retry_claim_preserved"


def test_v2_does_not_preserve_usable_retry_alternate_that_winner_covers():
    alternate = clip("alternate", 0, 5, "Creatine watermelon flavor mixes in one bottle", selected=False)
    winner = clip("winner", 10, 15, "Creatine watermelon flavor mixes in one bottle every day", selected=False)
    d = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(), alternates=(alternate, winner), discarded=(),
        diagnostics={"editorial_engine_v2_request": {"require_audiovisual_evidence": True}},
    )
    reasoner = FakeReasoner([
        v2_decision("alternate", "swap", "retry_alternate", .9, 0, "usable_alternate", 0),
        v2_decision("winner", "select", "retry_winner", .98, 0, "best_complete_take", 1),
    ])

    out = apply_unified_selection_reasoner(d, reasoner)

    assert [item.clip_id for item in out.selected] == ["winner"]
    assert [item.clip_id for item in out.alternates] == ["alternate"]


def test_v2_explicit_whole_take_coverage_removes_prior_selected_fragments():
    d = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(), alternates=(
            clip("fragment1", 0, 5, "Specific opening claim", selected=False),
            clip("fragment2", 5, 10, "Distinct middle demonstration", selected=False),
            clip("complete", 20, 35, "Specific opening claim distinct middle demonstration and close", selected=False),
        ), discarded=(), diagnostics={"editorial_engine_v2_request": {"require_audiovisual_evidence": True}},
    )
    class ContestReasoner(FakeReasoner):
        def __init__(self, relation="equivalent_take", confidence=.96, covered=("fragment1", "fragment2")):
            self.relation, self.confidence, self.covered = relation, confidence, covered

        def reason(self, _draft):
            return UnifiedSelectionPlan(
                decisions=(
                    v2_decision("fragment1", "select", "independent", .98, 0, "independent_story_coverage", 0),
                    v2_decision("fragment2", "select", "independent", .98, 1, "independent_story_coverage", 1),
                    v2_decision("complete", "select", "independent", .98, 2, "best_complete_take", 2),
                ), provider="fake", model="test",
                take_competitions=(UnifiedTakeCompetition(("complete",), self.covered, (),
                                                        self.relation, self.confidence, "covered"),),
            )

    result = apply_unified_selection_reasoner(d, ContestReasoner())
    assert [item.clip_id for item in result.selected] == ["complete"]
    assert result.diagnostics["v2_take_competitions"][0]["decision"] == "covered_alternates_removed"
    for relation, confidence in (("complementary", .99), ("equivalent_take", .89)):
        result = apply_unified_selection_reasoner(d, ContestReasoner(relation, confidence))
        assert [item.clip_id for item in result.selected] == ["fragment1", "fragment2", "complete"]
    result = apply_unified_selection_reasoner(d, ContestReasoner(covered=("fragment1",)))
    assert [item.clip_id for item in result.selected] == ["fragment2", "complete"]


def test_v2_cyclic_whole_take_claims_preserve_both_regardless_of_order():
    d = DraftTimeline(schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
                      selected=(clip("a", 0, 5, "opening", selected=True),
                                clip("b", 10, 15, "closing", selected=True)),
                      alternates=(), discarded=(),
                      diagnostics={"editorial_engine_v2_request": {"require_audiovisual_evidence": True}})
    class Cyclic(FakeReasoner):
        def __init__(self, contests):
            self.contests = contests

        def reason(self, _draft):
            return UnifiedSelectionPlan(decisions=(
                v2_decision("a", "select", "independent", .99, 0, "best_complete_take", 0),
                v2_decision("b", "select", "independent", .99, 1, "best_complete_take", 1),
            ), provider="fake", model="test", take_competitions=self.contests)

    contests = (UnifiedTakeCompetition(("a",), ("b",), (), "equivalent_take", .99),
                UnifiedTakeCompetition(("b",), ("a",), (), "equivalent_take", .99))
    for order in (contests, contests[::-1]):
        result = apply_unified_selection_reasoner(d, Cyclic(order))
        assert [item.clip_id for item in result.selected] == ["a", "b"]
        assert all(c["decision"] == "contradictory_competitions_preserved"
                   for c in result.diagnostics["v2_take_competitions"])


def test_v2_three_way_and_chain_claims_do_not_depend_on_comparison_order():
    from cutsell_worker.unified_selection_reasoner import _apply_v2_take_competitions
    clips = tuple(clip(name, i * 10, i * 10 + 5, name, selected=True)
                  for i, name in enumerate(("a", "b", "c")))
    decisions = {item.clip_id: v2_decision(item.clip_id, "select", "independent", .99, i,
                                           "best_complete_take", i) for i, item in enumerate(clips)}
    a_b = UnifiedTakeCompetition(("a",), ("b",), (), "equivalent_take", .99)
    b_c = UnifiedTakeCompetition(("b",), ("c",), (), "equivalent_take", .99)
    c_a = UnifiedTakeCompetition(("c",), ("a",), (), "equivalent_take", .99)
    for contests in ((a_b, b_c), (a_b, b_c, c_a)):
        for order in (contests, contests[::-1]):
            actions, overrides = ["select"] * 3, [None] * 3
            _apply_v2_take_competitions(clips, decisions, actions, overrides, order)
            assert actions == ["select"] * 3


def test_v2_whole_take_cannot_claim_to_cover_absent_purchase_action():
    from cutsell_worker.unified_selection_reasoner import _apply_v2_take_competitions, _has_purchase_action
    body = clip("body", 120, 144, "Este jumpsuit es el mejor que encontre en TikTok Shop", selected=True)
    cta = clip("cta", 145, 147, "Lo puedes encontrar aqui en el carrito naranja", selected=False)
    assert not _has_purchase_action(body.text)
    assert _has_purchase_action(cta.text)
    assert _has_purchase_action("Puedes encontrarlo en el carrito")
    assert _has_purchase_action("Puedes comprarlo en la tienda")
    assert _has_purchase_action("Puedes conseguirlo en el enlace")
    assert not _has_purchase_action("I did not buy it at that store")
    assert not _has_purchase_action("No puedes encontrarlo en el carrito")
    assert not _has_purchase_action("Don't click the link")
    assert _has_purchase_action("No tiene azúcar. Puedes comprarlo en el carrito")
    decisions = {"body": v2_decision("body", "select", "retry_winner", .99, 0, "best_complete_take", 0),
                 "cta": v2_decision("cta", "discard", "retry_alternate", .95, 0, "redundant_retry", 1)}
    actions, overrides = ["select", "discard"], [None, None]
    audit = _apply_v2_take_competitions((body, cta), decisions, actions, overrides,
                                        (UnifiedTakeCompetition(("body",), ("cta",), (), "equivalent_take", .98),))
    assert actions == ["select", "select"]
    assert overrides[1] == "purchase_action_not_covered_by_winner"
    assert audit[0]["decision"] == "purchase_action_coverage_conflict"
    complete = replace(body, text="Compra este jumpsuit ahora en la tienda")
    actions, overrides = ["select", "discard"], [None, None]
    _apply_v2_take_competitions((complete, cta), decisions, actions, overrides,
                                 (UnifiedTakeCompetition(("body",), ("cta",), (), "equivalent_take", .98),))
    assert actions == ["select", "discard"]
    final_cta = replace(cta, clip_id="final_cta", start=150, end=153,
                        text="Puedes comprarlo ahora en el carrito")
    decisions["final_cta"] = v2_decision("final_cta", "select", "independent", .96, 2,
                                         "necessary_continuation", 2)
    actions, overrides = ["select", "discard", "select"], [None] * 3
    _apply_v2_take_competitions((body, cta, final_cta), decisions, actions, overrides,
                                 (UnifiedTakeCompetition(("body",), ("cta",), (), "equivalent_take", .98),))
    assert actions == ["select", "discard", "select"]
    different_destination = replace(final_cta, text="Puedes comprarlo por el enlace de la bio")
    actions, overrides = ["select", "discard", "select"], [None] * 3
    _apply_v2_take_competitions((body, cta, different_destination), decisions, actions, overrides,
                                 (UnifiedTakeCompetition(("body",), ("cta",), (), "equivalent_take", .98),))
    assert actions == ["select", "select", "select"]


def test_v2_unique_selected_cta_prevents_rescuing_duplicate_discarded_cta():
    from cutsell_worker.unified_selection_reasoner import _apply_v2_take_competitions
    chosen = clip("chosen", 80, 96,
                  "Lo puedes encontrar en el carrito y dura muchísimo", selected=True)
    duplicate = clip("duplicate", 63, 79,
                     "Lo puedes encontrar en el carrito y déjame comentarios", selected=False)
    decisions = {
        "chosen": v2_decision("chosen", "select", "retry_winner", .98, 0, "best_complete_take", 1),
        "duplicate": v2_decision("duplicate", "discard", "retry_alternate", .95, 0,
                                  "redundant_retry", 0),
    }
    actions, overrides = ["select", "discard"], [None, None]
    audit = _apply_v2_take_competitions(
        (duplicate, chosen), decisions, actions, overrides,
        (UnifiedTakeCompetition(("chosen",), ("duplicate",), (),
                                "equivalent_take", .95, "Same delivery and destination"),),
    )
    assert actions == ["select", "discard"]
    assert overrides == [None, None]
    assert audit[0]["decision"] == "winner_not_confirmed_selected"


def test_v2_equivalence_cannot_erase_failed_delivery_decision():
    from cutsell_worker.unified_selection_reasoner import _apply_v2_take_competitions
    failed = clip("failed", 51, 69, "We were laughing and could barely stay in character", selected=True)
    winner = clip("winner", 28, 50, "We ordered a pizza and had fun making the video", selected=True)
    decisions = {
        "failed": v2_decision("failed", "discard", "failed", .90, 0, "failed_delivery", 1),
        "winner": v2_decision("winner", "select", "retry_winner", .98, 1, "best_complete_take", 0),
    }
    actions, overrides = ["select", "select"], [None, None]
    audit = _apply_v2_take_competitions(
        (failed, winner), decisions, actions, overrides,
        (UnifiedTakeCompetition(("winner",), ("failed",), (),
                                "equivalent_take", .95, "Same overall narrative"),),
    )
    assert actions == ["select", "select"]
    assert overrides == [None, None]
    assert audit[0]["decision"] == "failed_delivery_preserved_against_equivalence"


def test_v2_confirmed_winner_removes_failed_take_rescued_by_broad_av():
    from cutsell_worker.unified_selection_reasoner import _apply_v2_take_competitions
    failed = clip("failed", 4, 15, "This perfume makes the room smell like cherry", selected=True)
    winner = clip("winner", 85, 105, "This perfume makes the room smell like cherry", selected=True)
    decisions = {
        "failed": v2_decision("failed", "discard", "failed", .96, 0, "failed_delivery", 0),
        "winner": v2_decision("winner", "select", "retry_winner", .98, 0, "best_complete_take", 1),
    }
    actions = ["select", "select"]
    overrides = ["av_audience_unique_content_overrides_failed_label", None]
    audit = _apply_v2_take_competitions(
        (failed, winner), decisions, actions, overrides,
        (UnifiedTakeCompetition(("winner",), ("failed",), (),
                                "equivalent_take", .96, "Same complete claim"),),
    )
    assert actions == ["discard", "select"]
    assert overrides[0] == "whole_take_equivalent_covered"
    assert audit[0]["decision"] == "covered_alternates_removed"


def test_v2_broad_av_rescue_retains_uncovered_quantity_despite_competition():
    from cutsell_worker.unified_selection_reasoner import _apply_v2_take_competitions
    failed = clip("failed", 4, 15, "Get stronger in 30 days", selected=True)
    winner = clip("winner", 85, 105, "Get stronger with daily exercise", selected=True)
    decisions = {
        "failed": v2_decision("failed", "discard", "failed", .96, 0, "failed_delivery", 0),
        "winner": v2_decision("winner", "select", "retry_winner", .98, 0, "best_complete_take", 1),
    }
    actions = ["select", "select"]
    overrides = ["av_audience_unique_content_overrides_failed_label", None]
    audit = _apply_v2_take_competitions(
        (failed, winner), decisions, actions, overrides,
        (UnifiedTakeCompetition(("winner",), ("failed",), (),
                                "equivalent_take", .96, "Equivalent claim"),),
    )
    assert actions == ["select", "select"]
    assert audit[0]["decision"] == "material_claim_not_covered_by_winner"


def test_v2_cta_support_cannot_depend_on_another_proposed_deletion():
    from cutsell_worker.unified_selection_reasoner import _apply_v2_take_competitions
    body_a = clip("body_a", 0, 5, "Product review", selected=True)
    body_b = clip("body_b", 10, 15, "Product details", selected=True)
    a = clip("cta_a", 6, 8, "Puedes encontrarlo en el carrito", selected=True)
    b = clip("cta_b", 16, 18, "Puedes comprarlo en el carrito", selected=True)
    clips = (body_a, a, body_b, b)
    decisions = {item.clip_id: v2_decision(item.clip_id, "select", "independent", .98, i,
                                           "best_complete_take", i) for i, item in enumerate(clips)}
    contests = (UnifiedTakeCompetition(("body_a",), ("cta_a",), (), "equivalent_take", .98),
                UnifiedTakeCompetition(("body_b",), ("cta_b",), (), "equivalent_take", .98))
    for order in (contests, contests[::-1]):
        actions, overrides = ["select"] * 4, [None] * 4
        _apply_v2_take_competitions(clips, decisions, actions, overrides, order)
        assert actions == ["select"] * 4


def test_v2_preserves_redundant_retry_label_when_information_is_not_redundant():
    detail = clip("detail", 0, 5, "Stronger at the gym with a noticeable difference in 30 days", selected=False)
    winner = clip("winner", 10, 15, "Watermelon creatine mixes into one daily bottle", selected=False)
    d = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(), alternates=(detail, winner), discarded=(),
        diagnostics={"editorial_engine_v2_request": {"require_audiovisual_evidence": True}},
    )
    reasoner = FakeReasoner([
        v2_decision("detail", "discard", "retry_alternate", .95, 0, "redundant_retry", 0),
        v2_decision("winner", "select", "retry_winner", .98, 0, "best_complete_take", 1),
    ])

    out = apply_unified_selection_reasoner(d, reasoner)

    assert [item.clip_id for item in out.selected] == ["detail", "winner"]


def test_v2_verbal_summary_cannot_cover_distinct_selected_silent_action():
    from cutsell_worker.unified_selection_reasoner import _apply_v2_take_competitions
    action = replace(clip('action', 10, 15, '', selected=False), audio_muted=True)
    later = clip('later', 17, 26, 'Mix the product in water daily', selected=False)
    clips = (action, later)
    decisions = {item.clip_id: v2_decision(item.clip_id, 'select', 'independent',
                                         .95, i, 'independent_story_coverage', i)
                 for i, item in enumerate(clips)}
    contest = UnifiedTakeCompetition(('later',), ('action',), (), 'equivalent_take', .97,
                                     'Narration summarizes mixing')
    actions, overrides = ['select', 'select'], [None, None]
    audit = _apply_v2_take_competitions(clips, decisions, actions, overrides, (contest,))
    assert actions == ['select', 'select']
    assert audit[0]['decision'] == 'distinct_focused_visual_action_preserved'
    # The same original frames already included in the winner can be folded.
    later = replace(later, start=9, end=26)
    clips = (action, later)
    actions, overrides = ['select', 'select'], [None, None]
    _apply_v2_take_competitions(clips, decisions, actions, overrides, (contest,))
    assert actions == ['discard', 'select']


def test_v2_equivalent_take_cannot_erase_missing_amount_or_audience_condition():
    from cutsell_worker.unified_selection_reasoner import _apply_v2_take_competitions
    condition = clip('condition', 0, 6, 'Si tu estas usando creatina tienes mas energia', selected=False)
    amount = clip('amount', 6, 10, 'En 30 dias vas a ver resultados', selected=False)
    winner = clip('winner', 30, 40, 'El suplemento te da energia y resultados', selected=False)
    clips = (condition, amount, winner)
    decisions = {item.clip_id: v2_decision(item.clip_id, 'select', 'independent',
                                         .98, i, 'independent_story_coverage', i)
                 for i, item in enumerate(clips)}
    contest = UnifiedTakeCompetition(('winner',), ('condition', 'amount'), (),
                                     'equivalent_take', .99, 'Everything is covered')
    actions, overrides = ['select'] * 3, [None] * 3
    audit = _apply_v2_take_competitions(clips, decisions, actions, overrides, (contest,))
    assert actions == ['select'] * 3
    assert audit[0]['decision'] == 'material_claim_not_covered_by_winner'
    # A winner that actually states the same audience and quantity may win.
    winner = replace(winner, text='Si tu estas usando creatina, en 30 dias tienes energia')
    actions, overrides = ['select'] * 3, [None] * 3
    _apply_v2_take_competitions((condition, amount, winner), decisions,
                                actions, overrides, (contest,))
    assert actions == ['discard'] * 2 + ['select']
    winner = replace(winner, text='Si tu estas usando creatina, con 30 gramos tienes energia')
    actions, overrides = ['select'] * 3, [None] * 3
    _apply_v2_take_competitions((condition, amount, winner), decisions,
                                actions, overrides, (contest,))
    assert actions == ['discard', 'select', 'select']


def test_v2_av_audience_unique_content_overrides_false_failed_label():
    hook = clip("hook", 5, 16, "GLP users get more repetitions energy and stronger muscles", selected=False)
    winner = clip("winner", 50, 70, "Watermelon creatine mixes into one bottle daily", selected=False)
    d = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(), alternates=(hook, winner), discarded=(),
        diagnostics={
            "editorial_engine_v2_request": {"require_audiovisual_evidence": True},
            "whole_video_context": {"sources": [{
                "source_asset_id": "src",
                "audiovisual_evidence": '{"regions":[{"start":5.5,"end":15.5,"role":"audience","confidence":0.96}]}',
            }]},
        },
    )
    reasoner = FakeReasoner([
        v2_decision("hook", "discard", "failed", .9, 0, "failed_delivery", 0),
        v2_decision("winner", "select", "retry_winner", .98, 0, "best_complete_take", 1),
    ])

    out = apply_unified_selection_reasoner(d, reasoner)

    assert [item.clip_id for item in out.selected] == ["hook", "winner"]
    row = next(row for row in out.diagnostics["unified_selection_reasoner"]["decisions"]
               if row["clip_id"] == "hook")
    assert row["safety_override"] == "av_audience_unique_content_overrides_failed_label"


def test_v2_av_majority_audience_preserves_unique_delivery_with_short_failed_tail():
    mixed = clip("mixed", 19, 29, "Thirty day results make muscles stronger with creatine", selected=False)
    winner = clip("winner", 50, 70, "Watermelon flavor mixes into one bottle daily", selected=False)
    d = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(), alternates=(mixed, winner), discarded=(),
        diagnostics={
            "editorial_engine_v2_request": {"require_audiovisual_evidence": True},
            "whole_video_context": {"sources": [{
                "source_asset_id": "src",
                "audiovisual_evidence": '{"regions":[{"start":19.3,"end":26.5,"role":"audience","confidence":0.95}]}',
            }]},
        },
    )
    reasoner = FakeReasoner([
        v2_decision("mixed", "discard", "failed", .9, 0, "failed_delivery", 0),
        v2_decision("winner", "select", "retry_winner", .98, 0, "best_complete_take", 1),
    ])
    out = apply_unified_selection_reasoner(d, reasoner)
    assert [item.clip_id for item in out.selected] == ["mixed", "winner"]


def test_v2_focused_av_rescues_only_word_aligned_clean_delivery_from_mixed_candidate():
    from cutsell_worker.contracts import Word

    mixed = replace(
        clip("mixed", 19.218, 28.14,
             "Y aparte tus músculos van a estar más fuerte en 30 días tú vas a ver resultados", selected=False),
        words=(Word("Y", 19.22, 19.35), Word("aparte", 19.36, 19.72),
               Word("tus", 19.73, 19.91), Word("músculos", 19.92, 20.42),
               Word("van", 20.43, 20.64), Word("a", 20.65, 20.72),
               Word("estar", 20.73, 21.10), Word("más", 21.11, 21.34),
               Word("fuerte", 21.35, 21.78), Word("en", 21.79, 21.91),
               Word("30", 21.92, 22.20), Word("días", 22.21, 22.56),
               Word("resultados", 23.5, 24.1), Word("reset", 26.4, 27.0)),
    )
    winner = clip("winner", 50, 70,
                  "Watermelon flavor mixes into one bottle daily", selected=False)
    d = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(), alternates=(mixed, winner), discarded=(),
        diagnostics={
            "editorial_engine_v2_request": {"require_audiovisual_evidence": True},
            "whole_video_context": {"sources": [{
                "source_asset_id": "src",
                "audiovisual_evidence": json.dumps({
                    "regions": [{"start": 17.5, "end": 26, "role": "mixed", "confidence": .9}],
                    "focused_delivery_regions": [{
                        "start": 18.6, "end": 24.3, "role": "audience", "confidence": .96,
                    }],
                }),
            }]},
        },
    )
    reasoner = FakeReasoner([
        v2_decision("mixed", "discard", "failed", .9, 0, "failed_delivery", 0),
        v2_decision("winner", "select", "retry_winner", .98, 0, "best_complete_take", 1),
    ])

    out = apply_unified_selection_reasoner(d, reasoner)

    refined = next(item for item in out.selected if item.clip_id == "mixed")
    assert refined.start == 19.22
    assert refined.end == 24.1
    assert "reset" not in refined.text
    row = next(row for row in out.diagnostics["unified_selection_reasoner"]["decisions"]
               if row["clip_id"] == "mixed")
    assert row["safety_override"] == "focused_av_clean_delivery_preserved"


def test_unified_request_requires_one_structured_human_style_decision_per_candidate():
    payload = build_unified_selection_payload(draft())
    request = build_unified_selection_request(payload, max_output_tokens=1000)

    schema = request["generationConfig"]["responseJsonSchema"]
    decisions = schema["properties"]["decisions"]
    # No exact-length array bound: an isolation probe
    # (scripts/isolate_unified_selection_schema.py) proved Gemini's
    # structured-output validator rejects minItems==maxItems at whole-video
    # scale (works at 5 candidates, 400s at 90) even with this exact model and
    # even with a much simpler schema. GoogleUnifiedSelectionReasoner.reason()
    # still enforces exactly one decision per candidate downstream in Python
    # ("unified Selection ordered decision count mismatch"), so the wire schema
    # must never re-add this bound.
    assert "minItems" not in decisions
    assert "maxItems" not in decisions
    properties = decisions["items"]["properties"]
    assert set(properties["action"]["enum"]) == {"select", "swap", "discard"}
    assert "composite_piece" in properties["relation"]["enum"]
    assert "continuation" in properties["relation"]["enum"]
    # RAW #120: a normal-STOP response undercounted by one with no length
    # bound to catch it. candidate_index (validated downstream in
    # GoogleUnifiedSelectionReasoner._call_once) is required so a short,
    # reordered, or duplicated response is always caught with the exact
    # index named, not just a bare count mismatch.
    assert "candidate_index" in properties
    assert "candidate_index" in decisions["items"]["required"]


def test_v2_verified_continuation_restores_adjacent_member_after_competition():
    from cutsell_worker.contracts import Word
    left = replace(clip('left', 73.8, 77.07, 'You will feel it in your', selected=False),
                   words=(Word('your', 76.8, 77.07),))
    right = replace(clip('right', 77.07, 86.5, 'first week using it', selected=True),
                    words=(Word('first', 77.07, 77.4),))
    d = DraftTimeline(schema_version=SCHEMA_VERSION, project_id='p',
                      strategy=EditStrategy.STORYTELLING, selected=(right,),
                      alternates=(), discarded=(left,),
                      diagnostics={'editorial_engine_v2_request': {'require_audiovisual_evidence': True}})
    class Verified:
        def reason(self, _draft):
            return UnifiedSelectionPlan(decisions=(
                v2_decision('left', 'discard', 'retry_alternate', .9, 1, 'redundant_retry', 0),
                v2_decision('right', 'select', 'independent', .95, 2, 'independent_story_coverage', 1),
            ), provider='test', model='test', continuation_links=(('left', 'right'),))
    output = apply_unified_selection_reasoner(d, Verified())
    assert [c.clip_id for c in output.selected] == ['left', 'right']
    assert output.diagnostics['unified_selection_reasoner']['decisions'][0]['safety_override'] == 'source_verified_necessary_continuation'
    for invalid in (replace(left, source_asset_id='other'),
                    replace(left, end=76.5), replace(left, words=())):
        broken = replace(d, discarded=(invalid,))
        assert apply_unified_selection_reasoner(broken, Verified()).diagnostics['unified_selection_reasoner']['status'] == 'provider_error_fail_open'

    class Reversed(Verified):
        def reason(self, candidate_draft):
            original = super().reason(candidate_draft)
            return replace(original, decisions=tuple(replace(decision,
                sequence_index=1-decision.sequence_index) for decision in original.decisions))
    assert apply_unified_selection_reasoner(d, Reversed()).diagnostics['unified_selection_reasoner']['status'] == 'provider_error_fail_open'


def test_v2_lexical_novelty_in_retry_does_not_override_selected_complete_take():
    earlier = clip('earlier', 20, 28,
                   'Available in every size and a lavender jacket pairs beautifully with it', selected=False)
    later = clip('later', 50, 69,
                 'This suit comes in every color and size and fits comfortably', selected=True)
    d = DraftTimeline(schema_version=SCHEMA_VERSION, project_id='p',
                      strategy=EditStrategy.STORYTELLING, selected=(later,),
                      alternates=(), discarded=(earlier,),
                      diagnostics={'editorial_engine_v2_request': True})
    class SemanticChoice:
        def reason(self, _draft):
            return UnifiedSelectionPlan(decisions=(
                v2_decision('earlier', 'discard', 'retry_alternate', .8, 1,
                            'redundant_retry', 0),
                v2_decision('later', 'select', 'retry_winner', .98, 0,
                            'best_complete_take', 1),
            ), provider='test', model='test')
    output = apply_unified_selection_reasoner(d, SemanticChoice())
    assert [c.clip_id for c in output.selected] == ['later']
    assert [c.clip_id for c in output.discarded] == ['earlier']


def test_v2_objective_quantity_survives_an_erroneous_retry_label():
    earlier = clip('earlier', 20, 28, 'Results in 30 days', selected=False)
    later = clip('later', 50, 69, 'Results with daily use', selected=True)
    d = DraftTimeline(schema_version=SCHEMA_VERSION, project_id='p',
                      strategy=EditStrategy.STORYTELLING, selected=(later,),
                      alternates=(), discarded=(earlier,),
                      diagnostics={'editorial_engine_v2_request': True})
    class SemanticChoice:
        def reason(self, _draft):
            return UnifiedSelectionPlan(decisions=(
                v2_decision('earlier', 'discard', 'retry_alternate', .8, 1,
                            'redundant_retry', 0),
                v2_decision('later', 'select', 'retry_winner', .98, 0,
                            'best_complete_take', 1),
            ), provider='test', model='test')
    output = apply_unified_selection_reasoner(d, SemanticChoice())
    assert [c.clip_id for c in output.selected] == ['earlier', 'later']


def test_v2_retry_claim_guard_handles_written_numbers_negation_and_health_facts():
    from cutsell_worker.unified_selection_reasoner import _preserve_retry_alternates_with_unique_information
    examples = ('resultados en treinta días', 'sin azúcar para el desayuno',
                'contiene alérgenos importantes', 'apto para diabéticos diagnosticados')
    for sentence in examples:
        earlier = clip('earlier', 10, 18, sentence, selected=False)
        later = clip('later', 30, 45, 'Mezcla diaria con agua', selected=True)
        choices = {'earlier': v2_decision('earlier', 'discard', 'retry_alternate', .8, 1,
                                          'redundant_retry', 0),
                   'later': v2_decision('later', 'select', 'retry_winner', .98, 0,
                                        'best_complete_take', 1)}
        actions, overrides = ['discard', 'select'], [None, None]
        audit = _preserve_retry_alternates_with_unique_information(
            (earlier, later), choices, actions, overrides)
        assert actions == ['select', 'select'], (sentence, audit)
        assert audit[0]['status'] == 'material_claim_preserved'


def test_v2_other_source_cannot_cover_material_quantity_and_lexical_conflict_is_audited():
    from cutsell_worker.unified_selection_reasoner import _preserve_retry_alternates_with_unique_information
    earlier = clip('earlier', 10, 18, 'Resultados en 30 días', selected=False)
    other_source = replace(clip('other', 1, 12, 'Resultados en 30 días', selected=True),
                           source_asset_id='other-source')
    later = clip('later', 30, 45, 'Mezcla diaria con agua', selected=True)
    choices = {'earlier': v2_decision('earlier', 'discard', 'retry_alternate', .8, 1,
                                      'redundant_retry', 0),
               'other': v2_decision('other', 'select', 'independent', .9, 2,
                                    'independent_story_coverage', 1),
               'later': v2_decision('later', 'select', 'retry_winner', .98, 0,
                                    'best_complete_take', 2)}
    actions, overrides = ['discard', 'select', 'select'], [None] * 3
    audit = _preserve_retry_alternates_with_unique_information(
        (earlier, other_source, later), choices, actions, overrides)
    assert actions[0] == 'select'
    assert audit[0]['missing_quantity'] == [('30', 'dias')]
    earlier = replace(earlier, text='Una chaqueta brillante combina muy bien')
    actions, overrides = ['discard', 'select', 'select'], [None] * 3
    audit = _preserve_retry_alternates_with_unique_information(
        (earlier, other_source, later), choices, actions, overrides)
    assert actions[0] == 'discard'
    assert audit[0]['status'] == 'model_discard_pending_review'


def test_v2_spanish_impersonal_uno_ve_does_not_resurrect_discarded_take():
    from cutsell_worker.unified_selection_reasoner import _preserve_retry_alternates_with_unique_information
    alternate = clip('alternate', 62, 82, 'a veces uno ve esos cuerpos transformados', selected=False)
    winner = clip('winner', 120, 147, 'un jumpsuit que te queda bien', selected=True)
    decisions = {'alternate': v2_decision('alternate', 'discard', 'retry_alternate', .9, 0,
                                           'redundant_retry', 0),
                 'winner': v2_decision('winner', 'select', 'retry_winner', .98, 0,
                                       'best_complete_take', 1)}
    actions, overrides = ['discard', 'select'], [None, None]
    audit = _preserve_retry_alternates_with_unique_information(
        (alternate, winner), decisions, actions, overrides)
    assert actions == ['discard', 'select']
    assert overrides == [None, None]
    assert audit[0]['missing_quantity'] == []

    for phrase, unit in (('one scoop', 'scoop'), ('one capsule', 'capsule'),
                         ('one tablet', 'tablet'), ('one dose', 'dose'),
                         ('one serving', 'serving'), ('one milliliter', 'milliliter'),
                         ('una dosis', 'dosis'), ('uno mililitro', 'mililitro'),
                         ('uno ve', None)):
        spoken = replace(alternate, text=f'agrega {phrase} de creatina')
        actions, overrides = ['discard', 'select'], [None, None]
        audit = _preserve_retry_alternates_with_unique_information(
            (spoken, winner), decisions, actions, overrides)
        if unit:
            assert actions[0] == 'select', phrase
            assert audit[0]['missing_quantity'] == [('1', unit)]
        else:
            assert actions[0] == 'discard', phrase


def test_quantity_preservation_does_not_authorize_demo_bridge_without_av():
    import json
    from cutsell_worker.unified_selection_reasoner import _preserve_continuous_demonstration
    left = clip('instruction', 96, 99, 'pones una cucharadita en el agua', selected=False)
    right = clip('explanation', 106, 120, 'pones un scoop en el agua', selected=True)
    decisions = {'instruction': v2_decision('instruction', 'discard', 'retry_alternate', .8, 0,
                                             'redundant_retry', 0),
                 'explanation': v2_decision('explanation', 'select', 'retry_winner', .98, 0,
                                             'best_complete_take', 1)}
    region = {'role': 'audience', 'start': 90, 'end': 121, 'confidence': .95,
              'visual_observation': 'Creator talks about a bottle'}
    d = replace(draft(), diagnostics={'whole_video_context': {'sources': [
        {'source_asset_id': 'src', 'audiovisual_evidence': json.dumps({'regions': [region]})}]}})
    actions, overrides = ['select', 'select'], ['material_retry_claim_preserved', None]
    _preserve_continuous_demonstration(d, (left, right), decisions, actions, overrides)
    assert overrides[0] == 'material_retry_claim_preserved'
    region['visual_observation'] = 'Creator mixing the product in a bottle'
    d = replace(d, diagnostics={'whole_video_context': {'sources': [
        {'source_asset_id': 'src', 'audiovisual_evidence': json.dumps({'regions': [region]})}]}})
    _preserve_continuous_demonstration(d, (left, right), decisions, actions, overrides)
    assert overrides[0] == 'av_continuous_demonstration_preserved'


def test_v2_different_protected_health_fact_is_not_covered_by_same_category_word():
    from cutsell_worker.unified_selection_reasoner import _preserve_retry_alternates_with_unique_information
    for earlier_text, winner_text in (
        ('Allergic to peanuts', 'Allergic to milk'),
        ('Diagnosed with diabetes', 'Diagnosed with asthma'),
        ('Ingredient includes nuts', 'Ingredient includes oats'),
    ):
        earlier = clip('early', 1, 5, earlier_text, selected=False)
        winner = clip('winner', 8, 14, winner_text, selected=True)
        choices = {'early': v2_decision('early', 'discard', 'retry_alternate', .9, 1,
                                        'redundant_retry', 0),
                   'winner': v2_decision('winner', 'select', 'retry_winner', .9, 0,
                                         'best_complete_take', 1)}
        actions, overrides = ['discard', 'select'], [None, None]
        audit = _preserve_retry_alternates_with_unique_information(
            (earlier, winner), choices, actions, overrides)
        assert actions[0] == 'select', audit
