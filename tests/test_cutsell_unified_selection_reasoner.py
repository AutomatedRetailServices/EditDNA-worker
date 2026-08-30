from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.unified_selection_google import (
    build_unified_selection_payload,
    build_unified_selection_request,
)
from cutsell_worker.unified_selection_reasoner import (
    UnifiedSelectionDecision,
    UnifiedSelectionPlan,
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
