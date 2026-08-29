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
