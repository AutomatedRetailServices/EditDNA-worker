"""Provider-reliability coverage for the Unified Selection Gemini transport.

RAW #119 saw Gemini return a MAX_TOKENS-truncated, unparseable response with
no retry available, so the reasoner failed open on the first hiccup even
though selection_reasoner_status had been "applied" moments earlier (RAW
#118). This file targets exactly that class of failure: truncated/malformed
provider responses, the output token budget that made truncation more likely
than it needed to be, and the retry policy added to recover from a transient
one. No editorial Selection rule is exercised or asserted on here -- these
tests only construct trivial two/three-candidate drafts to drive the
transport, never real story content.
"""
from dataclasses import replace
import json

import pytest

from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, MediaSignals, SCHEMA_VERSION
from cutsell_worker.hybrid_google_transport import DollarBudgetLedger
from cutsell_worker.hybrid_payload import estimate_tokens_from_chars
from cutsell_worker.hybrid_provider_settings import HybridProviderSettings
from cutsell_worker.unified_selection_google import (
    GoogleUnifiedSelectionReasoner,
    UnifiedSelectionUnreliableResponseError,
    build_unified_selection_payload,
    output_token_reserve,
    parse_unified_selection_response,
)


def clip(i: int) -> DraftClip:
    text = f"Independent story beat number {i} with some unique audience-facing detail."
    return DraftClip(
        clip_id=f"c{i}",
        source_asset_id="src",
        source_order=i,
        start=float(i * 5),
        end=float(i * 5 + 4),
        text=text,
        caption_text=text,
    )


def grouped_clip(i: int, *, source: str = "src", group: str | None = "take-1",
                 start: float | None = None, end: float | None = None,
                 bucket_text: str | None = None) -> DraftClip:
    start = float(i * 5) if start is None else start
    end = start + 4.0 if end is None else end
    text = bucket_text or f"Delivery piece {i}"
    return DraftClip(
        clip_id=f"g{i}-{source}", source_asset_id=source, source_order=i,
        start=start, end=end, text=text, caption_text=text,
        take_group_id=group,
    )


def draft(candidate_count: int) -> DraftTimeline:
    clips = tuple(clip(i) for i in range(candidate_count))
    return DraftTimeline(
        schema_version=SCHEMA_VERSION,
        project_id="p",
        strategy=EditStrategy.STORYTELLING,
        selected=clips,
        alternates=(),
        discarded=(),
    )


def timeline_with(*, selected=(), alternates=(), discarded=()) -> DraftTimeline:
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p",
        strategy=EditStrategy.STORYTELLING, selected=tuple(selected),
        alternates=tuple(alternates), discarded=tuple(discarded),
        diagnostics={"editorial_engine_v2_request": True},
    )


def test_take_group_summary_reconstructs_split_delivery_without_mutating_candidates():
    payload = build_unified_selection_payload(timeline_with(
        selected=(grouped_clip(0, start=10.0, end=12.0, bucket_text="first half"),),
        alternates=(grouped_clip(1, start=12.4, end=15.0, bucket_text="second half"),),
    ))
    group = payload["take_groups"][0]
    assert group["candidate_indices"] == [0, 1]
    assert [row["gap_before"] for row in group["intervals"]] == [None, 0.4]
    assert "first half" in group["combined_text_with_gaps"]
    assert "second half" in group["combined_text_with_gaps"]
    assert [row["clip_id"] for row in payload["candidates"]] == ["g0-src", "g1-src"]


def test_same_take_group_id_in_different_sources_never_merges():
    payload = build_unified_selection_payload(timeline_with(selected=(
        grouped_clip(0, source="a", group="shared"),
        grouped_clip(1, source="b", group="shared"),
    )))
    assert len(payload["take_groups"]) == 2
    assert {group["source_asset_id"] for group in payload["take_groups"]} == {"a", "b"}


def test_missing_take_group_ids_are_independent_singletons():
    payload = build_unified_selection_payload(timeline_with(selected=(
        grouped_clip(0, group=None), grouped_clip(1, group=None),
    )))
    assert len(payload["take_groups"]) == 2
    assert all(len(group["candidate_indices"]) == 1 for group in payload["take_groups"])


def test_noncontiguous_same_group_preserves_gap_instead_of_claiming_continuity():
    payload = build_unified_selection_payload(timeline_with(selected=(
        grouped_clip(0, group="provisional", start=0.0, end=2.0),
        grouped_clip(1, group="provisional", start=20.0, end=22.0),
    )))
    group = payload["take_groups"][0]
    assert group["intervals"][1]["gap_before"] == 18.0
    assert "[GAP 18.000s]" in group["combined_text_with_gaps"]
    assert group["evidence_only"] is True


def test_take_group_contract_separates_delivery_evidence_from_retry_authority():
    payload = build_unified_selection_payload(timeline_with(selected=(grouped_clip(0),)))
    contract = " ".join(payload["take_group_contract"])
    assert "not retry families" in contract
    assert "return one decision for every original candidate" in contract


def test_take_group_context_is_v2_only_and_does_not_change_legacy_payload():
    payload = build_unified_selection_payload(draft(2))
    assert "take_groups" not in payload
    assert "take_group_contract" not in payload


def test_real_group_id_cannot_collide_with_missing_id_singleton():
    payload = build_unified_selection_payload(timeline_with(selected=(
        grouped_clip(0, group=None), grouped_clip(1, group="__missing_0"),
    )))
    assert len(payload["take_groups"]) == 2
    assert len({group["take_evidence_id"] for group in payload["take_groups"]}) == 2


def test_v2_does_not_expose_provisional_buckets_or_hybrid_votes_to_selector():
    source = timeline_with(
        selected=(grouped_clip(0, group="take"),),
        discarded=(grouped_clip(1, group="take"),),
    )
    payload = build_unified_selection_payload(source)
    assert [row["clip_id"] for row in payload["candidates"]] == ["g0-src", "g1-src"]
    assert all("current_bucket" not in row and "hybrid_votes" not in row
               for row in payload["candidates"])
    assert all("current_bucket" not in interval
               for group in payload["take_groups"] for interval in group["intervals"])
    assert {row.clip_id for row in (*source.selected, *source.discarded)} == {
        "g0-src", "g1-src",
    }


def decisions_json(candidate_count: int, *, index_offset: int = 0) -> str:
    """index_offset lets a test deliberately misalign candidate_index from
    position, e.g. index_offset=1 makes every candidate_index wrong by one."""
    return json.dumps({
        "decisions": [
            {
                "candidate_index": i + index_offset,
                "action": "select",
                "relation": "independent",
                "confidence": 1.0,
                "family_index": i,
                "reason_code": "independent_story_coverage",
            }
            for i in range(candidate_count)
        ]
    })


def v2_decisions_json(candidate_count: int, *, duplicate_sequence: bool = False) -> str:
    data = json.loads(decisions_json(candidate_count))
    data["competitions"] = []
    for i, item in enumerate(data["decisions"]):
        item["sequence_index"] = 0 if duplicate_sequence else i
    return json.dumps(data)


def gemini_response(text: str, *, finish_reason: str = "STOP", output_tokens: int = 100) -> dict:
    return {
        "candidates": [{
            "finishReason": finish_reason,
            "content": {"parts": [{"text": text}]},
        }],
        "usageMetadata": {"candidatesTokenCount": output_tokens},
    }


def truncated_response(*, finish_reason: str = "MAX_TOKENS") -> dict:
    # A realistic truncation: valid JSON prefix, cut off mid-string with no
    # closing delimiters -- exactly the shape json.loads chokes on.
    text = (
        '{"decisions":[{"action":"select","relation":"independent",'
        '"confidence":1.0,"family_index":0,"reason_code":"in'
    )
    return gemini_response(text, finish_reason=finish_reason)


class FakeResponse:
    def __init__(self, status_code: int, body: dict):
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body)

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, bodies: list[dict]):
        self._bodies = bodies
        self.calls: list[tuple] = []

    def post(self, url, *, headers, json, timeout):
        idx = len(self.calls)
        self.calls.append((url, headers, json, timeout))
        body = self._bodies[idx] if idx < len(self._bodies) else self._bodies[-1]
        return FakeResponse(200, body)


def make_reasoner(fake: FakeSession, ledger: DollarBudgetLedger | None = None) -> GoogleUnifiedSelectionReasoner:
    settings = HybridProviderSettings(enabled=True)
    return GoogleUnifiedSelectionReasoner(
        api_key="fake-key",
        model=settings.primary_model,
        settings=settings,
        ledger=ledger or DollarBudgetLedger(1.0),
        session=fake,
    )


# --- output token budget -----------------------------------------------

def test_output_token_reserve_exceeds_the_old_flat_heuristic_that_truncated_in_raw119():
    old_flat_heuristic = max(640, 36 * 32)  # the exact formula RAW #119 ran with
    assert output_token_reserve(32, ceiling=4096) > old_flat_heuristic


def test_output_token_reserve_respects_an_explicit_ceiling():
    assert output_token_reserve(10_000, ceiling=4096) == 4096


def test_output_token_reserve_has_a_floor_for_tiny_candidate_counts():
    assert output_token_reserve(0, ceiling=4096) == 640
    assert output_token_reserve(1, ceiling=4096) == 640


def test_output_token_reserve_accounts_for_gemini_pretty_printed_responses():
    # RAW run 33316711594 (attempts 1 and 2, head da8bd80) truncated at
    # MAX_TOKENS with 32 candidates against the PREVIOUS compact-JSON
    # worst-case estimate (1640 tokens). An isolation probe (scripts/
    # isolate_unified_selection_output_budget.py) proved Gemini's real
    # structured-output responses are pretty-printed with indentation, not
    # compact -- and reproduced the identical truncation with an unmodified
    # pre-fix prompt too, ruling out anything else as the cause. This pins
    # the budget comfortably above the empirically observed near-truncation
    # output token usage (~1622-1623 tokens, response still incomplete) at
    # the real Video00 candidate count.
    assert output_token_reserve(32, ceiling=4096) > 2000


# --- parser: truncated/malformed responses never look like a result -----

def test_parse_raises_unreliable_error_on_truncated_json_and_names_finish_reason():
    with pytest.raises(UnifiedSelectionUnreliableResponseError, match="MAX_TOKENS"):
        parse_unified_selection_response(truncated_response())


def test_parse_raises_unreliable_error_when_candidates_missing():
    with pytest.raises(UnifiedSelectionUnreliableResponseError, match="candidates"):
        parse_unified_selection_response({"candidates": []})
    with pytest.raises(UnifiedSelectionUnreliableResponseError, match="blockReason='SAFETY'.*promptTokenCount=321"):
        parse_unified_selection_response({"candidates": [],
            "promptFeedback": {"blockReason": "SAFETY", "blockReasonMessage": "private input"},
            "usageMetadata": {"promptTokenCount": 321}})


def test_parse_raises_unreliable_error_when_content_missing():
    with pytest.raises(UnifiedSelectionUnreliableResponseError, match="content"):
        parse_unified_selection_response({"candidates": [{"finishReason": "STOP"}]})


def test_parse_raises_unreliable_error_when_decisions_key_missing():
    raw = gemini_response(json.dumps({"not_decisions": []}))
    with pytest.raises(UnifiedSelectionUnreliableResponseError, match="decisions"):
        parse_unified_selection_response(raw)


def test_parse_succeeds_and_reports_finish_reason_on_a_clean_response():
    raw = gemini_response(decisions_json(2), finish_reason="STOP")
    decisions, output_tokens, finish_reason = parse_unified_selection_response(raw)
    assert len(decisions) == 2
    assert finish_reason == "STOP"
    assert output_tokens == 100


# --- reasoner: retry recovers from a truncated first attempt -------------

def test_reason_retries_once_after_a_truncated_first_attempt_and_succeeds():
    fake = FakeSession([truncated_response(), gemini_response(decisions_json(2))])
    reasoner = make_reasoner(fake)

    plan = reasoner.reason(draft(2))

    assert len(fake.calls) == 2
    assert len(plan.decisions) == 2
    assert all(d.action == "select" for d in plan.decisions)
    # retry asks for a larger response budget in case truncation was the cause
    first_budget = fake.calls[0][2]["generationConfig"]["maxOutputTokens"]
    second_budget = fake.calls[1][2]["generationConfig"]["maxOutputTokens"]
    assert second_budget > first_budget


def test_reason_raises_after_exhausting_retries_rather_than_returning_a_partial_result():
    fake = FakeSession([truncated_response(), truncated_response()])
    reasoner = make_reasoner(fake)

    with pytest.raises(UnifiedSelectionUnreliableResponseError):
        reasoner.reason(draft(2))

    # exactly one retry (max_retries defaults to 1) -- not an unbounded loop,
    # and the failure is a real raised exception, never a plan built from the
    # truncated response.
    assert len(fake.calls) == 2


def test_reason_retries_on_decision_count_mismatch_not_only_on_parse_failure():
    # A response that parses cleanly but is short one decision is exactly as
    # untrustworthy as a truncated one and must be treated the same way.
    short = gemini_response(decisions_json(1))  # draft below has 2 candidates
    good = gemini_response(decisions_json(2))
    fake = FakeSession([short, good])
    reasoner = make_reasoner(fake)

    plan = reasoner.reason(draft(2))

    assert len(fake.calls) == 2
    assert len(plan.decisions) == 2


def test_v2_reason_stably_normalizes_duplicate_sequence_index():
    d = replace(draft(3), diagnostics={
        "editorial_engine_v2_request": {"require_audiovisual_evidence": True},
        "whole_video_context": {
            "sources": [{
                "source_asset_id": "src",
                "audiovisual_evidence": '{"regions":[{"role":"audience"}]}',
            }],
        },
    })
    fake = FakeSession([gemini_response(v2_decisions_json(3, duplicate_sequence=True))])
    reasoner = make_reasoner(fake)

    plan = reasoner.reason(d)

    assert len(fake.calls) == 1
    assert [decision.sequence_index for decision in plan.decisions] == [0, 1, 2]


def test_v2_parses_explicit_take_competition_indices():
    from cutsell_worker.unified_selection_google import _parse_take_competitions, UnifiedSelectionUnreliableResponseError
    import pytest
    rows = [{"clip_id": name} for name in ("prior", "unique", "final")]
    data = {"competitions": [{
        "winner_candidate_indices": [2], "covered_candidate_indices": [0],
        "material_unique_candidate_indices": [1], "relation": "equivalent_take",
        "confidence": .97, "reason": "Complete delivery preserves prior claim",
    }]}
    raw = gemini_response(json.dumps(data))
    contests = _parse_take_competitions(raw, rows)
    assert contests[0].winner_clip_ids == ("final",)
    assert contests[0].material_unique_clip_ids == ("unique",)
    data["competitions"][0]["covered_candidate_indices"] = [1]
    with pytest.raises(UnifiedSelectionUnreliableResponseError):
        _parse_take_competitions(gemini_response(json.dumps(data)), rows)


def test_schema_requires_candidate_index_on_every_decision():
    # RAW #120: a normal-STOP response returned 31 decisions for 32
    # candidates. An isolation probe (scripts/isolate_unified_selection_
    # cardinality.py) found requiring candidate_index gets 8/8 trials with
    # the exactly right count, and it lets a reordered/duplicated response
    # (same length, wrong mapping) be caught too -- a bare length check
    # cannot see that at all.
    from cutsell_worker.unified_selection_google import unified_selection_response_schema
    schema = unified_selection_response_schema(5)
    item_schema = schema["properties"]["decisions"]["items"]
    assert "candidate_index" in item_schema["properties"]
    assert "candidate_index" in item_schema["required"]


def test_reason_raises_on_reordered_response_with_the_correct_count():
    # Same length as expected, but candidate_index values are shifted by one
    # -- exactly the "right count, wrong mapping" case a bare length check
    # would silently accept and apply to the wrong clips.
    reordered = gemini_response(decisions_json(2, index_offset=1))
    fake = FakeSession([reordered, reordered])
    reasoner = make_reasoner(fake)

    with pytest.raises(UnifiedSelectionUnreliableResponseError, match="candidate_index mismatch"):
        reasoner.reason(draft(2))

    assert len(fake.calls) == 2  # still retried once, still fails both times


def test_reason_names_the_specific_mismatched_indices_in_the_error():
    reordered = gemini_response(decisions_json(3, index_offset=1))
    fake = FakeSession([reordered, reordered])
    reasoner = make_reasoner(fake)

    with pytest.raises(UnifiedSelectionUnreliableResponseError) as excinfo:
        reasoner.reason(draft(3))

    # every one of the 3 positions is wrong (shifted by 1); the error names
    # them rather than only reporting a generic mismatch.
    assert "(0, 1)" in str(excinfo.value)
    assert "(1, 2)" in str(excinfo.value)
    assert "(2, 3)" in str(excinfo.value)


def test_failed_first_attempt_releases_its_ledger_reservation_before_retrying():
    settings = HybridProviderSettings(enabled=True)
    d = draft(2)
    payload = build_unified_selection_payload(d)
    input_tokens = estimate_tokens_from_chars(len(json.dumps(payload, ensure_ascii=False)))
    first_reserve = output_token_reserve(2, ceiling=4096)
    retry_reserve = min(4096, max(first_reserve, int(first_reserve * 1.5)))
    retry_cost = settings.estimate_cost_usd(input_tokens=input_tokens, output_tokens=retry_reserve, escalation=False)

    # Sized so there is only ever enough budget for ONE reservation at a
    # time (at the larger, bumped retry size) -- if the failed first
    # attempt's reservation were not released, the retry's own reservation
    # would not fit and reason() would raise a budget error instead of
    # succeeding.
    ledger = DollarBudgetLedger(retry_cost * 1.2)
    fake = FakeSession([truncated_response(), gemini_response(decisions_json(2))])
    reasoner = GoogleUnifiedSelectionReasoner(
        api_key="fake-key", model=settings.primary_model, settings=settings,
        ledger=ledger, session=fake,
    )

    plan = reasoner.reason(d)

    assert len(plan.decisions) == 2


def test_retry_budget_is_capped_to_what_the_ledger_can_actually_afford():
    # RAW #121: adding candidate_index (the RAW #120 fix) grew the schema
    # just enough that the naive 1.5x retry bump alone could exceed the tiny
    # default per-edit cost cap (max_cost_per_edit_usd, $0.0075) at the real
    # Video00 candidate count (32) -- so a retryable candidate_index mismatch
    # (nothing to do with token budget) could never get a second attempt at
    # all: it died on "unified Selection edit dollar budget exhausted" before
    # ever making the retry's HTTP call. Sized deterministically here (the
    # midpoint between what attempt 1 costs and what the naive bump would
    # cost) so the test reproduces the exact boundary regardless of exactly
    # how many tokens this file's synthetic payload happens to serialize to.
    settings = HybridProviderSettings(enabled=True)
    d = draft(32)
    payload = build_unified_selection_payload(d)
    input_tokens = estimate_tokens_from_chars(len(json.dumps(payload, ensure_ascii=False)))
    reserve = output_token_reserve(32, ceiling=4096)
    attempt1_cost = settings.estimate_cost_usd(input_tokens=input_tokens, output_tokens=reserve, escalation=False)
    naive_bumped_cost = settings.estimate_cost_usd(
        input_tokens=input_tokens, output_tokens=max(reserve, int(reserve * 1.5)), escalation=False,
    )
    assert naive_bumped_cost > attempt1_cost  # sanity: the bump really is bigger
    ledger_max = (attempt1_cost + naive_bumped_cost) / 2  # affords attempt 1, not the naive bump

    reordered = gemini_response(decisions_json(32, index_offset=1))
    good = gemini_response(decisions_json(32))
    fake = FakeSession([reordered, good])
    ledger = DollarBudgetLedger(ledger_max)
    reasoner = GoogleUnifiedSelectionReasoner(
        api_key="fake-key", model=settings.primary_model, settings=settings,
        ledger=ledger, session=fake,
    )

    plan = reasoner.reason(d)  # must not raise "budget exhausted"

    assert len(fake.calls) == 2
    assert len(plan.decisions) == 32
    first_budget = fake.calls[0][2]["generationConfig"]["maxOutputTokens"]
    second_budget = fake.calls[1][2]["generationConfig"]["maxOutputTokens"]
    assert first_budget == reserve
    assert reserve < second_budget < max(reserve, int(reserve * 1.5))  # more headroom, but capped below the naive bump


def test_max_affordable_output_tokens_reflects_ledger_remaining_balance():
    # Direct unit coverage of the helper reason() uses to cap a retry's
    # bumped budget: given the ledger's ACTUAL remaining balance (not the
    # amount that funded some earlier, now-released reservation -- a full
    # release restores exactly what was reserved, so a same-size repeat is
    # always affordable again by construction), it must compute the largest
    # output budget a fresh call at this input size could still reserve.
    settings = HybridProviderSettings(enabled=True)
    ledger = DollarBudgetLedger(max_usd=0.01, reserved_usd=0.008)  # $0.002 left
    reasoner = GoogleUnifiedSelectionReasoner(
        api_key="fake-key", model=settings.primary_model, settings=settings, ledger=ledger,
    )
    input_tokens = 1000
    input_cost = settings.estimate_cost_usd(input_tokens=input_tokens, output_tokens=0, escalation=False)
    expected_budget_for_output = ledger.remaining_usd - input_cost
    expected_tokens = int(expected_budget_for_output / (settings.primary_output_per_million_usd / 1_000_000.0))

    assert reasoner._max_affordable_output_tokens(input_tokens) == expected_tokens
    assert expected_tokens < output_token_reserve(32, ceiling=4096)  # tighter than a real Video00-scale need


# --- non-retryable preflight failures never spend a retry ----------------

def test_missing_api_key_raises_before_any_http_call_and_is_never_retried():
    fake = FakeSession([gemini_response(decisions_json(2))])
    settings = HybridProviderSettings(enabled=True)
    reasoner = GoogleUnifiedSelectionReasoner(
        api_key="", model=settings.primary_model, settings=settings,
        ledger=DollarBudgetLedger(1.0), session=fake,
    )
    with pytest.raises(ValueError, match="Gemini API key required"):
        reasoner.reason(draft(2))
    assert len(fake.calls) == 0


def test_disallowed_model_raises_before_any_http_call_and_is_never_retried():
    fake = FakeSession([gemini_response(decisions_json(2))])
    settings = HybridProviderSettings(enabled=True)
    reasoner = GoogleUnifiedSelectionReasoner(
        api_key="fake-key", model="not-an-approved-model", settings=settings,
        ledger=DollarBudgetLedger(1.0), session=fake,
    )
    with pytest.raises(ValueError, match="not approved"):
        reasoner.reason(draft(2))
    assert len(fake.calls) == 0


# --- visual/performance evidence reaches the payload ---------------------
#
# RAW #122 audit: local_performance.py computes real per-take face/pose/
# motion evidence (MediaSignals) and pipeline.py's take->DraftClip conversion
# was silently dropping it -- DraftClip had no field to carry it at all, so
# not one candidate the reasoner ever saw carried anything beyond transcript
# text and timing. These tests pin the fix: DraftClip.signals is threaded
# into the payload as `visual_evidence` when present, and omitted (never
# zeroed -- a zero would read as "confirmed bad", not "no evidence") when a
# clip has no signals.

def test_payload_omits_visual_evidence_when_clip_has_no_signals():
    payload = build_unified_selection_payload(draft(2))
    for row in payload["candidates"]:
        assert "visual_evidence" not in row


def test_payload_includes_visual_evidence_when_clip_signals_are_present():
    signals = MediaSignals(
        source_asset_id="src", start=0.0, end=4.0,
        face_visibility=0.9, eye_contact=0.8, motion_stability=0.2,
        visual_fumble=0.7, expression_naturalness=0.3, gesture_naturalness=0.4,
        distraction_risk=0.6,
    )
    stumbled = replace(clip(0), signals=signals)
    d = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(stumbled, clip(1)), alternates=(), discarded=(),
    )

    payload = build_unified_selection_payload(d)

    rows_by_id = {row["clip_id"]: row for row in payload["candidates"]}
    assert "visual_evidence" not in rows_by_id["c1"]
    evidence = rows_by_id["c0"]["visual_evidence"]
    assert evidence == {
        "face_visibility": 0.9,
        "eye_contact": 0.8,
        "motion_stability": 0.2,
        "visual_fumble": 0.7,
        "expression_naturalness": 0.3,
        "gesture_naturalness": 0.4,
        "distraction_risk": 0.6,
    }


def test_editorial_contract_requires_exactly_one_select_per_retry_family():
    payload = build_unified_selection_payload(draft(2))
    contract_text = " ".join(payload["editorial_contract"])
    assert "exactly ONE SELECT" in contract_text


def test_editorial_contract_instructs_use_of_visual_evidence():
    payload = build_unified_selection_payload(draft(2))
    contract_text = " ".join(payload["editorial_contract"])
    assert "visual_evidence" in contract_text


def test_native_selection_counts_same_media_sent_to_generation():
    fake = FakeSession([{"totalTokens": 1200}, gemini_response(decisions_json(2))])
    reasoner = make_reasoner(fake)
    reasoner.audiovisual_parts = ({"inlineData": {"mimeType": "video/mp4", "data": "TEST"}},)
    plan = reasoner.reason(draft(2))
    assert fake.calls[0][0].endswith(":countTokens")
    assert fake.calls[1][0].endswith(":generateContent")
    assert fake.calls[0][2]["contents"] == fake.calls[1][2]["contents"]
    assert plan.estimated_input_tokens == 1200


@pytest.mark.parametrize("count", [None, True, -1, 100000])
def test_native_preflight_failure_never_generates_or_falls_back_to_text(count):
    fake = FakeSession([{"totalTokens": count}])
    reasoner = make_reasoner(fake)
    reasoner.audiovisual_parts = ({"inlineData": {"mimeType": "video/mp4", "data": "TEST"}},)
    with pytest.raises(ValueError):
        reasoner.reason(draft(2))
    assert len(fake.calls) == 1


def test_v2_contract_prefers_complete_take_over_fragment_patchwork():
    d = replace(draft(2), diagnostics={
        "editorial_engine_v2_request": {"require_audiovisual_evidence": True},
    })
    payload = build_unified_selection_payload(d)
    contract = " ".join(payload["editorial_contract"])
    assert "continuous complete take" in contract
    assert "immediately adjacent continuation candidates" in contract
    assert "unique audience information" in contract
    assert "useful visual actions" in contract
    assert "audiovisual evidence shows it failed" in contract
    assert "reconstruct whole delivery attempts" in contract
    assert "Compare complete takes as units" in contract
    assert "material fact, number, condition, personality moment" in contract
