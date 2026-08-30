"""GPU-free real-Gemini isolation probe for the Phase 2 semantic-equivalence
arbiter (cutsell_worker/semantic_idea_equivalence_google.py).

This never touches RunPod and launches no paid RAW. It makes one small,
real, billed batch call to Gemini using the exact production request
builder (build_semantic_equivalence_request) and response parser
(parse_semantic_equivalence_response) shipped in that module, over a fixed
set of general, video-agnostic text-pair fixtures with a known expected
same_idea label for each pair -- never Video00-specific content, per the
architecture rebalance's Phase 2 constraint that this arbiter (and its
tests) must remain provider-agnostic and clip/video-identity-free.

Fixtures cover:
  - same-idea paraphrases with very different wording (the exact class of
    pair retry_similarity()'s word-containment floor cannot reliably
    distinguish from unrelated text -- see take_grouping_provider.
    reconcile_semantic_idea_equivalence's docstring for why a numeric
    lexical-similarity band was rejected as this feature's ambiguity gate).
  - an incomplete/false-start retry paired with its complete continuation.
  - a same-idea pair in Spanish with numbers, in the same
    RAW-transcript-derived style already used by this repo's own
    hybrid_story_guard tests (fixture style reused, not Video00 content).
  - genuinely distinct story beats that merely share surface vocabulary.

Reports per-pair classification correctness against the expected label,
aggregate accuracy, real token usage, real cost, and latency. Capped at
HARD_CAP_USD; this is one batched call, not per-pair calls, matching the
production arbiter's bounded-batching design.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests  # noqa: E402

from cutsell_worker.semantic_idea_equivalence import IdeaEquivalencePair, IdeaEquivalenceRequest  # noqa: E402
from cutsell_worker.semantic_idea_equivalence_google import (  # noqa: E402
    build_semantic_equivalence_request,
    output_token_reserve,
    parse_semantic_equivalence_response,
)

MODEL = "gemini-3.5-flash-lite"
HARD_CAP_USD = 0.10
INPUT_PER_MILLION_USD = 0.30
OUTPUT_PER_MILLION_USD = 2.50
MAX_OUTPUT_TOKENS_CEILING = 1_500  # GoogleSemanticEquivalenceArbiter default

# (left_text, right_text, expected_same_idea, category) -- all synthetic,
# general, video-agnostic. No clip_id/timestamp/video identity anywhere, by
# construction (IdeaEquivalencePair carries text only).
FIXTURES: list[tuple[str, str, bool, str]] = [
    ("We just launched our new skincare line today.",
     "Today marks the official launch of our brand new skincare line.",
     True, "paraphrase_low_lexical_overlap"),
    ("This product completely changed my morning routine.",
     "Um, so, this thing totally changed how I do mornings now.",
     True, "paraphrase_with_filler_words"),
    ("I want to tell you why I—",
     "Let me explain why I decided to switch to this brand.",
     True, "incomplete_retry_and_complete_continuation"),
    ("Here's how you set it up in three easy steps.",
     "Setting this up literally only takes three steps.",
     True, "paraphrase_low_lexical_overlap"),
    ("It's really affordable for what you get.",
     "Honestly, the price is such a good deal for the quality.",
     True, "paraphrase_low_lexical_overlap"),
    ("Today we finally launched our newest product.",
     "We are excited to share that our newest product launched today.",
     True, "paraphrase_reordered_words"),
    ("Solo un cinco a diez por ciento de los casos son hereditarios.",
     "Nada más un cinco a diez por ciento de los casos son hereditarios.",
     True, "paraphrase_non_english_with_numbers"),
    ("Here's how the packaging looks up close.",
     "This is why the price point makes sense for beginners.",
     False, "distinct_story_beats"),
    ("I had a really rough morning before filming this.",
     "Let's talk about the return policy for a second.",
     False, "distinct_story_beats"),
    ("This is my honest review after one month of use.",
     "Here's a quick tip for storing it properly.",
     False, "distinct_story_beats"),
    ("It comes in three colors: black, white, and blue.",
     "The battery life is genuinely impressive for daily use.",
     False, "distinct_story_beats_shared_topic"),
    ("Thanks so much for watching, see you next time.",
     "Let's get right into the unboxing process.",
     False, "distinct_story_beats"),
    ("My dog kept barking during the whole recording, sorry about that.",
     "I already ordered a second one because I liked it so much.",
     False, "distinct_story_beats"),
    ("Setup took about ten minutes once I found the right cable for it.",
     "The instructions included a QR code linking to a video walkthrough.",
     False, "distinct_story_beats_shared_topic"),
]


@dataclass
class PairOutcome:
    index: int
    category: str
    expected_same_idea: bool
    predicted_same_idea: bool | None
    confidence: float | None
    reason: str
    correct: bool | None


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing")

    request = IdeaEquivalenceRequest(pairs=tuple(
        IdeaEquivalencePair(left_text=left, right_text=right) for left, right, _, _ in FIXTURES
    ))
    budget = output_token_reserve(len(FIXTURES), ceiling=MAX_OUTPUT_TOKENS_CEILING)
    body = build_semantic_equivalence_request(request, max_output_tokens=budget)

    # No clip/video identity ever reaches the wire -- confirm it directly on
    # the built request body, not just by construction.
    body_text = json.dumps(body)
    assert "clip_id" not in body_text and "source_asset_id" not in body_text, (
        "semantic equivalence request leaked non-text identity fields"
    )

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
    started = time.monotonic()
    response = requests.post(
        endpoint,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json=body,
        timeout=60.0,
    )
    latency = round(time.monotonic() - started, 3)
    response.raise_for_status()
    raw = response.json()

    decisions, output_tokens, finish_reason = parse_semantic_equivalence_response(raw)
    usage = raw.get("usageMetadata") or {}
    input_tokens = int(usage.get("promptTokenCount") or 0)
    cost_usd = round(
        (input_tokens / 1_000_000.0) * INPUT_PER_MILLION_USD
        + (output_tokens / 1_000_000.0) * OUTPUT_PER_MILLION_USD,
        6,
    )
    if cost_usd > HARD_CAP_USD:
        raise RuntimeError(f"cost ${cost_usd} exceeded hard cap ${HARD_CAP_USD}")

    by_index = {int(d.get("pair_index")): d for d in decisions if isinstance(d, dict)}
    outcomes: list[PairOutcome] = []
    for i, (_, _, expected, category) in enumerate(FIXTURES):
        decision = by_index.get(i)
        if decision is None:
            outcomes.append(PairOutcome(i, category, expected, None, None, "", None))
            continue
        predicted = bool(decision.get("same_idea"))
        outcomes.append(PairOutcome(
            index=i,
            category=category,
            expected_same_idea=expected,
            predicted_same_idea=predicted,
            confidence=float(decision.get("confidence", -1.0)),
            reason=str(decision.get("reason") or ""),
            correct=predicted == expected,
        ))

    scored = [o for o in outcomes if o.correct is not None]
    correct_count = sum(1 for o in scored if o.correct)
    accuracy = round(correct_count / len(scored), 4) if scored else None

    report = {
        "model": MODEL,
        "hard_cap_usd": HARD_CAP_USD,
        "spend_usd": cost_usd,
        "pair_count": len(FIXTURES),
        "output_token_budget": budget,
        "finish_reason": finish_reason,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_sec": latency,
        "decision_count_returned": len(decisions),
        "missing_decision_count": sum(1 for o in outcomes if o.predicted_same_idea is None),
        "accuracy": accuracy,
        "correct_count": correct_count,
        "scored_count": len(scored),
        "pairs": [
            {
                "index": o.index,
                "category": o.category,
                "expected_same_idea": o.expected_same_idea,
                "predicted_same_idea": o.predicted_same_idea,
                "confidence": o.confidence,
                "reason": o.reason,
                "correct": o.correct,
            }
            for o in outcomes
        ],
    }
    print(json.dumps(report, indent=2))

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/semantic-equivalence-classification-isolation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if accuracy is not None and accuracy < 1.0:
        print(f"NOTE: accuracy {accuracy} < 1.0 -- see per-pair 'correct': false rows above for misclassifications.")


if __name__ == "__main__":
    main()
