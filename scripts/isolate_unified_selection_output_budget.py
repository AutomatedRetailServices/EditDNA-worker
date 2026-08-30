"""Isolate whether the retry-family policy fix (editorial_contract prompt
growth + optional visual_evidence per candidate) is what made RAW run
33316711594 (attempts 1 and 2, head da8bd80) truncate at MAX_TOKENS twice in
a row, immediately after that fix landed, at the real Video00 candidate
count (32) -- without spending a third paid RunPod GPU RAW.

Both CI-red attempts failed identically in class:

    "error": "UnifiedSelectionUnreliableResponseError: Gemini unified
    response was not valid JSON (finishReason='MAX_TOKENS'): ..."

truncating at a nearly identical byte offset (~5900 chars) both times, which
looks like a genuinely marginal output budget rather than pure random
variance. Nothing in this fix touched unified_selection_response_schema(),
output_token_reserve(), or the decision object's fields (still
candidate_index/action/relation/confidence/family_index/reason_code, all
enum/number-bounded) -- so if the budget math is unchanged and the schema
is unchanged, the only two things that grew are (1) the editorial_contract
prompt text (several new sentences) and (2) the optional `visual_evidence`
object now added per candidate when a clip has local_performance.py
signals.

This probe reuses the real production request builders (build_unified_
selection_payload/build_unified_selection_request, the exact code shipped
in this fix) and constructs three variants at the same candidate count and
the same production output_token_reserve() budget:

  - old_prompt_no_visual_evidence: the editorial_contract text exactly as
    it read before this fix (reconstructed literally below), no
    visual_evidence on any candidate -- the closest approximation of what
    RAW #122 (the last successful run) actually sent.
  - new_prompt_no_visual_evidence: this fix's editorial_contract text, no
    visual_evidence -- isolates the prompt-growth axis alone.
  - new_prompt_with_visual_evidence: this fix's editorial_contract text,
    every candidate carries a synthetic visual_evidence object -- isolates
    the payload-growth axis alone (production may or may not populate this
    depending on cv2/mediapipe availability in the RunPod image; this
    checks the worst case).

If only the new-prompt variants show materially higher truncation/output-
token usage than the old-prompt variant, that proves this fix ate into a
margin that was already thin. If all three truncate at a similar rate, the
two RAW failures were provider variance unrelated to this fix's content.

Every call is real and billed, capped at HARD_CAP_USD. No semantic
Selection logic is imported for its behavior -- only the request-building
functions are exercised with synthetic, non-Video00 candidate text.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from dataclasses import dataclass, replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests  # noqa: E402

from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, MediaSignals, SCHEMA_VERSION  # noqa: E402
from cutsell_worker.unified_selection_google import (  # noqa: E402
    build_unified_selection_payload,
    build_unified_selection_request,
    output_token_reserve,
)

MODEL = "gemini-3.5-flash-lite"
HARD_CAP_USD = 0.40
INPUT_PER_MILLION_USD = 0.30
OUTPUT_PER_MILLION_USD = 2.50
CANDIDATE_COUNT = 32  # the real Video00 candidate_count seen in every prior RAW
TRIALS_PER_CELL = 5
OUTPUT_TOKENS_CEILING = 4096  # GoogleUnifiedSelectionReasoner.max_output_tokens default

# Reconstructed literally from unified_selection_google.py as it read before
# this fix (git history: the commit immediately prior to "Fix Unified
# Selection retry-family policy gap found in RAW #122 audit").
_OLD_EDITORIAL_CONTRACT = [
    "Understand the full creator message before deciding any individual take.",
    "First infer idea families and retry relationships across the entire timeline.",
    "SELECT independent valid story coverage, the best retry, necessary continuations, and every clean piece needed for a composite best take.",
    "SWAP a usable alternative or redundant delivery that should not play by default but remains useful for manual replacement.",
    "DISCARD only recording-process BTS, failed/abandoned delivery, or an inferior retry with no unique audience-facing information.",
    "Do not prefer a monolithic take merely because it is longer; a human-quality composite of cleaner micro-deliveries may be better.",
    "Do not treat adjacent valid statements as retries just because they share topic words.",
    "Preserve numbers, negations, names, causal claims, and genuinely new story facts.",
    "Natural source story order is authoritative; do not reorder candidates.",
    "WHEN UNCERTAIN, preserve content rather than destructively deleting it.",
]

_STORY_BEATS = [
    "I picked up the package from the front porch this morning before work.",
    "The box was noticeably larger than what I was expecting for the order.",
    "There was a handwritten note thanking me for being a repeat customer.",
    "I opened it carefully since the packaging looked a little fragile.",
    "The color in person was a shade different from the website photos.",
    "I tested it right away to make sure everything worked as advertised.",
    "Setup took about ten minutes once I found the right cable for it.",
    "Overall the whole process was smoother than I thought it would be.",
    "I would recommend this to a friend who asked me about it later.",
    "The instructions included a QR code linking to a video walkthrough.",
]
_STUMBLE_FILLERS = ["Um, ", "So, ", "Like, ", "", "Okay, ", "Wait, "]


def _retry_family_heavy_candidate_texts(n: int) -> list[str]:
    texts: list[str] = []
    i = 0
    beat_idx = 0
    while len(texts) < n:
        base = _STORY_BEATS[beat_idx % len(_STORY_BEATS)]
        beat_idx += 1
        if i % 2 == 0 and len(texts) + 2 <= n:
            family_size = 2 + (i % 3)
            for k in range(min(family_size, n - len(texts))):
                filler = _STUMBLE_FILLERS[k % len(_STUMBLE_FILLERS)]
                texts.append(f"{filler}{base}")
        else:
            texts.append(f"{base} (unique {len(texts)})")
        i += 1
    return texts[:n]


def make_draft(texts: list[str], *, with_signals: bool) -> DraftTimeline:
    clips = []
    for i, text in enumerate(texts):
        clip = DraftClip(
            clip_id=f"c{i}", source_asset_id="src", source_order=i,
            start=float(i * 5), end=float(i * 5 + 4), text=text, caption_text=text,
        )
        if with_signals:
            clip = replace(clip, signals=MediaSignals(
                source_asset_id="src", start=clip.start, end=clip.end,
                face_visibility=0.7, eye_contact=0.55, motion_stability=0.6,
                visual_fumble=0.3, expression_naturalness=0.5, gesture_naturalness=0.5,
                distraction_risk=0.2,
            ))
        clips.append(clip)
    third = max(1, len(clips) // 3)
    selected = tuple(clips[:third])
    alternates = tuple(clips[third:2 * third])
    discarded = tuple(clips[2 * third:])
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="output-budget-probe",
        strategy=EditStrategy.STORYTELLING,
        selected=selected or (clips[0],), alternates=alternates, discarded=discarded,
    )


def build_variant_body(*, variant: str, n: int, budget: int) -> dict:
    with_signals = variant == "new_prompt_with_visual_evidence"
    texts = _retry_family_heavy_candidate_texts(n)
    draft = make_draft(texts, with_signals=with_signals)
    payload = build_unified_selection_payload(draft)  # real production code, current shipped fix

    if variant == "old_prompt_no_visual_evidence":
        payload = copy.deepcopy(payload)
        payload["editorial_contract"] = _OLD_EDITORIAL_CONTRACT
    elif variant not in ("new_prompt_no_visual_evidence", "new_prompt_with_visual_evidence"):
        raise ValueError(f"unknown variant {variant!r}")

    return build_unified_selection_request(payload, max_output_tokens=budget)


@dataclass
class CallResult:
    variant: str
    trial: int
    status_code: int | None
    ok: bool
    finish_reason: str
    truncated: bool
    input_tokens: int
    output_tokens: int
    latency_sec: float
    error_body: str
    notes: str = ""


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000.0) * INPUT_PER_MILLION_USD + (output_tokens / 1_000_000.0) * OUTPUT_PER_MILLION_USD


def run_one(*, variant: str, trial: int, body: dict, spend: list[float]) -> CallResult:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing")
    if sum(spend) >= HARD_CAP_USD:
        raise RuntimeError(f"hard cap ${HARD_CAP_USD} reached before {variant}/trial{trial}")

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
    started = time.monotonic()
    try:
        response = requests.post(
            endpoint, headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=body, timeout=90.0,
        )
    except requests.exceptions.RequestException as exc:
        latency = round(time.monotonic() - started, 3)
        return CallResult(
            variant=variant, trial=trial, status_code=None, ok=False, finish_reason="",
            truncated=False, input_tokens=0, output_tokens=0, latency_sec=latency,
            error_body=f"{exc.__class__.__name__}: {exc}"[:800], notes="network_error",
        )
    latency = round(time.monotonic() - started, 3)
    ok = response.status_code == 200
    finish_reason, input_tokens, output_tokens, error_body, notes = "", 0, 0, "", ""
    truncated = False

    if not ok:
        error_body = response.text[:800]
        spend.append(0.0)
    else:
        raw = response.json()
        usage = raw.get("usageMetadata") or {}
        input_tokens = int(usage.get("promptTokenCount") or 0)
        output_tokens = int(usage.get("candidatesTokenCount") or 0)
        spend.append(estimate_cost(input_tokens, output_tokens))
        candidates = raw.get("candidates") or []
        first = candidates[0] if candidates else {}
        finish_reason = str(first.get("finishReason") or "")
        parts = (first.get("content") or {}).get("parts") or []
        text = "".join(str(p.get("text") or "") for p in parts if isinstance(p, dict))
        try:
            parsed = json.loads(text)
            decisions = parsed.get("decisions")
            if not isinstance(decisions, list) or len(decisions) != CANDIDATE_COUNT:
                notes += f"unexpected_decision_count={len(decisions) if isinstance(decisions, list) else 'n/a'} "
        except json.JSONDecodeError as exc:
            truncated = True
            notes += f"json_decode_error={exc} "
        if finish_reason == "MAX_TOKENS":
            truncated = True

    return CallResult(
        variant=variant, trial=trial, status_code=response.status_code,
        ok=ok, finish_reason=finish_reason, truncated=truncated,
        input_tokens=input_tokens, output_tokens=output_tokens, latency_sec=latency,
        error_body=error_body, notes=notes.strip(),
    )


def main() -> None:
    spend: list[float] = []
    results: list[CallResult] = []
    budget = output_token_reserve(CANDIDATE_COUNT, ceiling=OUTPUT_TOKENS_CEILING)
    print(f"production output_token_reserve({CANDIDATE_COUNT}, ceiling={OUTPUT_TOKENS_CEILING}) = {budget}")

    variants = ["old_prompt_no_visual_evidence", "new_prompt_no_visual_evidence", "new_prompt_with_visual_evidence"]
    for variant in variants:
        for trial in range(TRIALS_PER_CELL):
            body = build_variant_body(variant=variant, n=CANDIDATE_COUNT, budget=budget)
            try:
                result = run_one(variant=variant, trial=trial, body=body, spend=spend)
            except RuntimeError as exc:
                print(f"stopping early: {exc}")
                break
            results.append(result)
            print(
                f"{variant:32s} trial{trial} status={result.status_code} "
                f"finish={result.finish_reason!r} truncated={result.truncated} "
                f"output_tokens={result.output_tokens}/{budget} {result.notes}"
            )

    by_variant: dict[str, list[CallResult]] = {}
    for r in results:
        by_variant.setdefault(r.variant, []).append(r)
    summary = {
        variant: {
            "trials": len(rows),
            "truncated_count": sum(1 for r in rows if r.truncated),
            "avg_output_tokens": round(sum(r.output_tokens for r in rows) / len(rows), 1) if rows else None,
            "max_output_tokens": max((r.output_tokens for r in rows), default=None),
        }
        for variant, rows in by_variant.items()
    }

    report = {
        "model": MODEL,
        "hard_cap_usd": HARD_CAP_USD,
        "spend_usd": round(sum(spend), 6),
        "candidate_count": CANDIDATE_COUNT,
        "output_token_budget": budget,
        "trials_per_cell": TRIALS_PER_CELL,
        "summary_by_variant": summary,
        "cases": [
            {
                "variant": r.variant, "trial": r.trial, "status_code": r.status_code, "ok": r.ok,
                "finish_reason": r.finish_reason, "truncated": r.truncated,
                "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                "latency_sec": r.latency_sec, "notes": r.notes, "error_body": r.error_body,
            }
            for r in results
        ],
    }
    print(json.dumps(report, indent=2))

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/unified-selection-output-budget-isolation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
