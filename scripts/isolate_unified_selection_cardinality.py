"""Isolate why a normal-STOP Gemini response can return fewer decisions than
candidates, without spending a paid RunPod RAW.

RAW #120 (the confirmatory run for the provider-reliability fix in
unified_selection_google.py) reached a NEW failure distinct from RAW #119's
MAX_TOKENS truncation:

    "error": "UnifiedSelectionUnreliableResponseError: unified Selection
    ordered decision count mismatch (expected 32, got 31,
    finishReason='STOP')"

finishReason='STOP' rules out truncation -- the model completed normally and
still returned one fewer decision than candidates. RAW #118's fix (dropping
the schema's exact minItems==maxItems bound, which 400'd at scale) also
removed the only thing forcing exact cardinality, so this could be:

  (a) schema cardinality enforcement -- no bound at all vs. a bound loose
      enough to not 400 but still pressure the model toward exact counts;
  (b) prompt ambiguity -- the prompt never states the target count as a
      number, only "one decision for every candidate";
  (c) candidate ordering/serialization -- near-duplicate/retry-family
      candidates could plausibly get merged by the model;
  (d) model omission/merge behavior at this candidate count, independent of
      schema or prompt;
  (e) parser assumptions -- ruled out separately (the parser already raises
      instead of silently accepting a short array; this probe is about the
      provider's behavior, not the parser);
  (f) another provider-layer issue this probe's variants don't cover.

This script reuses the real production request builders from
cutsell_worker.unified_selection_google (not a reimplementation) for the
baseline case, and constructs a small number of controlled variants (schema
cardinality hints, an index-echo field, and a prompt reinforcement) to
isolate which axis actually changes the model's compliance, at the real
Video00 candidate count (32) with synthetic, generic, non-Video00 content in
two text modes (all-distinct vs. retry-family-heavy) to test hypothesis (c).

Every call is real and billed (same API as the existing bake-off/isolation
scripts), capped at HARD_CAP_USD. No semantic Selection logic is imported
for its behavior -- only request-building functions are exercised with
synthetic data, and the results only inform a later, separately-reviewed
code change.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests  # noqa: E402

from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION  # noqa: E402
from cutsell_worker.unified_selection_google import (  # noqa: E402
    build_unified_selection_payload,
    build_unified_selection_request,
    unified_selection_response_schema,
)

MODEL = "gemini-3.5-flash-lite"
HARD_CAP_USD = 0.30
INPUT_PER_MILLION_USD = 0.30
OUTPUT_PER_MILLION_USD = 2.50
CANDIDATE_COUNT = 32  # RAW #118/#120's real Video00 candidate_count
TRIALS_PER_CELL = 2
# Generous and identical across every variant so truncation can never be
# confused with a cardinality failure -- this investigation is about
# whether the model returns the right COUNT, not whether it has enough room.
OUTPUT_TOKENS_CEILING = 3000

# Generic, deliberately non-Video00 narrative beats (a product-unboxing
# story), reused for both text modes.
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


def _distinct_candidate_texts(n: int) -> list[str]:
    return [f"{_STORY_BEATS[i % len(_STORY_BEATS)]} (beat {i})" for i in range(n)]


def _retry_family_heavy_candidate_texts(n: int) -> list[str]:
    """~40% of candidates are near-duplicate retries of a shared idea within
    a family of 2-4 consecutive candidates -- the shape a real raw take with
    stumbles/retakes actually produces, to test whether the model merges
    near-identical adjacent candidates into one decision."""
    texts: list[str] = []
    i = 0
    beat_idx = 0
    while len(texts) < n:
        base = _STORY_BEATS[beat_idx % len(_STORY_BEATS)]
        beat_idx += 1
        if i % 2 == 0 and len(texts) + 2 <= n:
            family_size = 2 + (i % 3)  # 2, 3, or 4
            for k in range(min(family_size, n - len(texts))):
                filler = _STUMBLE_FILLERS[k % len(_STUMBLE_FILLERS)]
                texts.append(f"{filler}{base}")
        else:
            texts.append(f"{base} (unique {len(texts)})")
        i += 1
    return texts[:n]


def make_draft(texts: list[str]) -> DraftTimeline:
    clips = []
    for i, text in enumerate(texts):
        clips.append(DraftClip(
            clip_id=f"c{i}",
            source_asset_id="src",
            source_order=i,
            start=float(i * 5),
            end=float(i * 5 + 4),
            text=text,
            caption_text=text,
        ))
    third = max(1, len(clips) // 3)
    selected = tuple(clips[:third])
    alternates = tuple(clips[third:2 * third])
    discarded = tuple(clips[2 * third:])
    return DraftTimeline(
        schema_version=SCHEMA_VERSION,
        project_id="cardinality-probe",
        strategy=EditStrategy.STORYTELLING,
        selected=selected or (clips[0],),
        alternates=alternates,
        discarded=discarded,
    )


def _with_exact_bound(schema: dict, n: int) -> dict:
    out = copy.deepcopy(schema)
    out["properties"]["decisions"]["minItems"] = n
    out["properties"]["decisions"]["maxItems"] = n
    return out


def _with_loose_band(schema: dict, n: int) -> dict:
    out = copy.deepcopy(schema)
    out["properties"]["decisions"]["minItems"] = max(1, n - 2)
    out["properties"]["decisions"]["maxItems"] = n + 2
    return out


def _with_index_echo(schema: dict) -> dict:
    out = copy.deepcopy(schema)
    item_props = out["properties"]["decisions"]["items"]["properties"]
    item_props["candidate_index"] = {"type": "integer", "minimum": 0}
    out["properties"]["decisions"]["items"]["required"] = (
        ["candidate_index"] + out["properties"]["decisions"]["items"]["required"]
    )
    return out


_PROMPT_REINFORCEMENT = (
    " You MUST return exactly {n} decisions in the response array, one per "
    "candidate, in the same order as the candidates array. Never merge two "
    "candidates into a single decision object and never omit any candidate, "
    "even if two candidates look nearly identical -- they still each need "
    "their own decision."
)


def build_variant_body(payload: dict, *, variant: str, n: int) -> dict:
    body = build_unified_selection_request(payload, max_output_tokens=OUTPUT_TOKENS_CEILING)
    base_schema = unified_selection_response_schema(n)  # production: unbounded

    if variant == "unbounded_baseline":
        schema = base_schema
    elif variant == "exact_bound_old":
        schema = _with_exact_bound(base_schema, n)
    elif variant == "loose_band":
        schema = _with_loose_band(base_schema, n)
    elif variant == "index_echo":
        schema = _with_index_echo(base_schema)
    elif variant == "prompt_reinforced":
        schema = base_schema
        body = copy.deepcopy(body)
        text = body["contents"][0]["parts"][0]["text"]
        marker = "Output only the requested JSON schema."
        assert marker in text, "production prompt text changed; update this probe's insertion point"
        body["contents"][0]["parts"][0]["text"] = text.replace(
            marker, marker + _PROMPT_REINFORCEMENT.format(n=n)
        )
    else:
        raise ValueError(f"unknown variant {variant!r}")

    body = copy.deepcopy(body)
    body["generationConfig"]["responseJsonSchema"] = schema
    return body


@dataclass
class CallResult:
    variant: str
    text_mode: str
    trial: int
    status_code: int | None
    ok: bool
    finish_reason: str
    expected_count: int
    returned_count: int | None
    returned_indices: list[int] | None
    error_body: str
    input_tokens: int
    output_tokens: int
    latency_sec: float
    notes: str = ""


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000.0) * INPUT_PER_MILLION_USD + (output_tokens / 1_000_000.0) * OUTPUT_PER_MILLION_USD


def run_one(
    *, variant: str, text_mode: str, trial: int, body: dict, expected_count: int, spend: list[float],
) -> CallResult:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing")
    if sum(spend) >= HARD_CAP_USD:
        raise RuntimeError(f"hard cap ${HARD_CAP_USD} reached before {variant}/{text_mode}/trial{trial}")

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
    started = time.monotonic()
    response = requests.post(
        endpoint,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json=body,
        timeout=90.0,
    )
    latency = round(time.monotonic() - started, 3)
    ok = response.status_code == 200

    finish_reason = ""
    returned_count = None
    returned_indices: list[int] | None = None
    error_body = ""
    input_tokens = 0
    output_tokens = 0
    notes = ""

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
            if isinstance(decisions, list):
                returned_count = len(decisions)
                if decisions and isinstance(decisions[0], dict) and "candidate_index" in decisions[0]:
                    returned_indices = sorted(
                        int(d.get("candidate_index", -1)) for d in decisions if isinstance(d, dict)
                    )
                    missing = sorted(set(range(expected_count)) - set(returned_indices))
                    duplicated = sorted({i for i in returned_indices if returned_indices.count(i) > 1})
                    if missing:
                        notes += f"missing_indices={missing} "
                    if duplicated:
                        notes += f"duplicated_indices={duplicated} "
            else:
                notes += "decisions_key_missing_or_not_list "
        except json.JSONDecodeError as exc:
            notes += f"json_decode_error={exc} "

    return CallResult(
        variant=variant, text_mode=text_mode, trial=trial,
        status_code=response.status_code, ok=ok, finish_reason=finish_reason,
        expected_count=expected_count, returned_count=returned_count,
        returned_indices=returned_indices, error_body=error_body,
        input_tokens=input_tokens, output_tokens=output_tokens,
        latency_sec=latency, notes=notes.strip(),
    )


def main() -> None:
    spend: list[float] = []
    results: list[CallResult] = []

    variants = ["unbounded_baseline", "exact_bound_old", "loose_band", "index_echo", "prompt_reinforced"]
    text_modes = {
        "distinct": _distinct_candidate_texts(CANDIDATE_COUNT),
        "retry_family_heavy": _retry_family_heavy_candidate_texts(CANDIDATE_COUNT),
    }

    for text_mode, texts in text_modes.items():
        draft = make_draft(texts)
        payload = build_unified_selection_payload(draft)
        for variant in variants:
            for trial in range(TRIALS_PER_CELL):
                body = build_variant_body(payload, variant=variant, n=CANDIDATE_COUNT)
                try:
                    result = run_one(
                        variant=variant, text_mode=text_mode, trial=trial,
                        body=body, expected_count=CANDIDATE_COUNT, spend=spend,
                    )
                except RuntimeError as exc:
                    print(f"stopping early: {exc}")
                    break
                results.append(result)
                print(
                    f"{variant:20s} {text_mode:18s} trial{trial} "
                    f"status={result.status_code} finish={result.finish_reason!r} "
                    f"expected={result.expected_count} returned={result.returned_count} "
                    f"{result.notes}"
                )

    report = {
        "model": MODEL,
        "hard_cap_usd": HARD_CAP_USD,
        "spend_usd": round(sum(spend), 6),
        "candidate_count": CANDIDATE_COUNT,
        "trials_per_cell": TRIALS_PER_CELL,
        "cases": [
            {
                "variant": r.variant,
                "text_mode": r.text_mode,
                "trial": r.trial,
                "status_code": r.status_code,
                "ok": r.ok,
                "finish_reason": r.finish_reason,
                "expected_count": r.expected_count,
                "returned_count": r.returned_count,
                "returned_indices": r.returned_indices,
                "notes": r.notes,
                "latency_sec": r.latency_sec,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "error_body": r.error_body,
            }
            for r in results
        ],
    }
    print(json.dumps(report, indent=2))

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/unified-selection-cardinality-isolation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
