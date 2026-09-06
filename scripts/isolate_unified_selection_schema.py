"""Isolate the Gemini 400 seen in RAW #116/#117 without spending a paid RunPod RAW.

RAW #116 and #117 both produced:
    selection_reasoner_status: "provider_error_fail_open"
    diagnostics.unified_selection_reasoner.error:
        "HTTPError: 400 Client Error: Bad Request for url:
         https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent"

cutsell-hybrid-llm-bakeoff.yml already proves gemini-3.5-flash-lite supports this
exact responseJsonSchema/thinkingConfig structured-output mechanism at small scale
(93.5% label accuracy, 2026-08-16 run). So the model and the general mechanism are
not the problem. This script reuses the REAL production request builders from
cutsell_worker.unified_selection_google (not a reimplementation) and sweeps the one
axis that differs between the working bakeoff calls and the Unified Selection
reasoner's whole-video call: candidate count, and whether the array is bound to an
exact minItems==maxItems length.

It makes a small number of cheap, real Gemini calls (same API, same billing as the
bakeoff) and prints the outcome and any error body for each, so the incompatible
request shape can be identified from evidence instead of another guess.

No semantic Selection logic is touched or imported for its behavior: only the
request-building functions are exercised, deterministically, with synthetic data.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests  # noqa: E402

from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION  # noqa: E402
from cutsell_worker.unified_selection_google import (  # noqa: E402
    build_unified_selection_payload,
    build_unified_selection_request,
    unified_selection_response_schema,
)

MODEL = "gemini-3.5-flash-lite"
HARD_CAP_USD = 0.25
INPUT_PER_MILLION_USD = 0.30
OUTPUT_PER_MILLION_USD = 2.50

_SENTENCES = [
    "So I went to the doctor because I noticed a small lump on the side of my neck.",
    "They did an ultrasound first and then recommended a biopsy to be safe.",
    "I was really nervous waiting for the results to come back from the lab.",
    "It turned out to be a papillary thyroid carcinoma, which was scary to hear.",
    "Nobody else in my family has ever had thyroid problems, so this was a shock.",
    "The surgery went well and recovery has been slower than I expected honestly.",
    "I want to share this so other people know to get things checked out early.",
]


def make_draft(candidate_count: int) -> DraftTimeline:
    clips = []
    for i in range(candidate_count):
        text = _SENTENCES[i % len(_SENTENCES)] + f" (segment {i})"
        clips.append(DraftClip(
            clip_id=f"c{i}",
            source_asset_id="src",
            source_order=i,
            start=float(i * 5),
            end=float(i * 5 + 4),
            text=text,
            caption_text=text,
        ))
    third = max(1, candidate_count // 3)
    selected = tuple(clips[:third])
    alternates = tuple(clips[third:2 * third])
    discarded = tuple(clips[2 * third:])
    return DraftTimeline(
        schema_version=SCHEMA_VERSION,
        project_id="isolation-probe",
        strategy=EditStrategy.STORYTELLING,
        selected=selected or (clips[0],),
        alternates=alternates,
        discarded=discarded,
    )


def bakeoff_style_schema(candidate_count: int | None, *, bounded: bool) -> dict:
    """The proven-working schema shape from scripts/hybrid_llm_bakeoff.py's SCHEMA,
    optionally adding the exact-length array bound to isolate that one variable."""
    decisions: dict = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "clip_id": {"type": "string"},
                "label": {"type": "string", "enum": ["winner", "alternate", "failed", "bts", "uncertain", "keep"]},
                "confidence": {"type": "number"},
                "reason_code": {"type": "string"},
            },
            "required": ["clip_id", "label", "confidence", "reason_code"],
            "additionalProperties": False,
        },
    }
    if bounded and candidate_count is not None:
        decisions["minItems"] = int(candidate_count)
        decisions["maxItems"] = int(candidate_count)
    return {
        "type": "object",
        "properties": {"decisions": decisions},
        "required": ["decisions"],
        "additionalProperties": False,
    }


@dataclass
class CallResult:
    name: str
    candidate_count: int
    status_code: int | None
    ok: bool
    error_body: str
    input_tokens: int
    output_tokens: int
    latency_sec: float


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000.0) * INPUT_PER_MILLION_USD + (output_tokens / 1_000_000.0) * OUTPUT_PER_MILLION_USD


def call(name: str, body: dict, spend: list[float], candidate_count: int) -> CallResult:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing")
    if sum(spend) >= HARD_CAP_USD:
        raise RuntimeError(f"hard cap ${HARD_CAP_USD} reached before running case {name!r}")

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
    error_body = "" if ok else response.text[:1200]
    input_tokens = 0
    output_tokens = 0
    if ok:
        raw = response.json()
        usage = raw.get("usageMetadata") or {}
        input_tokens = int(usage.get("promptTokenCount") or 0)
        output_tokens = int(usage.get("candidatesTokenCount") or 0)
        spend.append(estimate_cost(input_tokens, output_tokens))
    else:
        # Failed requests are not billed for generation; still track a small
        # nominal reserve so a string of failures cannot loop unboundedly here.
        spend.append(0.0)

    return CallResult(
        name=name,
        candidate_count=candidate_count,
        status_code=response.status_code,
        ok=ok,
        error_body=error_body,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_sec=latency,
    )


def main() -> None:
    spend: list[float] = []
    results: list[CallResult] = []

    # 1. Unified's real schema (5 fields incl. two enums, unbounded family_index int),
    #    exact-length bound, SMALL N -- sanity: does the richer schema work at all?
    draft_small = make_draft(5)
    payload_small = build_unified_selection_payload(draft_small)
    body_small = build_unified_selection_request(payload_small, max_output_tokens=640)
    results.append(call("unified_schema_bounded_small_N5", body_small, spend, 5))

    # 2. Unified's real schema, exact-length bound, LARGE N (matches whole-video scale).
    draft_large = make_draft(90)
    payload_large = build_unified_selection_payload(draft_large)
    body_large_bounded = build_unified_selection_request(payload_large, max_output_tokens=3200)
    results.append(call("unified_schema_bounded_large_N90", body_large_bounded, spend, 90))

    # 3. Same Unified per-item schema fields, LARGE N, but WITHOUT minItems/maxItems --
    #    isolates whether the exact-length array bound itself is the incompatible part.
    schema_unbounded = unified_selection_response_schema(90)
    schema_unbounded["properties"]["decisions"].pop("minItems", None)
    schema_unbounded["properties"]["decisions"].pop("maxItems", None)
    body_large_unbounded = dict(body_large_bounded)
    body_large_unbounded["generationConfig"] = dict(body_large_bounded["generationConfig"])
    body_large_unbounded["generationConfig"]["responseJsonSchema"] = schema_unbounded
    results.append(call("unified_schema_unbounded_large_N90", body_large_unbounded, spend, 90))

    # 4. Proven-working bakeoff-style small schema, but at LARGE N, no bound --
    #    isolates whether payload/candidate SCALE alone (independent of schema
    #    richness) is what breaks, regardless of schema shape.
    body_simple_unbounded = dict(body_large_bounded)
    body_simple_unbounded["generationConfig"] = dict(body_large_bounded["generationConfig"])
    body_simple_unbounded["generationConfig"]["responseJsonSchema"] = bakeoff_style_schema(90, bounded=False)
    results.append(call("simple_schema_unbounded_large_N90", body_simple_unbounded, spend, 90))

    # 5. Proven-working bakeoff-style small schema, LARGE N, WITH the exact-length
    #    bound -- isolates whether "any schema + exact bound at N=90" fails.
    body_simple_bounded = dict(body_large_bounded)
    body_simple_bounded["generationConfig"] = dict(body_large_bounded["generationConfig"])
    body_simple_bounded["generationConfig"]["responseJsonSchema"] = bakeoff_style_schema(90, bounded=True)
    results.append(call("simple_schema_bounded_large_N90", body_simple_bounded, spend, 90))

    report = {
        "model": MODEL,
        "hard_cap_usd": HARD_CAP_USD,
        "spend_usd": round(sum(spend), 6),
        "cases": [
            {
                "name": r.name,
                "candidate_count": r.candidate_count,
                "status_code": r.status_code,
                "ok": r.ok,
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
    with open("artifacts/unified-selection-schema-isolation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
