"""D-136 (D-127 Phase 2) -- provider-backed OFFLINE evaluation harness.

docs/CUTSELL_DECISIONS.md D-136. Reuses the SAME 9 D-127 Section 20 fixture
cases already proven in `multimodal_besttake_eval.py` (Phase 1's own eval
set, unchanged, still gating `detect_class_b_trigger` alone) -- this module
does not redefine them, it builds a bounded `MultimodalBestTakeRequest` per
case and runs it through `safe_arbitrate` against a CONCRETE arbiter (e.g.
`multimodal_besttake_openai.OpenAIMultimodalBestTakeArbiter`), scoring the
result against an independently-declared expected outcome (never derived
from the provider's own response -- see `_EXPECTED_OUTCOMES` below, fixed
before any provider call is made).

EVALUATION ONLY. Nothing here is imported by `pipeline.py`; nothing here
can change a production winner, score, rank, grouping, or Boundary
decision -- every request is built from already-existing D-127/D-128 eval
fixtures, and every result is a plain report row, never written back to
any authoritative structure.

Synthetic media only: each candidate gets ONE tiny synthetic stub JPEG
frame (an ffmpeg `lavfi` solid-color source -- the SAME `ffmpeg` binary
`frame_sampling.py` already depends on, no new dependency, no real source
media, no RAW, no paid GPU). This mirrors `frame_sampling.FrameSample`'s
shape closely enough for `MultimodalFinalistInput.sampled_frame_
references` without building a second, parallel frame extractor.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .case_b_performance_evidence import CaseBPerformanceEvidence
from .multimodal_besttake_arbiter import (
    ABSTAINED,
    BEST_TAKE,
    EQUIVALENT,
    GOOD_TAKE_TRIM_EXIT,
    MEANING_SAFETY_MISMATCH,
    NOT_INVOKED,
    UNCERTAIN,
    WOULD_INVOKE_SHADOW,
    MultimodalBestTakeArbiter,
    MultimodalBestTakeRequest,
    MultimodalFinalistInput,
    safe_arbitrate,
)
from .multimodal_besttake_eval import EvalCase, eval_cases

# --- D-136 Expected Outcome Policy vocabulary (directive-required) --------
SHOULD_SELECT_A = "SHOULD_SELECT_A"
SHOULD_SELECT_B = "SHOULD_SELECT_B"
SHOULD_EQUIVALENT = "SHOULD_EQUIVALENT"
SHOULD_TRIM_EXIT = "SHOULD_TRIM_EXIT"
SHOULD_UNCERTAIN = "SHOULD_UNCERTAIN"
NOT_APPLICABLE_SINGLE_MEMBER = "NOT_APPLICABLE_SINGLE_MEMBER"

#: Fixed BEFORE any provider call -- derived from the SAME D-127/D-128
#: doctrine each fixture's own docstring already states, never from
#: provider output. `structured_engine_winner` is the candidate id the
#: EXISTING structured system would keep (agreement cases: the agreed
#: winner; the one real disagreement case this eval set traces to a real
#: RAW -- `semantic_deliveryscore_disagreement_negative`, D-126's own
#: `tg_ef754f8f610ab360df` shape -- resolved to "A" via the pre-existing
#: `critical_coverage_dominance` authority, confirmed in D-126;
#: `complementary_content_negative` is a generic, not-RAW-traced modeled
#: disagreement, so its assumed structured outcome is the DeliveryScorer
#: top pick, honestly labeled `assumed=True`).
_EXPECTED_OUTCOMES: dict[str, dict] = {
    "pimples_shaped_positive": {
        "expected": SHOULD_SELECT_B, "structured_engine_winner": "A", "assumed": False,
        "would_improve_if_correct": True,
    },
    "papillary_equivalent_realization_negative": {
        "expected": SHOULD_EQUIVALENT, "structured_engine_winner": "A", "assumed": False,
        "would_improve_if_correct": False,
    },
    "stomach_retry_negative": {
        "expected": SHOULD_UNCERTAIN, "structured_engine_winner": "A", "assumed": False,
        "would_improve_if_correct": False,
    },
    "complementary_content_negative": {
        "expected": SHOULD_UNCERTAIN, "structured_engine_winner": "B", "assumed": True,
        "would_improve_if_correct": False,
    },
    "polarity_negation_safety_negative": {
        "expected": SHOULD_SELECT_A, "structured_engine_winner": "A", "assumed": False,
        "would_improve_if_correct": False,
    },
    "legitimate_clean_retry_negative": {
        "expected": NOT_APPLICABLE_SINGLE_MEMBER, "structured_engine_winner": "A", "assumed": False,
        "would_improve_if_correct": False,
    },
    "semantic_deliveryscore_disagreement_negative": {
        "expected": SHOULD_SELECT_A, "structured_engine_winner": "A", "assumed": False,
        "would_improve_if_correct": False,
    },
    "ambiguous_tied_performance_negative": {
        "expected": SHOULD_EQUIVALENT, "structured_engine_winner": "A", "assumed": False,
        "would_improve_if_correct": False,
    },
    "boundary_only_exit_negative": {
        "expected": SHOULD_TRIM_EXIT, "structured_engine_winner": "A", "assumed": False,
        "would_improve_if_correct": False,
    },
}

#: Cases that carry a real, meaningful positive Class B trigger (per D-128's
#: own `detect_class_b_trigger` classification) -- exactly one, matching
#: Phase 1's own eval set.
_POSITIVE_CASE_NAMES = frozenset({"pimples_shaped_positive"})


def _synthetic_stub_frame(directory: Path, candidate_id: str, hue: str) -> str:
    """One tiny, deterministic, synthetic stub frame -- ffmpeg `lavfi`
    solid-color source, the SAME ffmpeg binary `frame_sampling.py` already
    depends on. No real source media, no RAW, no paid GPU."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{candidate_id}.jpg"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"color=c={hue}:s=64x64:d=1",
        "-frames:v", "1", "-q:v", "5", str(destination),
    ]
    subprocess.run(command, capture_output=True, check=True)
    return str(destination)


_HUES = ("gray", "steelblue", "darkorange")


def _finalist_from_fixture(
    case: EvalCase,
    candidate_id: str,
    *,
    frame_dir: Path,
    hue: str,
    boundary_note: str | None = None,
) -> MultimodalFinalistInput:
    kwargs = case.kwargs
    evidence: CaseBPerformanceEvidence = kwargs["case_b_evidence_by_id"].get(candidate_id) or CaseBPerformanceEvidence(
        candidate_id=candidate_id, source_asset_id="src", delivery_available=False,
        delivery_start=None, delivery_end=None, delivery_span_duration=None,
        delivery_events=(), delivery_event_count=0, delivery_event_duration_total=0.0,
        count_by_kind={}, duration_by_kind={}, event_density=None,
    )
    semantic_winner = kwargs.get("semantic_winner")
    deliveryscore_winner = kwargs.get("deliveryscore_winner")
    is_semantic_winner = candidate_id == semantic_winner
    return MultimodalFinalistInput(
        candidate_id=candidate_id,
        source_asset_id="eval-src",
        source_start=0.0,
        source_end=10.0,
        # D-136: synthetic, generic, non-Video00-specific per-candidate
        # description -- this eval set has no real ASR transcript (its
        # underlying CaseBPerformanceEvidence fixtures are hand-built
        # synthetic values, not real footage); labeled honestly as such.
        transcript_text=f"(synthetic eval fixture) candidate {candidate_id} of case {case.name}.",
        meaning_sufficient=candidate_id in kwargs.get("meaning_sufficient_ids", set()),
        case_b_evidence=evidence,
        semantic_label=("winner" if is_semantic_winner else "other"),
        semantic_confidence=(0.95 if is_semantic_winner else 0.0),
        deliveryscore_summary=(1.0 if candidate_id == deliveryscore_winner else 0.0),
        boundary_editability_note=boundary_note,
        sampled_frame_references=(_synthetic_stub_frame(frame_dir, candidate_id, hue),),
    )


@dataclass(frozen=True)
class Phase2CaseResult:
    name: str
    trigger_class: str
    candidate_ids: tuple[str, ...]
    structured_engine_winner: str
    expected_outcome: str
    arbiter_outcome: str | None
    arbiter_candidate_id: str | None
    confidence: float | None
    safe_call_status: str
    verdict: str  # "correct" | "incorrect" | "abstained" | "skipped" | "error"
    winner_would_change: bool
    would_improve_structured_result: bool
    meaning_safety_violation: bool
    latency_ms: float | None
    call_metadata: dict | None


def _score(case_name: str, expected: str, status: str, response) -> tuple[str, bool, bool]:
    """Returns (verdict, winner_would_change, would_improve). Never reads
    provider output to determine the EXPECTED value -- only to determine
    whether the ACTUAL output matched it."""
    meta = _EXPECTED_OUTCOMES[case_name]
    structured_winner = meta["structured_engine_winner"]
    would_improve_if_correct = meta["would_improve_if_correct"]

    if status == MEANING_SAFETY_MISMATCH:
        return "incorrect", True, False
    if status in (NOT_INVOKED,) or response is None:
        return ("skipped" if expected == NOT_APPLICABLE_SINGLE_MEMBER else "error"), False, False
    if status not in (WOULD_INVOKE_SHADOW, ABSTAINED):
        return "error", False, False

    outcome = response.outcome
    candidate = response.best_take_candidate_id
    winner_would_change = bool(outcome == BEST_TAKE and candidate != structured_winner)

    if expected in (SHOULD_SELECT_A, SHOULD_SELECT_B):
        expected_candidate = "A" if expected == SHOULD_SELECT_A else "B"
        if outcome == BEST_TAKE and candidate == expected_candidate:
            return "correct", winner_would_change, (would_improve_if_correct and winner_would_change)
        if outcome == BEST_TAKE:
            return "incorrect", winner_would_change, False
        return "abstained", False, False

    if expected == SHOULD_EQUIVALENT:
        if outcome in (EQUIVALENT, UNCERTAIN):
            return "correct", False, False
        return "incorrect", winner_would_change, False

    if expected == SHOULD_UNCERTAIN:
        if outcome in (UNCERTAIN, EQUIVALENT):
            return "correct", False, False
        return "incorrect", winner_would_change, False

    if expected == SHOULD_TRIM_EXIT:
        if outcome == GOOD_TAKE_TRIM_EXIT:
            return "correct", False, False
        if outcome == BEST_TAKE:
            return "incorrect", winner_would_change, False
        return "abstained", False, False

    return "error", False, False


def run_phase2_eval(
    arbiter: MultimodalBestTakeArbiter,
    *,
    workdir: str,
) -> list[Phase2CaseResult]:
    """Runs every D-127 Section 20 case through the given concrete arbiter
    (via `safe_arbitrate`, never a direct unguarded call). One result row
    per case. Never invoked from the live pipeline."""
    results: list[Phase2CaseResult] = []
    frame_root = Path(workdir)
    for case in eval_cases():
        kwargs = case.kwargs
        member_ids = tuple(kwargs["member_ids"])
        meta = _EXPECTED_OUTCOMES[case.name]

        if meta["expected"] == NOT_APPLICABLE_SINGLE_MEMBER:
            results.append(Phase2CaseResult(
                name=case.name,
                trigger_class=("CLASS_B" if case.name in _POSITIVE_CASE_NAMES else "NEGATIVE_CONTROL"),
                candidate_ids=member_ids,
                structured_engine_winner=meta["structured_engine_winner"],
                expected_outcome=meta["expected"], arbiter_outcome=None,
                arbiter_candidate_id=None, confidence=None, safe_call_status=NOT_INVOKED,
                verdict="skipped", winner_would_change=False, would_improve_structured_result=False,
                meaning_safety_violation=False, latency_ms=None, call_metadata=None,
            ))
            continue

        boundary_note = (
            "trailing debris after spoken content ends, otherwise clean and usable"
            if case.name == "boundary_only_exit_negative" else None
        )
        finalists = tuple(
            _finalist_from_fixture(
                case, candidate_id,
                frame_dir=frame_root / case.name,
                hue=_HUES[index % len(_HUES)],
                boundary_note=(boundary_note if (candidate_id == "A" and boundary_note) else None),
            )
            for index, candidate_id in enumerate(member_ids)
        )
        request = MultimodalBestTakeRequest(
            family_id=kwargs["family_id"], proposition_context="", finalists=finalists,
        )

        started = time.monotonic()
        status, response = safe_arbitrate(arbiter, request)
        latency_ms = round((time.monotonic() - started) * 1000, 1)
        call_metadata = getattr(arbiter, "last_call_metadata", None)

        verdict, winner_would_change, would_improve = _score(case.name, meta["expected"], status, response)
        results.append(Phase2CaseResult(
            name=case.name,
            trigger_class=("CLASS_B" if case.name in _POSITIVE_CASE_NAMES else "NEGATIVE_CONTROL"),
            candidate_ids=member_ids,
            structured_engine_winner=meta["structured_engine_winner"],
            expected_outcome=meta["expected"],
            arbiter_outcome=(response.outcome if response is not None else None),
            arbiter_candidate_id=(response.best_take_candidate_id if response is not None else None),
            confidence=(response.confidence if response is not None else None),
            safe_call_status=status,
            verdict=verdict,
            winner_would_change=winner_would_change,
            would_improve_structured_result=would_improve,
            meaning_safety_violation=(status == MEANING_SAFETY_MISMATCH),
            latency_ms=latency_ms,
            call_metadata=call_metadata,
        ))
    return results


def summarize_phase2_eval(results: list[Phase2CaseResult]) -> dict:
    """D-136 PASS METRICS -- factual counts only, never a production
    activation threshold."""
    total = len(results)
    positive = sum(1 for r in results if r.trigger_class == "CLASS_B")
    negative = total - positive
    correct = sum(1 for r in results if r.verdict == "correct")
    incorrect = sum(1 for r in results if r.verdict == "incorrect")
    abstained = sum(1 for r in results if r.verdict == "abstained")
    skipped = sum(1 for r in results if r.verdict == "skipped")
    errored = sum(1 for r in results if r.verdict == "error")
    positive_improvement = sum(
        1 for r in results if r.trigger_class == "CLASS_B" and r.verdict == "correct" and r.would_improve_structured_result
    )
    negative_regression = sum(
        1 for r in results if r.trigger_class == "NEGATIVE_CONTROL" and r.winner_would_change
    )
    meaning_violations = sum(1 for r in results if r.meaning_safety_violation)
    provider_failures = sum(
        1 for r in results
        if r.safe_call_status not in (WOULD_INVOKE_SHADOW, ABSTAINED, NOT_INVOKED, MEANING_SAFETY_MISMATCH)
    )
    return {
        "total_cases": total,
        "positive_cases": positive,
        "negative_controls": negative,
        "correct": correct,
        "incorrect": incorrect,
        "abstained": abstained,
        "skipped_single_member": skipped,
        "invalid_or_error": errored,
        "positive_case_improvement_count": positive_improvement,
        "negative_control_regression_count": negative_regression,
        "meaning_safety_violations": meaning_violations,
        "provider_failure_count": provider_failures,
    }
