"""D-128 Phase 1 -- bounded `MultimodalBestTakeArbiter` contract (SHADOW ONLY).

docs/CUTSELL_DECISIONS.md D-128, docs/CUTSELL_MULTIMODAL_FALLBACK_ARBITER_
FORENSIC_D127.md Section 17. A DISTINCT interface from `semantic_idea_
equivalence.SemanticEquivalenceArbiter` -- that contract is deliberately
TEXT-ONLY (no clip_id/timestamp/video identity, by design, so it can never
become a Video00-specific guard); a multimodal BestTake arbiter needs the
OPPOSITE shape (clip identity, source spans, media references), so it gets
its own contract rather than bolting an unrelated payload onto the
equivalence arbiter's intentionally narrow one.

Mirrors the SAME proven request/response/gate-policy/safe-call pattern
already used throughout this codebase (`semantic_idea_equivalence.py`,
`visual_analysis.py`, `hybrid_editorial.py`) -- not a new architectural
shape.

THIS MODULE NEVER CALLS A PROVIDER. `NullMultimodalBestTakeArbiter` is an
inert placeholder (mirrors `visual_analysis.NoopVisualProvider`'s exact
convention); `safe_arbitrate` exists for the offline eval harness and any
future Phase 2/3 wiring -- the LIVE pipeline (`pipeline.py`'s per-family
diagnostics loop) never calls either of them; it only records, via
`multimodal_besttake_fallback.fallback_trigger_diagnostics`, whether
fallback WOULD be eligible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Protocol, Tuple

from .case_b_performance_evidence import CaseBPerformanceEvidence
from .multimodal_besttake_fallback import FallbackTriggerDecision

SCHEMA_VERSION = "cutsell.multimodal_besttake_arbiter.v1"

# --- bounded output vocabulary (D-127 Section 8, D-128's own tightened set) ---
# Deliberately EXCLUDES KEEP_BOTH_COMPLEMENTARY/NEW_COMPOSITE/NEW_CANDIDATE/
# REWRITE/MERGE_SPEECH -- none of those are legal outcomes for the Cut.ai
# milestone (D-019's KEEP/DISCARD-only doctrine; D-127 Section 8/9).
BEST_TAKE = "BEST_TAKE"
EQUIVALENT = "EQUIVALENT"
GOOD_TAKE_TRIM_ENTRY = "GOOD_TAKE_TRIM_ENTRY"
GOOD_TAKE_TRIM_EXIT = "GOOD_TAKE_TRIM_EXIT"
UNCERTAIN = "UNCERTAIN"
VALID_OUTCOMES = frozenset({BEST_TAKE, EQUIVALENT, GOOD_TAKE_TRIM_ENTRY, GOOD_TAKE_TRIM_EXIT, UNCERTAIN})

# --- shadow-call outcome vocabulary (D-128) ---
NOT_INVOKED = "NOT_INVOKED"
WOULD_INVOKE_SHADOW = "WOULD_INVOKE_SHADOW"
ABSTAINED = "ABSTAINED"
ERROR = "ERROR"
INVALID_RESPONSE = "INVALID_RESPONSE"

# --- named future failure modes (D-127 Section 18). Never exercised in
# Phase 1 -- no provider call exists yet to time out, error, or exceed a
# cost ceiling. Named here so a future Phase 2/3 implementation has a
# stable vocabulary to raise/return, not invented ad hoc later. ---
TIMEOUT = "TIMEOUT"
PROVIDER_ERROR = "PROVIDER_ERROR"
UNSUPPORTED_MEDIA = "UNSUPPORTED_MEDIA"
LOW_CONFIDENCE = "LOW_CONFIDENCE"
COST_CEILING = "COST_CEILING"
MEANING_SAFETY_MISMATCH = "MEANING_SAFETY_MISMATCH"


@dataclass(frozen=True)
class MultimodalFinalistInput:
    """One bounded finalist. References and metadata only -- D-127 Section
    6/9: never raw media bytes, never full RAW. `video_span_reference`/
    `audio_span_reference`/`sampled_frame_references` are placeholders for
    a future provider integration (D-127 Section 7 confirmed real frame
    extraction already exists via `frame_sampling.py`; genuine audio
    perception does not exist yet) -- Phase 1 never populates them with
    anything beyond `None`/`()`."""
    candidate_id: str
    source_asset_id: str
    source_start: float
    source_end: float
    transcript_text: str
    meaning_sufficient: bool
    case_b_evidence: CaseBPerformanceEvidence
    semantic_label: str
    semantic_confidence: float
    deliveryscore_summary: float | None
    boundary_editability_note: str | None = None
    video_span_reference: str | None = None
    audio_span_reference: str | None = None
    sampled_frame_references: Tuple[str, ...] = ()


@dataclass(frozen=True)
class MultimodalBestTakeRequest:
    family_id: str
    proposition_context: str
    finalists: Tuple[MultimodalFinalistInput, ...]


@dataclass(frozen=True)
class MultimodalBestTakeResponse:
    family_id: str
    outcome: str
    best_take_candidate_id: Optional[str]
    confidence: float
    reason: str
    provider: str
    model: str
    requested: bool
    available: bool


class MultimodalBestTakeArbiter(Protocol):
    def arbitrate(self, request: MultimodalBestTakeRequest) -> MultimodalBestTakeResponse: ...


@dataclass(frozen=True)
class MultimodalBestTakeGatePolicy:
    """Call-shape safety bound only -- NEVER a decision threshold. `3` is
    the definitional finalist ceiling this task's own scope already states
    ("2-3 legitimate finalists", D-127/D-128), not an invented calibration
    number. Token/cost/latency ceilings are deliberately NOT set here --
    D-127 Section 16 requires measuring real trigger frequency and
    provider cost/latency before any such number is fixed; setting one
    now would be exactly the threshold invention this task forbids."""
    max_finalists_per_request: int = 3


def should_request_multimodal_arbitration(
    request: MultimodalBestTakeRequest,
    policy: MultimodalBestTakeGatePolicy = MultimodalBestTakeGatePolicy(),
) -> bool:
    if not request.finalists:
        return False
    return len(request.finalists) <= policy.max_finalists_per_request


def validate_multimodal_besttake_response(
    request: MultimodalBestTakeRequest,
    response: MultimodalBestTakeResponse,
) -> MultimodalBestTakeResponse:
    """Mirrors `semantic_idea_equivalence.validate_idea_equivalence_result`'s
    own raise-then-caught-by-safe-call pattern. `BEST_TAKE` MUST reference
    exactly one of the supplied candidate ids -- never a candidate outside
    the bounded request (D-128's own explicit requirement)."""
    if response.outcome not in VALID_OUTCOMES:
        raise ValueError(f"multimodal arbiter returned an unknown outcome: {response.outcome!r}")
    valid_ids = {finalist.candidate_id for finalist in request.finalists}
    if response.outcome == BEST_TAKE and response.best_take_candidate_id not in valid_ids:
        raise ValueError("BEST_TAKE response must reference exactly one supplied candidate id")
    if not 0.0 <= float(response.confidence) <= 1.0:
        raise ValueError("multimodal arbiter confidence outside 0..1")
    return response


class NullMultimodalBestTakeArbiter:
    """Inert shadow placeholder -- no network, no provider, always
    UNCERTAIN/unavailable. Mirrors `visual_analysis.NoopVisualProvider`'s
    exact convention. Gives the protocol a concrete, testable, zero-risk
    implementation; the live pipeline never calls even this one (see
    module docstring)."""
    def arbitrate(self, request: MultimodalBestTakeRequest) -> MultimodalBestTakeResponse:
        return MultimodalBestTakeResponse(
            family_id=request.family_id,
            outcome=UNCERTAIN,
            best_take_candidate_id=None,
            confidence=0.0,
            reason="null_arbiter_shadow_placeholder_no_provider_call",
            provider="none",
            model="none",
            requested=False,
            available=False,
        )


def _classify_provider_call_exception(exc: Exception) -> str:
    """D-136 Phase 2: minimal, additive classification of an exception
    raised by `arbiter.arbitrate(...)` itself (never a response-validation
    failure, which `safe_arbitrate` classifies separately as `INVALID_
    RESPONSE`). These named modes (D-127 Section 18) were declared but
    explicitly unexercised in Phase 1 -- "no provider call exists yet to
    time out, error, or exceed a cost ceiling." Phase 2 is the first
    context that actually calls a provider, so this mapping is now
    exercised for real. Falls back to the pre-existing generic `ERROR`
    only for an exception class this minimal, name-based mapping does not
    recognize -- never a new retry loop, never invented API-specific
    exception types beyond matching on class name (works whether or not
    the `openai` package's own exception classes are importable)."""
    name = exc.__class__.__name__
    if "Timeout" in name:
        return TIMEOUT
    if "APIError" in name or "APIConnection" in name or "APIStatus" in name or "RateLimit" in name or "AuthenticationError" in name:
        return PROVIDER_ERROR
    if isinstance(exc, ValueError):
        # A malformed/non-JSON provider payload raises here (inside
        # `arbiter.arbitrate(...)`, via `openai_json.parse_json_object`)
        # rather than from `validate_multimodal_besttake_response` -- both
        # shapes mean the same thing: an invalid response, not a transport
        # or auth failure.
        return INVALID_RESPONSE
    return ERROR


def _meaning_safety_violation(request: MultimodalBestTakeRequest, response: MultimodalBestTakeResponse) -> bool:
    """D-136 Phase 2 meaning-safety check (directive-required, new):
    every finalist entering a request already carries its own `meaning_
    sufficient` flag (computed upstream, never re-derived here -- same
    "classify, never re-derive" discipline as `detect_class_b_trigger`).
    A `BEST_TAKE` response selecting a finalist whose own flag is `False`
    is a safety mismatch regardless of the provider's stated confidence --
    this never overrides an existing deterministic safety rule, it only
    catches the provider re-introducing a candidate the structured system
    had already excluded."""
    if response.outcome != BEST_TAKE or response.best_take_candidate_id is None:
        return False
    finalist = next(
        (item for item in request.finalists if item.candidate_id == response.best_take_candidate_id),
        None,
    )
    return finalist is not None and not finalist.meaning_sufficient


def safe_arbitrate(
    arbiter: MultimodalBestTakeArbiter | None,
    request: MultimodalBestTakeRequest,
    policy: MultimodalBestTakeGatePolicy = MultimodalBestTakeGatePolicy(),
) -> tuple[str, MultimodalBestTakeResponse | None]:
    """Fail-open safe-call wrapper (mirrors `safe_check_idea_equivalence`/
    `safe_visual_analyze`). Returns `(shadow_call_outcome, response|None)`.
    NOT called anywhere in the live pipeline -- provided for the offline
    eval harness (`multimodal_besttake_eval.py`/`multimodal_besttake_eval_
    phase2.py`) and any future Phase 3 wiring.

    D-136 Phase 2 extension (minimal, additive -- the Protocol/dataclass
    interfaces are unchanged): a raised `arbiter.arbitrate(...)` exception
    is now classified via `_classify_provider_call_exception` (TIMEOUT/
    PROVIDER_ERROR/generic ERROR) instead of always collapsing to `ERROR`;
    a response-validation failure (`validate_multimodal_besttake_response`
    raising `ValueError`) is reported as `INVALID_RESPONSE` specifically
    (previously also generic `ERROR` -- Phase 1's own test asserting the
    old generic value was updated to this more specific, correct code, see
    D-136 decision entry); and every response is checked for a meaning-
    safety violation (`_meaning_safety_violation`) BEFORE any other
    classification -- a mismatch reports `MEANING_SAFETY_MISMATCH` and has
    no authoritative effect (fail open, per directive)."""
    if arbiter is None or not should_request_multimodal_arbitration(request, policy):
        return NOT_INVOKED, None
    try:
        raw_response = arbiter.arbitrate(request)
    except Exception as exc:
        return _classify_provider_call_exception(exc), None
    try:
        response = validate_multimodal_besttake_response(request, raw_response)
    except ValueError:
        return INVALID_RESPONSE, None
    if _meaning_safety_violation(request, response):
        return MEANING_SAFETY_MISMATCH, response
    if not response.available:
        return NOT_INVOKED, response
    if response.outcome == UNCERTAIN:
        return ABSTAINED, response
    return WOULD_INVOKE_SHADOW, response


def build_multimodal_besttake_request(
    trigger: FallbackTriggerDecision,
    finalists_by_id: Mapping[str, MultimodalFinalistInput],
    *,
    proposition_context: str = "",
) -> MultimodalBestTakeRequest | None:
    """Assembles the bounded request from an ELIGIBLE `FallbackTriggerDecision`
    and pre-built per-candidate finalist inputs. Returns `None` when the
    trigger itself is not eligible, or when fewer than 2 of its candidate
    ids have a corresponding finalist input available -- callers must
    never construct or send a request for a non-eligible/under-populated
    family."""
    if not trigger.eligible or trigger.structured_winner is None:
        return None
    candidate_ids = (trigger.structured_winner,) + tuple(trigger.alternative_candidates)
    finalists = tuple(finalists_by_id[cid] for cid in candidate_ids if cid in finalists_by_id)
    if len(finalists) < 2:
        return None
    return MultimodalBestTakeRequest(
        family_id=trigger.family_id or "",
        proposition_context=proposition_context,
        finalists=finalists,
    )
