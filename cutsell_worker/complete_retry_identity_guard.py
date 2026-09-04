"""Conservative identity guard for complete-take retry supersession.

A complete spoken idea may be auto-deleted as a superseded retry only when the later
candidate is strongly identifiable as the same delivery, not merely a continuation on
the same topic. This keeps the engine aligned with NEVER REMOVE SPOKEN INFORMATION and
WHEN UNCERTAIN, KEEP without hardcoding benchmark phrases or timestamps.

D-072 (observability only -- see docs/CUTSELL_DECISIONS.md): D-070 proved that a real,
previously-observed run could show `later_retry_semantic_overlap > 0` alongside
`later_retry_replacement_id == null` in hybrid_editorial_chunks -- not a bug, but this
guard's own additional sequence-identity veto discarding a candidate the base function
found, while the base function's `overlap` return value survives unchanged. That made
the diagnostic surface look self-contradictory. This module now also records, per call,
WHY a candidate was or wasn't certified -- via the same ContextVar-side-channel pattern
`hybrid_session_cleanup._LAST_SEMANTIC_COMPUTE_PLAN` already establishes for exactly this
class of problem (produced here, read-and-cleared once by hybrid_session_cleanup.py's
own per-decision loop). Purely additive: no return value, gate, or threshold below
changes as a result.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
import unicodedata

_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_COMPLETE_RETRY_MIN_OVERLAP = 0.64
_COMPLETE_RETRY_MIN_SEQUENCE = 0.52

# D-072: bounded, explicit rejection-reason vocabulary -- never free text.
# NOT_APPLICABLE is used only by hybrid_session_cleanup.py's own call site
# for decisions this guard was never invoked for (decision.label != "failed").
NO_CANDIDATE = "NO_CANDIDATE"
SEMANTIC_OVERLAP_BELOW_THRESHOLD = "SEMANTIC_OVERLAP_BELOW_THRESHOLD"
NUMBER_PRESERVATION_FAILED = "NUMBER_PRESERVATION_FAILED"
SEQUENCE_IDENTITY_BELOW_THRESHOLD = "SEQUENCE_IDENTITY_BELOW_THRESHOLD"
LEXICAL_REPLACEMENT_VERIFIED = "LEXICAL_REPLACEMENT_VERIFIED"
INCOMPLETE_RETRY_LOOSER_MATCH = "INCOMPLETE_RETRY_LOOSER_MATCH"
NOT_APPLICABLE = "NOT_APPLICABLE"

_VALID_REPLACEMENT_REJECTION_REASONS = frozenset({
    NO_CANDIDATE, SEMANTIC_OVERLAP_BELOW_THRESHOLD, NUMBER_PRESERVATION_FAILED,
    SEQUENCE_IDENTITY_BELOW_THRESHOLD, LEXICAL_REPLACEMENT_VERIFIED,
    INCOMPLETE_RETRY_LOOSER_MATCH, NOT_APPLICABLE,
})


@dataclass(frozen=True)
class ReplacementGuardDiagnostic:
    """D-072: everything needed to explain one _later_semantic_retry_
    replacement call's outcome, independent of the (candidate, overlap)
    tuple the function itself must keep returning unchanged."""
    replacement_candidate_clip_id_before_guard: str | None
    semantic_overlap: float
    sequence_identity: float | None
    sequence_identity_threshold: float
    lexical_identity_passed: bool | None
    replacement_rejection_reason: str

    def __post_init__(self) -> None:
        assert self.replacement_rejection_reason in _VALID_REPLACEMENT_REJECTION_REASONS


# D-072: same ContextVar side-channel pattern as hybrid_session_cleanup.py's
# own _LAST_SEMANTIC_COMPUTE_PLAN / hybrid_semantic_complementary_rescue.
# _SPLIT_IDS -- set once per _later_semantic_retry_replacement call, read
# and cleared once by hybrid_session_cleanup.py's per-decision loop.
# ContextVars are per-context (per async task / thread), never a shared
# global, so this can never leak between concurrent jobs or clips; explicit
# clearing on read additionally means a caller that forgets to consume it
# for one decision can never accidentally attribute a stale diagnostic to
# a later, unrelated decision.
_LAST_REPLACEMENT_GUARD_DIAGNOSTIC: "ContextVar[ReplacementGuardDiagnostic | None]" = ContextVar(
    "_LAST_REPLACEMENT_GUARD_DIAGNOSTIC", default=None,
)


def _consume_replacement_guard_diagnostic() -> "ReplacementGuardDiagnostic | None":
    """Read-and-clear. Returns None if no _later_semantic_retry_replacement
    call has run in this context since the last consumption."""
    value = _LAST_REPLACEMENT_GUARD_DIAGNOSTIC.get()
    _LAST_REPLACEMENT_GUARD_DIAGNOSTIC.set(None)
    return value


def _diagnose_no_replacement_reason(
    failed_take,
    members,
    compatible_members,
    decisions_by_id,
    *,
    minimum_label_confidence: float,
    minimum_overlap: float,
    maximum_delay_sec: float,
) -> str:
    """D-072: called only when original() (on compatible_members) returned
    no candidate -- distinguishes NO_CANDIDATE (nothing structurally
    eligible existed) from SEMANTIC_OVERLAP_BELOW_THRESHOLD (something
    eligible existed but never reached the overlap floor) from
    NUMBER_PRESERVATION_FAILED (the number-preservation filter is what
    removed every eligible candidate). Pure, side-effect-free re-scan
    (hybrid_session_cleanup._scan_replacement_candidates) -- observability
    only, never consulted by the actual certification decision above."""
    from .hybrid_session_cleanup import _scan_replacement_candidates

    filtered_scan = _scan_replacement_candidates(
        failed_take, compatible_members, decisions_by_id,
        minimum_label_confidence=minimum_label_confidence,
        maximum_delay_sec=maximum_delay_sec,
    )
    if filtered_scan.eligible_candidate_count == 0:
        if len(compatible_members) < len(members):
            full_scan = _scan_replacement_candidates(
                failed_take, members, decisions_by_id,
                minimum_label_confidence=minimum_label_confidence,
                maximum_delay_sec=maximum_delay_sec,
            )
            if full_scan.eligible_candidate_count > filtered_scan.eligible_candidate_count:
                return NUMBER_PRESERVATION_FAILED
        return NO_CANDIDATE
    if filtered_scan.best_overlap_seen < minimum_overlap:
        return SEMANTIC_OVERLAP_BELOW_THRESHOLD
    # Eligible and overlap cleared the floor, yet original() still returned
    # no candidate -- should not happen given original()'s own invariant
    # (D-070); fall back to the most honest available label instead of
    # fabricating a more specific one.
    return NO_CANDIDATE


def _numbers(text: str) -> frozenset[str]:
    return frozenset(match.group(0).replace(",", ".") for match in _NUMBER_RE.finditer(str(text or "")))


def _normalized_text(text: str) -> str:
    raw = unicodedata.normalize("NFKD", str(text or "").lower())
    raw = "".join(char for char in raw if not unicodedata.combining(char))
    raw = re.sub(r"[^\w]+", " ", raw, flags=re.UNICODE)
    return " ".join(raw.split())


def _sequence_identity(left_text: str, right_text: str) -> float:
    left = _normalized_text(left_text)
    right = _normalized_text(right_text)
    if not left or not right:
        return 0.0
    return float(SequenceMatcher(None, left, right).ratio())


def install_complete_retry_identity_guard() -> None:
    from . import hybrid_session_cleanup as cleanup

    original = cleanup._later_semantic_retry_replacement
    if getattr(original, "_cutsell_complete_retry_identity_guard", False):
        return

    def protected(
        failed_take,
        members,
        decisions_by_id,
        *,
        minimum_label_confidence: float = 0.68,
        minimum_overlap: float = 0.50,
        maximum_delay_sec: float = 24.0,
    ):
        # Incomplete retries can still use the original looser matching: by definition
        # they often contain only a fragment of the later successful delivery.
        if not bool(getattr(failed_take, "complete_idea", False)):
            replacement, overlap = original(
                failed_take,
                members,
                decisions_by_id,
                minimum_label_confidence=minimum_label_confidence,
                minimum_overlap=minimum_overlap,
                maximum_delay_sec=maximum_delay_sec,
            )
            # D-072: this branch never applies the stricter guard at all --
            # report accordingly rather than implying a rejection happened.
            _LAST_REPLACEMENT_GUARD_DIAGNOSTIC.set(ReplacementGuardDiagnostic(
                replacement_candidate_clip_id_before_guard=(
                    replacement.clip_id if replacement is not None else None
                ),
                semantic_overlap=overlap,
                sequence_identity=None,
                sequence_identity_threshold=_COMPLETE_RETRY_MIN_SEQUENCE,
                lexical_identity_passed=None,
                replacement_rejection_reason=(
                    INCOMPLETE_RETRY_LOOSER_MATCH if replacement is not None else NO_CANDIDATE
                ),
            ))
            return replacement, overlap

        # Complete deliveries carry information authority. A later take must preserve
        # every numeric fact already spoken before it can supersede the earlier take.
        failed_numbers = _numbers(getattr(failed_take, "text", ""))
        compatible_members = tuple(
            candidate
            for candidate in members
            if candidate.clip_id == failed_take.clip_id
            or not failed_numbers
            or failed_numbers.issubset(_numbers(getattr(candidate, "text", "")))
        )
        effective_min_overlap = max(float(minimum_overlap), _COMPLETE_RETRY_MIN_OVERLAP)

        replacement, overlap = original(
            failed_take,
            compatible_members,
            decisions_by_id,
            minimum_label_confidence=minimum_label_confidence,
            minimum_overlap=effective_min_overlap,
            maximum_delay_sec=maximum_delay_sec,
        )
        if replacement is None:
            # D-072: distinguish NO_CANDIDATE / SEMANTIC_OVERLAP_BELOW_THRESHOLD /
            # NUMBER_PRESERVATION_FAILED -- observability only, does not
            # change the (None, overlap) already decided above.
            reason = _diagnose_no_replacement_reason(
                failed_take, members, compatible_members, decisions_by_id,
                minimum_label_confidence=minimum_label_confidence,
                minimum_overlap=effective_min_overlap,
                maximum_delay_sec=maximum_delay_sec,
            )
            _LAST_REPLACEMENT_GUARD_DIAGNOSTIC.set(ReplacementGuardDiagnostic(
                replacement_candidate_clip_id_before_guard=None,
                semantic_overlap=overlap,
                sequence_identity=None,
                sequence_identity_threshold=_COMPLETE_RETRY_MIN_SEQUENCE,
                lexical_identity_passed=None,
                replacement_rejection_reason=reason,
            ))
            return None, overlap

        # Topic overlap alone is not enough. Require recognizable delivery-level
        # sequence identity as a second independent signal. This rejects narrative
        # continuations that reuse nouns while introducing a new fact or event.
        sequence_identity = _sequence_identity(
            getattr(failed_take, "text", ""),
            getattr(replacement, "text", ""),
        )
        lexical_identity_passed = sequence_identity >= _COMPLETE_RETRY_MIN_SEQUENCE
        if not lexical_identity_passed:
            # D-072: this is the exact D-070 shape -- a real candidate was
            # found (nonzero overlap) and then vetoed here. Recording it
            # explicitly is the whole point of this directive: the
            # decision itself (return None, overlap) is unchanged.
            _LAST_REPLACEMENT_GUARD_DIAGNOSTIC.set(ReplacementGuardDiagnostic(
                replacement_candidate_clip_id_before_guard=replacement.clip_id,
                semantic_overlap=overlap,
                sequence_identity=sequence_identity,
                sequence_identity_threshold=_COMPLETE_RETRY_MIN_SEQUENCE,
                lexical_identity_passed=False,
                replacement_rejection_reason=SEQUENCE_IDENTITY_BELOW_THRESHOLD,
            ))
            return None, overlap

        _LAST_REPLACEMENT_GUARD_DIAGNOSTIC.set(ReplacementGuardDiagnostic(
            replacement_candidate_clip_id_before_guard=replacement.clip_id,
            semantic_overlap=overlap,
            sequence_identity=sequence_identity,
            sequence_identity_threshold=_COMPLETE_RETRY_MIN_SEQUENCE,
            lexical_identity_passed=True,
            replacement_rejection_reason=LEXICAL_REPLACEMENT_VERIFIED,
        ))
        return replacement, overlap

    protected._cutsell_complete_retry_identity_guard = True
    cleanup._later_semantic_retry_replacement = protected
