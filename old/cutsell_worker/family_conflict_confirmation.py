"""D-291 -- family-scoped confirmation of a complete-window label conflict.

docs/CUTSELL_DECISIONS.md D-291 (RAW #118/#124 pimples family). The hybrid
editorial judge labels overlapping 10-candidate windows (stride 5). A retry
family that straddles the overlap is judged twice, in two different
contexts, and the two family-complete windows can disagree: RAW #124 chunk 3
said the monolith was the winner (0.92) and the later take an alternate;
chunk 4 said the opposite (0.95). RAW #118 chunk 3 said monolith winner,
chunk 4 found no winner at all. D-150 correctly abstains on that conflict
(`ABSTAIN_CONFLICT`); before D-289.10 the ladder then fell to the raw
DeliveryScore tie-break (NON_DECISIVE, monolith), after D-289.10 a two-winner
conflict blocks the render as REVIEW_REQUIRED. Neither is the editorial
answer: the D-062.2 hierarchy places *semantic arbiter confirmation* (layer 5)
BEFORE performance tie-breaks (layer 7) and long before human review (layer
11), and the doctrine names Gemini as the bounded arbiter for exactly this
residual ambiguity.

This module asks the SAME editorial judge (same prompt, schema, temperature 0,
per-edit DollarBudgetLedger, `safe_editorial_judge` gate) ONE more bounded
question per conflicted family: a session containing exactly the family's
members plus at most one already-known neighbour on each side for context
(their labels are ignored). The answer resolves the conflict only when it is
an unambiguous family-level verdict: exactly one member labelled "winner" at
or above the same 0.85 floor `_semantic_best_take` applies, every other
member labelled something other than "winner"/"uncertain", and the winner
among the family's meaning-sufficient members. Anything else (uncertain, two
winners, a decline, a budget refusal, a transport error) leaves the recorded
conflict exactly as D-289.10 left it. The confirmed labels then flow through
the UNCHANGED `_semantic_best_take` ladder (D-101 safety veto, D-103 required
realization, D-123 CASE B gate, dominance, contradiction) -- this module
never selects a winner itself and never touches membership or Boundary.

No "last wins", no "highest confidence wins", no clip id, timestamp or phrase:
the judge decides; the module only bounds, records and validates the shape of
its answer. Default ON for the conflict class only; `CUTSELL_FAMILY_CONFLICT_
CONFIRMATION=0` disables it. At most `MAX_CONFIRMATIONS_PER_RUN` requests per
video, each priced and enforced by the transport's own ledger.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Iterable, Mapping

from .contracts import CandidateTake
from .hybrid_editorial import EditorialJudge, HybridGatePolicy, safe_editorial_judge
from .hybrid_session_cleanup import _BUDGET_EXHAUSTED_PROVIDER_PREFIX, _editorial_session
from .semantic_authority_observability import AUTHORITY_ABSTAIN_CONFLICT, stable_request_hash
from .whole_video_analysis import WholeVideoContext

SCHEMA_VERSION = "cutsell.family_conflict_confirmation.v1"

FAMILY_CONFLICT_CONFIRMATION_ENV = "CUTSELL_FAMILY_CONFLICT_CONFIRMATION"
MAX_CONFIRMATIONS_PER_RUN = 4
NEIGHBOUR_CONTEXT_EACH_SIDE = 1
# The SAME winner-label floor `_semantic_best_take` applies (never a new bar).
WINNER_CONFIDENCE_FLOOR = 0.85
CONFIRMATION_CHUNK_INDEX = -1

# compact reason vocabulary
NOT_A_CONFLICT = "not_a_complete_window_conflict"
DISABLED_BY_ENV = "disabled_by_env"
NO_EDITORIAL_JUDGE = "no_editorial_judge"
CAP_REACHED = "confirmation_cap_reached"
NOT_REQUESTED = "judge_gate_did_not_request"
BUDGET_EXHAUSTED = "budget_exhausted"
PROVIDER_UNAVAILABLE = "provider_unavailable"
NO_SINGLE_WINNER = "no_single_family_winner"
WINNER_NOT_MEANING_SUFFICIENT = "winner_not_meaning_sufficient"
OTHER_MEMBER_NOT_RULED_OUT = "other_member_winner_or_uncertain"
CONFIRMED = "family_winner_confirmed"


def family_conflict_confirmation_enabled(env: Mapping[str, str] | None = None) -> bool:
    values = env if env is not None else os.environ
    raw = str(values.get(FAMILY_CONFLICT_CONFIRMATION_ENV, "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class FamilyConflictConfirmation:
    """The full, read-only outcome for one family. `resolved` is True only on
    `CONFIRMED`; `decisions` then carries the family members' confirmed
    (label, confidence) pairs and is empty otherwise."""
    attempted: bool
    resolved: bool
    reason: str
    winner_clip_id: str | None
    decisions: Mapping[str, tuple[str, float]] = field(default_factory=dict)
    row: Mapping[str, object] = field(default_factory=dict)


def _not_attempted(reason: str, family_ids: tuple[str, ...]) -> FamilyConflictConfirmation:
    return FamilyConflictConfirmation(False, False, reason, None, {}, {
        "schema_version": SCHEMA_VERSION, "attempted": False, "resolved": False, "reason": reason,
        "family_member_ids": list(family_ids),
    })


def _neighbours(
    members: tuple[CandidateTake, ...],
    ordered_takes: Iterable[CandidateTake],
) -> tuple[tuple[CandidateTake, ...], tuple[CandidateTake, ...]]:
    """At most `NEIGHBOUR_CONTEXT_EACH_SIDE` same-source takes immediately
    before the first member and after the last member, in source order --
    context only, never labelled by this module."""
    family_ids = {member.clip_id for member in members}
    source = members[0].source_asset_id
    first_start = min(float(member.start) for member in members)
    last_end = max(float(member.end) for member in members)
    same_source = sorted(
        (take for take in ordered_takes if take.source_asset_id == source and take.clip_id not in family_ids),
        key=lambda take: (float(take.start), float(take.end), take.clip_id),
    )
    before = tuple(take for take in same_source if float(take.end) <= first_start + 1e-6)
    after = tuple(take for take in same_source if float(take.start) >= last_end - 1e-6)
    return before[-NEIGHBOUR_CONTEXT_EACH_SIDE:] if NEIGHBOUR_CONTEXT_EACH_SIDE else (), after[:NEIGHBOUR_CONTEXT_EACH_SIDE]


def confirm_family_winner_conflict(
    members: tuple[CandidateTake, ...],
    *,
    semantic_authority_gate_status: str,
    meaning_sufficient_ids: Iterable[str],
    editorial_judge: EditorialJudge | None,
    whole_video_context: WholeVideoContext | None,
    ordered_takes: Iterable[CandidateTake],
    run_counter: dict,
    policy: HybridGatePolicy = HybridGatePolicy(),
    winner_confidence: float = WINNER_CONFIDENCE_FLOOR,
    env: Mapping[str, str] | None = None,
) -> FamilyConflictConfirmation:
    """Ask the editorial judge once about ONE conflicted family. Pure with
    respect to everything except the judge call and `run_counter["used"]`
    (the per-run cap). Never raises."""
    family_ids = tuple(member.clip_id for member in members)
    if len(members) < 2 or str(semantic_authority_gate_status or "") != AUTHORITY_ABSTAIN_CONFLICT:
        return _not_attempted(NOT_A_CONFLICT, family_ids)
    if not family_conflict_confirmation_enabled(env):
        return _not_attempted(DISABLED_BY_ENV, family_ids)
    if editorial_judge is None:
        return _not_attempted(NO_EDITORIAL_JUDGE, family_ids)
    used = int(run_counter.get("used", 0))
    if used >= MAX_CONFIRMATIONS_PER_RUN:
        return _not_attempted(CAP_REACHED, family_ids)

    ordered_members = tuple(sorted(members, key=lambda take: (float(take.start), float(take.end), take.clip_id)))
    before, after = _neighbours(ordered_members, ordered_takes)
    session_members = (*before, *ordered_members, *after)
    session = _editorial_session(
        session_members, whole_video_context,
        partition_index=CONFIRMATION_CHUNK_INDEX, chunk_index=CONFIRMATION_CHUNK_INDEX,
    )
    result = safe_editorial_judge(editorial_judge, session, policy)
    if result.requested:
        run_counter["used"] = used + 1

    row: dict = {
        "schema_version": SCHEMA_VERSION,
        "attempted": True,
        "session_id": session.session_id,
        "chunk_index": CONFIRMATION_CHUNK_INDEX,
        "family_member_ids": list(family_ids),
        "context_member_ids": [take.clip_id for take in (*before, *after)],
        "member_ids": [take.clip_id for take in session_members],
        "requested": bool(result.requested),
        "available": bool(result.available),
        "provider": result.provider,
        "model": result.model,
        "request_hash": stable_request_hash(
            [take.clip_id for take in session_members],
            {take.clip_id: take.text for take in session_members},
            {take.clip_id: take.start for take in session_members},
            {take.clip_id: take.end for take in session_members},
            result.model,
        ),
        "budget_exhausted": str(result.provider or "").startswith(_BUDGET_EXHAUSTED_PROVIDER_PREFIX),
        "run_confirmations_used": int(run_counter.get("used", 0)),
        "decisions": [
            {"clip_id": d.clip_id, "label": d.label, "confidence": d.confidence, "reason_code": d.reason_code,
             "context_only": d.clip_id not in family_ids}
            for d in result.decisions
        ],
    }

    def _outcome(reason: str, winner: str | None, decisions: Mapping[str, tuple[str, float]]):
        row.update({"resolved": reason == CONFIRMED, "reason": reason, "winner_clip_id": winner,
                    "family_labels": {cid: list(pair) for cid, pair in decisions.items()}})
        return FamilyConflictConfirmation(True, reason == CONFIRMED, reason, winner, dict(decisions) if reason == CONFIRMED else {}, row)

    if not result.requested:
        return _outcome(NOT_REQUESTED, None, {})
    if not result.available:
        return _outcome(BUDGET_EXHAUSTED if row["budget_exhausted"] else PROVIDER_UNAVAILABLE, None, {})

    family_labels = {
        d.clip_id: (str(d.label), float(d.confidence)) for d in result.decisions if d.clip_id in family_ids
    }
    if set(family_labels) != set(family_ids):
        return _outcome(PROVIDER_UNAVAILABLE, None, family_labels)
    winners = [cid for cid, (label, conf) in family_labels.items() if label == "winner" and conf >= winner_confidence]
    if len(winners) != 1:
        return _outcome(NO_SINGLE_WINNER, None, family_labels)
    winner = winners[0]
    if winner not in set(meaning_sufficient_ids):
        return _outcome(WINNER_NOT_MEANING_SUFFICIENT, None, family_labels)
    for cid, (label, _conf) in family_labels.items():
        if cid != winner and label in {"winner", "uncertain"}:
            return _outcome(OTHER_MEMBER_NOT_RULED_OUT, None, family_labels)
    return _outcome(CONFIRMED, winner, family_labels)
