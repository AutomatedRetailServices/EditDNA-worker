"""Cost-efficient semantic cleanup across bounded creator mini-sessions.

Instead of paying once per retry group/singleton, this stage batches the candidates that
belong to one inferred creator mini-session. The model judges every candidate against a
compact whole-source transcript and overlapping local windows so a failed attempt and a
later clean retake are less likely to fall into unrelated semantic calls.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import os
import re
from typing import Iterable, Mapping, Tuple

from .complete_retry_identity_guard import (
    NOT_APPLICABLE as _REPLACEMENT_REASON_NOT_APPLICABLE,
    _consume_replacement_guard_diagnostic,
)
from .contracts import CandidateTake
from .final_sibling_grouping import _content as _content_tokens
from .final_sibling_grouping import _negations, _numbers
from .hybrid_editorial import (
    EditorialCandidate,
    EditorialJudge,
    EditorialSession,
    HybridGatePolicy,
    safe_editorial_judge,
)
from .hybrid_payload import estimate_tokens_from_chars
from .hybrid_provider_settings import HybridProviderSettings
from .semantic_compute_planner import (
    SemanticComputePlan,
    SemanticWorkItem,
    SemanticWorkPriority,
    build_semantic_compute_plan,
)
from .session_boundaries import partition_takes_by_sessions
from .temporal_editing import harmful_events_for_take
from .whole_video_analysis import WholeVideoContext

# D-053 Section 11: composite_resolver.py's 19-hook monkeypatch chain wraps
# apply_hybrid_session_cleanup's own HybridSessionCleanupResult repeatedly;
# several hooks reconstruct it via `type(result)(kept=..., deleted=...,
# requested_chunk_count=..., available_chunk_count=..., diagnostics=...,
# semantic_decisions=...)` -- a keyword constructor call, not
# dataclasses.replace() -- which silently drops any field the hook's own
# (pre-D-052) code doesn't name, reverting semantic_compute_plan to its
# dataclass default (None) the moment the first such hook fires. Rather than
# hand-edit all 20 affected hook files (a real regression risk on already
# load-bearing scoring logic, and exactly the kind of change this directive
# says must not touch semantic execution ordering or provider decisions),
# this uses the SAME ContextVar side-channel pattern
# hybrid_semantic_complementary_rescue._SPLIT_IDS /
# hybrid_composite_best_take._COMPOSITE_SPLIT_IDS already establish for
# exactly this class of problem (composite_resolver.py's own
# _composite_split_ids() reads and clears those two the same way). Set once
# here, read-and-cleared once by composite_resolver.apply_composite_
# resolution -- no hook file needs to know this exists.
_LAST_SEMANTIC_COMPUTE_PLAN: "ContextVar[SemanticComputePlan | None]" = ContextVar(
    "_LAST_SEMANTIC_COMPUTE_PLAN", default=None,
)

_TOKEN_RE = re.compile(r"[\w'’-]+", re.UNICODE)
_RESET_CANDIDATES = frozenset({"body_reset_candidate", "hand_motion_reset_candidate"})
_BREAK_CANDIDATES = frozenset({"camera_disengagement_candidate", "facial_expression_shift_candidate"})

# D-052 Part B Section 14: OFF by default -- current pure chunk_index-order
# dispatch (whichever chunk happens to be requested fifth silently loses the
# shared budget) is unchanged unless explicitly opted in. When enabled,
# every window in this call is classified into a SemanticWorkPriority using
# ONLY signals already available before any paid call is made, planned via
# semantic_compute_planner.build_semantic_compute_plan, and dispatched in
# planned (priority) order instead of enumeration order. See
# tests/test_cutsell_d052_semantic_compute_planner.py and
# CUTSELL_DECISIONS.md D-052.
_SEMANTIC_COMPUTE_PLANNER_ENV = "CUTSELL_SEMANTIC_COMPUTE_PLANNER"


def _semantic_compute_planner_enabled(env: Mapping[str, str] | None = None) -> bool:
    values = env if env is not None else os.environ
    return str(values.get(_SEMANTIC_COMPUTE_PLANNER_ENV, "")).strip().lower() in {"1", "true", "yes", "on"}


def _has_negation_or_number_conflict(members: Tuple[CandidateTake, ...]) -> bool:
    """Pre-call, fully deterministic P0 signal: does this window already
    contain two members that look like the same idea (meaningful content-
    token overlap) but disagree on negation or number presence? This is
    the exact same shape final_story_coherence_validation later checks
    (D-052 deliberately reuses final_sibling_grouping's own extractors) --
    here it is used only to reserve semantic-compute budget ahead of a
    contradiction risk, never to make an editorial decision itself."""
    texts = [member.text for member in members]
    for i in range(len(texts)):
        left_content = _content_tokens(texts[i])
        if len(left_content) < 4:
            continue
        for j in range(i + 1, len(texts)):
            right_content = _content_tokens(texts[j])
            if len(right_content) < 4:
                continue
            shared = len(left_content & right_content)
            if shared < 3:
                continue
            left_negations, right_negations = _negations(texts[i]), _negations(texts[j])
            left_numbers, right_numbers = _numbers(texts[i]), _numbers(texts[j])
            if bool(left_negations) != bool(right_negations):
                return True
            if left_numbers and right_numbers and left_numbers != right_numbers:
                return True
    return False


def _classify_window_priority(
    members: Tuple[CandidateTake, ...],
    context: WholeVideoContext | None,
) -> SemanticWorkPriority:
    """D-052 Section 7: classify one window's semantic-compute priority
    using only pre-call signals -- StoryValidator's own contradiction check
    runs downstream of hybrid editorial and is not available yet, so P0
    here is a conservative, cheaper, pre-call proxy for the same risk
    shape, not a duplicate of that later check."""
    if _has_negation_or_number_conflict(members):
        return SemanticWorkPriority.P0_SAFETY_CRITICAL
    for member in members:
        corroborated, _reasons = _failed_local_evidence(member, context)
        if corroborated:
            return SemanticWorkPriority.P1_RETRY_EQUIVALENCE
    return SemanticWorkPriority.P2_EDITORIAL_QUALITY


_COST_ESTIMATE_SETTINGS = HybridProviderSettings()
_CONTEXT_OVERHEAD_CHARS = 600  # rough allowance for the session's fixed schema/context fields


def _estimate_output_token_ceiling(member_count: int) -> int:
    """Same tiering hybrid_google_transport._compact_output_token_ceiling
    uses for the real compact structured-decision schema -- duplicated
    (not imported) because that function additionally needs the live
    request's ``requested_max`` ceiling, which this planning-time estimate
    has no equivalent for yet."""
    if member_count <= 2:
        return 192
    if member_count <= 4:
        return 256
    if member_count <= 6:
        return 320
    return min(500, 80 + (32 * member_count))


def _estimate_window_cost_usd(members: Tuple[CandidateTake, ...]) -> float:
    """D-053 Section 11: real, token-based cost estimate using the SAME
    formula the live transport bills against
    (HybridProviderSettings.estimate_cost_usd), not a flat per-member
    dollar guess. D-052's original flat estimate (0.0015 USD/member)
    overshot every real window's actual cost badly enough that
    build_semantic_compute_plan's own ``planned_calls``/
    ``deferred_optional_calls`` bookkeeping showed 0 planned calls in a
    live 3-run battery even though the real ledger (a genuinely different,
    authoritative accounting) accepted 4 of 6 -- this fixes the ESTIMATE
    only. Priority classification and dispatch ORDER (the actual D-052
    Part B fix) never depended on this number being accurate; only the
    plan's own "predicted planned/deferred" diagnostic labels did."""
    total_chars = sum(len(str(member.text or "")) for member in members) + _CONTEXT_OVERHEAD_CHARS
    input_tokens = estimate_tokens_from_chars(total_chars)
    output_tokens = _estimate_output_token_ceiling(len(members))
    return round(_COST_ESTIMATE_SETTINGS.estimate_cost_usd(input_tokens=input_tokens, output_tokens=output_tokens), 6)


@dataclass(frozen=True)
class HybridSessionCleanupResult:
    kept: Tuple[CandidateTake, ...]
    deleted: Tuple[CandidateTake, ...]
    requested_chunk_count: int
    available_chunk_count: int
    diagnostics: tuple[dict, ...]
    semantic_decisions: tuple[tuple[str, str, float], ...] = ()
    # D-052 Part B: present only when CUTSELL_SEMANTIC_COMPUTE_PLANNER is
    # enabled for this call -- None otherwise (today's default), never a
    # behavior change to any existing caller that only reads the fields
    # above.
    semantic_compute_plan: "SemanticComputePlan | None" = None


def _token_count(text: str) -> int:
    return len(_TOKEN_RE.findall(str(text or "")))


def _normalized_tokens(text: str) -> frozenset[str]:
    return frozenset(token.lower() for token in _TOKEN_RE.findall(str(text or "")) if len(token) >= 2)


def _semantic_overlap(left: CandidateTake, right: CandidateTake) -> float:
    left_tokens = _normalized_tokens(left.text)
    right_tokens = _normalized_tokens(right.text)
    if len(left_tokens) < 3 or len(right_tokens) < 3:
        return 0.0
    shared = len(left_tokens & right_tokens)
    return shared / max(1, min(len(left_tokens), len(right_tokens)))


def _later_semantic_retry_replacement(
    failed_take: CandidateTake,
    members: Tuple[CandidateTake, ...],
    decisions_by_id: dict[str, tuple[str, float]],
    *,
    minimum_label_confidence: float = 0.68,
    minimum_overlap: float = 0.50,
    maximum_delay_sec: float = 24.0,
) -> tuple[CandidateTake | None, float]:
    """Find a later complete retake of substantially the same spoken idea.

    This is not general semantic deletion. It only corroborates a Hybrid `failed`
    decision when the same bounded creator session contains a later complete delivery
    with strong lexical/semantic overlap. That closes the common case where the failed
    first take has no obvious visual fumble but the creator immediately records a clean
    replacement.
    """
    best: CandidateTake | None = None
    best_overlap = 0.0
    for candidate in members:
        if candidate.clip_id == failed_take.clip_id:
            continue
        if candidate.source_asset_id != failed_take.source_asset_id:
            continue
        if float(candidate.start) <= float(failed_take.end):
            continue
        if float(candidate.start) - float(failed_take.end) > maximum_delay_sec:
            continue
        if not bool(candidate.complete_idea):
            continue
        label, confidence = decisions_by_id.get(candidate.clip_id, ("", 0.0))
        if label not in {"winner", "alternate", "keep"} or confidence < minimum_label_confidence:
            continue
        overlap = _semantic_overlap(failed_take, candidate)
        if overlap >= minimum_overlap and overlap > best_overlap:
            best = candidate
            best_overlap = overlap
    return best, best_overlap


@dataclass(frozen=True)
class _ReplacementCandidateScan:
    """D-072: pure, side-effect-free re-scan of `members` for observability
    only -- mirrors _later_semantic_retry_replacement's own gate cascade
    verbatim (never duplicated with independent logic that could drift)
    but additionally reports the best-seen overlap and the count of
    structurally eligible candidates REGARDLESS of whether any candidate
    actually cleared `minimum_overlap` -- information the decision-making
    function itself has no reason to keep once it returns. Never consulted
    by any decision; complete_retry_identity_guard.py's own no-replacement
    diagnostics (D-072) are the only reader."""
    eligible_candidate_count: int
    best_candidate_clip_id: str | None
    best_overlap_seen: float


def _scan_replacement_candidates(
    failed_take: CandidateTake,
    members: Tuple[CandidateTake, ...],
    decisions_by_id: dict[str, tuple[str, float]],
    *,
    minimum_label_confidence: float = 0.68,
    maximum_delay_sec: float = 24.0,
) -> _ReplacementCandidateScan:
    eligible_count = 0
    best_overlap = 0.0
    best_clip_id: str | None = None
    for candidate in members:
        if candidate.clip_id == failed_take.clip_id:
            continue
        if candidate.source_asset_id != failed_take.source_asset_id:
            continue
        if float(candidate.start) <= float(failed_take.end):
            continue
        if float(candidate.start) - float(failed_take.end) > maximum_delay_sec:
            continue
        if not bool(candidate.complete_idea):
            continue
        label, confidence = decisions_by_id.get(candidate.clip_id, ("", 0.0))
        if label not in {"winner", "alternate", "keep"} or confidence < minimum_label_confidence:
            continue
        eligible_count += 1
        overlap = _semantic_overlap(failed_take, candidate)
        if overlap > best_overlap:
            best_overlap = overlap
            best_clip_id = candidate.clip_id
    return _ReplacementCandidateScan(eligible_count, best_clip_id, best_overlap)


def _source_events(context: WholeVideoContext | None, source_asset_id: str):
    if context is None:
        return ()
    for source in context.sources:
        if source.source_asset_id == source_asset_id:
            return tuple(source.events)
    return ()


def _performance_event_summary(take: CandidateTake, context: WholeVideoContext | None) -> dict[str, int | float | bool]:
    events = tuple(
        event for event in _source_events(context, take.source_asset_id)
        if event.end >= take.start - 0.20 and event.start <= take.end + 0.20
    )
    resets = [event for event in events if str(event.kind) in _RESET_CANDIDATES and event.confidence >= 0.88]
    breaks = [event for event in events if str(event.kind) in _BREAK_CANDIDATES and event.confidence >= 0.76]
    return {
        "strong_reset_count": len(resets),
        "strong_break_count": len(breaks),
        "max_reset_confidence": round(max((float(event.confidence) for event in resets), default=0.0), 4),
        "max_break_confidence": round(max((float(event.confidence) for event in breaks), default=0.0), 4),
        "multimodal_reset": bool(resets and breaks),
    }


def _evidence(take: CandidateTake, context: WholeVideoContext | None) -> tuple[tuple[str, float | str | bool | int], ...]:
    performance = _performance_event_summary(take, context)
    signals = take.signals
    base: list[tuple[str, float | str | bool | int]] = [
        ("complete_idea", bool(take.complete_idea)),
        *(performance.items()),
    ]
    if signals is None:
        return tuple(base)
    base.extend((
        ("audio_quality", round(float(signals.audio_quality), 4)),
        ("eye_contact", round(float(signals.eye_contact), 4)),
        ("visual_fumble", round(float(signals.visual_fumble), 4)),
        ("expression_naturalness", round(float(signals.expression_naturalness), 4)),
        ("gesture_naturalness", round(float(signals.gesture_naturalness), 4)),
        ("delivery_energy", round(float(signals.delivery_energy), 4)),
        ("distraction_risk", round(float(signals.distraction_risk), 4)),
    ))
    return tuple(base)


def _source_context(
    context: WholeVideoContext | None,
    source_asset_id: str,
) -> tuple[tuple[str, str | float], ...]:
    if context is None:
        return ()
    for source in context.sources:
        if source.source_asset_id != source_asset_id:
            continue
        summary = " ".join(str(source.summary or "").split())[:3600]
        return (
            ("summary", summary),
            ("creator_intent", str(source.creator_intent or "")[:500]),
            ("main_topic", str(source.main_topic or "")[:500]),
            ("product_or_subject", str(source.product_or_subject or "")[:500]),
            ("story_logic", str(source.story_logic or "")[:900]),
            ("edit_mode", str(source.edit_mode or "natural")),
            ("sales_intent", round(float(source.sales_intent), 4)),
        )
    return ()


def _failed_local_evidence(
    take: CandidateTake,
    context: WholeVideoContext | None,
) -> tuple[bool, tuple[str, ...]]:
    """Independent Watch+Listen evidence that a take belongs to recording failure."""
    reasons: list[str] = []
    for event in harmful_events_for_take(take, context, minimum_confidence=0.80):
        reasons.append(f"event:{event.kind}:{event.confidence:.2f}")

    performance = _performance_event_summary(take, context)
    reset_count = int(performance["strong_reset_count"])
    break_count = int(performance["strong_break_count"])
    if reset_count >= 2 and break_count >= 1:
        reasons.append(f"multimodal_reset_cluster:{reset_count}:{break_count}")
    elif reset_count >= 4:
        reasons.append(f"dense_physical_reset:{reset_count}")

    signals = take.signals
    if signals is not None:
        if float(signals.visual_fumble) >= 0.68:
            reasons.append(f"visual_fumble:{float(signals.visual_fumble):.2f}")
        if float(signals.distraction_risk) >= 0.78:
            reasons.append(f"distraction_risk:{float(signals.distraction_risk):.2f}")
        if float(signals.expression_naturalness) <= 0.32:
            reasons.append(f"expression_naturalness:{float(signals.expression_naturalness):.2f}")
        if float(signals.gesture_naturalness) <= 0.32:
            reasons.append(f"gesture_naturalness:{float(signals.gesture_naturalness):.2f}")

    return bool(reasons), tuple(reasons)


def _overlapping_windows(
    items: Tuple[CandidateTake, ...],
    *,
    size: int,
    stride: int,
) -> tuple[Tuple[CandidateTake, ...], ...]:
    if size <= 0 or stride <= 0:
        raise ValueError("hybrid session window size/stride must be positive")
    if len(items) <= size:
        return (items,) if items else ()
    starts = list(range(0, max(1, len(items) - size + 1), stride))
    final_start = len(items) - size
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    return tuple(tuple(items[start : start + size]) for start in starts)


def _editorial_session(
    members: Tuple[CandidateTake, ...],
    context: WholeVideoContext | None,
    *,
    partition_index: int,
    chunk_index: int,
) -> EditorialSession:
    source_id = members[0].source_asset_id
    member_key = "|".join(member.clip_id for member in members)
    session_id = "hc_" + hashlib.sha256(
        f"{source_id}|{partition_index}|{chunk_index}|{member_key}".encode()
    ).hexdigest()[:18]
    return EditorialSession(
        session_id=session_id,
        source_asset_id=source_id,
        candidates=tuple(EditorialCandidate(
            clip_id=member.clip_id,
            text=member.text,
            start=member.start,
            end=member.end,
            local_label="keep",
            local_confidence=0.50,
            evidence=_evidence(member, context),
        ) for member in members),
        local_confidence=0.50,
        conflict_score=0.50,
        task="classify_recording_process_within_single_creator_session",
        source_context=_source_context(context, source_id),
    )


def _decision_priority(label: str, confidence: float) -> tuple[int, float]:
    order = {"failed": 5, "bts": 5, "winner": 4, "alternate": 3, "keep": 2, "uncertain": 1}
    return order.get(str(label), 0), float(confidence)


def apply_hybrid_session_cleanup(
    takes: Iterable[CandidateTake],
    context: WholeVideoContext | None,
    editorial_judge: EditorialJudge | None,
    *,
    policy: HybridGatePolicy = HybridGatePolicy(),
    delete_confidence: float = 0.94,
    corroborated_failed_confidence: float = 0.82,
    corroborated_bts_confidence: float = 0.84,
    micro_failed_confidence: float = 0.80,
    clustered_bts_confidence: float = 0.84,
    retry_replaced_failed_confidence: float = 0.84,
    chunk_size: int = 10,
    chunk_stride: int = 5,
    env: Mapping[str, str] | None = None,
) -> HybridSessionCleanupResult:
    """Classify overlapping creator-session windows while failing open on uncertainty."""
    take_tuple = tuple(takes)
    if not take_tuple or editorial_judge is None:
        _LAST_SEMANTIC_COMPUTE_PLAN.set(None)
        return HybridSessionCleanupResult(take_tuple, (), 0, 0, (), ())

    take_map = {take.clip_id: take for take in take_tuple}
    partitions = partition_takes_by_sessions(take_tuple, context)
    if not partitions:
        partitions = (take_tuple,)

    deleted_ids: set[str] = set()
    requested_chunks = 0
    available_chunks = 0
    diagnostics = []
    best_semantic: dict[str, tuple[str, float]] = {}

    effective_size = min(chunk_size, policy.max_candidates_per_request)
    effective_stride = min(max(1, chunk_stride), effective_size)

    # D-052 Part B: build every window across every partition FIRST -- the
    # planner (when enabled) needs the complete set of eligible work before
    # it can decide an execution order, exactly per Section 6 ("Introduce a
    # provider-neutral SemanticComputePlan generated BEFORE paid hybrid
    # calls begin").
    all_windows: list[tuple[int, int, Tuple[CandidateTake, ...], EditorialSession]] = []
    for partition_index, partition in enumerate(partitions):
        ordered = tuple(sorted(partition, key=lambda item: (item.start, item.end, item.clip_id)))
        windows = _overlapping_windows(ordered, size=effective_size, stride=effective_stride)
        for chunk_index, members in enumerate(windows):
            session = _editorial_session(
                members,
                context,
                partition_index=partition_index,
                chunk_index=chunk_index,
            )
            all_windows.append((partition_index, chunk_index, members, session))

    execution_order = list(range(len(all_windows)))
    plan: SemanticComputePlan | None = None
    if _semantic_compute_planner_enabled(env) and all_windows:
        from .hybrid_provider_settings import load_hybrid_provider_settings

        cost_ceiling_usd = load_hybrid_provider_settings(dict(env) if env is not None else None).max_cost_per_edit_usd
        work_items = tuple(
            SemanticWorkItem(
                work_id=str(position),
                priority=_classify_window_priority(members, context),
                estimated_cost_usd=_estimate_window_cost_usd(members),
                reason=f"partition={partition_index} chunk={chunk_index} size={len(members)}",
            )
            for position, (partition_index, chunk_index, members, _session) in enumerate(all_windows)
        )
        plan = build_semantic_compute_plan(work_items, cost_ceiling_usd=cost_ceiling_usd)
        # Every window is still attempted (the transport's own DollarBudgetLedger
        # remains the sole authoritative enforcement -- see module docstring);
        # only the ORDER changes, from plain enumeration to planned/priority
        # order. This is what guarantees a P0 window is never starved merely
        # because it happened to be requested fifth.
        execution_order = [int(work_id) for work_id in plan.planned_calls] + [
            int(work_id) for work_id in plan.deferred_optional_calls
        ]

    for position in execution_order:
        partition_index, chunk_index, members, session = all_windows[position]
        # D-052: the block below is unchanged from before the planner existed
        # (deliberately kept at its original nesting depth to minimize diff
        # risk on this already load-bearing scoring logic) -- only WHICH
        # window is visited on each loop iteration changed, via
        # `execution_order` above.
        if True:
            result = safe_editorial_judge(editorial_judge, session, policy)
            if result.requested:
                requested_chunks += 1
            if result.available:
                available_chunks += 1

            chunk_deleted = []
            decisions = []
            local_by_id: dict[str, tuple[bool, tuple[str, ...]]] = {}
            if result.available:
                decisions_by_id = {
                    decision.clip_id: (str(decision.label), float(decision.confidence))
                    for decision in result.decisions
                }
                for decision in result.decisions:
                    candidate = (decision.label, float(decision.confidence))
                    current = best_semantic.get(decision.clip_id)
                    if current is None or _decision_priority(*candidate) > _decision_priority(*current):
                        best_semantic[decision.clip_id] = candidate
                    local_by_id[decision.clip_id] = _failed_local_evidence(take_map[decision.clip_id], context)

                harmful = [
                    decision for decision in result.decisions
                    if decision.label in {"failed", "bts"} and decision.confidence >= 0.82
                ]
                corroborated_harmful = [
                    decision for decision in harmful
                    if local_by_id.get(decision.clip_id, (False, ()))[0]
                ]
                dense_semantic_failure_cluster = len(harmful) >= 3 and len(corroborated_harmful) >= 2

                for decision in result.decisions:
                    take = take_map[decision.clip_id]
                    corroborated, local_reasons = local_by_id[decision.clip_id]
                    replacement, replacement_overlap = _later_semantic_retry_replacement(
                        take, members, decisions_by_id
                    ) if decision.label == "failed" else (None, 0.0)
                    # D-072: read-and-clear the guard's own observability
                    # side channel (set only when the call above actually
                    # ran, i.e. decision.label == "failed" -- consuming it
                    # in the same branch prevents a stale value from a
                    # PRIOR decision ever leaking onto this one). Additive
                    # only: nothing below reads these fields to make a
                    # decision.
                    guard_diagnostic = (
                        _consume_replacement_guard_diagnostic()
                        if decision.label == "failed" else None
                    )
                    retry_replaced_failed_delete = bool(
                        decision.label == "failed"
                        and decision.confidence >= retry_replaced_failed_confidence
                        and replacement is not None
                    )
                    hard_semantic_delete = bool(
                        decision.label in {"failed", "bts"}
                        and decision.confidence >= delete_confidence
                    )
                    corroborated_failed_delete = bool(
                        decision.label == "failed"
                        and decision.confidence >= corroborated_failed_confidence
                        and corroborated
                    )
                    corroborated_bts_delete = bool(
                        decision.label == "bts"
                        and decision.confidence >= corroborated_bts_confidence
                        and corroborated
                    )
                    micro_failed_delete = bool(
                        decision.label == "failed"
                        and decision.confidence >= micro_failed_confidence
                        and corroborated
                        and take.duration_sec <= 1.25
                        and _token_count(take.text) <= 2
                    )
                    clustered_bts_delete = bool(
                        decision.label == "bts"
                        and decision.confidence >= clustered_bts_confidence
                        and dense_semantic_failure_cluster
                    )
                    applied_delete = (
                        hard_semantic_delete
                        or retry_replaced_failed_delete
                        or corroborated_failed_delete
                        or corroborated_bts_delete
                        or micro_failed_delete
                        or clustered_bts_delete
                    )
                    if hard_semantic_delete:
                        delete_basis = "high_confidence_semantic"
                    elif retry_replaced_failed_delete:
                        delete_basis = "semantic_failed_plus_later_overlapping_complete_retake"
                    elif micro_failed_delete:
                        delete_basis = "micro_failed_plus_local_performance"
                    elif corroborated_failed_delete:
                        delete_basis = "semantic_failed_plus_local_performance"
                    elif corroborated_bts_delete:
                        delete_basis = "semantic_bts_plus_local_performance"
                    elif clustered_bts_delete:
                        delete_basis = "semantic_bts_inside_corroborated_failure_cluster"
                    else:
                        delete_basis = "kept_fail_open"
                    if applied_delete:
                        deleted_ids.add(decision.clip_id)
                        chunk_deleted.append(decision.clip_id)
                    decisions.append({
                        "clip_id": decision.clip_id,
                        "label": decision.label,
                        "confidence": decision.confidence,
                        "reason_code": decision.reason_code,
                        "local_failure_corroborated": corroborated,
                        "local_failure_reasons": list(local_reasons),
                        "later_retry_replacement_id": replacement.clip_id if replacement is not None else None,
                        "later_retry_semantic_overlap": round(float(replacement_overlap), 4),
                        "dense_semantic_failure_cluster": dense_semantic_failure_cluster,
                        "delete_basis": delete_basis,
                        "applied_delete": applied_delete,
                        # D-072: additive observability only -- explains
                        # WHY later_retry_replacement_id is null even when
                        # later_retry_semantic_overlap is nonzero (D-070's
                        # finding). Never read by any decision below this
                        # point or anywhere else in the pipeline (D-072
                        # Section 5 requires and verifies this).
                        "replacement_candidate_clip_id_before_guard": (
                            guard_diagnostic.replacement_candidate_clip_id_before_guard
                            if guard_diagnostic is not None else None
                        ),
                        "sequence_identity": (
                            guard_diagnostic.sequence_identity if guard_diagnostic is not None else None
                        ),
                        "sequence_identity_threshold": (
                            guard_diagnostic.sequence_identity_threshold
                            if guard_diagnostic is not None else None
                        ),
                        "lexical_identity_passed": (
                            guard_diagnostic.lexical_identity_passed if guard_diagnostic is not None else None
                        ),
                        "replacement_rejection_reason": (
                            guard_diagnostic.replacement_rejection_reason
                            if guard_diagnostic is not None else _REPLACEMENT_REASON_NOT_APPLICABLE
                        ),
                    })

            plan_outcome = plan.outcome_for(str(position)) if plan is not None else None
            diagnostics.append({
                "partition_index": partition_index,
                "chunk_index": chunk_index,
                "session_id": session.session_id,
                "member_ids": [member.clip_id for member in members],
                "source_context_available": bool(session.source_context),
                "window_size": len(members),
                "window_stride": effective_stride,
                "requested": bool(result.requested),
                "available": bool(result.available),
                "provider": result.provider,
                "model": result.model,
                "deleted_ids": chunk_deleted,
                "decisions": decisions,
                # D-052 Part B: present only when CUTSELL_SEMANTIC_COMPUTE_PLANNER
                # is enabled -- the plan's own priority classification and
                # predicted planned/deferred status for this exact window,
                # independent of whether the transport's own ledger actually
                # accepted or rejected the call above.
                "planner_execution_rank": execution_order.index(position) if plan is not None else None,
                "planner_priority": plan_outcome.priority.name if plan_outcome is not None else None,
                "planner_predicted_planned": plan_outcome.planned if plan_outcome is not None else None,
            })

    kept = tuple(take for take in take_tuple if take.clip_id not in deleted_ids)
    deleted = tuple(take for take in take_tuple if take.clip_id in deleted_ids)
    semantic_decisions = tuple(
        (clip_id, label, confidence)
        for clip_id, (label, confidence) in best_semantic.items()
    )
    # D-053 Section 11: set the ContextVar side-channel unconditionally
    # (None when the flag is off, exactly like the field itself) so
    # composite_resolver.apply_composite_resolution can recover this exact
    # plan even if a downstream monkeypatch hook in the chain reconstructs
    # HybridSessionCleanupResult without naming this field -- see this
    # module's own docstring on _LAST_SEMANTIC_COMPUTE_PLAN above.
    _LAST_SEMANTIC_COMPUTE_PLAN.set(plan)
    return HybridSessionCleanupResult(
        kept=kept,
        deleted=deleted,
        requested_chunk_count=requested_chunks,
        available_chunk_count=available_chunks,
        diagnostics=tuple(diagnostics),
        semantic_decisions=semantic_decisions,
        semantic_compute_plan=plan,
    )
