"""D-146: FAMILY-COMPLETE SEMANTIC AUTHORITY GATE -- PHASE A (OBSERVABILITY ONLY).

Per docs/CUTSELL_DECISIONS.md D-145 (design) and D-146 (this module): makes the
structural gap D-144 root-caused (SEMANTIC_PROVIDER_VARIANCE / FAMILY_FORMATION_
VARIANCE) VISIBLE before any authority change is attempted. This module adds
zero new authority, zero new deletion/selection/winner logic, and calls no
provider. Every function here is a pure projection of data ALREADY computed by
`hybrid_session_cleanup.apply_hybrid_session_cleanup` (its per-window
`diagnostics` rows -- `member_ids`, `decisions`, `provider`, `model`,
`session_id`) and `pipeline.py::family_scoped_semantic_decisions` (its
`source_info` return value). Nothing here is read by any decision -- it is
wired into `judge_group_diagnostics` (pipeline.py) and the window diagnostics
rows (hybrid_session_cleanup.py) as ADDITIVE dict keys only.

## Why this module exists standalone (not inline in pipeline.py)

`family_scoped_semantic_decisions` (D-094.3 F8) already computes a narrower,
CIRCULAR-SAFE version of "is there a family-complete window" as a means to an
end (which labels to trust). Its `source_info` return is `None` whenever no
complete window exists -- collapsing two very different honest answers
("we have window evidence for this family, but none of it is complete" vs
"we have NO window evidence for this family at all") into the same falsy
value. D-146 requires a THREE-WAY answer (true/false/unknown) reported
honestly, so this module reimplements the completeness *test* itself (same
`family_set <= set(row["member_ids"])` rule, verified byte-identical) as a
side, observation-only detector -- never replacing or being consulted by
`family_scoped_semantic_decisions`, which remains the sole decision authority
unchanged.

## Today's actual behavior (D-146 must describe it honestly)

Per D-146's own mandatory pipeline-ordering inspection (docs/CUTSELL_DECISIONS.md
D-146): `apply_composite_resolution` (which runs `hybrid_session_cleanup
.apply_hybrid_session_cleanup`'s sliding per-window editorial judge) executes
on `kept` takes BEFORE `safe_group_takes_by_sessions` assigns retry-family
membership (pipeline.py, `apply_composite_resolution` at line ~833 precedes
`safe_group_takes_by_sessions` at line ~882). The sliding windows themselves
(`_overlapping_windows`, chunk_size=10/chunk_stride=5) are built over the
GLOBAL kept-take pool, oblivious to family boundaries that do not exist yet.
This means: a per-window "winner"/"alternate" label is NEVER minted with
knowledge of its own family-completeness -- `family_complete_context` can only
be answered RETROACTIVELY, after grouping, by checking whether any of the
(already-fixed) windows' recorded `member_ids` happen to be a superset of the
now-known family. That is exactly what this module's `family_complete_context`
does, and exactly what `family_scoped_semantic_decisions` already
opportunistically exploits for its own labeling preference. Today's authority
is therefore NEVER gated on family completeness (D-145's future Phase B
contract is not implemented here) -- it is silently better when a
family-complete window happens to exist, and falls back to the historically
demonstrated-unsafe global cross-window max-priority merge
(`_decision_priority`) otherwise. `provider_authority_applied` below reports
this honestly as one of exactly two values.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Iterable, Mapping, Sequence

PROVIDER_TEMPERATURE_UNKNOWN = "UNKNOWN"
PROVIDER_PROMPT_VERSION_UNKNOWN = "UNKNOWN"

AUTHORITY_FAMILY_COMPLETE_WINDOW_PREFERRED = "FAMILY_COMPLETE_WINDOW_PREFERRED"
AUTHORITY_GLOBAL_CROSS_WINDOW_MERGE = "GLOBAL_CROSS_WINDOW_MERGE"


def _window_key(row: Mapping) -> str:
    session_id = row.get("session_id")
    if session_id:
        return str(session_id)
    return f"{row.get('partition_index')}:{row.get('chunk_index')}"


def _rows_touching_family(family_set: set, window_rows: Iterable[Mapping] | None) -> list[Mapping]:
    if not window_rows:
        return []
    touched = []
    for row in window_rows:
        if not isinstance(row, Mapping):
            continue
        member_ids = set(row.get("member_ids") or ())
        if family_set & member_ids:
            touched.append(row)
    return touched


def family_complete_context(
    family_member_ids: Sequence[str],
    window_rows: Iterable[Mapping] | None,
) -> str:
    """Three-way, non-circular completeness detector.

    "true"    -- at least one recorded window's member_ids is a SUPERSET of
                 the whole family (the same test family_scoped_semantic_
                 decisions already uses to prefer that window's labels).
    "false"   -- at least one window touches this family, but none is a
                 complete superset (the D-094.3 F8 unsafe shape).
    "unknown" -- no window evidence touches this family at all (window_rows
                 is empty/None), or the family has fewer than 2 members (the
                 completeness question does not apply to a non-contest).

    Never reads a decision, never influences one -- observation only.
    """
    family_set = set(str(m) for m in family_member_ids)
    if len(family_set) < 2:
        return "unknown"
    touched = _rows_touching_family(family_set, window_rows)
    if not touched:
        return "unknown"
    complete = [row for row in touched if family_set <= set(row.get("member_ids") or ())]
    return "true" if complete else "false"


def complete_window_ids(
    family_member_ids: Sequence[str],
    window_rows: Iterable[Mapping] | None,
) -> tuple[str, ...]:
    """Stable window ids (session_id, falling back to partition:chunk) of every
    recorded window that is a family-complete superset for this family."""
    family_set = set(str(m) for m in family_member_ids)
    if len(family_set) < 2 or not window_rows:
        return ()
    return tuple(
        _window_key(row) for row in window_rows
        if isinstance(row, Mapping) and family_set <= set(row.get("member_ids") or ())
    )


def window_ids_touching_family(
    family_member_ids: Sequence[str],
    window_rows: Iterable[Mapping] | None,
) -> tuple[str, ...]:
    """Stable window ids of every recorded window that saw at least one member
    of this family (complete or partial)."""
    family_set = set(str(m) for m in family_member_ids)
    return tuple(_window_key(row) for row in _rows_touching_family(family_set, window_rows))


def omitted_candidate_ids(
    family_member_ids: Sequence[str],
    window_rows: Iterable[Mapping] | None,
) -> dict[str, tuple[str, ...]]:
    """Per partial window (one that saw SOME but not ALL family members), the
    family member ids that window never saw -- keyed by the window's stable
    id. Windows that saw none of the family, or all of it, are absent."""
    family_set = set(str(m) for m in family_member_ids)
    result: dict[str, tuple[str, ...]] = {}
    for row in _rows_touching_family(family_set, window_rows):
        member_ids = set(row.get("member_ids") or ())
        omitted = family_set - member_ids
        if omitted:
            result[_window_key(row)] = tuple(sorted(omitted))
    return result


def partial_window_conflict(
    family_member_ids: Sequence[str],
    window_rows: Iterable[Mapping] | None,
) -> dict | None:
    """Detects the exact D-094.3 F8 shape: two DIFFERENT family members each
    labelled "winner" by two DIFFERENT windows, with NO family-complete window
    present to arbitrate between them. Returns None whenever a family-complete
    window exists (the family-scoped labels are trusted, so a partial window's
    disagreement is moot by construction) or when fewer than two distinct
    "winner" labels were ever recorded for this family. Never used to change
    a decision -- reporting the same shape family_scoped_semantic_decisions
    already silently tolerates when no complete window exists."""
    family_set = set(str(m) for m in family_member_ids)
    if len(family_set) < 2 or not window_rows:
        return None
    touched = _rows_touching_family(family_set, window_rows)
    if not touched:
        return None
    if any(family_set <= set(row.get("member_ids") or ()) for row in touched):
        return None
    winners_by_window: dict[str, set] = {}
    for row in touched:
        for decision in row.get("decisions") or ():
            if not isinstance(decision, Mapping):
                continue
            clip_id = decision.get("clip_id")
            if clip_id in family_set and str(decision.get("label") or "") == "winner":
                winners_by_window.setdefault(_window_key(row), set()).add(str(clip_id))
    distinct_winner_ids: set = set()
    for ids in winners_by_window.values():
        distinct_winner_ids |= ids
    if len(distinct_winner_ids) < 2:
        return None
    return {
        "conflicting_winner_ids": tuple(sorted(distinct_winner_ids)),
        "winner_window_ids": {key: tuple(sorted(ids)) for key, ids in winners_by_window.items()},
    }


def provider_config_from_window_rows(
    family_member_ids: Sequence[str],
    window_rows: Iterable[Mapping] | None,
) -> dict:
    """Provider/model actually recorded on windows touching this family, plus
    temperature/prompt_version -- reported as the literal string "UNKNOWN"
    because neither field exists anywhere in the current EditorialJudgeResult/
    EditorialJudge contract (hybrid_editorial.py). Never invented, never
    defaulted to a guessed value."""
    family_set = set(str(m) for m in family_member_ids)
    seen: list[tuple[str, str]] = []
    for row in _rows_touching_family(family_set, window_rows):
        pair = (str(row.get("provider") or ""), str(row.get("model") or ""))
        if pair not in seen:
            seen.append(pair)
    return {
        "providers_observed": tuple(seen),
        "temperature": PROVIDER_TEMPERATURE_UNKNOWN,
        "prompt_version": PROVIDER_PROMPT_VERSION_UNKNOWN,
    }


def stable_request_hash(
    member_ids: Iterable[str],
    texts_by_id: Mapping[str, str],
    starts_by_id: Mapping[str, float],
    ends_by_id: Mapping[str, float],
    model: str,
    prompt_version: str | None = None,
) -> str:
    """A stable, content-sensitive, candidate-order-insensitive identity for
    one semantic window request. Two windows with the SAME candidate set
    (same clip_id/start/end/text) built in a different iteration order hash
    identically; any change to a candidate's text/start/end, or a change in
    which model answered, changes the hash. prompt_version is never invented
    -- when the caller does not have one, it is recorded as the literal
    string "UNKNOWN" inside the hashed payload itself (never silently
    substituted with a fixed constant that would make two runs with a real,
    but different, prompt version collide)."""
    rows = sorted(
        (
            str(clip_id),
            round(float(starts_by_id.get(clip_id, 0.0)), 3),
            round(float(ends_by_id.get(clip_id, 0.0)), 3),
            str(texts_by_id.get(clip_id, "")),
        )
        for clip_id in member_ids
    )
    payload = {
        "candidates": rows,
        "model": str(model or "unknown"),
        "prompt_version": str(prompt_version) if prompt_version else PROVIDER_PROMPT_VERSION_UNKNOWN,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "rh_" + hashlib.sha256(blob.encode()).hexdigest()[:24]


AGREEMENT_NO_COMPLETE_WINDOW = "NO_COMPLETE_WINDOW"
AGREEMENT_ONE_COMPLETE_WINDOW = "ONE_COMPLETE_WINDOW"
AGREEMENT_MULTIPLE_AGREE = "MULTIPLE_COMPLETE_WINDOWS_AGREE"
AGREEMENT_MULTIPLE_DISAGREE = "MULTIPLE_COMPLETE_WINDOWS_DISAGREE"
AGREEMENT_UNKNOWN = "UNKNOWN"

CONFLICT_REASON_COMPLETE_WINDOWS_DISAGREE = "multiple_family_complete_windows_disagree_on_comparative_winner"


def _normalized_window_outcome(family_set: set, row: Mapping) -> dict:
    """D-149: a deterministic, STRUCTURED-FIELDS-ONLY comparison
    representation for one family-complete window -- never raw prose. Built
    only from fields `hybrid_session_cleanup.py`'s diagnostics rows already
    carry (`decisions[].clip_id/label/confidence`, `request_hash`,
    `session_id`). `raw_relation` is reported as `None` today because no
    component anywhere in this codebase records a typed relation
    (retry_of/corrects/complements/etc, D-098 Section 13.6) on a window row
    -- never invented."""
    winner_ids: set = set()
    alternate_ids: set = set()
    per_member: dict = {}
    for decision in row.get("decisions") or ():
        if not isinstance(decision, Mapping):
            continue
        clip_id = decision.get("clip_id")
        if clip_id not in family_set:
            continue
        label = str(decision.get("label") or "")
        per_member[str(clip_id)] = {
            "label": label,
            "confidence": decision.get("confidence"),
        }
        if label == "winner":
            winner_ids.add(str(clip_id))
        elif label == "alternate":
            alternate_ids.add(str(clip_id))
    return {
        "window_id": _window_key(row),
        "request_hash": row.get("request_hash"),
        "member_ids": tuple(sorted(family_set)),
        "provider_outcome_by_member": per_member,
        "normalized_winner_ids": tuple(sorted(winner_ids)),
        "normalized_alternate_ids": tuple(sorted(alternate_ids)),
        "raw_relation": row.get("raw_relation"),
    }


def complete_window_outcomes(
    family_member_ids: Sequence[str],
    window_rows: Iterable[Mapping] | None,
) -> tuple[dict, ...]:
    """The normalized outcome (13.3.1-shaped, structured-fields-only) of
    every window that is a family-complete superset for this family --
    reuses the EXACT same completeness test as `complete_window_ids`
    (D-146's retroactive, post-grouping semantics: evaluated only after
    final family membership is known, never lets a window define the
    family and then certify itself complete)."""
    family_set = set(str(m) for m in family_member_ids)
    if len(family_set) < 2 or not window_rows:
        return ()
    complete_rows = [
        row for row in window_rows
        if isinstance(row, Mapping) and family_set <= set(row.get("member_ids") or ())
    ]
    return tuple(_normalized_window_outcome(family_set, row) for row in complete_rows)


def _outcome_signature(outcome: Mapping) -> tuple:
    """The comparison key two complete windows must match on to AGREE:
    the same normalized winner set AND the same normalized alternate set.
    Never compares raw prose/confidence -- structured fields only."""
    return (outcome["normalized_winner_ids"], outcome["normalized_alternate_ids"])


def complete_window_agreement(
    family_member_ids: Sequence[str],
    window_rows: Iterable[Mapping] | None,
) -> dict:
    """D-149 (Phase A.2): classifies how the family's family-complete
    windows (if any) relate to each other, and DETECTS (never enforces)
    the new named state `COMPLETE_CONTEXT_CONFLICT` docs/CUTSELL_
    CANONICAL_ENGINE_ARCHITECTURE_D098.md Section 13.8.1 names: all
    relevant competitors were present in multiple independent requests,
    but those requests disagree. `family_complete_context=true` is
    necessary but NOT sufficient for a trustworthy single winner -- this
    is the second check D-147's real-media finding proved necessary.

    Status values: `NO_COMPLETE_WINDOW` / `ONE_COMPLETE_WINDOW` /
    `MULTIPLE_COMPLETE_WINDOWS_AGREE` / `MULTIPLE_COMPLETE_WINDOWS_
    DISAGREE` / `UNKNOWN`. Only the DISAGREE status sets
    `complete_context_conflict=True` -- a single window's own internal
    ambiguity (e.g. it alone labels two members both "winner") is not,
    by itself, a cross-window disagreement; with >=2 complete windows, an
    ambiguous window's outcome simply fails to match any other window's
    signature and the family falls into DISAGREE via the same equality
    check, never a separate rule. Never merges into a winner: this
    function only classifies and reports, it changes nothing."""
    completeness = family_complete_context(family_member_ids, window_rows)
    outcomes = complete_window_outcomes(family_member_ids, window_rows)
    if completeness == "unknown":
        status = AGREEMENT_UNKNOWN
    elif completeness == "false":
        status = AGREEMENT_NO_COMPLETE_WINDOW
    elif len(outcomes) <= 1:
        status = AGREEMENT_ONE_COMPLETE_WINDOW
    else:
        signatures = {_outcome_signature(outcome) for outcome in outcomes}
        status = AGREEMENT_MULTIPLE_AGREE if len(signatures) == 1 else AGREEMENT_MULTIPLE_DISAGREE

    conflict = status == AGREEMENT_MULTIPLE_DISAGREE
    if conflict:
        conflict_window_ids = tuple(outcome["window_id"] for outcome in outcomes)
        conflict_winner_sets = tuple(sorted({outcome["normalized_winner_ids"] for outcome in outcomes}))
        conflict_reason = CONFLICT_REASON_COMPLETE_WINDOWS_DISAGREE
    else:
        conflict_window_ids = ()
        conflict_winner_sets = ()
        conflict_reason = None

    sig_counts: Counter = Counter(_outcome_signature(outcome) for outcome in outcomes)
    majority_count = max(sig_counts.values()) if sig_counts else 0

    return {
        "complete_window_count": len(outcomes),
        "complete_window_ids": tuple(outcome["window_id"] for outcome in outcomes),
        "complete_window_request_hashes": tuple(outcome["request_hash"] for outcome in outcomes),
        "complete_window_outcomes": outcomes,
        "complete_window_winner_sets": tuple(outcome["normalized_winner_ids"] for outcome in outcomes),
        "complete_window_agreement_status": status,
        "complete_context_conflict": conflict,
        "complete_context_conflict_window_ids": conflict_window_ids,
        "complete_context_conflict_winner_sets": conflict_winner_sets,
        "complete_context_conflict_reason": conflict_reason,
        # D-148 Section 13.9: provider CONSISTENCY itself is evidence. These
        # are plain counts -- never a score or a confidence threshold.
        "complete_window_consistency_count": majority_count,
        "complete_window_disagreement_count": len(outcomes) - majority_count,
    }


def family_authority_diagnostics(
    family_member_ids: Sequence[str],
    window_rows: Iterable[Mapping] | None,
    family_scoped_source_info: Mapping | None,
) -> dict:
    """One composed, additive diagnostics dict for a single retry family.
    `family_scoped_source_info` is the SECOND return value of pipeline.py's
    `family_scoped_semantic_decisions` (already surfaced today under
    `judge_group_diagnostics[...]["semantic_label_source"]`) -- exposed here
    verbatim under its own key so this module's report is self-contained,
    never a second, divergent computation of the same thing.

    D-149 (Phase A.2) additive fields: `complete_window_agreement`'s own
    dict is merged in verbatim under its own keys (`complete_window_count`,
    `complete_window_agreement_status`, `complete_context_conflict`, etc.)
    -- every D-146 key above is unchanged; `complete_window_ids` here is
    the SAME field D-146 already returns (D-149 reuses it, never
    duplicates it under a second name)."""
    window_rows = tuple(window_rows) if window_rows else ()
    d146_fields = {
        "family_complete_context": family_complete_context(family_member_ids, window_rows),
        "complete_window_ids": complete_window_ids(family_member_ids, window_rows),
        "semantic_window_ids": window_ids_touching_family(family_member_ids, window_rows),
        "omitted_candidate_ids": omitted_candidate_ids(family_member_ids, window_rows),
        "partial_window_conflict": partial_window_conflict(family_member_ids, window_rows),
        "provider_authority_applied": (
            AUTHORITY_FAMILY_COMPLETE_WINDOW_PREFERRED if family_scoped_source_info is not None
            else AUTHORITY_GLOBAL_CROSS_WINDOW_MERGE
        ),
        "provider_config": provider_config_from_window_rows(family_member_ids, window_rows),
        "family_scoped_source_info": family_scoped_source_info,
    }
    return {**d146_fields, **complete_window_agreement(family_member_ids, window_rows)}


def summarize_family_authority_observability(per_family_rows: Iterable[Mapping]) -> dict:
    """Tail-safe, counts-only CI summary (same pattern as D-119/D-125's
    compact qualification summaries) -- never the full per-family detail
    (no clip ids), which stays in the full diagnostics artifact. D-149
    adds complete-window-agreement counts, D-150 adds semantic-authority-
    gate counts, alongside D-146's existing counts, all in the SAME single
    pass over `per_family_rows` (materialized once so a one-shot iterable
    -- e.g. a generator -- is never silently exhausted before every count
    reads it); every existing key is unchanged."""
    rows = [row for row in per_family_rows if isinstance(row, Mapping)]
    counts: Counter = Counter()
    conflict_family_count = 0
    omitted_occurrences = 0
    authority_counts: Counter = Counter()
    agreement_counts: Counter = Counter()
    gate_status_counts: Counter = Counter()
    complete_context_conflict_family_count = 0
    any_conflict_family_count = 0
    for row in rows:
        counts[str(row.get("family_complete_context") or "unknown")] += 1
        authority_counts[str(row.get("provider_authority_applied") or "")] += 1
        has_partial_conflict = bool(row.get("partial_window_conflict"))
        has_complete_conflict = bool(row.get("complete_context_conflict"))
        if has_partial_conflict:
            conflict_family_count += 1
        if has_complete_conflict:
            complete_context_conflict_family_count += 1
        if has_partial_conflict or has_complete_conflict:
            any_conflict_family_count += 1
        for omitted in (row.get("omitted_candidate_ids") or {}).values():
            omitted_occurrences += len(omitted)
        agreement_counts[str(row.get("complete_window_agreement_status") or AGREEMENT_UNKNOWN)] += 1
        if "semantic_authority_gate_status" in row:
            gate_status_counts[str(row.get("semantic_authority_gate_status") or "")] += 1
    return {
        "family_count": sum(counts.values()),
        "family_complete_context_counts": dict(counts),
        "provider_authority_applied_counts": dict(authority_counts),
        "partial_window_conflict_family_count": conflict_family_count,
        "omitted_candidate_occurrences": omitted_occurrences,
        # D-149 (Phase A.2) additive counts -- counts only, no clip ids.
        "families_with_no_complete_window": agreement_counts.get(AGREEMENT_NO_COMPLETE_WINDOW, 0),
        "families_with_one_complete_window": agreement_counts.get(AGREEMENT_ONE_COMPLETE_WINDOW, 0),
        "families_with_multiple_complete_windows": (
            agreement_counts.get(AGREEMENT_MULTIPLE_AGREE, 0) + agreement_counts.get(AGREEMENT_MULTIPLE_DISAGREE, 0)
        ),
        "families_with_complete_window_agreement": agreement_counts.get(AGREEMENT_MULTIPLE_AGREE, 0),
        "families_with_complete_context_conflict": complete_context_conflict_family_count,
        "families_with_partial_window_conflict": conflict_family_count,
        "families_with_any_semantic_conflict": any_conflict_family_count,
        # D-150 (Phase B) additive counts -- counts only, no clip ids.
        # Populated only for rows that actually ran the gate (i.e. carry
        # `semantic_authority_gate_status`); absent entirely from a
        # summary built purely from D-146/D-149 rows, same backward-
        # compatible pattern D-149 itself used for D-146's summary.
        "semantic_authority_allowed_count": gate_status_counts.get(AUTHORITY_ALLOWED, 0),
        "semantic_authority_abstain_incomplete_count": gate_status_counts.get(
            AUTHORITY_ABSTAIN_INCOMPLETE_CONTEXT, 0
        ),
        "semantic_authority_abstain_conflict_count": gate_status_counts.get(AUTHORITY_ABSTAIN_CONFLICT, 0),
        "semantic_authority_advisory_count": gate_status_counts.get(AUTHORITY_ADVISORY, 0),
    }


# ---------------------------------------------------------------------------
# D-150: SEMANTIC AUTHORITY PHASE B -- FAMILY-COMPLETE + COMPLETE-CONTEXT-
# CONFLICT AUTHORITY GATE.
#
# Per docs/CUTSELL_DECISIONS.md D-145/D-148/D-149/D-150. D-146/D-149 made
# family_complete_context and complete_context_conflict OBSERVABLE. This
# section is the smallest general gate that makes them AUTHORITATIVE over
# whether a provider-backed comparative "winner" label may become
# `_semantic_best_take`'s decisive `single_semantic_winner` fast-path
# answer (pipeline.py). It controls ONLY that one fast path -- it never
# touches deterministic cleanup, mechanical deletion, ASR, proposition
# evidence, behavior evidence, DeliveryScorer, the rest of BestTake's own
# ladder, Boundary, or Pacing. When the gate abstains, `_semantic_best_
# take` falls through to the SAME general ladder it already falls through
# to for zero/multiple "winner" labels -- no new fallback algorithm, no
# provider call, no confidence threshold, no "pick anyway".
#
# SINGLE SOURCE OF TRUTH: `resolve_semantic_comparative_authority` below
# consumes `family_complete_context`/`complete_context_conflict` exactly
# as already computed by `family_authority_diagnostics` (D-146/D-149) --
# it never recomputes completeness or conflict independently.
# ---------------------------------------------------------------------------

AUTHORITY_ALLOWED = "AUTHORITATIVE"
AUTHORITY_ADVISORY = "ADVISORY"
AUTHORITY_ABSTAIN_CONFLICT = "ABSTAIN_CONFLICT"
AUTHORITY_ABSTAIN_INCOMPLETE_CONTEXT = "ABSTAIN_INCOMPLETE_CONTEXT"

REASON_NOT_A_CONTEST = "not_a_contest_single_member_family"
REASON_NO_COMPLETE_WINDOW = "no_family_complete_window"
REASON_FAMILY_COMPLETE_NO_CONFLICT = "family_complete_context_true_no_conflict"


def resolve_semantic_comparative_authority(
    family_member_count: int,
    family_complete_context_value: str,
    complete_context_conflict: bool,
) -> tuple[str, str]:
    """The Phase B gate's entire decision, in one pure function.

    family_member_count < 2:      AUTHORITATIVE, "not_a_contest..." -- a
        singleton is not a retry contest at all (D-146's own family_
        complete_context already reports "unknown" for this case, which
        would otherwise wrongly read as "incomplete" here -- explicitly
        exempted so existing singleton behavior, including a legitimate
        global-cross-window-merge "winner" label on a non-contest member,
        is byte-for-byte unchanged. Never touches family topology.
    complete_context_conflict:    ABSTAIN_CONFLICT -- D-147/D-149's proven
        shape: multiple independently family-complete windows disagree.
        Checked BEFORE the completeness check because a conflicted
        "complete" answer is strictly worse than an absent one -- it must
        never be read as "close enough to trustworthy".
    family_complete_context != "true" (i.e. "false" or "unknown"):
        ABSTAIN_INCOMPLETE_CONTEXT -- D-145's original doctrine: no
        complete family context, no authoritative comparative winner.
    otherwise (family_complete_context == "true", conflict == False):
        AUTHORITATIVE -- the provider-backed comparative label may inform
        `_semantic_best_take`'s existing decisive fast path exactly as it
        does today; no new validity condition is added here (existing
        checks -- winner_confidence floor, `_single_winner_safety_veto`,
        D-123's CASE B gate -- already run downstream, unchanged).

    Never calls a provider. Never reads window rows itself -- the caller
    passes in fields `family_authority_diagnostics` already computed."""
    if family_member_count < 2:
        return AUTHORITY_ALLOWED, REASON_NOT_A_CONTEST
    if complete_context_conflict:
        return AUTHORITY_ABSTAIN_CONFLICT, CONFLICT_REASON_COMPLETE_WINDOWS_DISAGREE
    if str(family_complete_context_value) != "true":
        return AUTHORITY_ABSTAIN_INCOMPLETE_CONTEXT, REASON_NO_COMPLETE_WINDOW
    return AUTHORITY_ALLOWED, REASON_FAMILY_COMPLETE_NO_CONFLICT


def would_be_decisive_semantic_winner(
    member_ids: Iterable[str],
    semantic_decisions: Mapping[str, tuple],
    winner_confidence: float = 0.85,
) -> bool:
    """Mirrors (never replaces) `pipeline._semantic_best_take`'s own
    `winners` computation exactly (label == "winner" and confidence >=
    floor), for OBSERVABILITY only -- lets the gate report whether the
    raw, ungated labels alone would have been decisive ("before"), which
    the actual gated call may then veto ("after"). Duplicated on purpose:
    this function decides nothing -- `_semantic_best_take` still owns the
    real check inline, unchanged, so a future edit to its own floor logic
    cannot silently desync from a shared import cycle back into pipeline.py
    (this module never imports pipeline.py)."""
    winners = [
        cid for cid in member_ids
        if str(semantic_decisions.get(cid, ("", 0.0))[0]) == "winner"
        and float(semantic_decisions.get(cid, ("", 0.0))[1]) >= winner_confidence
    ]
    return len(winners) == 1


def semantic_authority_gate_diagnostics(
    member_ids: Sequence[str],
    semantic_decisions: Mapping[str, tuple],
    family_authority_row: Mapping,
    *,
    winner_confidence: float = 0.85,
) -> dict:
    """The compact, additive diagnostics block D-150 adds to `judge_group_
    diagnostics` (pipeline.py) -- reuses `family_authority_row`'s existing
    `family_complete_context`/`complete_context_conflict`/`complete_
    window_agreement_status` fields VERBATIM (no duplicate window payload,
    no recomputation) plus the gate's own decision and a before/after
    comparison of what the raw labels alone would have decided."""
    member_ids = tuple(member_ids)
    status, reason = resolve_semantic_comparative_authority(
        len(set(str(m) for m in member_ids)),
        str(family_authority_row.get("family_complete_context") or "unknown"),
        bool(family_authority_row.get("complete_context_conflict", False)),
    )
    before_decisive = would_be_decisive_semantic_winner(member_ids, semantic_decisions, winner_confidence)
    after_decisive = before_decisive and status == AUTHORITY_ALLOWED
    return {
        "semantic_authority_gate_evaluated": True,
        "semantic_authority_gate_status": status,
        "semantic_authority_gate_reason": reason,
        "semantic_authority_before": "DECISIVE" if before_decisive else "NON_DECISIVE",
        "semantic_authority_after": "DECISIVE" if after_decisive else "NON_DECISIVE",
        # Reused verbatim -- single source of truth, D-146/D-149's own
        # already-computed fields, never a second computation.
        "complete_context_conflict": family_authority_row.get("complete_context_conflict"),
        "family_complete_context": family_authority_row.get("family_complete_context"),
        "complete_window_agreement_status": family_authority_row.get("complete_window_agreement_status"),
    }
