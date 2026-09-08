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
    never a second, divergent computation of the same thing."""
    window_rows = tuple(window_rows) if window_rows else ()
    return {
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


def summarize_family_authority_observability(per_family_rows: Iterable[Mapping]) -> dict:
    """Tail-safe, counts-only CI summary (same pattern as D-119/D-125's
    compact qualification summaries) -- never the full per-family detail,
    which stays in the full diagnostics artifact."""
    counts: Counter = Counter()
    conflict_family_count = 0
    omitted_occurrences = 0
    authority_counts: Counter = Counter()
    for row in per_family_rows:
        if not isinstance(row, Mapping):
            continue
        counts[str(row.get("family_complete_context") or "unknown")] += 1
        authority_counts[str(row.get("provider_authority_applied") or "")] += 1
        if row.get("partial_window_conflict"):
            conflict_family_count += 1
        for omitted in (row.get("omitted_candidate_ids") or {}).values():
            omitted_occurrences += len(omitted)
    return {
        "family_count": sum(counts.values()),
        "family_complete_context_counts": dict(counts),
        "provider_authority_applied_counts": dict(authority_counts),
        "partial_window_conflict_family_count": conflict_family_count,
        "omitted_candidate_occurrences": omitted_occurrences,
    }
