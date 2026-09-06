"""Video00 quality-ladder comparison: RAW vs Cut.ai vs Human Gold vs CutSell (D-095).

QA / oracle tooling ONLY. Nothing here is imported by production Selection,
Boundary, BestTake, retry grouping, rendering or any LLM prompt. The two
reference edits (the Cut.ai commercial baseline and the Human Gold edit)
are evaluation oracles; this module only ever runs AFTER a CutSell result
exists, to compare it against them.

Canonical quality ladder (D-095):

    RAW -> CUT.AI PARITY -> HUMAN GOLD PARITY -> HUMAN WATCH + LISTEN PASS

The module partitions the RAW timeline into editorial regions using every
boundary any of the three edits (Cut.ai, Human Gold, CutSell) draws plus
every CutSell candidate boundary, records for each region who kept it, and
classifies every CutSell discrepancy:

    LEVEL_1  CutSell is worse than Cut.ai              -> fix first
    LEVEL_2  CutSell ~= Cut.ai but Human Gold is better -> after Level 1 is stable
    LEVEL_3  CutSell matches or exceeds Human Gold      -> protect, do not touch

For every LEVEL_1 region the module also names the CutSell authority most
likely responsible (AttemptReconstructor / recording-process removal,
IdeaClusterer, BestTakeResolver, RealizationResolver, CompositeResolver,
BoundaryEngine, Renderer) from the engine's own diagnostics, so a Selection
error is not "fixed" in Boundary and vice versa. Attribution is a
heuristic read of the diagnostics, labelled as such in the output.

Media alignment (edited video -> RAW source ranges) reuses the audio
cross-correlation decision map in ``cutsell_worker.human_gold_decision_map_v2``
-- also QA-only -- so no transcript, ASR or LLM is needed to place a
reference edit back on the RAW timeline. It is imported lazily inside the
CLI path so the region/classification logic stays testable without numpy
or ffmpeg.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

SCHEMA_VERSION = "cutsell.video00.quality_ladder.v1"

LEVEL_1 = "LEVEL_1"
LEVEL_2 = "LEVEL_2"
LEVEL_3 = "LEVEL_3"

# Region kinds (selection-level).
CONSENSUS_KEEP = "consensus_keep"
CONSENSUS_DELETE = "consensus_delete"
MISSING_DELIVERY = "missing_delivery"            # both references keep it, CutSell does not
FALSE_KEEP = "false_keep"                        # both references delete it, CutSell keeps it
GOLD_REMOVES_CUTAI_KEEPS = "gold_removes_cutai_keeps"   # CutSell == Cut.ai, Gold is stricter
GOLD_KEEPS_CUTAI_DROPS = "gold_keeps_cutai_drops"       # neither Cut.ai nor CutSell has it, Gold does
MATCHES_GOLD_OVER_CUTAI = "matches_gold_over_cutai"     # CutSell agrees with Gold's deletion
MATCHES_GOLD_BEYOND_CUTAI = "matches_gold_beyond_cutai"  # CutSell agrees with Gold's keep

# Refinements of MISSING_DELIVERY / FALSE_KEEP (what CutSell actually did).
NO_CANDIDATE = "no_candidate_segmented"
FALSE_DELETE = "false_delete_outside_family"
LOST_FAMILY_COMPETITION = "lost_family_competition"
TAKE_CHOICE_AGAINST_REFERENCES = "take_choice_against_both_references"
REDUNDANT_REALIZATION = "redundant_realization_both_kept"
UNGROUPED_RETRY = "ungrouped_retry_of_kept_idea"
FAILED_MATERIAL_RETAINED = "failed_or_process_material_retained"
RESTORED_BY_RESOLVER = "restored_by_realization_resolver"

# Authorities (D-021 component map).
AUTH_ATTEMPT = "AttemptReconstructor/RecordingProcessRemoval"
AUTH_CLUSTERER = "IdeaClusterer/RetryFamilyFormation"
AUTH_BEST_TAKE = "BestTakeResolver"
AUTH_REALIZATION = "RealizationResolver"
AUTH_COMPOSITE = "CompositeResolver/PreResolverCleanup"
AUTH_BOUNDARY = "BoundaryEngine"
AUTH_RENDER = "Renderer"
AUTH_NONE = "none"

DEFAULT_BOUNDARY_TOLERANCE_SEC = 0.35
DEFAULT_MIN_REGION_SEC = 0.04

_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+")
_STOP = frozenset({
    "a", "al", "como", "con", "cuando", "de", "del", "el", "en", "es", "esta", "este", "la",
    "las", "le", "lo", "los", "me", "mi", "mis", "no", "o", "para", "pero", "por", "porque",
    "que", "se", "si", "su", "sus", "un", "una", "y", "ya", "the", "and", "to", "of", "in",
    "is", "it", "that", "this", "was", "with", "for", "on", "my", "i",
})


# ---------------------------------------------------------------------------
# Spans
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, float(self.end) - float(self.start))


def _round(value: float | None, places: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value) + 0.0, places)


def normalize_spans(spans: Iterable[Span], *, join_gap_sec: float = 0.0) -> tuple[Span, ...]:
    ordered = sorted((s for s in spans if s.end > s.start), key=lambda s: (s.start, s.end))
    out: list[Span] = []
    for span in ordered:
        if out and span.start <= out[-1].end + join_gap_sec:
            out[-1] = Span(out[-1].start, max(out[-1].end, span.end))
        else:
            out.append(span)
    return tuple(out)


def overlap_duration(left: Span, right: Span) -> float:
    return max(0.0, min(left.end, right.end) - max(left.start, right.start))


def covered_duration(span: Span, spans: Iterable[Span]) -> float:
    return sum(overlap_duration(span, other) for other in normalize_spans(spans))


def _content_tokens(text: str) -> frozenset[str]:
    raw = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return frozenset(t for t in _TOKEN_RE.findall(raw) if t not in _STOP and len(t) > 1)


def _token_overlap(left: str, right: str) -> float:
    a, b = _content_tokens(left), _content_tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, min(len(a), len(b)))


# ---------------------------------------------------------------------------
# Reference cuts
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReferenceCut:
    """An edited video projected back onto the RAW timeline.

    ``chunks`` are dicts with raw_start / raw_end (RAW seconds), edit_start /
    edit_end (seconds inside the edited video) and alignment_confidence.
    """

    name: str
    chunks: tuple[dict, ...]
    edit_duration_sec: float | None = None

    @property
    def spans(self) -> tuple[Span, ...]:
        return normalize_spans(Span(float(c["raw_start"]), float(c["raw_end"])) for c in self.chunks)

    @property
    def kept_duration_sec(self) -> float:
        return sum(s.duration for s in self.spans)

    @staticmethod
    def from_spans(name: str, spans: Iterable[tuple[float, float]], *, confidence: float = 1.0) -> "ReferenceCut":
        chunks = []
        cursor = 0.0
        for index, (start, end) in enumerate(spans, 1):
            duration = float(end) - float(start)
            chunks.append({
                "index": index,
                "raw_start": float(start),
                "raw_end": float(end),
                "edit_start": cursor,
                "edit_end": cursor + duration,
                "alignment_confidence": float(confidence),
            })
            cursor += duration
        return ReferenceCut(name=name, chunks=tuple(chunks), edit_duration_sec=cursor)


# ---------------------------------------------------------------------------
# Engine result readers
# ---------------------------------------------------------------------------
def parent_clip_id(clip_id: str) -> str:
    """Physical fragments minted by Boundary keep ``<parent>__psig...`` ids
    (D-037); Selection identity is the parent."""
    clip_id = str(clip_id or "")
    return clip_id.split("__", 1)[0] if "__" in clip_id else clip_id


def _span_of(item: dict) -> Span | None:
    try:
        start = float(item.get("start"))
        end = float(item.get("end"))
    except (TypeError, ValueError):
        return None
    return Span(start, end) if end > start else None


def engine_candidates(engine_result: dict) -> tuple[dict, ...]:
    out: list[dict] = []
    for bucket, status in (("selected", "selected"), ("alternates", "alternate"), ("discarded", "discarded")):
        for order, item in enumerate(engine_result.get(bucket) or ()):
            span = _span_of(item or {})
            if span is None:
                continue
            clip_id = str(item.get("clip_id") or "")
            out.append({
                "status": status,
                "order": order,
                "clip_id": clip_id,
                "parent_clip_id": parent_clip_id(clip_id),
                "start": span.start,
                "end": span.end,
                "text": str(item.get("text") or ""),
                "take_group_id": item.get("take_group_id"),
            })
    return tuple(out)


def engine_selected_spans(engine_result: dict) -> tuple[Span, ...]:
    return normalize_spans(
        span for span in (_span_of(item or {}) for item in (engine_result.get("selected") or ())) if span is not None
    )


def _diagnostics(engine_result: dict) -> dict:
    diag = engine_result.get("diagnostics")
    return diag if isinstance(diag, dict) else {}


def _judge_groups(engine_result: dict) -> tuple[dict, ...]:
    rows = _diagnostics(engine_result).get("take_judge_groups") or ()
    return tuple(row for row in rows if isinstance(row, dict))


def _group_membership(engine_result: dict) -> dict[str, dict]:
    """clip_id -> {group_id, selected_clip_id, member_ids} from take_judge_groups
    (multi-member families only; singletons are not listed there)."""
    membership: dict[str, dict] = {}
    for row in _judge_groups(engine_result):
        members = [str(c.get("clip_id") or "") for c in (row.get("semantic_candidates") or ())]
        if not members:
            members = [str(c.get("clip_id") or "") for c in (row.get("ranked") or ())]
        info = {
            "group_id": row.get("group_id"),
            "selected_clip_id": row.get("selected_clip_id"),
            "local_selected_clip_id": row.get("local_selected_clip_id"),
            "semantic_override_applied": bool(row.get("semantic_override_applied")),
            "member_ids": tuple(members),
        }
        for clip_id in members:
            membership[clip_id] = info
    return membership


def _plan_ideas(engine_result: dict) -> tuple[dict, ...]:
    plan = _diagnostics(engine_result).get("canonical_edit_plan") or {}
    ideas = plan.get("ideas") if isinstance(plan, dict) else None
    return tuple(i for i in (ideas or ()) if isinstance(i, dict))


def _idea_index(engine_result: dict) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for idea in _plan_ideas(engine_result):
        for key, role in (("winning_clip_ids", "winner"), ("discarded_clip_ids", "discarded"),
                          ("authoritative_composite_realization_ids", "composite_piece")):
            for clip_id in idea.get(key) or ():
                index.setdefault(str(clip_id), {
                    "idea_id": idea.get("idea_id"),
                    "role": role,
                    "is_composite": bool(idea.get("is_composite")),
                    "coverage_status": idea.get("coverage_status"),
                    "authoritative_resolution_status": idea.get("authoritative_resolution_status"),
                })
    return index


def _restored_clip_ids(engine_result: dict) -> frozenset[str]:
    placement = _diagnostics(engine_result).get("authoritative_story_placement")
    ids: set[str] = set()
    rows: Iterable = ()
    if isinstance(placement, dict):
        rows = placement.get("units") or placement.get("placements") or placement.get("rows") or ()
    elif isinstance(placement, list):
        rows = placement
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in ("clip_id", "clip_ids", "restored_clip_id", "restored_clip_ids"):
            value = row.get(key)
            if isinstance(value, str):
                ids.add(value)
            elif isinstance(value, (list, tuple)):
                ids.update(str(v) for v in value)
    return frozenset(ids)


def _freeze(engine_result: dict) -> dict:
    contract = _diagnostics(engine_result).get("selection_boundary_contract") or {}
    if not isinstance(contract, dict):
        return {}
    return {
        "plan_id": contract.get("plan_id"),
        "plan_version": contract.get("plan_version"),
        "status": contract.get("status"),
        "semantic_sha256": contract.get("semantic_sha256"),
    }


# ---------------------------------------------------------------------------
# Region map
# ---------------------------------------------------------------------------
def _atomic_intervals(raw_duration: float, *edge_sources: Iterable[float]) -> list[Span]:
    edges = {0.0, float(raw_duration)}
    for source in edge_sources:
        for value in source:
            value = float(value)
            if 0.0 <= value <= raw_duration:
                edges.add(round(value, 4))
    ordered = sorted(edges)
    return [Span(a, b) for a, b in zip(ordered, ordered[1:]) if b > a]


def _covered(span: Span, spans: Sequence[Span], *, ratio: float = 0.5) -> bool:
    if span.duration <= 0:
        return False
    return covered_duration(span, spans) / span.duration >= ratio


def classify_triple(cutai_keep: bool, gold_keep: bool, cutsell_keep: bool) -> tuple[str, str]:
    """Return (level, kind) for one region from who kept it."""
    if cutai_keep and gold_keep and cutsell_keep:
        return LEVEL_3, CONSENSUS_KEEP
    if not cutai_keep and not gold_keep and not cutsell_keep:
        return LEVEL_3, CONSENSUS_DELETE
    if cutai_keep and gold_keep and not cutsell_keep:
        return LEVEL_1, MISSING_DELIVERY
    if not cutai_keep and not gold_keep and cutsell_keep:
        return LEVEL_1, FALSE_KEEP
    if cutai_keep and not gold_keep and cutsell_keep:
        return LEVEL_2, GOLD_REMOVES_CUTAI_KEEPS
    if not cutai_keep and gold_keep and not cutsell_keep:
        return LEVEL_2, GOLD_KEEPS_CUTAI_DROPS
    if cutai_keep and not gold_keep and not cutsell_keep:
        return LEVEL_3, MATCHES_GOLD_OVER_CUTAI
    return LEVEL_3, MATCHES_GOLD_BEYOND_CUTAI  # (not cutai, gold, cutsell)


def _refine_and_attribute(
    kind: str,
    overlapping: Sequence[dict],
    *,
    membership: dict[str, dict],
    idea_index: dict[str, dict],
    restored_ids: frozenset[str],
    selected_candidates: Sequence[dict],
) -> tuple[str | None, str, str]:
    """Return (refinement, authority, rationale) for a LEVEL_1 region."""
    selected = [c for c in overlapping if c["status"] == "selected"]
    alternates = [c for c in overlapping if c["status"] == "alternate"]
    discarded = [c for c in overlapping if c["status"] == "discarded"]

    if kind == MISSING_DELIVERY:
        if not overlapping:
            return NO_CANDIDATE, AUTH_ATTEMPT, "no CutSell candidate (any status) covers a region both references keep"
        probe = (alternates or discarded)[0]
        pid = probe["parent_clip_id"]
        family = membership.get(probe["clip_id"]) or membership.get(pid)
        idea = idea_index.get(probe["clip_id"]) or idea_index.get(pid)
        if family is not None:
            return (
                TAKE_CHOICE_AGAINST_REFERENCES,
                AUTH_BEST_TAKE,
                f"candidate {probe['clip_id']} lost family {family.get('group_id')} to "
                f"{family.get('selected_clip_id')} while both references chose this realization",
            )
        if idea is not None and idea.get("role") == "discarded":
            return (
                LOST_FAMILY_COMPETITION,
                AUTH_REALIZATION,
                f"candidate {probe['clip_id']} discarded inside idea {idea.get('idea_id')} by the authoritative resolver",
            )
        return (
            FALSE_DELETE,
            AUTH_COMPOSITE,
            f"candidate {probe['clip_id']} was deleted outside any retry family (pre-resolver cleanup / hybrid delete)",
        )

    if kind == FALSE_KEEP:
        if not selected:
            return None, AUTH_BOUNDARY, "kept span without a selected candidate: physical boundary slack"
        probe = selected[0]
        pid = probe["parent_clip_id"]
        if probe["clip_id"] in restored_ids or pid in restored_ids:
            return RESTORED_BY_RESOLVER, AUTH_REALIZATION, f"{probe['clip_id']} was restored by the RealizationResolver"
        family = membership.get(probe["clip_id"]) or membership.get(pid)
        if family is not None:
            others = [
                c for c in selected_candidates
                if c["parent_clip_id"] != pid and (c["clip_id"] in family["member_ids"] or c["parent_clip_id"] in family["member_ids"])
            ]
            if others:
                return (
                    REDUNDANT_REALIZATION,
                    AUTH_BEST_TAKE,
                    f"family {family.get('group_id')} has two kept realizations ({probe['clip_id']} and {others[0]['clip_id']})",
                )
            return (
                TAKE_CHOICE_AGAINST_REFERENCES,
                AUTH_BEST_TAKE,
                f"family {family.get('group_id')} winner {probe['clip_id']} is a realization both references rejected",
            )
        # Singleton: is it an ungrouped retry of something CutSell also kept?
        for other in selected_candidates:
            if other["parent_clip_id"] == pid:
                continue
            if _token_overlap(probe["text"], other["text"]) >= 0.5:
                return (
                    UNGROUPED_RETRY,
                    AUTH_CLUSTERER,
                    f"{probe['clip_id']} was never grouped with {other['clip_id']} although their content overlaps",
                )
        return (
            FAILED_MATERIAL_RETAINED,
            AUTH_ATTEMPT,
            f"{probe['clip_id']} is a lone candidate both references remove (failed attempt / recording process)",
        )
    return None, AUTH_NONE, ""


def build_region_map(
    *,
    raw_duration_sec: float,
    cutai: ReferenceCut,
    gold: ReferenceCut,
    engine_result: dict,
    rendered: ReferenceCut | None = None,
    boundary_tolerance_sec: float = DEFAULT_BOUNDARY_TOLERANCE_SEC,
    min_region_sec: float = DEFAULT_MIN_REGION_SEC,
) -> dict:
    raw_duration = float(raw_duration_sec)
    cutai_spans = cutai.spans
    gold_spans = gold.spans
    candidates = engine_candidates(engine_result)
    selected_candidates = [c for c in candidates if c["status"] == "selected"]
    cutsell_spans = engine_selected_spans(engine_result)
    membership = _group_membership(engine_result)
    idea_index = _idea_index(engine_result)
    restored_ids = _restored_clip_ids(engine_result)
    freeze = _freeze(engine_result)

    atoms = _atomic_intervals(
        raw_duration,
        (e for s in cutai_spans for e in (s.start, s.end)),
        (e for s in gold_spans for e in (s.start, s.end)),
        (e for c in candidates for e in (c["start"], c["end"])),
    )

    # Merge consecutive atoms sharing the same keep-triple AND the same set of
    # selected CutSell fragments, so one region never straddles two CutSell
    # fragments or a change of ownership.
    merged: list[dict] = []
    for atom in atoms:
        if atom.duration < 1e-6:
            continue
        overlapping = [c for c in candidates if overlap_duration(atom, Span(c["start"], c["end"])) > 1e-6]
        selected_ids = frozenset(c["clip_id"] for c in overlapping if c["status"] == "selected")
        triple = (_covered(atom, cutai_spans), _covered(atom, gold_spans), _covered(atom, cutsell_spans))
        key = (triple, selected_ids)
        if merged and merged[-1]["_key"] == key:
            merged[-1]["span"] = Span(merged[-1]["span"].start, atom.end)
            merged[-1]["_overlapping"] = {c["clip_id"]: c for c in overlapping} | merged[-1]["_overlapping"]
        else:
            merged.append({"_key": key, "span": atom, "_overlapping": {c["clip_id"]: c for c in overlapping}})

    regions: list[dict] = []
    for index, row in enumerate(merged, 1):
        span: Span = row["span"]
        (cutai_keep, gold_keep, cutsell_keep), _ = row["_key"]
        overlapping = sorted(row["_overlapping"].values(), key=lambda c: (c["start"], c["end"]))
        level, kind = classify_triple(cutai_keep, gold_keep, cutsell_keep)
        is_boundary = span.duration < boundary_tolerance_sec and kind not in (CONSENSUS_KEEP, CONSENSUS_DELETE)
        refinement = None
        authority = AUTH_NONE
        rationale = ""
        if level == LEVEL_1:
            if is_boundary:
                authority = AUTH_BOUNDARY
                rationale = "sub-tolerance edge difference against both references"
            else:
                refinement, authority, rationale = _refine_and_attribute(
                    kind, overlapping, membership=membership, idea_index=idea_index,
                    restored_ids=restored_ids, selected_candidates=selected_candidates,
                )
        cutsell_rows = []
        for c in overlapping:
            fam = membership.get(c["clip_id"]) or membership.get(c["parent_clip_id"])
            idea = idea_index.get(c["clip_id"]) or idea_index.get(c["parent_clip_id"])
            cutsell_rows.append({
                "clip_id": c["clip_id"],
                "status": c["status"],
                "start": _round(c["start"]),
                "end": _round(c["end"]),
                "text": c["text"][:140],
                "retry_family": None if fam is None else fam.get("group_id"),
                "family_winner": None if fam is None else fam.get("selected_clip_id"),
                "idea_id": None if idea is None else idea.get("idea_id"),
                "idea_role": None if idea is None else idea.get("role"),
                "is_composite": None if idea is None else idea.get("is_composite"),
                "restored": c["clip_id"] in restored_ids or c["parent_clip_id"] in restored_ids,
            })
        regions.append({
            "region_index": index,
            "raw_start": _round(span.start),
            "raw_end": _round(span.end),
            "duration_sec": _round(span.duration),
            "cutai_keep": cutai_keep,
            "gold_keep": gold_keep,
            "cutsell_keep": cutsell_keep,
            "level": level,
            "kind": kind,
            "scope": "boundary" if is_boundary else ("negligible" if span.duration < min_region_sec else "selection"),
            "refinement": refinement,
            "attributed_authority": authority,
            "attribution_rationale": rationale,
            "cutsell_candidates": cutsell_rows,
        })

    # Traceability for every selected fragment.
    traceability = []
    rendered_spans = rendered.spans if rendered is not None else ()
    for c in sorted(selected_candidates, key=lambda c: c["order"]):
        span = Span(c["start"], c["end"])
        fam = membership.get(c["clip_id"]) or membership.get(c["parent_clip_id"])
        idea = idea_index.get(c["clip_id"]) or idea_index.get(c["parent_clip_id"])
        region_levels = [
            r["level"] for r in regions
            if r["scope"] == "selection" and any(cc["clip_id"] == c["clip_id"] and cc["status"] == "selected" for cc in r["cutsell_candidates"])
        ]
        worst = LEVEL_1 if LEVEL_1 in region_levels else (LEVEL_2 if LEVEL_2 in region_levels else LEVEL_3)
        render_cov = (covered_duration(span, rendered_spans) / span.duration) if (rendered is not None and span.duration) else None
        traceability.append({
            "clip_id": c["clip_id"],
            "parent_clip_id": c["parent_clip_id"],
            "raw_start": _round(c["start"]),
            "raw_end": _round(c["end"]),
            "text": c["text"][:160],
            "retry_family": None if fam is None else fam.get("group_id"),
            "family_members": None if fam is None else list(fam.get("member_ids") or ()),
            "family_local_winner": None if fam is None else fam.get("local_selected_clip_id"),
            "semantic_override_applied": None if fam is None else fam.get("semantic_override_applied"),
            "idea_id": None if idea is None else idea.get("idea_id"),
            "resolution": None if idea is None else idea.get("authoritative_resolution_status"),
            "is_composite": None if idea is None else idea.get("is_composite"),
            "restored_by_resolver": c["clip_id"] in restored_ids or c["parent_clip_id"] in restored_ids,
            "freeze_plan_id": freeze.get("plan_id"),
            "freeze_plan_version": freeze.get("plan_version"),
            "cutai_coverage": _round(covered_duration(span, cutai_spans) / span.duration if span.duration else 0.0, 4),
            "gold_coverage": _round(covered_duration(span, gold_spans) / span.duration if span.duration else 0.0, 4),
            "rendered_coverage": _round(render_cov, 4),
            "worst_level": worst,
        })

    summary = _summarize(regions, cutai, gold, cutsell_spans, rendered, raw_duration)
    return {
        "schema_version": SCHEMA_VERSION,
        "raw_duration_sec": _round(raw_duration, 6),
        "references": {
            "cutai": {"chunk_count": len(cutai.chunks), "kept_duration_sec": _round(cutai.kept_duration_sec), "edit_duration_sec": _round(cutai.edit_duration_sec)},
            "gold": {"chunk_count": len(gold.chunks), "kept_duration_sec": _round(gold.kept_duration_sec), "edit_duration_sec": _round(gold.edit_duration_sec)},
            "cutsell_rendered": None if rendered is None else {"chunk_count": len(rendered.chunks), "kept_duration_sec": _round(rendered.kept_duration_sec), "edit_duration_sec": _round(rendered.edit_duration_sec)},
        },
        "freeze": freeze,
        "summary": summary,
        "regions": regions,
        "traceability": traceability,
        "attribution_note": "attributed_authority is a heuristic read of CutSell diagnostics (take_judge_groups, canonical_edit_plan, authoritative_story_placement); confirm against the run log before fixing.",
    }


def _f1(reference: Sequence[Span], candidate: Sequence[Span]) -> dict:
    ref_total = sum(s.duration for s in reference)
    cand_total = sum(s.duration for s in candidate)
    overlap = sum(covered_duration(s, candidate) for s in reference)
    recall = overlap / ref_total if ref_total else 0.0
    precision = overlap / cand_total if cand_total else 0.0
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) else 0.0
    return {"recall": _round(recall, 4), "precision": _round(precision, 4), "f1": _round(f1, 4)}


def _summarize(regions, cutai: ReferenceCut, gold: ReferenceCut, cutsell_spans, rendered, raw_duration) -> dict:
    by_level: dict[str, dict] = {lvl: {"selection_count": 0, "selection_seconds": 0.0, "boundary_count": 0, "boundary_seconds": 0.0} for lvl in (LEVEL_1, LEVEL_2, LEVEL_3)}
    by_kind: dict[str, dict] = {}
    by_authority: dict[str, dict] = {}
    for r in regions:
        if r["scope"] == "negligible":
            continue
        bucket = by_level[r["level"]]
        if r["scope"] == "boundary":
            bucket["boundary_count"] += 1
            bucket["boundary_seconds"] += r["duration_sec"]
        else:
            bucket["selection_count"] += 1
            bucket["selection_seconds"] += r["duration_sec"]
            k = by_kind.setdefault(r["kind"], {"count": 0, "seconds": 0.0, "level": r["level"]})
            k["count"] += 1
            k["seconds"] += r["duration_sec"]
        if r["level"] == LEVEL_1:
            a = by_authority.setdefault(r["attributed_authority"], {"count": 0, "seconds": 0.0})
            a["count"] += 1
            a["seconds"] += r["duration_sec"]
    for bucket in by_level.values():
        bucket["selection_seconds"] = _round(bucket["selection_seconds"])
        bucket["boundary_seconds"] = _round(bucket["boundary_seconds"])
    for k in by_kind.values():
        k["seconds"] = _round(k["seconds"])
    for a in by_authority.values():
        a["seconds"] = _round(a["seconds"])

    cutai_spans, gold_spans = cutai.spans, gold.spans
    level1_seconds = by_level[LEVEL_1]["selection_seconds"] or 0.0
    cutai_keep = cutai.kept_duration_sec
    return {
        "by_level": by_level,
        "by_kind": by_kind,
        "level1_by_authority": by_authority,
        "durations_sec": {
            "raw": _round(raw_duration),
            "cutai_keep": _round(cutai_keep),
            "gold_keep": _round(gold.kept_duration_sec),
            "cutsell_keep": _round(sum(s.duration for s in cutsell_spans)),
            "cutsell_rendered": None if rendered is None else _round(rendered.kept_duration_sec),
        },
        "selection_parity": {
            "cutsell_vs_cutai": _f1(cutai_spans, cutsell_spans),
            "cutsell_vs_gold": _f1(gold_spans, cutsell_spans),
            "cutai_vs_gold": _f1(gold_spans, cutai_spans),
        },
        # Cut.ai parity score: Level-1 selection seconds relative to what Cut.ai keeps.
        # Reported, not gated -- the acceptance threshold is a Product Owner decision.
        "cutai_parity": {
            "level1_selection_seconds": _round(level1_seconds),
            "level1_share_of_cutai_keep": _round(level1_seconds / cutai_keep if cutai_keep else 0.0, 4),
            "level1_selection_regions": by_level[LEVEL_1]["selection_count"],
        },
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _flag(value: bool) -> str:
    return "K" if value else "-"


def render_markdown(report: dict, *, max_text: int = 70) -> str:
    s = report["summary"]
    lines = [
        f"# Video00 quality ladder ({report['schema_version']})",
        "",
        "Legend: C=Cut.ai G=Human Gold S=CutSell; K=kept, -=removed. Levels: 1=CutSell worse than Cut.ai (fix first), "
        "2=CutSell~Cut.ai but Gold better, 3=matches/exceeds Gold (protect).",
        "",
        "## Summary",
        "",
        "| level | selection regions | selection sec | boundary regions | boundary sec |",
        "|---|---|---|---|---|",
    ]
    for lvl in (LEVEL_1, LEVEL_2, LEVEL_3):
        b = s["by_level"][lvl]
        lines.append(f"| {lvl} | {b['selection_count']} | {b['selection_seconds']} | {b['boundary_count']} | {b['boundary_seconds']} |")
    d = s["durations_sec"]
    lines += [
        "",
        f"Durations (s): raw {d['raw']} | Cut.ai keep {d['cutai_keep']} | Gold keep {d['gold_keep']} | CutSell keep {d['cutsell_keep']}"
        + (f" | CutSell rendered {d['cutsell_rendered']}" if d.get("cutsell_rendered") is not None else ""),
        "",
        f"Selection F1: CutSell vs Cut.ai {s['selection_parity']['cutsell_vs_cutai']['f1']} | CutSell vs Gold {s['selection_parity']['cutsell_vs_gold']['f1']} | Cut.ai vs Gold {s['selection_parity']['cutai_vs_gold']['f1']}",
        "",
        f"Cut.ai parity: LEVEL_1 selection seconds {s['cutai_parity']['level1_selection_seconds']} "
        f"({s['cutai_parity']['level1_share_of_cutai_keep']} of Cut.ai keep) across {s['cutai_parity']['level1_selection_regions']} regions",
        "",
        "LEVEL_1 by authority: " + (", ".join(f"{k}: {v['count']} regions / {v['seconds']} s" for k, v in sorted(s["level1_by_authority"].items(), key=lambda kv: -kv[1]['seconds'])) or "none"),
        "",
        "## Regions (selection scope)",
        "",
        "| # | raw start | raw end | dur | C | G | S | level | kind | refinement | authority | CutSell candidates |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report["regions"]:
        if r["scope"] != "selection":
            continue
        cands = "; ".join(
            f"{c['status'][:3]}:{c['clip_id'][-8:]}" + (f"[{c['retry_family'][-6:]}]" if c.get("retry_family") else "") + f" \"{c['text'][:max_text]}\""
            for c in r["cutsell_candidates"]
        ) or "(none)"
        lines.append(
            f"| {r['region_index']} | {r['raw_start']} | {r['raw_end']} | {r['duration_sec']} | {_flag(r['cutai_keep'])} | {_flag(r['gold_keep'])} | {_flag(r['cutsell_keep'])} "
            f"| {r['level'][-1]} | {r['kind']} | {r['refinement'] or ''} | {r['attributed_authority'] if r['level']==LEVEL_1 else ''} | {cands} |"
        )
    boundary_rows = [r for r in report["regions"] if r["scope"] == "boundary"]
    lines += ["", f"## Boundary-scope regions (< tolerance): {len(boundary_rows)}", ""]
    if boundary_rows:
        lines += ["| # | raw start | raw end | dur | C | G | S | level | kind |", "|---|---|---|---|---|---|---|---|---|"]
        for r in boundary_rows:
            lines.append(f"| {r['region_index']} | {r['raw_start']} | {r['raw_end']} | {r['duration_sec']} | {_flag(r['cutai_keep'])} | {_flag(r['gold_keep'])} | {_flag(r['cutsell_keep'])} | {r['level'][-1]} | {r['kind']} |")
    lines += ["", "## Traceability (every CutSell selected fragment)", "",
              "| clip | raw range | family | idea / resolution | composite | restored | C cov | G cov | rendered cov | worst level | text |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in report["traceability"]:
        lines.append(
            f"| {t['clip_id'][-12:]} | {t['raw_start']}–{t['raw_end']} | {(t['retry_family'] or '')[-8:]} | {(t['idea_id'] or '')[-8:]} {t['resolution'] or ''} "
            f"| {t['is_composite']} | {t['restored_by_resolver']} | {t['cutai_coverage']} | {t['gold_coverage']} | {t['rendered_coverage']} | {t['worst_level'][-1]} | {t['text'][:max_text]} |"
        )
    lines += ["", f"_{report['attribution_note']}_"]
    return "\n".join(lines)


def write_csv(report: dict, path: str | Path) -> None:
    fields = ["region_index", "raw_start", "raw_end", "duration_sec", "cutai_keep", "gold_keep", "cutsell_keep",
              "level", "kind", "scope", "refinement", "attributed_authority", "attribution_rationale", "cutsell_candidates"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in report["regions"]:
            row = {k: r.get(k) for k in fields}
            row["cutsell_candidates"] = " | ".join(f"{c['status']}:{c['clip_id']}:{c['text'][:60]}" for c in r["cutsell_candidates"])
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Media alignment (CLI only)
# ---------------------------------------------------------------------------
def align_edit_to_raw(raw_path: str | Path, edit_path: str | Path, name: str) -> ReferenceCut:
    """Project an edited MP4 back onto RAW by audio cross-correlation
    (``human_gold_decision_map_v2``). QA-only; imported lazily."""
    from cutsell_worker.human_gold_decision_map import _ffprobe_duration
    from cutsell_worker.human_gold_decision_map_v2 import align_gold_audio_to_raw_v2

    edit_duration = _ffprobe_duration(edit_path)
    _anchors, chunks = align_gold_audio_to_raw_v2(raw_path, edit_path)
    rows = tuple({
        "index": c.index,
        "raw_start": float(c.raw_start),
        "raw_end": float(c.raw_end),
        "edit_start": float(c.gold_start),
        "edit_end": float(c.gold_end),
        "alignment_confidence": float(c.alignment_confidence),
    } for c in chunks)
    return ReferenceCut(name=name, chunks=rows, edit_duration_sec=float(edit_duration))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Video00 quality ladder: RAW vs Cut.ai vs Human Gold vs CutSell (QA-only)")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--cutai", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--engine-json", required=True)
    parser.add_argument("--engine-mp4", default=None, help="optional CutSell rendered/diagnostic preview to align back to RAW")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--boundary-tolerance-sec", type=float, default=DEFAULT_BOUNDARY_TOLERANCE_SEC)
    args = parser.parse_args(argv)

    from cutsell_worker.human_gold_decision_map import _ffprobe_duration

    raw_duration = _ffprobe_duration(args.raw)
    cutai = align_edit_to_raw(args.raw, args.cutai, "cutai")
    gold = align_edit_to_raw(args.raw, args.gold, "gold")
    rendered = None
    if args.engine_mp4:
        try:
            rendered = align_edit_to_raw(args.raw, args.engine_mp4, "cutsell_rendered")
        except Exception as exc:  # observability, never hide it
            print(json.dumps({"rendered_alignment_error": str(exc)[:300]}))
    engine_result = json.loads(Path(args.engine_json).read_text(encoding="utf-8"))
    report = build_region_map(
        raw_duration_sec=raw_duration, cutai=cutai, gold=gold, engine_result=engine_result,
        rendered=rendered, boundary_tolerance_sec=args.boundary_tolerance_sec,
    )
    report["inputs"] = {"raw": str(args.raw), "cutai": str(args.cutai), "gold": str(args.gold),
                        "engine_json": str(args.engine_json), "engine_mp4": args.engine_mp4}
    report["reference_chunks"] = {"cutai": list(cutai.chunks), "gold": list(gold.chunks),
                                  "cutsell_rendered": None if rendered is None else list(rendered.chunks)}
    Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    markdown = render_markdown(report)
    if args.out_md:
        Path(args.out_md).write_text(markdown, encoding="utf-8")
    if args.out_csv:
        write_csv(report, args.out_csv)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
