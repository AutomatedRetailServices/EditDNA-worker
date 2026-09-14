"""CLEAN RAW Watch+Listen checkpoint (D-097.13, bounded encargo).

D-096/D-095 established the canonical quality ladder RAW -> CUT.AI PARITY ->
HUMAN GOLD PARITY -> technical post-render QC -> perceptual SYSTEM WATCH +
LISTEN -> HUMAN WATCH + LISTEN PASS, and ruled that every claim of fixed
behavior must be proven on the RENDERED VIDEO the worker actually produced,
never inferred from plan timestamps or a selection diagnostic alone (CODE
EXISTS != VIDEO USED IT). `perceptual_watch_listen.py` already implements
the advisory, per-capability System Watch+Listen reviewer; this module adds
the missing link the D-097.12 stomach-family fix (and any future selection
fix) needs to become observable on real media: a bounded diagnostic render
of a CLEAN_RAW_SELECTION through the SAME canonical physical authorities
production uses (`render_plan.build_render_plan`,
`live_render_qc.render_with_post_render_qc`,
`perceptual_watch_listen.review_rendered_candidate`), plus an explicit,
audio-correlation-based PHYSICAL membership check (reusing
`video00_quality_ladder.verify_render_against_plan`'s own template-matching
primitive -- no new correlation math) that answers "is the clip that
should be in the file actually IN it, and is the clip that should be
ABSENT actually absent from the whole file" -- not merely "is it present
in the frozen plan".

This module NEVER edits selection. It takes a `DraftTimeline` as an
opaque, read-only input (the CLEAN_RAW_SELECTION some upstream editorial
authority already decided) and only ever calls render/QC/perceptual
functions on it. Watch+Listen output can only ROUTE a finding to the
authority that owns the concern it describes; it never deletes, restores,
regroups, or composites anything itself -- enforced structurally here by
never importing any grouping/resolver/story-validation function.

StageVerdict vocabulary (never anything else):
    PASS      -- every capability the checkpoint could evaluate came back
                 clean AND physical membership matched the selection.
    FAIL      -- a concrete physical defect was detected (membership
                 mismatch, or a routed perceptual/technical finding).
    UNCERTAIN -- at least one required condition could not be reliably
                 evaluated (missing evidence, NOT_IMPLEMENTED capability,
                 a `found=False` audio-correlation result too weak to
                 call either way) and nothing rose to FAIL.
    ERROR     -- evaluation itself failed (exception building the render,
                 reading media, or running a sub-check).
A capability the checkpoint cannot evaluate NEVER silently becomes PASS.
"""
from __future__ import annotations

import os
import traceback
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from cutsell_worker.live_boundary_repair import segment_output_windows
from cutsell_worker.live_render_qc import render_with_post_render_qc
from cutsell_worker.perceptual_watch_listen import error_review, review_rendered_candidate
from cutsell_worker.render_plan import build_render_plan

STAGE_PASS = "PASS"
STAGE_FAIL = "FAIL"
STAGE_UNCERTAIN = "UNCERTAIN"
STAGE_ERROR = "ERROR"
_ALLOWED_VERDICTS = frozenset({STAGE_PASS, STAGE_FAIL, STAGE_UNCERTAIN, STAGE_ERROR})

# Routing table (task-mandated): a finding names the authority that owns
# the concern, never a fix applied here.
ROUTE_RENDERER = "Renderer/PhysicalPlan"
ROUTE_REALIZATION_RESOLVER = "RealizationResolver/RetryFamilyAuthority"
ROUTE_BOUNDARY_ENGINE = "BoundaryEngine"
ROUTE_TECHNICAL_QC = "Renderer/TechnicalQC"
ROUTE_STORY_SAFETY = "MeaningPreservation/StorySafety"
ROUTE_UNKNOWN_CAPABILITY = "PerceptualCapabilityGap"


@dataclass(frozen=True)
class ExpectedPresence:
    """One clip the CLEAN_RAW_SELECTION says must be physically IN the
    rendered file, located by audio-correlation against its own RAW span."""

    clip_id: str
    label: str = ""


@dataclass(frozen=True)
class ExpectedAbsence:
    """One clip (typically a discarded realization/attempt) the
    CLEAN_RAW_SELECTION says must be physically ABSENT from the rendered
    file -- checked by correlating its RAW-source template against the
    ENTIRE render, not merely confirming it is missing from the frozen
    plan (a plan omission proves nothing about a renderer/repair bug that
    might still splice the wrong material in)."""

    clip_id: str
    source_asset_id: str
    raw_start: float
    raw_end: float
    label: str = ""


@dataclass(frozen=True)
class ExpectedIndependence:
    """A clip the CLEAN_RAW_SELECTION keeps as its OWN, separate
    realization (never absorbed into another family's rendered span)."""

    clip_id: str
    label: str = ""


@dataclass
class CleanRawCheckpointResult:
    verdict: str
    routed_owner: str | None
    diagnostic_record: dict[str, Any]

    def __post_init__(self) -> None:
        if self.verdict not in _ALLOWED_VERDICTS:
            raise ValueError(f"invalid CLEAN_RAW StageVerdict: {self.verdict!r}")


def _membership_check(
    *,
    raw_media_path: str,
    render_path: str,
    present: Sequence[ExpectedPresence],
    absent: Sequence[ExpectedAbsence],
    independent: Sequence[ExpectedIndependence],
    selected_spans: Mapping[str, tuple[float, float]],
    found_correlation: float = 0.60,
    absence_correlation: float = 0.55,
) -> dict[str, Any]:
    """Physical membership proof: locates every expected-present clip in
    the actual rendered file by audio-correlation against its own RAW
    span, and confirms every expected-absent clip's RAW audio template
    does NOT correlate strongly anywhere in the render. Reuses the exact
    correlation primitives `video00_quality_ladder.verify_render_against_
    plan` already uses -- no second implementation of the matching math."""
    from benchmarks.video00_quality_ladder import verify_render_against_plan
    from cutsell_worker.human_gold_decision_map import _audio_features, _normalized_window_correlation

    present_selected = [
        {"clip_id": item.clip_id, "start": selected_spans[item.clip_id][0], "end": selected_spans[item.clip_id][1]}
        for item in present
        if item.clip_id in selected_spans
    ]
    verification = verify_render_against_plan(
        raw_media_path, render_path, present_selected, found_correlation=found_correlation,
    )
    present_rows = []
    present_uncertain = False
    for item in present:
        row = next((f for f in verification["fragments"] if f["clip_id"] == item.clip_id), None)
        if row is None:
            present_rows.append({"clip_id": item.clip_id, "label": item.label, "status": "UNCERTAIN",
                                 "reason": "no RAW span recorded for this clip_id"})
            present_uncertain = True
            continue
        found = bool(row.get("found"))
        present_rows.append({"clip_id": item.clip_id, "label": item.label,
                             "status": "CONFIRMED" if found else "NOT_FOUND",
                             "correlation": row.get("correlation")})

    absent_rows = []
    absent_violation = False
    if absent:
        render_features = _audio_features(render_path, hop_sec=0.02)
        raw_cache: dict[str, Any] = {}
        for item in absent:
            if item.source_asset_id not in raw_cache:
                raw_cache[item.source_asset_id] = _audio_features(raw_media_path, hop_sec=0.02)
            raw_features = raw_cache[item.source_asset_id]
            a, b = int(round(item.raw_start / 0.02)), int(round(item.raw_end / 0.02))
            template = raw_features[a:b]
            if template.shape[0] == 0:
                absent_rows.append({"clip_id": item.clip_id, "label": item.label, "status": "UNCERTAIN",
                                    "reason": "empty RAW template for this span"})
                continue
            corr = _normalized_window_correlation(render_features, template)
            peak = float(corr.max()) if corr.shape[0] else 0.0
            physically_absent = peak < absence_correlation
            if not physically_absent:
                absent_violation = True
            absent_rows.append({"clip_id": item.clip_id, "label": item.label,
                                "status": "CONFIRMED_ABSENT" if physically_absent else "PHYSICALLY_PRESENT",
                                "peak_correlation": round(peak, 4)})

    independence_rows = []
    for item in independent:
        span = selected_spans.get(item.clip_id)
        independence_rows.append({
            "clip_id": item.clip_id, "label": item.label,
            "status": "PRESERVED_AS_OWN_SELECTION" if span is not None else "NOT_IN_SELECTION",
        })

    return {
        "render_verification": verification,
        "present": present_rows,
        "absent": absent_rows,
        "independent": independence_rows,
        "present_uncertain": present_uncertain,
        "present_all_confirmed": bool(present_rows) and all(r["status"] == "CONFIRMED" for r in present_rows),
        "absent_all_confirmed": (not absent) or all(r["status"] == "CONFIRMED_ABSENT" for r in absent_rows),
        "absent_violation": absent_violation,
    }


def run_clean_raw_checkpoint(
    draft,
    local_paths: Mapping[str, str],
    *,
    raw_media_path: str,
    output_path: str,
    family: str,
    run_identity: Mapping[str, Any],
    expected_present: Sequence[ExpectedPresence] = (),
    expected_absent: Sequence[ExpectedAbsence] = (),
    expected_independent: Sequence[ExpectedIndependence] = (),
) -> CleanRawCheckpointResult:
    """Run the CLEAN RAW checkpoint: render `draft` (the CLEAN_RAW_
    SELECTION) through the canonical physical authorities, run technical
    QC and perceptual Watch+Listen on the ACTUAL rendered file, verify
    physical membership by audio correlation against `raw_media_path`, and
    reduce to one CLEAN_RAW StageVerdict. Never mutates `draft`."""
    selected_ids_before = tuple(c.clip_id for c in draft.selected)
    discarded_ids_before = tuple(c.clip_id for c in draft.discarded)
    record: dict[str, Any] = {
        "run_identity": dict(run_identity),
        "stage": "CLEAN_RAW",
        "family": family,
        "selected_ids": list(selected_ids_before),
        "discarded_ids": list(discarded_ids_before),
        "source_ranges": {c.clip_id: [c.start, c.end] for c in (*draft.selected, *draft.discarded)},
    }
    try:
        plan = build_render_plan(draft, local_paths)
        qc_result = render_with_post_render_qc(draft, plan, output_path)
        record["diagnostic_mp4_path"] = qc_result.output_path
        record["technical_qc"] = {
            "status": qc_result.status,
            "render_attempt_count": len(qc_result.attempts),
            "deliverable": getattr(qc_result, "deliverable", None),
        }
        final_state = qc_result.attempts[-1].input_boundary_state if qc_result.attempts else ()
        by_id = {segment.clip_id: segment for segment in plan}
        current_segments = tuple(
            _replace_span(by_id[row["clip_id"]], row) for row in final_state if row.get("clip_id") in by_id
        ) or plan
        record["final_rendered_ranges"] = {seg.clip_id: [seg.start, seg.end] for seg in current_segments}

        media_path = qc_result.output_path
        if not media_path or not os.path.exists(media_path):
            record["watch_listen"] = None
            record["unsupported_capabilities"] = ["all -- no rendered media produced"]
            verdict, owner = STAGE_ERROR, ROUTE_TECHNICAL_QC
            record["verdict_reason"] = "technical QC produced no output_path/media to review"
            return _finish(record, verdict, owner)

        windows = segment_output_windows(current_segments)
        review = review_rendered_candidate(media_path, draft, current_segments, windows).as_dict()
        record["watch_listen"] = review

        selected_spans = {seg.clip_id: (seg.start, seg.end) for seg in current_segments}
        membership = _membership_check(
            raw_media_path=raw_media_path, render_path=media_path,
            present=expected_present, absent=expected_absent, independent=expected_independent,
            selected_spans=selected_spans,
        )
        record["physical_membership"] = membership

        verdict, owner, reason = _reduce_verdict(qc_result, review, membership)
        record["verdict_reason"] = reason
        record["unsupported_capabilities"] = [
            c["capability"] for c in review.get("capabilities", []) if c.get("status") in ("NOT_IMPLEMENTED", "UNCERTAIN", "ERROR")
        ]
    except Exception as exc:  # noqa: BLE001 -- ERROR is a reported status, never a silent pass
        record["watch_listen"] = error_review(f"clean_raw_checkpoint_failed: {exc}").as_dict()
        record["exception"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()[-2000:]
        return _finish(record, STAGE_ERROR, None)

    assert tuple(c.clip_id for c in draft.selected) == selected_ids_before, "checkpoint must never mutate selection"
    assert tuple(c.clip_id for c in draft.discarded) == discarded_ids_before, "checkpoint must never mutate selection"
    return _finish(record, verdict, owner)


def _replace_span(segment, row: Mapping[str, Any]):
    from dataclasses import replace
    return replace(segment, start=float(row["start"]), end=float(row["end"]))


def _reduce_verdict(qc_result, review: Mapping[str, Any], membership: Mapping[str, Any]) -> tuple[str, str | None, str]:
    """Deterministic reduction, in priority order. A FAIL always names the
    routed owner; PASS is scoped to capabilities ACTUALLY EVALUATED (a
    NOT_IMPLEMENTED capability is surfaced in `unsupported_capabilities`,
    never silently treated as passing, but also never by itself forces the
    whole checkpoint to UNCERTAIN -- that is reserved for a condition this
    checkpoint tried and failed to determine, i.e. an UNCERTAIN or ERROR
    capability result, or a membership check that could not be evaluated)."""
    if membership["absent_violation"]:
        return STAGE_FAIL, ROUTE_RENDERER, "an expected-absent clip's audio physically appears in the render"
    if not membership["present_all_confirmed"] and membership["present"]:
        # Distinguish a clean not-found (renderer dropped required content
        # -> Renderer/PhysicalPlan) from a genuinely uncertain template
        # (empty span, missing RAW range -> UNCERTAIN, never silently PASS).
        if any(r["status"] == "NOT_FOUND" for r in membership["present"]):
            return STAGE_FAIL, ROUTE_RENDERER, "an expected-present clip could not be located in the rendered media"
        return STAGE_UNCERTAIN, ROUTE_UNKNOWN_CAPABILITY, "physical presence of an expected clip could not be evaluated"

    if str(qc_result.status) != "PASS":
        return STAGE_FAIL, ROUTE_TECHNICAL_QC, f"technical QC status={qc_result.status}"

    routing = review.get("routing") or {}
    if review.get("status") == "FAIL" and routing:
        # Route by the first owner the reviewer itself named; never fixed here.
        owner_map = {
            "BoundaryEngine": ROUTE_BOUNDARY_ENGINE,
            "BestTakeResolver": ROUTE_REALIZATION_RESOLVER,
            "Renderer": ROUTE_RENDERER,
        }
        first_owner = next(iter(routing))
        return STAGE_FAIL, owner_map.get(first_owner, ROUTE_UNKNOWN_CAPABILITY), f"perceptual finding routed to {first_owner}"
    if review.get("status") == "FAIL":
        return STAGE_FAIL, ROUTE_UNKNOWN_CAPABILITY, "perceptual review FAIL with no routing recorded"

    counts = review.get("capability_status_counts") or {}
    if counts.get("UNCERTAIN") or counts.get("ERROR"):
        return STAGE_UNCERTAIN, ROUTE_UNKNOWN_CAPABILITY, "one or more perceptual capabilities could not evaluate the condition"

    if membership["absent"] and not membership["absent_all_confirmed"]:
        return STAGE_UNCERTAIN, ROUTE_UNKNOWN_CAPABILITY, "absence of a discarded clip could not be reliably confirmed"

    return STAGE_PASS, None, "technical QC PASS, physical membership confirmed, no unresolved perceptual finding"


def _finish(record: dict[str, Any], verdict: str, owner: str | None) -> CleanRawCheckpointResult:
    record["final_stage_verdict"] = verdict
    record["routed_owner"] = owner
    return CleanRawCheckpointResult(verdict=verdict, routed_owner=owner, diagnostic_record=record)
