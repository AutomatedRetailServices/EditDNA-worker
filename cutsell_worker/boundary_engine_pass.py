"""D-097 Priority C/E -- the ONE post-Freeze BoundaryEngine pass and the
physical cleanup OWNERSHIP CONTRACT.

D-096 (root cause #4, C-6/C-7/C-8/C-12, G-8, D-6) proved that physical
cleanup was owned by nobody in particular: five pre-Freeze editors, one
post-Freeze polish and two renderer-side trimmers each touched source
ranges, entries had no owner at all, and the D-095.2 interior dead-air
trimmer ran INSIDE `build_flow_b_draft` -- before the Resolver / Story
Validator / repair loop decided the final KEEP set -- so a realization
restored or re-shaped later never received the trim (run 34008386434: the
2.32 s pause that invalidated the render sat inside a clip the trimmer had
never seen). This module is that move, not a new layer: the SAME existing
functions (`trim_locked_selection_edges`, `split_selected_interior_
performance_gaps`) now run once, here, on the frozen final KEEP set, plus
the one owner that was missing (audio-evidenced entry/exit tightening).

## PHYSICAL CLEANUP OWNERSHIP CONTRACT (D-097.E)

| Physical concern | Owner | When | Evidence | Never |
|---|---|---|---|---|
| ENTRY (leading non-speech) | BoundaryEngine pass (this module) | after Selection Freeze | audio_silence_interval events at the leading edge; multimodal reset evidence (edge-only trim) | remove a word; change membership |
| EXIT (trailing non-speech) | BoundaryEngine pass; renderer `tighten_trailing_silence` is the LAST mechanical op, recorded per segment | after Freeze; at render | audio_silence_interval at the trailing edge; renderer silencedetect on the exact segment | remove a word |
| INTERIOR DEAD AIR (>= 1.2 s) | BoundaryEngine pass via `split_selected_interior_performance_gaps` | after Freeze | audio_silence_interval (primary and relaxed floor) | cut inside a word |
| RESET DEBRIS (micro word gaps with visual resets) | BoundaryEngine pass (performance-gap split) + `polish_human_boundaries_v5` | after Freeze | A-5 local performance events | change the token stream |
| MICRO CONTINUITY (re-joining over-segmented same-source pieces) | `post_selection_continuity_coalescer` (pre-Freeze draft wrapper: restores the micro-gap, identities preserved -- D-097.3) and `render_plan._coalesce_contiguous_segments` (mechanical, render) | draft / render | source adjacency, no reset evidence | merge across a Boundary-authorized cut; re-minting a clip identity |
| RENDERER MECHANICAL OPS | `render.render_preview` | render | trailing-silence tighten (recorded), 12 ms join fades, contiguous coalesce | any editorial decision |
| PRE-FREEZE EDITORS (delivery_edge_trim, speech_safe_dead_air_guard, final_boundary_authority envelope, temporal trims) | Selection-phase candidate shaping | before Freeze | word envelopes, harmful events | be the final physical authority |

Selection decides WHAT plays; this pass decides WHERE it starts and ends;
the renderer only executes. Every operation below preserves the frozen
ordered token stream (`enforce_selection_contract` verifies it afterwards).

## Reconciliation (C-12)

`reconcile_silence_findings` maps every LINGERING_ACCIDENTAL_SILENCE the
post-render QC measured on the OUTPUT timeline back to the source range of
the segment it fell in, and answers, per finding: did the source-level
measurement see it (which floor), did this pass see the clip, and what did
the trimmer decide. A run can then prove whether a remaining dead-air
finding is a measurement gap, an ordering gap, or a trimmer rejection --
instead of guessing (D-096 G-14).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Mapping, Sequence

from .contracts import DraftClip, ProcessingResult
from .post_selection_edge_only_boundary import trim_locked_selection_edges
from .post_selection_interior_gap_trim import (
    AUDIO_SILENCE_EVENT_KIND,
    LONG_AUDIO_SILENCE_SEC,
    split_selected_interior_performance_gaps,
)

SCHEMA_VERSION = "cutsell.boundary_engine_pass.v1"
STAGE_POST_FREEZE = "post_freeze"
BOUNDARY_REASON_AUDIO_ENTRY = "tighten_audio_entry"
BOUNDARY_REASON_AUDIO_EXIT = "tighten_audio_exit"

# Audio-evidenced edge tightening: a source silence interval that overlaps a
# selected clip's leading/trailing edge is recording-process slack. The edge
# moves to the silence boundary (plus a natural pause pad) but never past
# the first/last aligned word, and only when the removal is material.
AUDIO_EDGE_PAD_SEC = 0.10
AUDIO_EDGE_MINIMUM_TRIM_SEC = 0.20
AUDIO_EDGE_OVERLAP_TOLERANCE_SEC = 0.08
AUDIO_EDGE_MINIMUM_REMAINING_SEC = 0.35

PHYSICAL_OWNERSHIP_CONTRACT: tuple[dict[str, str], ...] = (
    {"concern": "entry", "owner": "boundary_engine_pass", "when": "post_freeze",
     "evidence": "audio_silence_interval leading edge; multimodal reset evidence"},
    {"concern": "exit", "owner": "boundary_engine_pass; render.tighten_trailing_silence (last mechanical op, recorded)",
     "when": "post_freeze; render", "evidence": "audio_silence_interval trailing edge; renderer silencedetect"},
    {"concern": "interior_dead_air", "owner": "boundary_engine_pass (split_selected_interior_performance_gaps)",
     "when": "post_freeze", "evidence": f"audio_silence_interval >= {LONG_AUDIO_SILENCE_SEC:.2f}s (primary + relaxed floor)"},
    {"concern": "reset_debris", "owner": "boundary_engine_pass (performance-gap split); polish_human_boundaries_v5",
     "when": "post_freeze", "evidence": "local performance reset/break events"},
    {"concern": "micro_continuity", "owner": "post_selection_continuity_coalescer (draft); render_plan coalesce (mechanical)",
     "when": "draft; render", "evidence": "same-source adjacency without reset evidence"},
    {"concern": "renderer_mechanical_ops", "owner": "render.render_preview",
     "when": "render", "evidence": "trailing-silence tighten (recorded), 12 ms join fades, contiguous coalesce"},
)


def _events_for_source(diagnostics: Mapping, source_asset_id: str) -> tuple[dict, ...]:
    whole = diagnostics.get("whole_video_context") or {}
    for source in whole.get("sources") or ():
        if isinstance(source, dict) and source.get("source_asset_id") == source_asset_id:
            return tuple(event for event in (source.get("events") or ()) if isinstance(event, dict))
    return ()


def _silences(events: Iterable[dict]) -> tuple[tuple[float, float, float], ...]:
    out = []
    for event in events:
        if str(event.get("kind") or "").strip().lower() != AUDIO_SILENCE_EVENT_KIND:
            continue
        out.append((float(event.get("start") or 0.0), float(event.get("end") or 0.0), float(event.get("confidence") or 0.0)))
    return tuple(sorted(out))


def tighten_selected_audio_edges(
    selected: Iterable[DraftClip],
    diagnostics: Mapping,
    *,
    pad_sec: float = AUDIO_EDGE_PAD_SEC,
    minimum_trim_sec: float = AUDIO_EDGE_MINIMUM_TRIM_SEC,
    overlap_tolerance_sec: float = AUDIO_EDGE_OVERLAP_TOLERANCE_SEC,
) -> tuple[tuple[DraftClip, ...], tuple[dict, ...]]:
    """Move a clip's leading edge to the end of a source silence that covers
    it, and its trailing edge to the start of one that covers it. Words are
    the hard floor: an edge never crosses the first/last aligned word."""
    output: list[DraftClip] = []
    audit: list[dict] = []
    for clip in selected:
        words = tuple(sorted(tuple(clip.words), key=lambda w: (float(w.start), float(w.end))))
        silences = _silences(_events_for_source(diagnostics, clip.source_asset_id))
        start, end = float(clip.start), float(clip.end)
        new_start, new_end = start, end
        actions: list[dict] = []
        first_word_start = float(words[0].start) if words else None
        last_word_end = float(words[-1].end) if words else None

        for s_start, s_end, confidence in silences:
            # Leading edge: the silence begins at/before the clip start and
            # reaches materially into the clip.
            if s_start <= start + overlap_tolerance_sec and s_end > start + minimum_trim_sec:
                candidate = s_end - pad_sec
                if first_word_start is not None:
                    candidate = min(candidate, first_word_start)
                if candidate - start >= minimum_trim_sec and candidate > new_start:
                    new_start = candidate
                    actions.append({
                        "action": BOUNDARY_REASON_AUDIO_ENTRY, "silence_start": round(s_start, 3),
                        "silence_end": round(s_end, 3), "silence_confidence": round(confidence, 3),
                        "trim_sec": round(candidate - start, 3),
                    })
            # Trailing edge: the silence reaches the clip end and starts
            # materially before it.
            if s_end >= end - overlap_tolerance_sec and s_start < end - minimum_trim_sec:
                candidate = s_start + pad_sec
                if last_word_end is not None:
                    candidate = max(candidate, last_word_end)
                if end - candidate >= minimum_trim_sec and candidate < new_end:
                    new_end = candidate
                    actions.append({
                        "action": BOUNDARY_REASON_AUDIO_EXIT, "silence_start": round(s_start, 3),
                        "silence_end": round(s_end, 3), "silence_confidence": round(confidence, 3),
                        "trim_sec": round(end - candidate, 3),
                    })

        if not actions or new_end - new_start < AUDIO_EDGE_MINIMUM_REMAINING_SEC:
            output.append(clip)
            continue
        reason = clip.boundary_reason or (
            BOUNDARY_REASON_AUDIO_ENTRY if actions[0]["action"] == BOUNDARY_REASON_AUDIO_ENTRY else BOUNDARY_REASON_AUDIO_EXIT
        )
        output.append(replace(clip, start=new_start, end=new_end, boundary_reason=reason))
        audit.append({
            "authority": "boundary_engine_pass",
            "clip_id": clip.clip_id,
            "original_start": round(start, 3), "original_end": round(end, 3),
            "result_start": round(new_start, 3), "result_end": round(new_end, 3),
            "actions": actions,
            "semantic_membership_changed": False,
        })
    return tuple(output), tuple(audit)


def apply_post_freeze_boundary_pass(result: ProcessingResult) -> ProcessingResult:
    """Run the one Boundary pass on the frozen final KEEP set: evidence-based
    edge trim -> interior dead-air / performance-gap split -> audio-evidenced
    entry/exit tightening. Membership and the ordered token stream are never
    changed (verified afterwards by `enforce_selection_contract`)."""
    draft = result.draft
    if not hasattr(draft, "selected") or not draft.selected:
        return result
    diagnostics = dict(draft.diagnostics or {})
    selected = tuple(draft.selected)

    selected, edge_audit = trim_locked_selection_edges(selected, diagnostics)
    selected, interior_audit = split_selected_interior_performance_gaps(
        selected, diagnostics, include_rejected_diagnostics=True,
    )
    selected, audio_edge_audit = tighten_selected_audio_edges(selected, diagnostics)

    splits = [row for row in interior_audit if row.get("decision") == "split"]
    rejects = [row for row in interior_audit if row.get("decision") == "reject"]
    # The historical keys stay (the ladder workflow and the continuity
    # coalescer read them); they now describe the post-Freeze pass.
    diagnostics["post_selection_edge_only_boundary"] = [*(diagnostics.get("post_selection_edge_only_boundary") or ()), *edge_audit]
    diagnostics["post_selection_interior_gap_trim"] = [*(diagnostics.get("post_selection_interior_gap_trim") or ()), *splits]
    diagnostics["post_selection_interior_gap_trace"] = [*(diagnostics.get("post_selection_interior_gap_trace") or ()), *rejects]
    diagnostics["boundary_engine_pass"] = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE_POST_FREEZE,
        "selected_count_in": len(tuple(draft.selected)),
        "selected_count_out": len(selected),
        "edge_trim_count": len(edge_audit),
        "interior_split_count": len(splits),
        "interior_reject_count": len(rejects),
        "audio_entry_trim_count": sum(1 for row in audio_edge_audit for a in row["actions"] if a["action"] == BOUNDARY_REASON_AUDIO_ENTRY),
        "audio_exit_trim_count": sum(1 for row in audio_edge_audit for a in row["actions"] if a["action"] == BOUNDARY_REASON_AUDIO_EXIT),
        "audio_edge_rows": list(audio_edge_audit),
        "ownership_contract": [dict(row) for row in PHYSICAL_OWNERSHIP_CONTRACT],
    }
    return replace(result, draft=replace(draft, selected=selected, diagnostics=diagnostics))


# --- C-12 reconciliation ---------------------------------------------------------

def reconcile_silence_findings(
    draft,
    segments: Sequence,
    findings: Iterable,
    output_windows: Sequence[tuple[float, float]],
    *,
    silence_kind: str = "LINGERING_ACCIDENTAL_SILENCE",
) -> tuple[dict, ...]:
    """For each post-render silence finding (OUTPUT seconds) return the source
    range it maps to and what the source-level evidence and the Boundary
    pass knew about it. Observability only."""
    diagnostics = dict(getattr(draft, "diagnostics", None) or {})
    splits = [row for row in diagnostics.get("post_selection_interior_gap_trim") or () if isinstance(row, dict)]
    rejects = [row for row in diagnostics.get("post_selection_interior_gap_trace") or () if isinstance(row, dict)]
    pass_ran = isinstance(diagnostics.get("boundary_engine_pass"), dict)
    rows: list[dict] = []
    for finding in findings:
        if str(getattr(finding, "kind", "")) != silence_kind:
            continue
        f_start, f_end = float(finding.start), float(finding.end)
        located = None
        for segment, (win_start, win_end) in zip(segments, output_windows):
            if f_start < win_end and f_end > win_start:
                located = (segment, win_start)
                break
        if located is None:
            rows.append({"finding_start": f_start, "finding_end": f_end, "verdict": "no_segment_located"})
            continue
        segment, win_start = located
        src_start = float(segment.start) + (f_start - win_start)
        src_end = float(segment.start) + (f_end - win_start)
        events = _silences(_events_for_source(diagnostics, segment.source_asset_id))
        overlapping = [
            {"start": s, "end": e, "confidence": c}
            for s, e, c in events if s < src_end and e > src_start
        ]
        parent = getattr(segment, "parent_semantic_clip_id", None) or segment.clip_id
        clip_ids = {segment.clip_id, parent}
        clip_splits = [r for r in splits if r.get("parent_clip_id") in clip_ids]
        clip_rejects = [r for r in rejects if r.get("parent_clip_id") in clip_ids]
        if not overlapping:
            verdict = "source_not_measured" if pass_ran else "source_not_measured_pass_not_run"
        elif clip_splits:
            verdict = "source_measured_and_split_elsewhere"
        elif clip_rejects:
            verdict = "source_measured_trimmer_rejected:" + ",".join(sorted({str(r.get("reason")) for r in clip_rejects}))
        elif not pass_ran:
            verdict = "source_measured_pass_not_run"
        else:
            verdict = "source_measured_no_trimmer_decision"
        rows.append({
            "finding_start": round(f_start, 3), "finding_end": round(f_end, 3),
            "clip_id": segment.clip_id, "parent_semantic_clip_id": parent,
            "source_asset_id": segment.source_asset_id,
            "source_start": round(src_start, 3), "source_end": round(src_end, 3),
            "source_silence_events_overlapping": overlapping,
            "trimmer_split_rows": clip_splits, "trimmer_reject_rows": clip_rejects,
            "verdict": verdict,
        })
    return tuple(rows)
