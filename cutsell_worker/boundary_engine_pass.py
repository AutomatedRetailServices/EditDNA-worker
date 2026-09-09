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

## PHYSICAL CLEANUP OWNERSHIP CONTRACT (D-097.E, visual consumption D-116)

| Physical concern | Owner | When | Evidence | Never |
|---|---|---|---|---|
| ENTRY (leading non-speech) | BoundaryEngine pass (this module) | after Selection Freeze | audio_silence_interval events at the leading edge; D-115 positioned visual/performance evidence classified ENTRY by `positioned_performance_evidence.classify_event_zone` (edge-only trim, D-116) | remove a word; change membership; trim a DELIVERY-zone event |
| EXIT (trailing non-speech) | BoundaryEngine pass; renderer `tighten_trailing_silence` is the LAST mechanical op, recorded per segment | after Freeze; at render | audio_silence_interval at the trailing edge; renderer silencedetect on the exact segment; D-115 positioned visual/performance evidence classified EXIT (edge-only trim, D-116) | remove a word; trim a DELIVERY-zone event |
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

## Visual CASE A edge consumption (D-116)

D-115 built the ONE canonical ENTRY/DELIVERY/EXIT temporal interpretation
(`positioned_performance_evidence.py`) but wired nothing to consume it.
`tighten_selected_visual_edges` is that consumer, CASE A only (D-111's
doctrine: a defect entirely outside the measured DELIVERY span is
BoundaryEngine's to trim; a defect overlapping DELIVERY is BestTake's, not
implemented here). It calls `compute_delivery_span`/`classify_event_zone`
directly on the selected clip's own words/events -- it never recomputes
ENTRY/DELIVERY/EXIT or "event overlap" itself, per D-115/D-116's shared-
temporal-authority rule. A DELIVERY-zone event that is genuinely embedded
in DELIVERY (does not straddle a boundary) is never trimmed -- that
remains BestTake/DeliveryScorer evidence, D-117 territory, not
implemented here.

## Partial-edge trim for a straddling event (D-177)

D-176's forensic proved a real, reproducible shape (three independent
Video00 RAWs): a selected candidate's own DELIVERY content is validated
correct by both quality references, but a visual/performance event
straddles the DELIVERY boundary by only a small amount -- `classify_
event_zone` already reports this exactly (`starts_before_delivery`/
`ends_after_delivery`, D-115's own straddle bookkeeping, never
recomputed here) -- and the pre-D-177 code below always treated ANY
overlap, however small, as a full DELIVERY-owned defect with zero trim
available. D-177 closes that named gap with ONE bounded rule, using NO
new numeric constant: when a straddling event's own portion INSIDE
DELIVERY is no larger than this module's existing `AUDIO_EDGE_OVERLAP_
TOLERANCE_SEC` (the same edge-touch tolerance already governing every
other "is this event close enough to the edge to trim" decision in this
module), the edge moves to the measured DELIVERY boundary itself
(`delivery_span.start` for an entry straddle, `delivery_span.end` for an
exit straddle) -- the exact same hard floor the pre-existing pure ENTRY/
EXIT loops already enforce, so the DELIVERY-side sliver (by definition no
larger than the tolerance) is always LEFT UNTOUCHED, never shaved: this
mechanism never removes so much as a millisecond of a real word,
satisfying the same speech-safety invariant as every other trim in this
module. A straddling event whose inside-DELIVERY portion
EXCEEDS that tolerance is a real DELIVERY defect and is left exactly as
before (`BOUNDARY_REASON_VISUAL_DELIVERY_OVERLAP_NO_TRIM`, same reason,
same behavior, same existing D-116 tests). An event that straddles BOTH
DELIVERY edges at once (wider than the whole measured span) is ambiguous
and always fails open, untrimmed. `AUDIO_EDGE_MINIMUM_REMAINING_SEC`
(the same existing per-clip floor) still applies afterward, exactly as
for every other trim this module performs. Entry/exit only: no interior
split, no composite, no membership change -- the ordered token stream
is untouched, verified by the same `enforce_selection_contract` call
this pass already runs under.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Mapping, Sequence

from .contracts import DraftClip, ProcessingResult
from .positioned_performance_evidence import (
    LOCAL_PERFORMANCE_EVENT_KINDS,
    ZONE_DELIVERY,
    ZONE_ENTRY,
    ZONE_EXIT,
    classify_event_zone,
    compute_delivery_span,
)
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

# D-116: visual CASE A edge trimming. Deliberately reuses the two audio-edge
# invariants above (edge-touch tolerance, per-clip remaining-duration floor)
# rather than inventing new visual-specific timing constants -- this task's
# own explicit "no fixed visual trim padding unless an already-existing
# Boundary invariant supplies such a value" constraint. No pad is added past
# a visual event's own boundary (unlike the audio path's natural-pause pad):
# the event's own start/end IS the evidence, clamped only by the measured
# DELIVERY floor.
BOUNDARY_REASON_VISUAL_ENTRY_EDGE_TRIM = "visual_entry_edge_trim"
BOUNDARY_REASON_VISUAL_EXIT_EDGE_TRIM = "visual_exit_edge_trim"
BOUNDARY_REASON_VISUAL_DELIVERY_OVERLAP_NO_TRIM = "visual_event_overlaps_delivery_no_trim"
BOUNDARY_REASON_VISUAL_NOT_AT_EDGE = "visual_event_not_at_edge"
BOUNDARY_REASON_VISUAL_BLOCKED_BY_DELIVERY_FLOOR = "visual_trim_blocked_by_delivery_floor"
BOUNDARY_REASON_VISUAL_TRIM_UNAVAILABLE = "visual_trim_unavailable"

# D-177: partial-edge trim for a visual/performance event that straddles the
# DELIVERY boundary by no more than this module's own existing
# AUDIO_EDGE_OVERLAP_TOLERANCE_SEC -- see module docstring's own "Partial-
# edge trim for a straddling event (D-177)" section. No new timing constant.
BOUNDARY_REASON_VISUAL_ENTRY_PARTIAL_EDGE_TRIM = "visual_entry_partial_edge_trim"
BOUNDARY_REASON_VISUAL_EXIT_PARTIAL_EDGE_TRIM = "visual_exit_partial_edge_trim"
# An event straddling BOTH DELIVERY edges at once (wider than the whole
# measured span) -- ambiguous, always fails open, never trimmed.
BOUNDARY_REASON_VISUAL_AMBIGUOUS_STRADDLE_NO_TRIM = "visual_ambiguous_straddle_no_trim"

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


def _visual_events_for_clip(clip: DraftClip, diagnostics: Mapping) -> tuple[dict, ...]:
    """The real, position-timestamped D-114 local-performance events that
    fall inside `clip`'s own currently-selected source window. Uses the
    same window-overlap test D-115's own `build_positioned_performance_
    evidence` already uses (`event.end > start and event.start < end`) --
    mechanical event-selection bookkeeping, not the ENTRY/DELIVERY/EXIT
    zone classification itself, which is never recomputed here (see
    `classify_event_zone`, imported from `positioned_performance_
    evidence.py`)."""
    start, end = float(clip.start), float(clip.end)
    out = []
    for event in _events_for_source(diagnostics, clip.source_asset_id):
        kind = str(event.get("kind") or "").strip()
        if kind not in LOCAL_PERFORMANCE_EVENT_KINDS:
            continue
        e_start = float(event.get("start") or 0.0)
        e_end = float(event.get("end") or 0.0)
        if not (e_end > start and e_start < end):
            continue
        out.append({
            "kind": kind, "start": e_start, "end": e_end,
            "confidence": float(event.get("confidence") or 0.0),
        })
    return tuple(sorted(out, key=lambda item: (item["start"], item["end"], item["kind"])))


def _visual_row(
    clip: DraftClip, event: dict, zone: str, overlaps: bool | None, delivery_span,
    *, reason: str, trim_side: str | None, trim_applied: bool,
    old_start: float | None = None, new_start_value: float | None = None,
    old_end: float | None = None, new_end_value: float | None = None,
    inside_delivery_overlap_sec: float | None = None,
) -> dict:
    return {
        "authority": "boundary_engine_pass",
        "clip_id": clip.clip_id,
        "event_kind": event["kind"],
        "event_start": round(event["start"], 3),
        "event_end": round(event["end"], 3),
        "event_confidence": round(event["confidence"], 4),
        "zone": zone,
        "overlaps_delivery": overlaps,
        "delivery_start": round(delivery_span.start, 3) if delivery_span.available else None,
        "delivery_end": round(delivery_span.end, 3) if delivery_span.available else None,
        "delivery_span_available": delivery_span.available,
        "old_start": round(old_start, 3) if old_start is not None else round(float(clip.start), 3),
        "new_start": round(new_start_value, 3) if new_start_value is not None else round(float(clip.start), 3),
        "old_end": round(old_end, 3) if old_end is not None else round(float(clip.end), 3),
        "new_end": round(new_end_value, 3) if new_end_value is not None else round(float(clip.end), 3),
        "trim_side": trim_side,
        "trim_applied": trim_applied,
        "reason": reason,
        "evidence_source": "local_performance",
        "semantic_membership_changed": False,
        # D-177: the portion of a STRADDLING event that falls inside the
        # measured DELIVERY span -- None for every non-straddle row (the
        # existing D-116 rows never set this). Never a fabricated value:
        # only ever the real event/delivery-span arithmetic.
        "inside_delivery_overlap_sec": (
            round(inside_delivery_overlap_sec, 3) if inside_delivery_overlap_sec is not None else None
        ),
    }


def tighten_selected_visual_edges(
    selected: Iterable[DraftClip],
    diagnostics: Mapping,
) -> tuple[tuple[DraftClip, ...], tuple[dict, ...]]:
    """D-116 CASE A: consume D-115's canonical ENTRY/DELIVERY/EXIT evidence
    to tighten a selected clip's leading/trailing edge past a real,
    positioned visual/performance defect that lies entirely outside the
    measured DELIVERY span. Never trims a DELIVERY-zone event (including
    any event that straddles a boundary -- D-115 classifies any overlap as
    DELIVERY) and never crosses `delivery_span.start`/`.end`, the hard
    safety floor. Edge-tightening only: an event that doesn't reach the
    clip's own current edge (within `AUDIO_EDGE_OVERLAP_TOLERANCE_SEC`,
    reused rather than a new constant) is left untouched -- this module
    never splits an interior event into a hole."""
    output: list[DraftClip] = []
    audit: list[dict] = []
    for clip in selected:
        delivery_span = compute_delivery_span(clip.words)
        events = _visual_events_for_clip(clip, diagnostics)
        if not events:
            output.append(clip)
            continue

        start, end = float(clip.start), float(clip.end)
        classified = [
            (event, *classify_event_zone(event["start"], event["end"], delivery_span))
            for event in events
        ]
        rows: list[dict] = []
        new_start, new_end = start, end
        entry_trim_applied = False
        exit_trim_applied = False

        if not delivery_span.available:
            # D-115 never fabricates a delivery span; without one there is
            # no safety floor to trim against, so no visual trim is ever
            # applied here -- diagnostics only.
            for event, zone, overlaps, before, after in classified:
                rows.append(_visual_row(
                    clip, event, zone, overlaps, delivery_span,
                    reason=BOUNDARY_REASON_VISUAL_TRIM_UNAVAILABLE, trim_side=None, trim_applied=False,
                ))
        else:
            for event, zone, overlaps, before, after in classified:
                if zone == ZONE_DELIVERY:
                    if before and after:
                        # D-177: straddles BOTH DELIVERY edges at once (an
                        # event wider than the whole measured span) --
                        # ambiguous, fails open, never trimmed.
                        rows.append(_visual_row(
                            clip, event, zone, overlaps, delivery_span,
                            reason=BOUNDARY_REASON_VISUAL_AMBIGUOUS_STRADDLE_NO_TRIM,
                            trim_side=None, trim_applied=False,
                        ))
                    elif not before and not after:
                        # Genuinely embedded in DELIVERY -- unchanged D-116
                        # behavior, same reason, same tests.
                        rows.append(_visual_row(
                            clip, event, zone, overlaps, delivery_span,
                            reason=BOUNDARY_REASON_VISUAL_DELIVERY_OVERLAP_NO_TRIM,
                            trim_side=None, trim_applied=False,
                        ))
                    # else: a one-sided straddle (before XOR after) -- its
                    # own eligibility (D-177) is decided below, alongside
                    # the ENTRY/EXIT edge loops, so it can chain correctly
                    # off whatever those loops already moved the running
                    # edge to.

            # ENTRY -- process outermost-in (ascending start) so contiguous
            # entry events correctly chain off the running (already-
            # tightened-this-loop) leading edge.
            entry_events = sorted(
                (item for item in classified if item[1] == ZONE_ENTRY),
                key=lambda item: (item[0]["start"], item[0]["end"]),
            )
            for event, zone, overlaps, before, after in entry_events:
                touches = event["start"] <= new_start + AUDIO_EDGE_OVERLAP_TOLERANCE_SEC and event["end"] > new_start
                if not touches:
                    rows.append(_visual_row(
                        clip, event, zone, overlaps, delivery_span,
                        reason=BOUNDARY_REASON_VISUAL_NOT_AT_EDGE, trim_side="ENTRY", trim_applied=False,
                    ))
                    continue
                candidate = min(event["end"], delivery_span.start)  # hard floor: never past delivery start
                if candidate <= new_start:
                    rows.append(_visual_row(
                        clip, event, zone, overlaps, delivery_span,
                        reason=BOUNDARY_REASON_VISUAL_BLOCKED_BY_DELIVERY_FLOOR, trim_side="ENTRY", trim_applied=False,
                    ))
                    continue
                old_start = new_start
                new_start = candidate
                entry_trim_applied = True
                rows.append(_visual_row(
                    clip, event, zone, overlaps, delivery_span,
                    reason=BOUNDARY_REASON_VISUAL_ENTRY_EDGE_TRIM, trim_side="ENTRY", trim_applied=True,
                    old_start=old_start, new_start_value=new_start,
                ))

            # EXIT -- process outermost-in (descending end) so contiguous
            # exit events correctly chain off the running trailing edge.
            exit_events = sorted(
                (item for item in classified if item[1] == ZONE_EXIT),
                key=lambda item: (item[0]["end"], item[0]["start"]),
                reverse=True,
            )
            for event, zone, overlaps, before, after in exit_events:
                touches = event["end"] >= new_end - AUDIO_EDGE_OVERLAP_TOLERANCE_SEC and event["start"] < new_end
                if not touches:
                    rows.append(_visual_row(
                        clip, event, zone, overlaps, delivery_span,
                        reason=BOUNDARY_REASON_VISUAL_NOT_AT_EDGE, trim_side="EXIT", trim_applied=False,
                    ))
                    continue
                candidate = max(event["start"], delivery_span.end)  # hard floor: never past delivery end
                if candidate >= new_end:
                    rows.append(_visual_row(
                        clip, event, zone, overlaps, delivery_span,
                        reason=BOUNDARY_REASON_VISUAL_BLOCKED_BY_DELIVERY_FLOOR, trim_side="EXIT", trim_applied=False,
                    ))
                    continue
                old_end = new_end
                new_end = candidate
                exit_trim_applied = True
                rows.append(_visual_row(
                    clip, event, zone, overlaps, delivery_span,
                    reason=BOUNDARY_REASON_VISUAL_EXIT_EDGE_TRIM, trim_side="EXIT", trim_applied=True,
                    old_end=old_end, new_end_value=new_end,
                ))

            # D-177 -- ENTRY partial-edge trim: a DELIVERY-zone event that
            # starts before delivery_span.start and ends inside it (before
            # XOR after in classify_event_zone's own terms). Runs AFTER the
            # pure-ENTRY loop above so it chains off whatever edge that
            # loop already established. Eligible only when the event's own
            # portion INSIDE DELIVERY is no larger than this module's
            # existing AUDIO_EDGE_OVERLAP_TOLERANCE_SEC -- no new constant.
            entry_straddle_events = sorted(
                (item for item in classified if item[1] == ZONE_DELIVERY and item[3] and not item[4]),
                key=lambda item: (item[0]["start"], item[0]["end"]),
            )
            for event, zone, overlaps, before, after in entry_straddle_events:
                # Materiality first (D-177's own eligibility test): a
                # straddling event whose inside-DELIVERY portion exceeds
                # this module's existing edge-touch tolerance is real
                # DELIVERY evidence regardless of where the clip's own
                # current edge happens to sit -- same reason D-116 already
                # uses for any DELIVERY-owned event, same D-116 tests.
                inside_overlap = max(0.0, event["end"] - delivery_span.start)
                if inside_overlap > AUDIO_EDGE_OVERLAP_TOLERANCE_SEC:
                    rows.append(_visual_row(
                        clip, event, zone, overlaps, delivery_span,
                        reason=BOUNDARY_REASON_VISUAL_DELIVERY_OVERLAP_NO_TRIM, trim_side="ENTRY", trim_applied=False,
                        inside_delivery_overlap_sec=inside_overlap,
                    ))
                    continue
                touches = event["start"] <= new_start + AUDIO_EDGE_OVERLAP_TOLERANCE_SEC and event["end"] > new_start
                if not touches:
                    rows.append(_visual_row(
                        clip, event, zone, overlaps, delivery_span,
                        reason=BOUNDARY_REASON_VISUAL_NOT_AT_EDGE, trim_side="ENTRY", trim_applied=False,
                        inside_delivery_overlap_sec=inside_overlap,
                    ))
                    continue
                # Same hard floor as the pure-ENTRY loop above: never past
                # delivery_span.start. For any genuine straddle this always
                # resolves to exactly delivery_span.start (the event's own
                # end is, by the materiality gate above, no more than
                # `tolerance` past it) -- the DELIVERY-side sliver itself is
                # left untouched, never shaved, per this module's own
                # speech-safety invariant (words are the hard floor).
                candidate = min(event["end"], delivery_span.start)
                if candidate <= new_start:
                    rows.append(_visual_row(
                        clip, event, zone, overlaps, delivery_span,
                        reason=BOUNDARY_REASON_VISUAL_BLOCKED_BY_DELIVERY_FLOOR, trim_side="ENTRY", trim_applied=False,
                        inside_delivery_overlap_sec=inside_overlap,
                    ))
                    continue
                old_start = new_start
                new_start = candidate
                entry_trim_applied = True
                rows.append(_visual_row(
                    clip, event, zone, overlaps, delivery_span,
                    reason=BOUNDARY_REASON_VISUAL_ENTRY_PARTIAL_EDGE_TRIM, trim_side="ENTRY", trim_applied=True,
                    old_start=old_start, new_start_value=new_start, inside_delivery_overlap_sec=inside_overlap,
                ))

            # D-177 -- EXIT partial-edge trim, symmetric to the ENTRY case
            # above.
            exit_straddle_events = sorted(
                (item for item in classified if item[1] == ZONE_DELIVERY and item[4] and not item[3]),
                key=lambda item: (item[0]["end"], item[0]["start"]),
                reverse=True,
            )
            for event, zone, overlaps, before, after in exit_straddle_events:
                inside_overlap = max(0.0, delivery_span.end - event["start"])
                if inside_overlap > AUDIO_EDGE_OVERLAP_TOLERANCE_SEC:
                    rows.append(_visual_row(
                        clip, event, zone, overlaps, delivery_span,
                        reason=BOUNDARY_REASON_VISUAL_DELIVERY_OVERLAP_NO_TRIM, trim_side="EXIT", trim_applied=False,
                        inside_delivery_overlap_sec=inside_overlap,
                    ))
                    continue
                touches = event["end"] >= new_end - AUDIO_EDGE_OVERLAP_TOLERANCE_SEC and event["start"] < new_end
                if not touches:
                    rows.append(_visual_row(
                        clip, event, zone, overlaps, delivery_span,
                        reason=BOUNDARY_REASON_VISUAL_NOT_AT_EDGE, trim_side="EXIT", trim_applied=False,
                        inside_delivery_overlap_sec=inside_overlap,
                    ))
                    continue
                # Symmetric hard floor: never past delivery_span.end -- the
                # DELIVERY-side sliver is left untouched, same speech-safety
                # invariant as the entry case above.
                candidate = max(event["start"], delivery_span.end)
                if candidate >= new_end:
                    rows.append(_visual_row(
                        clip, event, zone, overlaps, delivery_span,
                        reason=BOUNDARY_REASON_VISUAL_BLOCKED_BY_DELIVERY_FLOOR, trim_side="EXIT", trim_applied=False,
                        inside_delivery_overlap_sec=inside_overlap,
                    ))
                    continue
                old_end = new_end
                new_end = candidate
                exit_trim_applied = True
                rows.append(_visual_row(
                    clip, event, zone, overlaps, delivery_span,
                    reason=BOUNDARY_REASON_VISUAL_EXIT_PARTIAL_EDGE_TRIM, trim_side="EXIT", trim_applied=True,
                    old_end=old_end, new_end_value=new_end, inside_delivery_overlap_sec=inside_overlap,
                ))

        audit.extend(rows)

        if (new_start == start and new_end == end) or new_end - new_start < AUDIO_EDGE_MINIMUM_REMAINING_SEC:
            output.append(clip)
            continue

        reason = clip.boundary_reason or (
            BOUNDARY_REASON_VISUAL_ENTRY_EDGE_TRIM if entry_trim_applied else BOUNDARY_REASON_VISUAL_EXIT_EDGE_TRIM
        )
        output.append(replace(clip, start=new_start, end=new_end, boundary_reason=reason))
    return tuple(output), tuple(audit)


def _partial_edge_trim_summary(
    before_by_id: Mapping[str, DraftClip],
    after_by_id: Mapping[str, DraftClip],
    visual_edge_audit: Iterable[dict],
) -> tuple[dict, ...]:
    """D-177: one compact, per-clip summary row -- only for a clip where at
    least one straddling event was actually evaluated for partial-edge
    eligibility (`inside_delivery_overlap_sec is not None` on its own
    audit row). Never dumps a transcript; reuses the same real numbers
    the per-event `visual_edge_audit` rows already carry."""
    by_clip: dict[str, list[dict]] = {}
    for row in visual_edge_audit:
        if row.get("inside_delivery_overlap_sec") is None:
            continue
        by_clip.setdefault(row["clip_id"], []).append(row)

    def _pick(rows: list[dict]) -> dict | None:
        return next((r for r in rows if r.get("trim_applied")), rows[0] if rows else None)

    summary: list[dict] = []
    for clip_id, rows in by_clip.items():
        entry_rows = [r for r in rows if r.get("trim_side") == "ENTRY"]
        exit_rows = [r for r in rows if r.get("trim_side") == "EXIT"]
        entry_pick = _pick(entry_rows)
        exit_pick = _pick(exit_rows)
        entry_applied = bool(entry_pick and entry_pick.get("trim_applied"))
        exit_applied = bool(exit_pick and exit_pick.get("trim_applied"))
        before_clip = before_by_id.get(clip_id)
        after_clip = after_by_id.get(clip_id)
        reasons = sorted({r["reason"] for r in (entry_pick, exit_pick) if r})
        summary.append({
            "clip_id": clip_id,
            "partial_edge_trim_evaluated": True,
            "partial_edge_trim_applied": entry_applied or exit_applied,
            "entry_partial_edge_trim_applied": entry_applied,
            "exit_partial_edge_trim_applied": exit_applied,
            "entry_event_kind": entry_pick["event_kind"] if entry_pick else None,
            "exit_event_kind": exit_pick["event_kind"] if exit_pick else None,
            "entry_inside_delivery_overlap_sec": entry_pick["inside_delivery_overlap_sec"] if entry_pick else None,
            "exit_inside_delivery_overlap_sec": exit_pick["inside_delivery_overlap_sec"] if exit_pick else None,
            "boundary_before_start": round(float(before_clip.start), 3) if before_clip else None,
            "boundary_before_end": round(float(before_clip.end), 3) if before_clip else None,
            "boundary_after_start": round(float(after_clip.start), 3) if after_clip else None,
            "boundary_after_end": round(float(after_clip.end), 3) if after_clip else None,
            "partial_edge_trim_reason": "+".join(reasons) if reasons else None,
        })
    return tuple(summary)


def apply_post_freeze_boundary_pass(result: ProcessingResult) -> ProcessingResult:
    """Run the one Boundary pass on the frozen final KEEP set: evidence-based
    edge trim -> interior dead-air / performance-gap split -> audio-evidenced
    entry/exit tightening -> D-116 visual (CASE A) entry/exit tightening.
    Membership and the ordered token stream are never changed (verified
    afterwards by `enforce_selection_contract`)."""
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
    # D-116: visual CASE A edge trimming runs LAST in the same pass, on
    # whatever audio-tightening already left, so audio and visual evidence
    # combine through this one existing Boundary contract rather than
    # competing (no second pass). D-177's own partial-edge trim runs
    # inside this same call -- see tighten_selected_visual_edges's own
    # docstring.
    pre_visual_selected = {clip.clip_id: clip for clip in selected}
    selected, visual_edge_audit = tighten_selected_visual_edges(selected, diagnostics)
    post_visual_selected = {clip.clip_id: clip for clip in selected}
    partial_edge_trim_summary = _partial_edge_trim_summary(
        pre_visual_selected, post_visual_selected, visual_edge_audit,
    )

    splits = [row for row in interior_audit if row.get("decision") == "split"]
    rejects = [row for row in interior_audit if row.get("decision") == "reject"]
    # The historical keys stay (the ladder workflow and the continuity
    # coalescer read them); they now describe the post-Freeze pass.
    diagnostics["post_selection_edge_only_boundary"] = [*(diagnostics.get("post_selection_edge_only_boundary") or ()), *edge_audit]
    diagnostics["post_selection_interior_gap_trim"] = [*(diagnostics.get("post_selection_interior_gap_trim") or ()), *splits]
    diagnostics["post_selection_interior_gap_trace"] = [*(diagnostics.get("post_selection_interior_gap_trace") or ()), *rejects]
    diagnostics["boundary_visual_edge_trim"] = [*(diagnostics.get("boundary_visual_edge_trim") or ()), *visual_edge_audit]
    # D-177: the compact per-clip partial-edge-trim summary -- separate key
    # from the existing D-116 per-event `boundary_visual_edge_trim` list
    # above (which already carries every straddle sub-decision too, via
    # its own `inside_delivery_overlap_sec` field) so no existing consumer
    # of that list's row shape is affected.
    diagnostics["boundary_partial_edge_trim"] = [
        *(diagnostics.get("boundary_partial_edge_trim") or ()), *partial_edge_trim_summary,
    ]
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
        "visual_entry_trim_count": sum(1 for row in visual_edge_audit if row.get("reason") == BOUNDARY_REASON_VISUAL_ENTRY_EDGE_TRIM and row.get("trim_applied")),
        "visual_exit_trim_count": sum(1 for row in visual_edge_audit if row.get("reason") == BOUNDARY_REASON_VISUAL_EXIT_EDGE_TRIM and row.get("trim_applied")),
        "visual_edge_rows": list(visual_edge_audit),
        # D-177 run-level counts (module docstring's own "Partial-edge trim
        # for a straddling event" section).
        "partial_edge_trim_evaluated_count": len(partial_edge_trim_summary),
        "partial_edge_trim_applied_count": sum(1 for row in partial_edge_trim_summary if row["partial_edge_trim_applied"]),
        "entry_partial_edge_trim_count": sum(1 for row in partial_edge_trim_summary if row["entry_partial_edge_trim_applied"]),
        "exit_partial_edge_trim_count": sum(1 for row in partial_edge_trim_summary if row["exit_partial_edge_trim_applied"]),
        "partial_edge_trim_rows": list(partial_edge_trim_summary),
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
