"""D-142 Phase 1 -- Dialogue/Pacing Transition contract and planner.

Canonical order (D-098 Section 11 / D-129, restated here for this module's
own placement, never renumbered):

    Selection / BestTake -> Selection Freeze -> Boundary (boundary_engine_
    pass.py) -> Dialogue/Pacing Transition (THIS MODULE) -> Renderer
    (render_plan.py / render.py)

## What this module is

A PLANNING/DIAGNOSTICS layer, not a second editor and not a second render
pipeline. It consumes the already-frozen, already-Boundary-approved
`draft.selected` clip sequence plus the canonical D-134 `dialogue_overlap_
enabled` permission, and produces one `DialogueTransitionPlan` row per
adjacent pair of selected clips describing HOW that join is realized. It
never reopens BestTake, never changes membership/order/proposition
identity, never invents a source span outside a Boundary-approved edge, and
never mutates `draft.selected` -- Phase 1 output is attached to
`draft.diagnostics` only.

## Renderer support matrix (STATIC FORENSIC, D-142, verified by direct
inspection of `render.py`'s `_concat_render_command`/`render_preview`)

- `HARD_CUT` -- SUPPORTED_NOW. This is literally the renderer's existing
  baseline: every segment is seeked/trimmed independently
  (`trim=duration=exact` video, `apad`+`atrim` audio to the SAME exact
  duration) and concatenated back to back with the `concat` filter, zero
  inserted silence between segments, only a 12 ms afade in/out on each
  segment's OWN edges for click-avoidance (`_audio_join_fade_filters`,
  D-094.3 F14). Audio and video always share one matched cut instant per
  segment -- there is no separate audio/video offset today.
- `TIGHT_CUT` -- SUPPORTED_NOW. Two independent, already-proven mechanisms
  already remove safe dead air at a join: (1) `boundary_engine_pass.py`'s
  post-Freeze `tighten_selected_audio_edges`/`tighten_selected_visual_
  edges`, driven by real measured `audio_silence_interval` events and D-115
  positioned visual evidence, never crossing a DELIVERY-zone event or the
  per-clip `AUDIO_EDGE_MINIMUM_REMAINING_SEC` floor; (2) `render.
  tighten_trailing_silence`, a real `ffmpeg silencedetect` probe run at
  render time as the LAST mechanical op (recorded per segment, D-097.E
  ownership table). This module does not re-implement either mechanism --
  it reads Boundary's OWN already-applied trim evidence (Section
  "Deterministic TIGHT_CUT evidence" below) to LABEL a join TIGHT_CUT and
  report `gap_removed_duration`, purely attributive, never a second
  physical operation.
- `J_CUT` -- REQUIRES_RENDERER_EXTENSION. `_concat_render_command` gives
  every segment's audio and video streams the SAME exact trimmed duration
  (`trim=duration=exact` / `atrim=duration=exact`), synced to start
  together; there is no mechanism today for clip B's audio to begin before
  clip A's video ends without restructuring the filter graph across a
  segment boundary (e.g. `adelay`/`amix` spanning two segments).
- `L_CUT` -- REQUIRES_RENDERER_EXTENSION. Same reasoning, mirrored: clip
  A's audio continuing into clip B's video window requires an
  audio stream whose duration exceeds its own segment's trim window, which
  the current per-segment independent-trim model does not support.
- `MICRO_AUDIO_OVERLAP` -- REQUIRES_RENDERER_EXTENSION. Adjacent segments'
  audio streams are concatenated end-to-end (`concat` filter), never mixed;
  a true overlap needs `acrossfade`/`amix` between two neighboring audio
  streams, which is not present anywhere in `render.py` today.

## Gap/silence forensic (D-142, verified by direct inspection)

- No literal inter-segment silence gap is inserted by the current
  renderer: the `concat` filter joins `[v_i][a_i]` directly to
  `[v_{i+1}][a_{i+1}]` with nothing between them (confirmed HARD_CUT is
  already the baseline, not a gap-then-cut).
- A tiny, sub-frame TECHNICAL padding can occur per segment: `apad=
  whole_dur=exact` pads a segment's raw trimmed audio up to `exact`
  (`rendered_segment_duration_sec`, which rounds UP to a whole output
  frame at `fps`) when the raw audio is fractionally shorter -- at most
  `1/fps` (~33 ms at the default 30 fps), inaudible, purely a frame-
  rounding artifact, not editorial dead air, unchanged since D-097.2.
- No PTS-reset/encoder timing gap remains: the pre-D-097.2 renderer used
  the concat DEMUXER (stream-copy of independently-encoded parts, each
  carrying its own AAC priming frame) which DID advance the real output
  timeline past the plan (D-097.2's own root-cause). The current single
  `ffmpeg -filter_complex` pass with `setpts=PTS-STARTPTS`/`asetpts=
  PTS-STARTPTS` per segment feeding one `concat` filter has no such gap --
  this was proven, not merely asserted, by D-097.2's own fix and its
  regression coverage.
- Genuine SOURCE-LEVEL pauses can remain INSIDE a selected clip's own span
  (interior dead air) -- these are Boundary's `split_selected_interior_
  performance_gaps` territory, not this module's; Dialogue/Pacing
  Transition only ever looks at the two EDGES of an adjacent pair, never a
  clip's interior.

## Audio continuity (D-142, verified by direct inspection)

Sample rate (48000), channel layout (stereo), and audio codec (AAC,
160k) are all normalized once via `aformat=...`/the final `-c:a aac`
encode -- consistent across every segment already. PTS continuity is
exact (see above). Click-avoidance uses a 12 ms afade in/out on each
segment's own edges (D-094.3 F14) -- this is NOT a crossfade between
neighbors (no `acrossfade`/`amix` exists in the current filter graph),
which is exactly why `MICRO_AUDIO_OVERLAP` is REQUIRES_RENDERER_EXTENSION
above. No new broad DSP work is introduced by this module.

## Deterministic TIGHT_CUT evidence (no new threshold invented)

Per D-142's own instruction to prefer an existing canonical threshold over
inventing one: this module's TIGHT_CUT determination is not a new
silence/gap measurement at all -- it is a read of `boundary_engine_pass`'s
OWN diagnostics (`audio_edge_rows`, `visual_edge_rows`), which already
applied `post_selection_interior_gap_trim.LONG_AUDIO_SILENCE_SEC` (interior
floor) and this module's sibling constants `AUDIO_EDGE_MINIMUM_TRIM_SEC`/
`AUDIO_EDGE_MINIMUM_REMAINING_SEC` (edge floors) to decide, per clip, per
edge, whether a trim was safe and material enough to apply. If Boundary
already recorded a real, applied trim at a clip's EXIT edge (the left
member of a pair) or a clip's ENTRY edge (the right member), that
specific join is labelled TIGHT_CUT with `gap_removed_duration` equal to
the sum of those already-applied trim amounts; otherwise it is HARD_CUT.
This can never disagree with Boundary (it only reads Boundary's own
completed work) and can never touch a DELIVERY-zone event (Boundary
itself never produces a trim row for one -- `BOUNDARY_REASON_VISUAL_
DELIVERY_OVERLAP_NO_TRIM` rows carry no `trim_applied=True`).

## Overlap toggle contract (D-134's `dialogue_overlap_enabled`, D-098
Section 11.5)

`dialogue_overlap_enabled=False` still allows HARD_CUT/TIGHT_CUT (neither
involves intentional cross-boundary dialogue overlap) and blocks every
overlap mode outright -- recorded as `fallback_reason=
"overlap_disabled"`, never even hypothetically considered.
`dialogue_overlap_enabled=True` is a PERMISSION, never a command: this
module still never selects `J_CUT`/`L_CUT`/`MICRO_AUDIO_OVERLAP` as the
actual `mode` in Phase 1 (the renderer cannot execute them --
REQUIRES_RENDERER_EXTENSION above), so the chosen mode is unchanged; the
diagnostic `fallback_reason` becomes `"renderer_extension_required"`
instead, honestly distinguishing "the user disabled overlap" from "overlap
is permitted but the renderer cannot do it yet" without ever faking
execution of an unsupported mode. This module reads ONLY the canonical
`dialogue_overlap_enabled` field passed in by its caller -- it never reads
the legacy `ProcessingRequest.audio_overlap` field directly (D-134's own
normalization boundary in `serde.py` remains the only place legacy input
is consumed).

## No editorial authority

This module imports nothing from `deterministic_best_take_authority.py`,
`multimodal_besttake_fallback.py`, `multimodal_besttake_arbiter.py`,
`realization_resolver.py`, or any AttemptReconstructor/IdeaClusterer/
BestTake module, and writes to no field any of those modules read. It
consumes only `DraftClip.clip_id/start/end` (already-frozen, Boundary-
approved) and `boundary_engine_pass`'s own diagnostics rows.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Optional, Sequence, Tuple

from .boundary_engine_pass import (
    BOUNDARY_REASON_AUDIO_ENTRY,
    BOUNDARY_REASON_AUDIO_EXIT,
    BOUNDARY_REASON_VISUAL_ENTRY_EDGE_TRIM,
    BOUNDARY_REASON_VISUAL_EXIT_EDGE_TRIM,
)
from .contracts import DraftClip, ProcessingResult

SCHEMA_VERSION = "cutsell.dialogue_pacing_transition.v1"

# --- Canonical transition-mode vocabulary (D-098 Section 11.2) -------------

HARD_CUT = "HARD_CUT"
TIGHT_CUT = "TIGHT_CUT"
J_CUT = "J_CUT"
L_CUT = "L_CUT"
MICRO_AUDIO_OVERLAP = "MICRO_AUDIO_OVERLAP"

TRANSITION_MODES: Tuple[str, ...] = (HARD_CUT, TIGHT_CUT, J_CUT, L_CUT, MICRO_AUDIO_OVERLAP)

# --- Renderer support classification (STATIC RENDERER FORENSIC above) -----

SUPPORTED_NOW = "SUPPORTED_NOW"
REPRESENTABLE_BUT_NOT_RENDERABLE = "REPRESENTABLE_BUT_NOT_RENDERABLE"
REQUIRES_RENDERER_EXTENSION = "REQUIRES_RENDERER_EXTENSION"

RENDERER_SUPPORT_MATRIX: Mapping[str, str] = {
    HARD_CUT: SUPPORTED_NOW,
    TIGHT_CUT: SUPPORTED_NOW,
    J_CUT: REQUIRES_RENDERER_EXTENSION,
    L_CUT: REQUIRES_RENDERER_EXTENSION,
    MICRO_AUDIO_OVERLAP: REQUIRES_RENDERER_EXTENSION,
}

# Modes Phase 1 is ever permitted to actually SELECT as `mode`. J_CUT/L_CUT/
# MICRO_AUDIO_OVERLAP are contract-only in Phase 1 -- represented in the
# vocabulary above, never chosen (see module docstring's overlap section).
PHASE_1_EXECUTABLE_MODES: Tuple[str, ...] = (HARD_CUT, TIGHT_CUT)

# --- Safety status vocabulary -----------------------------------------------

SAFETY_SAFE = "SAFE"
SAFETY_FALLBACK = "FALLBACK_TO_BASELINE"

# --- Fallback reasons --------------------------------------------------------

FALLBACK_OVERLAP_DISABLED = "overlap_disabled"
FALLBACK_RENDERER_EXTENSION_REQUIRED = "renderer_extension_required"


@dataclass(frozen=True)
class DialogueTransitionPlan:
    """One adjacent-pair transition decision. Purely descriptive: building
    this never mutates `DraftClip`/`RenderSegment`/Boundary's own output.

    `left_audio_end`/`right_audio_start`/`visual_cut_time` are SOURCE-
    relative timestamps (the same coordinate convention as `DraftClip.
    start`/`.end`), not output-timeline positions -- consistent with
    `render_plan.RenderSegment`. For every Phase-1-executable mode
    (`HARD_CUT`/`TIGHT_CUT`) `visual_cut_time == left_audio_end` by
    construction: audio and video share one matched cut instant today: see
    the module docstring's renderer support matrix for why J_CUT/L_CUT
    (which would diverge them) are not yet selectable.
    """

    transition_index: int
    left_clip_id: str
    right_clip_id: str
    mode: str
    dialogue_overlap_enabled: bool
    visual_cut_time: float
    left_audio_end: float
    right_audio_start: float
    overlap_duration: float = 0.0
    gap_removed_duration: float = 0.0
    safety_status: str = SAFETY_SAFE
    fallback_reason: Optional[str] = None
    provenance: Tuple[str, ...] = ()


def _boundary_pass_diagnostics(diagnostics: Mapping) -> Mapping:
    value = diagnostics.get("boundary_engine_pass") if isinstance(diagnostics, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def _applied_edge_trim(diagnostics: Mapping, clip_id: str, *, side: str) -> Tuple[float, Tuple[str, ...]]:
    """Sum of Boundary's OWN already-applied trim seconds at one clip's
    named edge (`"ENTRY"`/`"EXIT"`), read from `boundary_engine_pass`'s own
    audit rows -- never a new measurement, never a new threshold. A
    DELIVERY-zone event can never appear here: Boundary itself never
    writes a `trim_applied=True` visual row for one."""
    boundary = _boundary_pass_diagnostics(diagnostics)
    total = 0.0
    reasons: list[str] = []

    audio_action = BOUNDARY_REASON_AUDIO_ENTRY if side == "ENTRY" else BOUNDARY_REASON_AUDIO_EXIT
    for row in boundary.get("audio_edge_rows") or ():
        if not isinstance(row, Mapping) or row.get("clip_id") != clip_id:
            continue
        for action in row.get("actions") or ():
            if isinstance(action, Mapping) and action.get("action") == audio_action:
                total += float(action.get("trim_sec") or 0.0)
                reasons.append(audio_action)

    visual_reason = BOUNDARY_REASON_VISUAL_ENTRY_EDGE_TRIM if side == "ENTRY" else BOUNDARY_REASON_VISUAL_EXIT_EDGE_TRIM
    for row in boundary.get("visual_edge_rows") or ():
        if not isinstance(row, Mapping) or row.get("clip_id") != clip_id:
            continue
        if row.get("trim_side") != side or not row.get("trim_applied") or row.get("reason") != visual_reason:
            continue
        if side == "ENTRY":
            total += max(0.0, float(row.get("new_start") or 0.0) - float(row.get("old_start") or 0.0))
        else:
            total += max(0.0, float(row.get("old_end") or 0.0) - float(row.get("new_end") or 0.0))
        reasons.append(visual_reason)

    return total, tuple(reasons)


def plan_dialogue_pacing_transitions(
    selected: Sequence[DraftClip],
    diagnostics: Mapping,
    *,
    dialogue_overlap_enabled: bool,
) -> Tuple[DialogueTransitionPlan, ...]:
    """Build one `DialogueTransitionPlan` per adjacent pair in `selected`'s
    existing order (the SAME order `render_plan.build_render_plan` already
    iterates -- no re-ordering, no re-grouping). Read-only: `selected` and
    `diagnostics` are never mutated or returned modified."""
    clips = tuple(selected)
    plans: list[DialogueTransitionPlan] = []
    overlap_enabled = bool(dialogue_overlap_enabled)

    for index in range(len(clips) - 1):
        left, right = clips[index], clips[index + 1]
        left_exit_trim, left_reasons = _applied_edge_trim(diagnostics, left.clip_id, side="EXIT")
        right_entry_trim, right_reasons = _applied_edge_trim(diagnostics, right.clip_id, side="ENTRY")
        gap_removed = round(left_exit_trim + right_entry_trim, 3)
        provenance = tuple(sorted(set(left_reasons) | set(right_reasons)))

        mode = TIGHT_CUT if gap_removed > 0.0 else HARD_CUT
        # J_CUT/L_CUT/MICRO_AUDIO_OVERLAP are never selected in Phase 1
        # (see module docstring) -- the fallback reason records WHY an
        # overlap mode was not attempted, distinguishing user-disabled from
        # renderer-incapable, without ever executing one.
        fallback_reason = FALLBACK_RENDERER_EXTENSION_REQUIRED if overlap_enabled else FALLBACK_OVERLAP_DISABLED

        plans.append(DialogueTransitionPlan(
            transition_index=index,
            left_clip_id=left.clip_id,
            right_clip_id=right.clip_id,
            mode=mode,
            dialogue_overlap_enabled=overlap_enabled,
            visual_cut_time=round(float(left.end), 3),
            left_audio_end=round(float(left.end), 3),
            right_audio_start=round(float(right.start), 3),
            overlap_duration=0.0,
            gap_removed_duration=gap_removed,
            safety_status=SAFETY_SAFE,
            fallback_reason=fallback_reason,
            provenance=provenance,
        ))
    return tuple(plans)


def _plan_row(plan: DialogueTransitionPlan) -> dict:
    return {
        "transition_index": plan.transition_index,
        "left_clip_id": plan.left_clip_id,
        "right_clip_id": plan.right_clip_id,
        "mode": plan.mode,
        "dialogue_overlap_enabled": plan.dialogue_overlap_enabled,
        "visual_cut_time": plan.visual_cut_time,
        "left_audio_end": plan.left_audio_end,
        "right_audio_start": plan.right_audio_start,
        "overlap_duration": plan.overlap_duration,
        "gap_removed_duration": plan.gap_removed_duration,
        "safety_status": plan.safety_status,
        "fallback_reason": plan.fallback_reason,
        "provenance": list(plan.provenance),
    }


def dialogue_pacing_transition_diagnostics(
    plans: Sequence[DialogueTransitionPlan],
    *,
    dialogue_overlap_enabled: bool,
) -> dict:
    """Compact, request-level (not per-family) diagnostics block -- the 15
    named fields D-142 requires, plus the bounded per-transition rows
    (bounded by construction: at most `len(selected) - 1` rows, the same
    order of magnitude as the family/judge-group diagnostics already
    printed elsewhere in this pipeline)."""
    rows = tuple(plans)
    mode_counts = {mode: 0 for mode in TRANSITION_MODES}
    fallback_reasons: dict[str, int] = {}
    total_gap_removed = 0.0
    total_overlap = 0.0
    for row in rows:
        mode_counts[row.mode] = mode_counts.get(row.mode, 0) + 1
        total_gap_removed += float(row.gap_removed_duration)
        total_overlap += float(row.overlap_duration)
        if row.fallback_reason:
            fallback_reasons[row.fallback_reason] = fallback_reasons.get(row.fallback_reason, 0) + 1

    return {
        "schema_version": SCHEMA_VERSION,
        "dialogue_pacing_evaluated": True,
        "dialogue_overlap_enabled": bool(dialogue_overlap_enabled),
        "transition_count": len(rows),
        "mode_counts": mode_counts,
        "hard_cut_count": mode_counts.get(HARD_CUT, 0),
        "tight_cut_count": mode_counts.get(TIGHT_CUT, 0),
        "j_cut_count": mode_counts.get(J_CUT, 0),
        "l_cut_count": mode_counts.get(L_CUT, 0),
        "micro_audio_overlap_count": mode_counts.get(MICRO_AUDIO_OVERLAP, 0),
        "total_gap_removed_sec": round(total_gap_removed, 3),
        "total_audio_overlap_sec": round(total_overlap, 3),
        "fallback_count": sum(1 for row in rows if row.fallback_reason),
        "fallback_reasons": fallback_reasons,
        "renderer_support_matrix": dict(RENDERER_SUPPORT_MATRIX),
        "transitions": [_plan_row(row) for row in rows],
    }


def apply_dialogue_pacing_transition_pass(
    result: ProcessingResult,
    *,
    dialogue_overlap_enabled: bool,
) -> ProcessingResult:
    """The ONE Dialogue/Pacing Transition call site (D-142 Phase 1),
    mirroring `boundary_engine_pass.apply_post_freeze_boundary_pass`'s own
    shape: takes the frozen, Boundary-approved `ProcessingResult`, computes
    the transition plan, and attaches it to `draft.diagnostics` only.
    `draft.selected` is NEVER reassigned here -- Phase 1 has no physical or
    semantic authority, matching the module docstring's canonical
    placement (strictly after Boundary, strictly before the renderer is
    ever invoked, which happens later, at export time, from a separately
    persisted draft -- see `export_job.py`/`validation.py`/`render_plan.
    build_render_plan`, all untouched by this task)."""
    draft = result.draft
    if not hasattr(draft, "selected") or not draft.selected:
        return result
    diagnostics = dict(draft.diagnostics or {})
    plans = plan_dialogue_pacing_transitions(
        draft.selected, diagnostics, dialogue_overlap_enabled=dialogue_overlap_enabled,
    )
    diagnostics["dialogue_pacing_transition"] = dialogue_pacing_transition_diagnostics(
        plans, dialogue_overlap_enabled=dialogue_overlap_enabled,
    )
    return replace(result, draft=replace(draft, diagnostics=diagnostics))
