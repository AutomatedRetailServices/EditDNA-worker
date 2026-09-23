"""V1 Manual Timeline Composition Executor -- D-278.

Post D-277 (V1 manual timeline architecture / composition contract).
This module is the ONE dedicated composition authority that EXECUTES a
`timeline_composition.TimelineComposition` against an already-resolved
CutSell base edit and already-resolved B-roll/voice-over media,
producing one final composed render. It never buries timeline policy
inside `render.py`, BestTake, Pacing, or Visual Finishing (Stage 1) --
`timeline_composition.py` remains the sole semantic-state authority;
this module only turns that state into ffmpeg commands.

## Architecture (Stage 2-3)

    AI BASE EDIT (already rendered, already qualified)
        + TimelineComposition (D-277 state)
        -> build_timeline_render_plan()   -- pure, resolves state to a
                                              plan, never touches ffmpeg
        -> execute_timeline_composition() -- the one ffmpeg pass
        -> final composed media
        -> output_format_qc.verify_output_format(..., FINAL_RENDER_
           OUTPUT_CONTRACT_V1)

This module NEVER recomputes BestTake/Ordering/Freeze/Boundary/Pacing
(Stage 3) -- it consumes an already-resolved base edit video file as
one opaque input. It never inspects face/product/hands/scene mode
(Stage 42) -- it composes timeline ROLES only, and the identical code
path runs for talking-head, faceless-product, hands/product, and
demo-action primary footage. It never ranks, scores, or places a
B-roll asset on the creator's behalf (Stage 43 post-launch AI
firewall) -- every placement in the `TimelineComposition` it receives
is already the creator's own manual decision.

## Video composition (Stage 4-6, 19, 22)

Every `BrollPlacement` becomes one full-frame overlay input, scaled to
the canonical output frame and shown via ffmpeg's own `overlay=...
enable='between(t,start,end)'` for exactly its
`[timeline_start_sec, timeline_end_sec)` window (Stage 19: a hard cut
at each boundary -- no cinematic transition, no AI transition
selection). Outside every B-roll window the base edit's own video is
what plays -- a placement never rewrites the base timeline (Stage 4/22:
deleting a placement is definitionally a no-op on the base video,
since nothing in this module ever mutates it).

## Audio composition (Stage 6-10, 23, 26-27)

Audio authority is resolved to a small number of contiguous,
non-overlapping timeline segments (breakpoints from every VOICE_OVER
placement plus every `USE_BROLL_AUDIO` B-roll placement only --
`KEEP_PRIMARY_VOICE` and `MUTE_BROLL_AUDIO` never change audio
authority, Stage 6/7's own "primary voice continues"). For each
segment this module picks EXACTLY ONE audio source
(`timeline_composition.caption_source_for_region`'s own precedence --
`VOICE_OVER` > `BROLL_SOURCE_AUDIO` > `ORIGINAL_PRIMARY_VOICE` --
reused verbatim, never re-derived) and extracts that source's own
audio for that exact timeline window (mapped through the covering
placement's own `source_in_sec` offset when the segment falls inside a
B-roll/VO window), then concatenates every segment in timeline order
with ffmpeg's own audio `concat` filter -- never `amix` (Stage 10: "no
accidental overlapping speech", by construction, since audio channels
are concatenated in series, never summed).

## Captions (Stage 11-13)

`resolve_caption_regions` computes the SAME authority partition as the
audio composition and returns it as ordered `CaptionRegion` rows. This
module does NOT render burned-in captions and does NOT run ASR (Stage
11's own explicit scope: "expose caption authority by region... do
NOT perform new ASR") -- no caption text is fabricated (Stage 13). A
future gate with real transcript strings per source may consume this
function's own output to actually generate an ASS/SRT track.

## Duration-mismatch policy (Stage 20-21) -- no silent time-stretch

`build_timeline_render_plan` rejects (`BROLL_DURATION_MISMATCH` /
`VOICEOVER_DURATION_MISMATCH`) any placement whose
`(source_out_sec - source_in_sec)` does not equal
`(timeline_end_sec - timeline_start_sec)` within a small floating-point
tolerance. D-277's own `validate_composition` never checked this (it
only checks each field's own bounds); this module adds the executor-
level check `timeline_composition.py` itself never needed for a pure
state model.

## Render safety (Stage 31, 34-35) -- reuses render.py's OWN doctrine,
never a second, independently-drifting copy

Timeout: `render.RENDER_FFMPEG_TIMEOUT_SEC` (1200.0), read by
reference, never duplicated as a literal (this module's own
`execute_timeline_composition` accepts an explicit override for tests
only, exactly like `render.py`'s and `source_normalization_executor.
py`'s own established `timeout_sec` parameter convention). Job-local
temp output + atomic `os.replace` promote (mirrors `render.py`'s own
D-266 doctrine; a private copy per this codebase's own established
per-module convention -- see `source_media_profile.py`'s own comment
on why modules own their own small helpers rather than reaching into
another module's private functions). `shell=False` throughout
(`subprocess.run` with a list argv, never `shell=True`). Bounded
stderr excerpt on any failure. Structured failure categories only
(Stage 35) -- never a bare exception message.

## What this module explicitly does NOT do

No microphone recording. No AI B-roll ranking/placement/suggestion of
any kind. No new source-normalization/timeout policy (uses only
already-qualified media, per Stage 14/32). No BestTake/Pacing/Boundary/
Freeze/Audio-Join-policy/Audio-Finishing-policy/Visual-Finishing-
policy/delivery-policy change -- this module imports NONE of those
modules. No mobile/UI code. No S3/asset-upload implementation (Stage
15's own future seam is named, not built, here).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import output_format_qc as ofq
from . import render as render_mod
from . import source_media_profile as smp
from . import timeline_composition as tc

# Stage 30/31 -- the canonical output frame this executor targets, matching
# FINAL_RENDER_OUTPUT_CONTRACT_V1 exactly (expected_width/height, fixed_fps,
# audio_sample_rate_hz) and render.py's own RENDER_FPS_DEFAULT. Read by
# reference from render.py wherever render.py already owns the number
# (FPS); the frame dimensions match render_preview's own defaults but are
# not re-exported from render.py, so they are named here once.
CANONICAL_OUTPUT_WIDTH = 1080
CANONICAL_OUTPUT_HEIGHT = 1920
CANONICAL_OUTPUT_AUDIO_SAMPLE_RATE_HZ = 48000

# Stage 20/21 -- floating-point tolerance for the duration-match check; not
# a new numeric POLICY, just float-comparison hygiene.
_DURATION_MATCH_TOLERANCE_SEC = 0.01

_STDERR_EXCERPT_MAX_CHARS = 4000

# =============================================================================
# Stage 35 -- structured failure categories (the ONLY strings this module
# ever returns as `outcome`/`error_category`; never a bare exception).
# =============================================================================

TIMELINE_INVALID = "TIMELINE_INVALID"
ASSET_MISSING = "ASSET_MISSING"
ASSET_BOUNDS_INVALID = "ASSET_BOUNDS_INVALID"
BROLL_DURATION_MISMATCH = "BROLL_DURATION_MISMATCH"
VOICEOVER_DURATION_MISMATCH = "VOICEOVER_DURATION_MISMATCH"
COMPOSITION_FFMPEG_FAILED = "COMPOSITION_FFMPEG_FAILED"
COMPOSITION_TIMEOUT = "COMPOSITION_TIMEOUT"
OUTPUT_MISSING = "OUTPUT_MISSING"
OUTPUT_EMPTY = "OUTPUT_EMPTY"
OUTPUT_FORMAT_QC_FAILED = "OUTPUT_FORMAT_QC_FAILED"

PLAN_BUILD_SUCCEEDED = "PLAN_BUILD_SUCCEEDED"
COMPOSITION_SUCCEEDED = "COMPOSITION_SUCCEEDED"
BASE_ONLY_BYPASS = "BASE_ONLY_BYPASS"


@dataclass(frozen=True)
class ResolvedTimelineAsset:
    """Stage 14/40 -- an already-media-safe, already-tenant-verified
    local asset the CALLER resolved (D-271/D-272/normalization/D-274E
    already applied, ownership already checked). This module never
    performs that verification itself and never accepts a raw,
    caller-controlled arbitrary path/S3 key from an untrusted client
    (Stage 40) -- it only consumes what the caller already vouches
    for."""

    asset_id: str
    local_path: str
    duration_sec: float
    has_audio: bool


@dataclass(frozen=True)
class CaptionRegion:
    """Stage 11/13 -- one caption-authority window. `authority` is one
    of `"VOICE_OVER"` / `"BROLL_SOURCE_AUDIO"` / `"ORIGINAL_PRIMARY_
    VOICE"` -- never a transcript string (no ASR here)."""

    start_sec: float
    end_sec: float
    authority: str


@dataclass(frozen=True)
class _AudioSegment:
    """Internal -- one contiguous audio-authority window resolved to an
    executable ffmpeg input reference. Not part of this module's own
    public contract; `CaptionRegion` is the public, caller-facing
    shape of the same partition."""

    start_sec: float
    end_sec: float
    authority: str
    input_index: int
    source_start_sec: float
    source_end_sec: float
    has_audio: bool


@dataclass(frozen=True)
class TimelineRenderPlan:
    """Stage 2 -- the immutable resolution of D-277 semantic state into
    explicit render operations. No AI decision anywhere in here -- pure
    arithmetic over already-supplied placements and already-resolved
    assets."""

    contract_version: int
    base_edit_identity: str
    timeline_revision_identity: str
    timeline_duration_sec: float
    base_edit_asset: ResolvedTimelineAsset
    broll_placements: tuple[tc.BrollPlacement, ...]
    voice_over_placements: tuple[tc.VoiceOverPlacement, ...]
    resolved_broll_assets: dict
    resolved_voice_over_assets: dict
    caption_regions: tuple[CaptionRegion, ...]
    plan_identity: str


@dataclass(frozen=True)
class TimelineRenderPlanResult:
    outcome: str
    plan: TimelineRenderPlan | None = None
    reason_codes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CompositionExecutionResult:
    outcome: str
    output_path: str | None = None
    diagnostics: dict = field(default_factory=dict)


def _compute_plan_identity(base_edit_identity: str, timeline_revision_identity: str) -> str:
    """Stage 28/29 -- deterministic, path-independent (never derived
    from a local resolved file path -- only from the two semantic
    identities D-277 already established). Changes exactly when the
    timeline's own semantic state changes (Stage 29); stable for an
    unchanged timeline regardless of which run resolved the assets."""
    payload = f"{base_edit_identity}:{timeline_revision_identity}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"tlplan_{digest}"


def _audio_authority_breakpoints(composition: tc.TimelineComposition) -> list[float]:
    """Stage 10 -- only VOICE_OVER and USE_BROLL_AUDIO placements ever
    change audio authority away from ORIGINAL_PRIMARY_VOICE (Stage 6/7:
    KEEP_PRIMARY_VOICE and MUTE_BROLL_AUDIO both leave the primary
    voice as-is)."""
    points = {0.0, composition.timeline_duration_sec}
    for vo in composition.voice_over_placements:
        points.add(vo.timeline_start_sec)
        points.add(vo.timeline_end_sec)
    for broll in composition.broll_placements:
        if broll.audio_mode == tc.TimelineAudioMode.USE_BROLL_AUDIO:
            points.add(broll.timeline_start_sec)
            points.add(broll.timeline_end_sec)
    return sorted(points)


def resolve_caption_regions(composition: tc.TimelineComposition) -> tuple[CaptionRegion, ...]:
    """Stage 11 -- the public, pure caption-authority partition. Reuses
    `timeline_composition.caption_source_for_region` (never a second,
    independently-drifting precedence rule) sampled at the midpoint of
    each authority-breakpoint interval."""
    points = _audio_authority_breakpoints(composition)
    regions = []
    for start, end in zip(points, points[1:]):
        if end - start <= 0:
            continue
        midpoint = (start + end) / 2.0
        authority = tc.caption_source_for_region(composition, midpoint)
        regions.append(CaptionRegion(start_sec=start, end_sec=end, authority=authority))
    return tuple(regions)


def build_timeline_render_plan(
    composition: tc.TimelineComposition,
    base_edit_asset: ResolvedTimelineAsset,
    resolved_broll_assets: dict,
    resolved_voice_over_assets: dict,
) -> TimelineRenderPlanResult:
    """Stage 2/20/21/39 -- the ONE pure function from D-277 state plus
    already-resolved assets to an executable plan. Never touches
    ffmpeg, never touches a filesystem beyond the caller-supplied
    `ResolvedTimelineAsset` records it is handed. Fail-closed: any
    validation failure returns a plan of `None` with a structured
    reason, never a partially-built plan."""
    validation = tc.validate_composition(composition)
    if not validation.valid:
        return TimelineRenderPlanResult(outcome=TIMELINE_INVALID, reason_codes=validation.errors)

    for broll in composition.broll_placements:
        asset = resolved_broll_assets.get(broll.asset.asset_id)
        if asset is None:
            return TimelineRenderPlanResult(
                outcome=ASSET_MISSING, reason_codes=(f"missing resolved broll asset: {broll.asset.asset_id}",)
            )
        if asset.duration_sec != broll.asset.duration_sec:
            return TimelineRenderPlanResult(
                outcome=ASSET_BOUNDS_INVALID,
                reason_codes=(f"resolved broll asset duration mismatch for {broll.asset.asset_id}",),
            )
        timeline_span = broll.timeline_end_sec - broll.timeline_start_sec
        source_span = broll.source_out_sec - broll.source_in_sec
        if abs(timeline_span - source_span) > _DURATION_MATCH_TOLERANCE_SEC:
            return TimelineRenderPlanResult(
                outcome=BROLL_DURATION_MISMATCH,
                reason_codes=(
                    f"broll {broll.placement_id}: timeline span {timeline_span:.3f}s != "
                    f"source span {source_span:.3f}s",
                ),
            )

    for vo in composition.voice_over_placements:
        asset = resolved_voice_over_assets.get(vo.asset.asset_id)
        if asset is None:
            return TimelineRenderPlanResult(
                outcome=ASSET_MISSING, reason_codes=(f"missing resolved voice_over asset: {vo.asset.asset_id}",)
            )
        if asset.duration_sec != vo.asset.duration_sec:
            return TimelineRenderPlanResult(
                outcome=ASSET_BOUNDS_INVALID,
                reason_codes=(f"resolved voice_over asset duration mismatch for {vo.asset.asset_id}",),
            )
        timeline_span = vo.timeline_end_sec - vo.timeline_start_sec
        source_span = vo.source_out_sec - vo.source_in_sec
        if abs(timeline_span - source_span) > _DURATION_MATCH_TOLERANCE_SEC:
            return TimelineRenderPlanResult(
                outcome=VOICEOVER_DURATION_MISMATCH,
                reason_codes=(
                    f"voice_over {vo.placement_id}: timeline span {timeline_span:.3f}s != "
                    f"source span {source_span:.3f}s",
                ),
            )

    revision = tc.derive_revision_identity(composition)
    plan = TimelineRenderPlan(
        contract_version=composition.contract_version,
        base_edit_identity=composition.base_edit_identity,
        timeline_revision_identity=revision.identity,
        timeline_duration_sec=composition.timeline_duration_sec,
        base_edit_asset=base_edit_asset,
        broll_placements=composition.broll_placements,
        voice_over_placements=composition.voice_over_placements,
        resolved_broll_assets=dict(resolved_broll_assets),
        resolved_voice_over_assets=dict(resolved_voice_over_assets),
        caption_regions=resolve_caption_regions(composition),
        plan_identity=_compute_plan_identity(composition.base_edit_identity, revision.identity),
    )
    return TimelineRenderPlanResult(outcome=PLAN_BUILD_SUCCEEDED, plan=plan)


# =============================================================================
# Execution -- Stage 31/34/35: mirrors render.py's OWN D-266 doctrine
# (job-local temp + atomic promote, shell=False, bounded stderr, structured
# failures) as this module's own private copy, per this codebase's
# established per-module convention (each module owns its small execution
# helpers rather than reaching into another module's private functions).
# =============================================================================


def _bounded_excerpt(text) -> str:
    if not text:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    return text.strip()[:_STDERR_EXCERPT_MAX_CHARS]


def _job_local_temp_output_path(destination: Path, execution_id: str) -> Path:
    return destination.with_name(f".{destination.name}.{execution_id}.composing{destination.suffix}")


def _video_filter_chain(input_label: str, output_label: str) -> str:
    return (
        f"[{input_label}]scale={CANONICAL_OUTPUT_WIDTH}:{CANONICAL_OUTPUT_HEIGHT}:"
        f"force_original_aspect_ratio=decrease,"
        f"pad={CANONICAL_OUTPUT_WIDTH}:{CANONICAL_OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
        f"setsar=1,fps={render_mod.RENDER_FPS_DEFAULT}[{output_label}]"
    )


def _build_composition_command(plan: TimelineRenderPlan, output_path: str) -> list[str]:
    """Stage 4-10 -- the one ffmpeg invocation for the whole composition
    (Stage 24/25: multiple B-roll/VO placements, one generation)."""
    inputs: list[str] = ["-i", plan.base_edit_asset.local_path]
    input_index_by_broll: dict[str, int] = {}
    input_index_by_vo: dict[str, int] = {}
    next_index = 1
    for broll in plan.broll_placements:
        asset = plan.resolved_broll_assets[broll.asset.asset_id]
        inputs += ["-i", asset.local_path]
        input_index_by_broll[broll.placement_id] = next_index
        next_index += 1
    for vo in plan.voice_over_placements:
        asset = plan.resolved_voice_over_assets[vo.asset.asset_id]
        inputs += ["-i", asset.local_path]
        input_index_by_vo[vo.placement_id] = next_index
        next_index += 1

    filters: list[str] = []

    # --- Video: base passthrough, then one full-frame overlay per B-roll
    # placement, each windowed to its own timeline interval (Stage 4-6,19).
    filters.append(_video_filter_chain("0:v", "vbase"))
    current_video = "vbase"
    for broll in plan.broll_placements:
        idx = input_index_by_broll[broll.placement_id]
        scaled_label = f"bv{idx}"
        filters.append(
            f"[{idx}:v]trim=start={broll.source_in_sec:.6f}:end={broll.source_out_sec:.6f},"
            f"setpts=PTS-STARTPTS+{broll.timeline_start_sec:.6f}/TB,"
            f"scale={CANONICAL_OUTPUT_WIDTH}:{CANONICAL_OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={CANONICAL_OUTPUT_WIDTH}:{CANONICAL_OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1[{scaled_label}]"
        )
        next_label = f"v{idx}"
        filters.append(
            f"[{current_video}][{scaled_label}]overlay=x=0:y=0:"
            f"enable='between(t,{broll.timeline_start_sec:.6f},{broll.timeline_end_sec:.6f})':"
            f"eof_action=pass[{next_label}]"
        )
        current_video = next_label
    final_video_label = current_video

    # --- Audio: one exclusive source per contiguous authority segment,
    # concatenated in series (Stage 8-10,26-27 -- never amix, never two
    # simultaneous speech sources).
    segments = _resolve_audio_segments(plan, input_index_by_broll, input_index_by_vo)
    audio_labels = []
    for i, seg in enumerate(segments):
        label = f"aseg{i}"
        duration = seg.end_sec - seg.start_sec
        if seg.has_audio:
            # Stage 30: FINAL_RENDER_OUTPUT_CONTRACT_V1 requires stereo --
            # a mono source (a real phone mic, or a synthetic mono test
            # tone) must be forced to stereo BEFORE concat, since ffmpeg's
            # audio `concat` filter requires every segment to share the
            # same channel layout, not just the same sample rate. Found
            # via a real smoke test (a mono synthetic tone produced a
            # mono final output, failing the contract's own
            # AUDIO_CHANNELS check) -- not a hypothetical edge case.
            filters.append(
                f"[{seg.input_index}:a]atrim=start={seg.source_start_sec:.6f}:"
                f"end={seg.source_end_sec:.6f},asetpts=PTS-STARTPTS,"
                f"aformat=channel_layouts=stereo,"
                f"aresample={CANONICAL_OUTPUT_AUDIO_SAMPLE_RATE_HZ}[{label}]"
            )
        else:
            filters.append(
                f"anullsrc=channel_layout=stereo:sample_rate={CANONICAL_OUTPUT_AUDIO_SAMPLE_RATE_HZ},"
                f"atrim=duration={duration:.6f}[{label}]"
            )
        audio_labels.append(f"[{label}]")
    if audio_labels:
        filters.append("".join(audio_labels) + f"concat=n={len(audio_labels)}:v=0:a=1[aout]")
    else:
        filters.append(
            f"anullsrc=channel_layout=stereo:sample_rate={CANONICAL_OUTPUT_AUDIO_SAMPLE_RATE_HZ},"
            f"atrim=duration={plan.timeline_duration_sec:.6f}[aout]"
        )

    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    command += inputs
    command += [
        "-filter_complex", ";".join(filters),
        "-map", f"[{final_video_label}]",
        "-map", "[aout]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        *render_mod.CANONICAL_OUTPUT_COLOR_METADATA_FLAGS,
        "-c:a", "aac", "-b:a", "160k", "-ar", str(CANONICAL_OUTPUT_AUDIO_SAMPLE_RATE_HZ),
        "-movflags", "+faststart",
        "-t", f"{plan.timeline_duration_sec:.6f}",
        output_path,
    ]
    return command


def _resolve_audio_segments(
    plan: TimelineRenderPlan, input_index_by_broll: dict, input_index_by_vo: dict,
) -> list[_AudioSegment]:
    points = sorted({0.0, plan.timeline_duration_sec}
                     | {p.timeline_start_sec for p in plan.voice_over_placements}
                     | {p.timeline_end_sec for p in plan.voice_over_placements}
                     | {p.timeline_start_sec for p in plan.broll_placements
                        if p.audio_mode == tc.TimelineAudioMode.USE_BROLL_AUDIO}
                     | {p.timeline_end_sec for p in plan.broll_placements
                        if p.audio_mode == tc.TimelineAudioMode.USE_BROLL_AUDIO})
    segments: list[_AudioSegment] = []
    for start, end in zip(points, points[1:]):
        if end - start <= 0:
            continue
        midpoint = (start + end) / 2.0
        vo_hit = next(
            (p for p in plan.voice_over_placements if p.timeline_start_sec <= midpoint < p.timeline_end_sec), None,
        )
        if vo_hit is not None:
            asset = plan.resolved_voice_over_assets[vo_hit.asset.asset_id]
            offset = start - vo_hit.timeline_start_sec
            src_start = vo_hit.source_in_sec + offset
            src_end = src_start + (end - start)
            segments.append(_AudioSegment(
                start_sec=start, end_sec=end, authority="VOICE_OVER",
                input_index=input_index_by_vo[vo_hit.placement_id],
                source_start_sec=src_start, source_end_sec=src_end, has_audio=asset.has_audio,
            ))
            continue
        broll_hit = next(
            (p for p in plan.broll_placements
             if p.audio_mode == tc.TimelineAudioMode.USE_BROLL_AUDIO
             and p.timeline_start_sec <= midpoint < p.timeline_end_sec),
            None,
        )
        if broll_hit is not None:
            asset = plan.resolved_broll_assets[broll_hit.asset.asset_id]
            offset = start - broll_hit.timeline_start_sec
            src_start = broll_hit.source_in_sec + offset
            src_end = src_start + (end - start)
            segments.append(_AudioSegment(
                start_sec=start, end_sec=end, authority="BROLL_SOURCE_AUDIO",
                input_index=input_index_by_broll[broll_hit.placement_id],
                source_start_sec=src_start, source_end_sec=src_end, has_audio=asset.has_audio,
            ))
            continue
        # ORIGINAL_PRIMARY_VOICE: base edit's own timeline time == final
        # timeline time (Stage 22/23 -- no timeline shift under the base).
        segments.append(_AudioSegment(
            start_sec=start, end_sec=end, authority="ORIGINAL_PRIMARY_VOICE",
            input_index=0, source_start_sec=start, source_end_sec=end,
            has_audio=plan.base_edit_asset.has_audio,
        ))
    return segments


def execute_timeline_composition(
    plan: TimelineRenderPlan,
    *,
    output_directory: str,
    timeout_sec: float | None = None,
) -> CompositionExecutionResult:
    """Stage 34/38 -- the one execution entry point. `timeout_sec`
    defaults to `None`, meaning "use the canonical
    `render.RENDER_FFMPEG_TIMEOUT_SEC` policy" (same `None`-means-
    canonical-default contract D-274F-A established for the
    normalization executor -- an explicit override, test code only,
    still wins)."""
    effective_timeout_sec = timeout_sec if timeout_sec is not None else render_mod.RENDER_FFMPEG_TIMEOUT_SEC

    if not plan.broll_placements and not plan.voice_over_placements:
        # Stage 38 -- base-only bypass: the timeline is semantically
        # identical to the base edit; never a needless re-encode, never a
        # mutation of the base edit file itself.
        return CompositionExecutionResult(
            outcome=BASE_ONLY_BYPASS,
            output_path=plan.base_edit_asset.local_path,
            diagnostics={"plan_identity": plan.plan_identity, "reason": "no manual placements"},
        )

    os.makedirs(output_directory, exist_ok=True)
    destination = Path(output_directory) / f"composed_{plan.plan_identity}.mp4"
    execution_id = hashlib.sha256(f"{plan.plan_identity}:{time.monotonic_ns()}".encode()).hexdigest()[:16]
    temp_output = _job_local_temp_output_path(destination, execution_id)

    command = _build_composition_command(plan, str(temp_output))
    fingerprint = hashlib.sha256(str(command).encode("utf-8")).hexdigest()[:24]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=effective_timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        if temp_output.exists():
            temp_output.unlink(missing_ok=True)
        return CompositionExecutionResult(
            outcome=COMPOSITION_TIMEOUT,
            diagnostics={
                "command_fingerprint": fingerprint, "timeout_sec": effective_timeout_sec,
                "stderr_excerpt": _bounded_excerpt(exc.stderr), "wall_time_sec": round(time.monotonic() - started, 3),
            },
        )
    wall_time_sec = time.monotonic() - started
    if completed.returncode != 0:
        if temp_output.exists():
            temp_output.unlink(missing_ok=True)
        return CompositionExecutionResult(
            outcome=COMPOSITION_FFMPEG_FAILED,
            diagnostics={
                "command_fingerprint": fingerprint, "return_code": completed.returncode,
                "stderr_excerpt": _bounded_excerpt(completed.stderr), "wall_time_sec": round(wall_time_sec, 3),
            },
        )

    if not temp_output.exists():
        return CompositionExecutionResult(outcome=OUTPUT_MISSING, diagnostics={"command_fingerprint": fingerprint})
    if temp_output.stat().st_size <= 0:
        temp_output.unlink(missing_ok=True)
        return CompositionExecutionResult(outcome=OUTPUT_EMPTY, diagnostics={"command_fingerprint": fingerprint})

    os.replace(temp_output, destination)  # Stage 34: atomic promote, same-directory guarantee.

    profile = smp.probe_source_media_profile(str(destination))
    qc = ofq.verify_output_format(profile, ofq.FINAL_RENDER_OUTPUT_CONTRACT_V1)
    if qc.status != ofq.STATUS_PASS:
        return CompositionExecutionResult(
            outcome=OUTPUT_FORMAT_QC_FAILED,
            output_path=str(destination),
            diagnostics={
                "command_fingerprint": fingerprint, "wall_time_sec": round(wall_time_sec, 3),
                "format_qc_status": qc.status, "format_qc_failed_checks": list(qc.failed_checks),
            },
        )

    return CompositionExecutionResult(
        outcome=COMPOSITION_SUCCEEDED,
        output_path=str(destination),
        diagnostics={
            "command_fingerprint": fingerprint, "wall_time_sec": round(wall_time_sec, 3),
            "plan_identity": plan.plan_identity, "format_qc_status": qc.status,
            "timeout_sec": effective_timeout_sec,
        },
    )
