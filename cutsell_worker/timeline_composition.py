"""V1 Manual Timeline Composition Contract -- D-277.

Post D-276 (canonical product-scope update: faceless/product/hands/demo
A-roll + V1 manual B-roll/timeline voice-over). This module implements
the ONE typed, deterministic contract for the V1 manual timeline: the
smallest set of immutable types and pure state-transition functions
capable of representing PRIMARY A-ROLL + MANUAL B-ROLL + ORIGINAL
PRIMARY VOICE + optional B-ROLL SOURCE AUDIO + RECORDED VOICE OVER,
without touching the already-closed CutSell editorial engine.

## Canonical timeline principle (D-277 Stage 1)

The AI edit remains the BASE TIMELINE (`TimelineComposition.
base_edit_identity`, an opaque, caller-supplied identity -- never a
local filesystem path, never re-derived here). Creator manual edits
are a NON-DESTRUCTIVE composition layer on top: every operation in
this module returns a NEW `TimelineComposition` (Stage 15 -- immutable
revisions, never an in-place mutation) and never touches BestTake, P1,
P2, Freeze, Boundary, Pacing, or the original AI render plan. Nothing
in this module reads a video file, calls ffmpeg, records a microphone,
or renders anything -- it is pure state and pure functions over that
state.

## Canonical timeline clock (Stage 2)

Every `timeline_start_sec` / `timeline_end_sec` value is seconds
relative to the FINAL EDITED TIMELINE's own start (0.0), never
source-file time or raw-take time. `source_in_sec` / `source_out_sec`
are always relative to the referenced asset's OWN duration
(`TimelineAssetReference.duration_sec`), never the timeline clock.
These two domains are never silently mixed.

## Canonical audio-mode vocabulary (Stage 7 -- smallest V1 set)

Exactly three B-roll audio modes: `KEEP_PRIMARY_VOICE`,
`USE_BROLL_AUDIO`, `MUTE_BROLL_AUDIO`. No
`MIX_PRIMARY_AND_BROLL_AUDIO` -- Stage 7's own "prefer smallest V1
set" instruction, and no V1 requirement demonstrated a need for it.
Voice-over carries no independent audio-mode field at all: Stage 33's
own V1 default (voice-over replaces/mutes the primary spoken voice for
its region, unconditionally) is the fixed V1 behavior, never a
per-placement choice, per the same smallest-set doctrine.

## Canonical overlap policy (Stage 23/24 -- chosen, documented, no
ambiguity)

B-roll visual placements may NEVER overlap each other in timeline
time, and voice-over placements may NEVER overlap each other in
timeline time (both are Stage 23/24's own "simplest deterministic
behavior" recommendation: disallow overlap rather than define a
layer-order/last-wins rule). A B-roll placement and a voice-over
placement MAY coexist over the same timeline interval -- they are
independent layers (visual vs. audio-priority), never mutually
exclusive.

## Canonical bounds policy (Stage 25/26 -- fail-closed, never implicit
truncation)

`source_in_sec >= 0`, `source_out_sec <= asset.duration_sec`,
`source_out_sec > source_in_sec`; `timeline_start_sec >= 0`,
`timeline_end_sec <= composition.timeline_duration_sec`,
`timeline_end_sec > timeline_start_sec`. Any violation makes the whole
candidate composition INVALID (`TimelineValidationResult.valid ==
False`) -- this module never silently clips/truncates an out-of-bounds
interval.

## What this module explicitly does NOT do (D-277's own binding scope)

No microphone recording, no B-roll rendering, no ffmpeg invocation, no
mobile UI, no upload/storage I/O, no AI engine change, no BestTake
change, no Visual Finishing change, no media-safety-stack bypass (a
manually uploaded B-roll asset must still pass the same D-271/D-272/
normalization/D-274E stack before its `TimelineAssetReference` can
exist -- enforced by the CALLER, not re-implemented here; see Stage
28's own "do not create a bypass" instruction). No AI B-roll placement,
suggestion, or scoring of any kind (Stage 41's post-launch firewall).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum

# Stage 16 -- single canonical owner of the timeline contract version.
TIMELINE_CONTRACT_VERSION = 1


class TimelineAudioMode(str, Enum):
    """Stage 7 -- the three canonical V1 B-roll audio modes."""

    KEEP_PRIMARY_VOICE = "KEEP_PRIMARY_VOICE"
    USE_BROLL_AUDIO = "USE_BROLL_AUDIO"
    MUTE_BROLL_AUDIO = "MUTE_BROLL_AUDIO"


@dataclass(frozen=True)
class TimelineAssetReference:
    """Stage 5/9/45 -- an opaque reference to a media asset already
    admitted through the existing media-safety stack (D-271 profile ->
    D-272 policy -> normalization if required -> D-274E QC). This
    module never re-implements or bypasses that stack (Stage 28); it
    only carries the identity and duration a caller has already
    established, never a raw filesystem path (Stage 17's own
    "never local file path / temp path / filename" rule applies
    identically here)."""

    asset_id: str
    source_media_identity: str
    duration_sec: float


@dataclass(frozen=True)
class BrollPlacement:
    """Stage 5/6/7 -- one manual B-roll visual placement. Visual
    coverage over `[timeline_start_sec, timeline_end_sec)`; the
    underlying primary A-roll audio track is untouched by this
    placement's mere existence -- `audio_mode` is the only thing that
    can change what is heard (Stage 6: "Do not automatically remove
    primary audio when visual B-roll is inserted")."""

    placement_id: str
    asset: TimelineAssetReference
    timeline_start_sec: float
    timeline_end_sec: float
    source_in_sec: float
    source_out_sec: float
    audio_mode: TimelineAudioMode = TimelineAudioMode.KEEP_PRIMARY_VOICE


@dataclass(frozen=True)
class VoiceOverPlacement:
    """Stage 9/45 -- one recorded voice-over placement. No microphone
    implementation here (Stage 9/10's own explicit scope boundary) --
    `asset` is an opaque reference a future recording handoff would
    populate. Carries no `audio_mode`: Stage 33's V1 default (mutes
    the primary voice for its region) is fixed, not configurable, per
    this module's own smallest-V1-set doctrine."""

    placement_id: str
    asset: TimelineAssetReference
    timeline_start_sec: float
    timeline_end_sec: float
    source_in_sec: float
    source_out_sec: float
    transcript_reference: str | None = None


@dataclass(frozen=True)
class TimelineComposition:
    """Stage 3/4/45 -- the whole V1 manual timeline state: the
    immutable AI base edit identity plus zero or more non-destructive
    B-roll and voice-over layers. Every state-transition function in
    this module takes one of these and returns a NEW one (Stage 15) --
    never mutates `broll_placements`/`voice_over_placements` in
    place (they are tuples, not lists, specifically to make in-place
    mutation impossible)."""

    contract_version: int
    base_edit_identity: str
    timeline_duration_sec: float
    broll_placements: tuple[BrollPlacement, ...] = ()
    voice_over_placements: tuple[VoiceOverPlacement, ...] = ()


@dataclass(frozen=True)
class TimelineValidationResult:
    """Stage 25/26 -- the ONE pure validation verdict. `errors` is
    always populated when `valid` is `False`; never a silent
    truncation or an exception raised out of `validate_composition`."""

    valid: bool
    errors: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TimelineRevisionIdentity:
    """Stage 17/18 -- distinguishes the immutable AI base-render
    identity from a creator's own timeline revision identity (Stage
    18: "Manual edits should not pretend to be the original AI
    render"). `identity` is deterministic over the composition's own
    ordered placement state and never derived from a local path."""

    identity: str
    base_edit_identity: str
    contract_version: int


def _bounds_errors_for(
    prefix: str,
    timeline_duration_sec: float,
    timeline_start_sec: float,
    timeline_end_sec: float,
    source_in_sec: float,
    source_out_sec: float,
    asset_duration_sec: float,
) -> list[str]:
    """Stage 25/26 -- the ONE bounds-check owner, shared by both
    placement kinds (they have byte-identical bounds semantics)."""
    errors: list[str] = []
    if source_in_sec < 0:
        errors.append(f"{prefix}: source_in_sec must be >= 0")
    if source_out_sec > asset_duration_sec:
        errors.append(f"{prefix}: source_out_sec exceeds asset duration")
    if source_out_sec <= source_in_sec:
        errors.append(f"{prefix}: source_out_sec must be > source_in_sec")
    if timeline_start_sec < 0:
        errors.append(f"{prefix}: timeline_start_sec must be >= 0")
    if timeline_end_sec > timeline_duration_sec:
        errors.append(f"{prefix}: timeline_end_sec exceeds timeline duration")
    if timeline_end_sec <= timeline_start_sec:
        errors.append(f"{prefix}: timeline_end_sec must be > timeline_start_sec")
    return errors


def _overlap_errors_for(prefix: str, placements) -> list[str]:
    """Stage 23/24 -- disallow-overlap, the ONE overlap-policy owner
    for both placement kinds. O(n^2) over a per-layer placement count
    that is, by construction, small (a manual creator timeline, never
    a machine-generated one)."""
    errors: list[str] = []
    ordered = sorted(placements, key=lambda p: p.timeline_start_sec)
    for earlier, later in zip(ordered, ordered[1:]):
        if later.timeline_start_sec < earlier.timeline_end_sec:
            errors.append(
                f"{prefix}: placements '{earlier.placement_id}' and "
                f"'{later.placement_id}' overlap in timeline time"
            )
    return errors


def validate_composition(composition: TimelineComposition) -> TimelineValidationResult:
    """Stage 25/26/23/24 -- the ONE pure validator every operation
    below runs before returning a candidate composition. Never
    mutates, never raises; every violation is collected, not just the
    first."""
    errors: list[str] = []
    seen_broll_ids: set[str] = set()
    for placement in composition.broll_placements:
        if placement.placement_id in seen_broll_ids:
            errors.append(f"duplicate broll placement_id: {placement.placement_id}")
        seen_broll_ids.add(placement.placement_id)
        errors.extend(
            _bounds_errors_for(
                f"broll[{placement.placement_id}]",
                composition.timeline_duration_sec,
                placement.timeline_start_sec,
                placement.timeline_end_sec,
                placement.source_in_sec,
                placement.source_out_sec,
                placement.asset.duration_sec,
            )
        )
    errors.extend(_overlap_errors_for("broll", composition.broll_placements))

    seen_vo_ids: set[str] = set()
    for placement in composition.voice_over_placements:
        if placement.placement_id in seen_vo_ids:
            errors.append(f"duplicate voice_over placement_id: {placement.placement_id}")
        seen_vo_ids.add(placement.placement_id)
        errors.extend(
            _bounds_errors_for(
                f"voice_over[{placement.placement_id}]",
                composition.timeline_duration_sec,
                placement.timeline_start_sec,
                placement.timeline_end_sec,
                placement.source_in_sec,
                placement.source_out_sec,
                placement.asset.duration_sec,
            )
        )
    errors.extend(_overlap_errors_for("voice_over", composition.voice_over_placements))

    return TimelineValidationResult(valid=not errors, errors=tuple(errors))


# =============================================================================
# Stage 14/19/20/21/22 -- the ten deterministic manual edit operations.
# Every one of these: (1) never mutates `composition` in place, (2)
# builds a candidate `TimelineComposition`, (3) validates it via
# `validate_composition`, (4) returns `(candidate_or_None,
# TimelineValidationResult)` -- fail-closed: an invalid candidate is
# never returned as the new composition, the CALLER decides what to do
# with the validation errors (Stage 26's own "prefer fail-closed
# contract").
# =============================================================================


def _replace_broll_list(
    composition: TimelineComposition, new_placements: tuple[BrollPlacement, ...]
) -> tuple[TimelineComposition, TimelineValidationResult]:
    candidate = TimelineComposition(
        contract_version=composition.contract_version,
        base_edit_identity=composition.base_edit_identity,
        timeline_duration_sec=composition.timeline_duration_sec,
        broll_placements=new_placements,
        voice_over_placements=composition.voice_over_placements,
    )
    result = validate_composition(candidate)
    return (candidate if result.valid else None), result


def _replace_voice_over_list(
    composition: TimelineComposition, new_placements: tuple[VoiceOverPlacement, ...]
) -> tuple[TimelineComposition, TimelineValidationResult]:
    candidate = TimelineComposition(
        contract_version=composition.contract_version,
        base_edit_identity=composition.base_edit_identity,
        timeline_duration_sec=composition.timeline_duration_sec,
        broll_placements=composition.broll_placements,
        voice_over_placements=new_placements,
    )
    result = validate_composition(candidate)
    return (candidate if result.valid else None), result


def add_broll(composition: TimelineComposition, placement: BrollPlacement):
    """Stage 14 -- ADD_BROLL."""
    return _replace_broll_list(composition, composition.broll_placements + (placement,))


def move_broll(
    composition: TimelineComposition,
    placement_id: str,
    timeline_start_sec: float,
    timeline_end_sec: float,
):
    """Stage 14/19 -- MOVE_BROLL. Changes the timeline interval only
    (Stage 19: "not source clip bytes"); source_in/source_out and
    asset identity are carried over unchanged."""
    existing = next((p for p in composition.broll_placements if p.placement_id == placement_id), None)
    if existing is None:
        return None, TimelineValidationResult(valid=False, errors=(f"unknown broll placement_id: {placement_id}",))
    moved = BrollPlacement(
        placement_id=existing.placement_id, asset=existing.asset,
        timeline_start_sec=timeline_start_sec, timeline_end_sec=timeline_end_sec,
        source_in_sec=existing.source_in_sec, source_out_sec=existing.source_out_sec,
        audio_mode=existing.audio_mode,
    )
    others = tuple(p for p in composition.broll_placements if p.placement_id != placement_id)
    return _replace_broll_list(composition, others + (moved,))


def trim_broll(
    composition: TimelineComposition,
    placement_id: str,
    source_in_sec: float,
    source_out_sec: float,
    timeline_start_sec: float,
    timeline_end_sec: float,
):
    """Stage 14/20 -- TRIM_BROLL. Every new bound is required
    explicitly from the caller (Stage 20: "source_in/source_out and/or
    timeline duration") -- this module never infers a trimmed duration
    on the caller's behalf."""
    existing = next((p for p in composition.broll_placements if p.placement_id == placement_id), None)
    if existing is None:
        return None, TimelineValidationResult(valid=False, errors=(f"unknown broll placement_id: {placement_id}",))
    trimmed = BrollPlacement(
        placement_id=existing.placement_id, asset=existing.asset,
        timeline_start_sec=timeline_start_sec, timeline_end_sec=timeline_end_sec,
        source_in_sec=source_in_sec, source_out_sec=source_out_sec,
        audio_mode=existing.audio_mode,
    )
    others = tuple(p for p in composition.broll_placements if p.placement_id != placement_id)
    return _replace_broll_list(composition, others + (trimmed,))


def replace_broll(composition: TimelineComposition, placement_id: str, new_placement: BrollPlacement):
    """Stage 14/21 -- REPLACE_BROLL. The caller supplies the entire new
    placement (Stage 21: "Do not automatically preserve incompatible
    duration without explicit policy") -- this function only requires
    `new_placement.placement_id == placement_id`, it never guesses
    bounds on the caller's behalf."""
    if new_placement.placement_id != placement_id:
        return None, TimelineValidationResult(
            valid=False, errors=(f"replace_broll: new_placement.placement_id must equal '{placement_id}'",)
        )
    if not any(p.placement_id == placement_id for p in composition.broll_placements):
        return None, TimelineValidationResult(valid=False, errors=(f"unknown broll placement_id: {placement_id}",))
    others = tuple(p for p in composition.broll_placements if p.placement_id != placement_id)
    return _replace_broll_list(composition, others + (new_placement,))


def delete_broll(composition: TimelineComposition, placement_id: str):
    """Stage 14/22 -- DELETE_BROLL. Simply removes the placement; the
    underlying primary A-roll is definitionally visible/audible again
    for that interval since it was never actually removed (Stage 4/22
    -- primary A-roll always remains intact underneath)."""
    if not any(p.placement_id == placement_id for p in composition.broll_placements):
        return None, TimelineValidationResult(valid=False, errors=(f"unknown broll placement_id: {placement_id}",))
    others = tuple(p for p in composition.broll_placements if p.placement_id != placement_id)
    return _replace_broll_list(composition, others)


def add_voice_over(composition: TimelineComposition, placement: VoiceOverPlacement):
    """Stage 14 -- ADD_VOICE_OVER."""
    return _replace_voice_over_list(composition, composition.voice_over_placements + (placement,))


def move_voice_over(
    composition: TimelineComposition,
    placement_id: str,
    timeline_start_sec: float,
    timeline_end_sec: float,
):
    """Stage 14 -- MOVE_VOICE_OVER (same contract as `move_broll`)."""
    existing = next((p for p in composition.voice_over_placements if p.placement_id == placement_id), None)
    if existing is None:
        return None, TimelineValidationResult(valid=False, errors=(f"unknown voice_over placement_id: {placement_id}",))
    moved = VoiceOverPlacement(
        placement_id=existing.placement_id, asset=existing.asset,
        timeline_start_sec=timeline_start_sec, timeline_end_sec=timeline_end_sec,
        source_in_sec=existing.source_in_sec, source_out_sec=existing.source_out_sec,
        transcript_reference=existing.transcript_reference,
    )
    others = tuple(p for p in composition.voice_over_placements if p.placement_id != placement_id)
    return _replace_voice_over_list(composition, others + (moved,))


def trim_voice_over(
    composition: TimelineComposition,
    placement_id: str,
    source_in_sec: float,
    source_out_sec: float,
    timeline_start_sec: float,
    timeline_end_sec: float,
):
    """Stage 14 -- TRIM_VOICE_OVER (same contract as `trim_broll`)."""
    existing = next((p for p in composition.voice_over_placements if p.placement_id == placement_id), None)
    if existing is None:
        return None, TimelineValidationResult(valid=False, errors=(f"unknown voice_over placement_id: {placement_id}",))
    trimmed = VoiceOverPlacement(
        placement_id=existing.placement_id, asset=existing.asset,
        timeline_start_sec=timeline_start_sec, timeline_end_sec=timeline_end_sec,
        source_in_sec=source_in_sec, source_out_sec=source_out_sec,
        transcript_reference=existing.transcript_reference,
    )
    others = tuple(p for p in composition.voice_over_placements if p.placement_id != placement_id)
    return _replace_voice_over_list(composition, others + (trimmed,))


def replace_voice_over(composition: TimelineComposition, placement_id: str, new_placement: VoiceOverPlacement):
    """Stage 14 -- REPLACE_VOICE_OVER (same contract as `replace_broll`).
    Stage 34's caption-regeneration seam: a caller replacing a VO
    placement is expected to also refresh that region's caption source
    to the new VO's own `transcript_reference` -- this module only
    carries the reference; it never regenerates a transcript itself."""
    if new_placement.placement_id != placement_id:
        return None, TimelineValidationResult(
            valid=False, errors=(f"replace_voice_over: new_placement.placement_id must equal '{placement_id}'",)
        )
    if not any(p.placement_id == placement_id for p in composition.voice_over_placements):
        return None, TimelineValidationResult(valid=False, errors=(f"unknown voice_over placement_id: {placement_id}",))
    others = tuple(p for p in composition.voice_over_placements if p.placement_id != placement_id)
    return _replace_voice_over_list(composition, others + (new_placement,))


def delete_voice_over(composition: TimelineComposition, placement_id: str):
    """Stage 14 -- DELETE_VOICE_OVER (same contract as `delete_broll`).
    Restores the original primary voice for that region (Stage 34's
    own implied inverse: captions for that region should then revert
    to the primary-voice transcript source -- again, a caller
    responsibility, not implemented here)."""
    if not any(p.placement_id == placement_id for p in composition.voice_over_placements):
        return None, TimelineValidationResult(valid=False, errors=(f"unknown voice_over placement_id: {placement_id}",))
    others = tuple(p for p in composition.voice_over_placements if p.placement_id != placement_id)
    return _replace_voice_over_list(composition, others)


def caption_source_for_region(
    composition: TimelineComposition, timeline_time_sec: float
) -> str:
    """Stage 13/34 -- deterministic caption-transcript-source policy
    for one instant on the timeline. A voice-over placement covering
    this instant always wins (Stage 33/34: VO replaces the primary
    voice, so its transcript must supply the captions there); B-roll
    `USE_BROLL_AUDIO` covering this instant is the second priority
    (the creator intentionally chose that clip's own audio); otherwise
    the original primary voice transcript applies. Returns one of
    `"VOICE_OVER"`, `"BROLL_SOURCE_AUDIO"`, `"ORIGINAL_PRIMARY_VOICE"`
    -- never a transcript string itself (Stage 13's own "decide caption
    transcript source", not "generate captions")."""
    for vo in composition.voice_over_placements:
        if vo.timeline_start_sec <= timeline_time_sec < vo.timeline_end_sec:
            return "VOICE_OVER"
    for broll in composition.broll_placements:
        if broll.timeline_start_sec <= timeline_time_sec < broll.timeline_end_sec:
            if broll.audio_mode == TimelineAudioMode.USE_BROLL_AUDIO:
                return "BROLL_SOURCE_AUDIO"
    return "ORIGINAL_PRIMARY_VOICE"


def compute_timeline_identity(composition: TimelineComposition) -> str:
    """Stage 17 -- a deterministic semantic identity over the base edit
    identity, every placement's own semantic fields (never a local
    path), and the contract version. Placements are sorted by
    `(timeline_start_sec, placement_id)` before hashing so identity is
    independent of caller-supplied tuple order (Stage 17's own
    "path-independent identity", generalized to order-independence)."""

    def _broll_key(p: BrollPlacement):
        return {
            "placement_id": p.placement_id,
            "asset_id": p.asset.asset_id,
            "source_media_identity": p.asset.source_media_identity,
            "timeline_start_sec": round(p.timeline_start_sec, 6),
            "timeline_end_sec": round(p.timeline_end_sec, 6),
            "source_in_sec": round(p.source_in_sec, 6),
            "source_out_sec": round(p.source_out_sec, 6),
            "audio_mode": p.audio_mode.value,
        }

    def _vo_key(p: VoiceOverPlacement):
        return {
            "placement_id": p.placement_id,
            "asset_id": p.asset.asset_id,
            "source_media_identity": p.asset.source_media_identity,
            "timeline_start_sec": round(p.timeline_start_sec, 6),
            "timeline_end_sec": round(p.timeline_end_sec, 6),
            "source_in_sec": round(p.source_in_sec, 6),
            "source_out_sec": round(p.source_out_sec, 6),
            "transcript_reference": p.transcript_reference,
        }

    payload = {
        "contract_version": composition.contract_version,
        "base_edit_identity": composition.base_edit_identity,
        "timeline_duration_sec": round(composition.timeline_duration_sec, 6),
        "broll": sorted(
            (_broll_key(p) for p in composition.broll_placements),
            key=lambda d: (d["timeline_start_sec"], d["placement_id"]),
        ),
        "voice_over": sorted(
            (_vo_key(p) for p in composition.voice_over_placements),
            key=lambda d: (d["timeline_start_sec"], d["placement_id"]),
        ),
    }
    canonical_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(canonical_bytes).hexdigest()[:24]
    return f"timeline_{digest}"


def derive_revision_identity(composition: TimelineComposition) -> TimelineRevisionIdentity:
    """Stage 17/18 -- the ONE function that turns a composition into
    its own revision identity, kept distinct from
    `base_edit_identity` (Stage 18: "Manual edits should not pretend
    to be the original AI render")."""
    return TimelineRevisionIdentity(
        identity=compute_timeline_identity(composition),
        base_edit_identity=composition.base_edit_identity,
        contract_version=composition.contract_version,
    )
