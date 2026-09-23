"""Canonical Source Normalization Contract + Plan Types -- D-274A.

Post D-273 (source normalization architecture design, Verdict A). This
module implements the ONE typed source of truth a future normalization
executor (D-274B onward) will consume: the canonical target media
contract (`CanonicalSourceMediaContract`) and an immutable, deterministic
`SourceNormalizationPlan` built from a `NORMALIZE_REQUIRED` D-272
decision.

## What this module explicitly does NOT do (D-274A's own binding scope)

No normalization execution, no ffmpeg command, no filtergraph string, no
pixel rotation, no VFR->CFR resample, no HDR tonemap, no HEVC->H264
transcode, no 10-bit->8-bit conversion, no audio conversion, no media
bytes read or written. Every function here is a pure function of typed
inputs (a `SourceMediaProfile`, a `SourceFormatPolicyDecision`, an
opaque caller-supplied source identity) to a typed, immutable plan or
verification result. `probe_source_media_profile` and
`evaluate_source_format_policy` remain the sole owners of media-fact and
policy-decision authority respectively; this module never re-derives or
duplicates either.

## A gap D-274A surfaced, reconciled by D-272B

D-274A's own first version recorded a real gap here: D-272's policy did
not emit a normalization reason for "codec is HEVC" by itself, so a
capability-confirmed HEVC source with no other defect resolved to plain
`ACCEPT`, never reaching this plan builder. D-272B (a narrow, Product-
Owner-authorized reconciliation, not a general D-272 reopening) closed
this: `evaluate_source_format_policy` now emits `REASON_HEVC_TO_H264_
NORMALIZATION_REQUIRED` itself whenever a confirmed-capability HEVC
source needs normalizing, even when otherwise clean -- HEVC is never
ACCEPT-native in the canonical V1 source contract. `codec_action` in
this module reads that reason directly (`sfp.REASON_HEVC_TO_H264_
NORMALIZATION_REQUIRED in reasons`), never re-deriving the HEVC-needs-
normalization fact from `profile.video_codec` independently -- D-272
remains the sole authority deciding normalization is required; this
module only maps that decision to a plan action (Stage 4's own "do not
duplicate HEVC policy here"). `timeline_action` still has no
corresponding D-272 reason code (D-272B's own scope was HEVC only) and
remains computed only inside an already-`NORMALIZE_REQUIRED` plan,
exactly as before.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from . import source_format_policy as sfp
from . import source_media_profile as smp

# =============================================================================
# Stage 1 -- normalization contract version, single canonical owner
# =============================================================================

SOURCE_NORMALIZATION_CONTRACT_VERSION = 1

# =============================================================================
# Stage 3 -- canonical V1 contract values (D-273's own decisions, encoded
# verbatim -- never silently widened here)
# =============================================================================

CONTRACT_CONTAINER_MP4 = "MP4"
CONTRACT_VIDEO_CODEC_H264 = "H264"
CONTRACT_PIXEL_FORMAT_YUV420P = "YUV420P"
CONTRACT_BIT_DEPTH_8 = 8

CONTRACT_ORIENTATION_PHYSICAL_PIXELS_ORIENTED = "PHYSICAL_PIXELS_ORIENTED"
CONTRACT_ROTATION_METADATA_ZERO_OR_ABSENT = "0_OR_ABSENT"

# Stage 4: deliberately NOT a flat "always 30fps" policy -- CFR sources
# of any real rate are preserved untouched (the renderer's own fps=
# filter already retimes to 30fps at render time); only VFR normalizes,
# to the source's own measured average rate.
CONTRACT_FRAME_RATE_POLICY_PRESERVE_CFR_NORMALIZE_VFR_TO_SOURCE_RATE = (
    "PRESERVE_CFR_NORMALIZE_VFR_TO_SOURCE_RATE"
)

CONTRACT_TIMELINE_START_AT_ZERO = "START_AT_ZERO"
CONTRACT_TIMELINE_MONOTONIC_TIMESTAMPS = "MONOTONIC_TIMESTAMPS"
CONTRACT_TIMELINE_NO_NEGATIVE_PTS_DTS = "NO_NEGATIVE_PTS_DTS"

CONTRACT_COLOR_PRIMARIES_BT709 = "BT709"
CONTRACT_COLOR_TRANSFER_BT709 = "BT709"
CONTRACT_COLOR_SPACE_BT709 = "BT709"
CONTRACT_COLOR_RANGE_TV = "TV"
CONTRACT_HDR_TARGET_SDR = "SDR"

# Stage 5: D-273 decided NO mandatory source-layer audio normalization
# for V1 -- represented explicitly as a policy value, not an omission.
CONTRACT_AUDIO_POLICY_PRESERVE_SOURCE_NO_MANDATORY_NORMALIZATION = (
    "PRESERVE_SOURCE_NO_MANDATORY_NORMALIZATION"
)
CONTRACT_AUDIO_STREAM_POLICY_SINGLE_STREAM_OR_ABSENT = "SINGLE_STREAM_OR_ABSENT"
CONTRACT_EXPECTED_VIDEO_STREAM_COUNT = 1


@dataclass(frozen=True)
class CanonicalSourceMediaContract:
    """D-274A Stage 2: the immutable target every normalized source is
    planned against. Pure data -- no execution semantics beyond what
    D-273 already decided."""

    container: str
    video_codec: str
    pixel_format: str
    bit_depth: int

    orientation_mode: str
    rotation_metadata_expected: str

    frame_rate_policy: str
    timeline_start_policy: tuple[str, ...]

    color_primaries: str
    color_transfer: str
    color_space: str
    color_range: str
    hdr_target: str

    audio_policy: str
    expected_video_stream_count: int
    expected_audio_stream_policy: str

    contract_version: int = SOURCE_NORMALIZATION_CONTRACT_VERSION


CANONICAL_SOURCE_MEDIA_CONTRACT_V1 = CanonicalSourceMediaContract(
    container=CONTRACT_CONTAINER_MP4,
    video_codec=CONTRACT_VIDEO_CODEC_H264,
    pixel_format=CONTRACT_PIXEL_FORMAT_YUV420P,
    bit_depth=CONTRACT_BIT_DEPTH_8,
    orientation_mode=CONTRACT_ORIENTATION_PHYSICAL_PIXELS_ORIENTED,
    rotation_metadata_expected=CONTRACT_ROTATION_METADATA_ZERO_OR_ABSENT,
    frame_rate_policy=CONTRACT_FRAME_RATE_POLICY_PRESERVE_CFR_NORMALIZE_VFR_TO_SOURCE_RATE,
    timeline_start_policy=(
        CONTRACT_TIMELINE_START_AT_ZERO,
        CONTRACT_TIMELINE_MONOTONIC_TIMESTAMPS,
        CONTRACT_TIMELINE_NO_NEGATIVE_PTS_DTS,
    ),
    color_primaries=CONTRACT_COLOR_PRIMARIES_BT709,
    color_transfer=CONTRACT_COLOR_TRANSFER_BT709,
    color_space=CONTRACT_COLOR_SPACE_BT709,
    color_range=CONTRACT_COLOR_RANGE_TV,
    hdr_target=CONTRACT_HDR_TARGET_SDR,
    audio_policy=CONTRACT_AUDIO_POLICY_PRESERVE_SOURCE_NO_MANDATORY_NORMALIZATION,
    expected_video_stream_count=CONTRACT_EXPECTED_VIDEO_STREAM_COUNT,
    expected_audio_stream_policy=CONTRACT_AUDIO_STREAM_POLICY_SINGLE_STREAM_OR_ABSENT,
)

# =============================================================================
# Stage 6 -- normalization action vocabulary (bounded, never proliferated)
# =============================================================================

ACTION_NO_ACTION = "NO_ACTION"

ACTION_ROTATE_90 = "ROTATE_90"
ACTION_ROTATE_180 = "ROTATE_180"
ACTION_ROTATE_270 = "ROTATE_270"

ACTION_VFR_TO_CFR = "VFR_TO_CFR"

ACTION_HDR_PQ_TO_SDR_BT709 = "HDR_PQ_TO_SDR_BT709"
ACTION_HDR_HLG_TO_SDR_BT709 = "HDR_HLG_TO_SDR_BT709"

ACTION_HEVC_TO_H264 = "HEVC_TO_H264"

ACTION_TEN_BIT_TO_EIGHT_BIT = "TEN_BIT_TO_EIGHT_BIT"
ACTION_PIXEL_FORMAT_TO_YUV420P = "PIXEL_FORMAT_TO_YUV420P"

ACTION_TIMELINE_TO_ZERO = "TIMELINE_TO_ZERO"

# Stage 31: the semantic canonical order for both plan-identity hashing
# and any future executor's own action sequencing. Design-only -- no
# ffmpeg filtergraph is emitted anywhere in this module (Stage 32).
_CANONICAL_ACTION_FIELD_ORDER: tuple[str, ...] = (
    "timeline_action",
    "rotation_action",
    "codec_action",
    "hdr_action",
    "bit_depth_action",
    "pixel_format_action",
    "frame_rate_action",
    "container_action",
    "audio_action",
)

# =============================================================================
# Stage 16 -- plan executability vocabulary
# =============================================================================

EXECUTABILITY_EXECUTABLE = "EXECUTABLE"
EXECUTABILITY_NOT_REQUIRED = "NOT_REQUIRED"
EXECUTABILITY_UNSUPPORTED = "UNSUPPORTED"
EXECUTABILITY_CAPABILITY_UNVERIFIED = "CAPABILITY_UNVERIFIED"
EXECUTABILITY_INVALID_SOURCE_STATE = "INVALID_SOURCE_STATE"

# =============================================================================
# Stage 17 -- normalization outcome vocabulary (immutable, typed)
# =============================================================================

NORMALIZATION_NOT_REQUIRED = "NORMALIZATION_NOT_REQUIRED"
NORMALIZATION_PLANNED = "NORMALIZATION_PLANNED"
NORMALIZATION_SUCCEEDED = "NORMALIZATION_SUCCEEDED"
NORMALIZATION_FAILED = "NORMALIZATION_FAILED"
NORMALIZATION_UNSUPPORTED = "NORMALIZATION_UNSUPPORTED"
NORMALIZATION_VERIFICATION_FAILED = "NORMALIZATION_VERIFICATION_FAILED"

# Stage 36 -- future executor failure categories (vocabulary only, never
# raised/constructed by this offline, execution-free gate).
NORMALIZATION_FFMPEG_FAILED = "NORMALIZATION_FFMPEG_FAILED"
NORMALIZATION_TIMEOUT = "NORMALIZATION_TIMEOUT"
NORMALIZATION_OUTPUT_MISSING = "NORMALIZATION_OUTPUT_MISSING"
NORMALIZATION_OUTPUT_EMPTY = "NORMALIZATION_OUTPUT_EMPTY"
NORMALIZATION_PROFILE_MISMATCH = "NORMALIZATION_PROFILE_MISMATCH"
NORMALIZATION_POLICY_STILL_BLOCKED = "NORMALIZATION_POLICY_STILL_BLOCKED"
NORMALIZATION_UNSUPPORTED_HDR = "NORMALIZATION_UNSUPPORTED_HDR"
NORMALIZATION_CODEC_UNAVAILABLE = "NORMALIZATION_CODEC_UNAVAILABLE"

# =============================================================================
# Stage 20 -- one-pass firewall
# =============================================================================

MAX_NORMALIZATION_ATTEMPTS = 1


def is_normalization_attempt_allowed(attempt_count: int) -> bool:
    """Stage 20/28: at most ONE normalization pass per source per job,
    ever. This module defines the invariant; it builds no retry loop."""
    return int(attempt_count) < MAX_NORMALIZATION_ATTEMPTS


# =============================================================================
# Stage 8 -- the plan type itself
# =============================================================================

@dataclass(frozen=True)
class SourceNormalizationPlan:
    """D-274A Stage 8: one immutable, deterministic normalization plan.
    No executor fields -- no temp paths, no subprocess handles, no
    filesystem state of any kind (Stage 8's own explicit instruction)."""

    source_identity: str
    source_profile_reference: str
    contract_version: int

    container_action: str
    codec_action: str
    rotation_action: str
    frame_rate_action: str
    hdr_action: str
    bit_depth_action: str
    pixel_format_action: str
    timeline_action: str
    audio_action: str

    target_contract: CanonicalSourceMediaContract
    reason_codes: tuple[str, ...]

    executability: str

    plan_identity: str

    # D-274B Stage 10: a narrow, additive field this gate's own executor
    # needs and D-274A's original type did not carry -- the NUMERIC VFR
    # target rate, derived ONLY from `profile.effective_fps` (D-271),
    # never a hardcoded 30. `None` whenever `frame_rate_action` is
    # `ACTION_NO_ACTION` (no VFR normalization requested). Defaulted so
    # every existing D-274A call site/test that never mentions this field
    # is unaffected.
    target_fps: float | None = None

    @property
    def is_executable(self) -> bool:
        return self.executability == EXECUTABILITY_EXECUTABLE


@dataclass(frozen=True)
class SourceNormalizationPlanResult:
    """D-274A: the return type of `build_source_normalization_plan`.
    `plan` is `None` exactly when `outcome` is `NORMALIZATION_NOT_
    REQUIRED` (ACCEPT) -- every other outcome still carries a plan
    describing what WOULD be needed, even when not currently
    executable (Stage 35's own diagnostics requirement)."""

    outcome: str
    plan: SourceNormalizationPlan | None
    reason_codes: tuple[str, ...]
    blocking_capability_gaps: tuple[str, ...]


# =============================================================================
# Stage 9 -- deterministic plan identity
# =============================================================================

def _compute_plan_identity(
    *,
    source_identity: str,
    contract_version: int,
    ordered_actions: tuple[str, ...],
    target_contract: CanonicalSourceMediaContract,
) -> str:
    """D-274A Stage 9: SHA-256 over source identity + contract version +
    the ORDERED semantic action tuple + the target contract -- reusing
    `render_delivery.py`'s own `json.dumps(sort_keys=True, default=str)`
    -> SHA-256 -> bounded-digest[:24] identity convention (D-267), never
    a new scheme. Path/filename never enter this payload (Stage 9/10's
    own explicit instruction) -- `source_identity` is an opaque,
    caller-supplied string, never a filesystem path."""
    payload = {
        "source_identity": source_identity,
        "contract_version": int(contract_version),
        "actions": list(ordered_actions),
        "target_contract": asdict(target_contract),
    }
    normalized = json.dumps(payload, sort_keys=True, default=str)
    return "normplan_" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


# =============================================================================
# Stage 11-15/25-29 -- the plan builder itself
# =============================================================================

def build_source_normalization_plan(
    source_identity: str,
    profile: "smp.SourceMediaProfile",
    policy_decision: "sfp.SourceFormatPolicyDecision",
    *,
    runtime_capability: "sfp.RuntimeCapabilityInput" | None = None,
    tonemap_available: bool = False,
    target_contract: CanonicalSourceMediaContract = CANONICAL_SOURCE_MEDIA_CONTRACT_V1,
) -> SourceNormalizationPlanResult:
    """D-274A Stage 11: pure function from a D-271 profile + a D-272
    decision to a `SourceNormalizationPlanResult`. Never re-probes,
    never re-evaluates policy, never touches a filesystem.

    Stage 10: `source_identity` must be a non-empty, caller-supplied
    opaque identity -- never a local filesystem path, never fabricated
    from a filename here.

    `tonemap_available` is a SEPARATE, explicit capability signal from
    `runtime_capability` (D-272's own `RuntimeCapabilityInput`, which
    only carries `hevc_decode_confirmed`/`av1_decode_confirmed`) --
    D-273 Stage 15's own "must never invent a production capability
    source" discipline applies here identically: defaults to `False`
    (unconfirmed); no caller in this codebase constructs a `True` value
    today.
    """
    if not isinstance(source_identity, str) or not source_identity.strip():
        raise ValueError("source_identity must be a non-empty, caller-supplied opaque identity")

    capability = runtime_capability or sfp.RuntimeCapabilityInput()

    # Stage 12: ACCEPT always means NOT_REQUIRED, no plan, no fake actions.
    if policy_decision.decision == sfp.DECISION_ACCEPT:
        return SourceNormalizationPlanResult(
            outcome=NORMALIZATION_NOT_REQUIRED,
            plan=None,
            reason_codes=(),
            blocking_capability_gaps=(),
        )

    # Stage 11: never build a plan for REJECT or INSUFFICIENT_EVIDENCE --
    # a source blocked for reasons other than "needs normalization" has
    # no safe normalization target.
    if policy_decision.decision in (sfp.DECISION_REJECT, sfp.DECISION_INSUFFICIENT_EVIDENCE):
        return SourceNormalizationPlanResult(
            outcome=NORMALIZATION_UNSUPPORTED,
            plan=None,
            reason_codes=policy_decision.blocking_reasons,
            blocking_capability_gaps=(),
        )

    # From here: policy_decision.decision == DECISION_NORMALIZE_REQUIRED
    reasons = policy_decision.normalization_reasons
    capability_gaps: list[str] = []
    unsupported = False
    invalid_state = False

    # Stage 13/27 -- rotation
    rotation_action = ACTION_NO_ACTION
    if sfp.REASON_ROTATION_NORMALIZATION_REQUIRED in reasons:
        rotation_map = {90: ACTION_ROTATE_90, 180: ACTION_ROTATE_180, 270: ACTION_ROTATE_270}
        mapped = rotation_map.get(profile.rotation_degrees)
        if mapped is None:
            # Stage 27: malformed/unknown rotation -- no executable action.
            invalid_state = True
        else:
            rotation_action = mapped

    # Stage 13/26 -- VFR
    frame_rate_action = ACTION_NO_ACTION
    target_fps: float | None = None
    if sfp.REASON_VFR_NORMALIZATION_REQUIRED in reasons:
        if profile.effective_fps is not None and profile.effective_fps > 0:
            frame_rate_action = ACTION_VFR_TO_CFR
            # D-274B Stage 10: the numeric target ONLY ever comes from the
            # source's own measured average rate -- never a hardcoded 30.
            target_fps = float(profile.effective_fps)
        else:
            # Stage 26: effective fps unavailable -- non-executable / insufficient evidence.
            invalid_state = True

    # Stage 13/15/28 -- HDR (capability-gated, Dolby Vision/OTHER unsupported)
    hdr_action = ACTION_NO_ACTION
    if sfp.REASON_HDR_NORMALIZATION_REQUIRED in reasons:
        if profile.hdr_status == smp.HDR_STATUS_HDR_PQ:
            hdr_action = ACTION_HDR_PQ_TO_SDR_BT709
            if not tonemap_available:
                capability_gaps.append("tonemap_available")
        elif profile.hdr_status == smp.HDR_STATUS_HDR_HLG:
            hdr_action = ACTION_HDR_HLG_TO_SDR_BT709
            if not tonemap_available:
                capability_gaps.append("tonemap_available")
        else:
            # Stage 7/14: Dolby Vision / HDR_OTHER -- never a supported
            # action; do not create a plan implying DV support.
            unsupported = True

    # Stage 13 -- 10-bit / pixel format
    bit_depth_action = ACTION_NO_ACTION
    if sfp.REASON_TEN_BIT_NORMALIZATION_REQUIRED in reasons:
        bit_depth_action = ACTION_TEN_BIT_TO_EIGHT_BIT

    pixel_format_action = ACTION_NO_ACTION
    if sfp.REASON_PIXEL_FORMAT_NORMALIZATION_REQUIRED in reasons:
        pixel_format_action = ACTION_PIXEL_FORMAT_TO_YUV420P

    # Stage 14 -- HEVC. D-272B closed the D-272/D-273 reconciliation gap
    # this module's own docstring used to record: D-272 now emits
    # REASON_HEVC_TO_H264_NORMALIZATION_REQUIRED itself whenever a
    # confirmed-capability HEVC source needs normalizing (even when
    # otherwise clean) -- D-272 remains the sole authority deciding
    # normalization is required; this module only maps that decision to
    # a plan action, never re-deriving it from the profile's own codec
    # field directly (Stage 4's own "do not duplicate HEVC policy here").
    codec_action = ACTION_NO_ACTION
    if sfp.REASON_HEVC_TO_H264_NORMALIZATION_REQUIRED in reasons:
        codec_action = ACTION_HEVC_TO_H264
        if not getattr(capability, "hevc_decode_confirmed", False):
            capability_gaps.append("hevc_decode_confirmed")

    # Stage 25 -- timeline (D-272 itself has no reason code for this yet;
    # this module inspects the profile's own start-time evidence directly,
    # but only as an ADDITIVE action within an already-triggered plan --
    # never independently promoting an ACCEPT source, per Stage 12).
    timeline_action = ACTION_NO_ACTION
    non_zero_start = any(
        value is not None and abs(value) > 0.0
        for value in (profile.format_start_time, profile.video_stream_start_time, profile.audio_stream_start_time)
    )
    if non_zero_start:
        timeline_action = ACTION_TIMELINE_TO_ZERO

    container_action = ACTION_NO_ACTION
    audio_action = ACTION_NO_ACTION

    actions_by_field = {
        "timeline_action": timeline_action,
        "rotation_action": rotation_action,
        "codec_action": codec_action,
        "hdr_action": hdr_action,
        "bit_depth_action": bit_depth_action,
        "pixel_format_action": pixel_format_action,
        "frame_rate_action": frame_rate_action,
        "container_action": container_action,
        "audio_action": audio_action,
    }
    ordered_actions = tuple(actions_by_field[name] for name in _CANONICAL_ACTION_FIELD_ORDER)

    plan_identity = _compute_plan_identity(
        source_identity=source_identity,
        contract_version=target_contract.contract_version,
        ordered_actions=ordered_actions,
        target_contract=target_contract,
    )

    if unsupported:
        executability = EXECUTABILITY_UNSUPPORTED
        outcome = NORMALIZATION_UNSUPPORTED
    elif invalid_state:
        executability = EXECUTABILITY_INVALID_SOURCE_STATE
        outcome = NORMALIZATION_UNSUPPORTED
    elif capability_gaps:
        executability = EXECUTABILITY_CAPABILITY_UNVERIFIED
        outcome = NORMALIZATION_UNSUPPORTED
    else:
        executability = EXECUTABILITY_EXECUTABLE
        outcome = NORMALIZATION_PLANNED

    plan = SourceNormalizationPlan(
        source_identity=source_identity,
        source_profile_reference=policy_decision.source_profile_status,
        contract_version=target_contract.contract_version,
        container_action=container_action,
        codec_action=codec_action,
        rotation_action=rotation_action,
        frame_rate_action=frame_rate_action,
        hdr_action=hdr_action,
        bit_depth_action=bit_depth_action,
        pixel_format_action=pixel_format_action,
        timeline_action=timeline_action,
        audio_action=audio_action,
        target_contract=target_contract,
        reason_codes=tuple(reasons),
        executability=executability,
        plan_identity=plan_identity,
        target_fps=target_fps,
    )

    return SourceNormalizationPlanResult(
        outcome=outcome,
        plan=plan,
        reason_codes=tuple(reasons),
        blocking_capability_gaps=tuple(capability_gaps),
    )


# =============================================================================
# Stage 23/24 -- derived normalized-source identity design (types only)
# =============================================================================

@dataclass(frozen=True)
class NormalizedSourceReference:
    """D-274A Stage 23/24: the identity shape a FUTURE executor (D-274B
    onward) will populate once normalization actually runs. Fields only
    -- never a local temp path, never a reuse of the original source's
    own hash."""

    original_source_identity: str
    normalization_plan_identity: str
    normalized_output_sha256: str | None = None


# =============================================================================
# Stage 18/19 -- verification contract (pure comparison, no execution)
# =============================================================================

@dataclass(frozen=True)
class NormalizationVerificationResult:
    """D-274A Stage 18: a pure comparison type. A future executor calls
    D-271's `probe_source_media_profile` and D-272's `evaluate_source_
    format_policy` on its OWN normalized output, then hands both results
    here -- this module never re-probes or re-evaluates itself, and
    never reads media bytes."""

    normalized_profile: "smp.SourceMediaProfile"
    normalized_policy_decision: "sfp.SourceFormatPolicyDecision"
    contract_matches: bool
    profile_matches_target: bool
    policy_accepts: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    verified: bool


def verify_normalized_source(
    normalized_profile: "smp.SourceMediaProfile",
    normalized_policy_decision: "sfp.SourceFormatPolicyDecision",
) -> NormalizationVerificationResult:
    """D-274A Stage 19: the mandatory re-probe/re-evaluate contract,
    encoded as a pure function. Only `ACCEPT` on re-evaluation is
    `verified=True` -- a still-blocked normalized output (Stage 21:
    NORMALIZE_REQUIRED/REJECT/INSUFFICIENT_EVIDENCE) is `verified=False`
    with the still-blocking reasons carried as `errors`, never silently
    treated as success."""
    accepted = normalized_policy_decision.decision == sfp.DECISION_ACCEPT
    still_blocking = normalized_policy_decision.blocking_reasons or normalized_policy_decision.normalization_reasons
    return NormalizationVerificationResult(
        normalized_profile=normalized_profile,
        normalized_policy_decision=normalized_policy_decision,
        contract_matches=accepted,
        profile_matches_target=accepted,
        policy_accepts=accepted,
        errors=() if accepted else tuple(still_blocking),
        warnings=tuple(normalized_policy_decision.warnings),
        verified=accepted,
    )


def verification_outcome(result: NormalizationVerificationResult) -> str:
    """Stage 17/21: only `NORMALIZATION_SUCCEEDED` or `NORMALIZATION_
    VERIFICATION_FAILED` -- never fabricates success from an unverified
    result."""
    return NORMALIZATION_SUCCEEDED if result.verified else NORMALIZATION_VERIFICATION_FAILED


# =============================================================================
# Stage 35 -- bounded, structured plan diagnostics (no media bytes, no secrets)
# =============================================================================

def plan_diagnostics(result: SourceNormalizationPlanResult) -> dict:
    """D-274A Stage 35: bounded, machine-readable diagnostics -- no
    filesystem paths, no credentials, no raw media data."""
    plan = result.plan
    return {
        "outcome": result.outcome,
        "contract_version": SOURCE_NORMALIZATION_CONTRACT_VERSION,
        "plan_identity": plan.plan_identity if plan else None,
        "source_identity": plan.source_identity if plan else None,
        "actions": (
            {
                "container": plan.container_action,
                "codec": plan.codec_action,
                "rotation": plan.rotation_action,
                "frame_rate": plan.frame_rate_action,
                "hdr": plan.hdr_action,
                "bit_depth": plan.bit_depth_action,
                "pixel_format": plan.pixel_format_action,
                "timeline": plan.timeline_action,
                "audio": plan.audio_action,
            }
            if plan
            else {}
        ),
        "executability": plan.executability if plan else None,
        "blocking_capability_gaps": list(result.blocking_capability_gaps),
        "reason_codes": list(result.reason_codes),
    }
