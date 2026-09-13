"""D-274A -- CANONICAL SOURCE NORMALIZATION CONTRACT + PLAN TYPES.

Post D-273 (source normalization architecture design, Verdict A).
Proves `cutsell_worker.source_normalization_plan`'s
`CanonicalSourceMediaContract`, `SourceNormalizationPlan`,
`build_source_normalization_plan`, the normalization action/outcome/
executability vocabularies, `NormalizationVerificationResult`, the
one-pass firewall, and plan-identity determinism -- entirely via typed,
in-memory `SourceMediaProfile`/`SourceFormatPolicyDecision` fixtures.

No normalization execution, no ffmpeg command, no filtergraph string,
no pixel rotation, no VFR->CFR resample, no HDR tonemap, no HEVC->H264
transcode, no 10-bit->8-bit conversion, no audio conversion, no RAW, no
provider anywhere in this file.
"""
from __future__ import annotations

import ast
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from cutsell_worker import source_format_policy as sfp
from cutsell_worker import source_media_profile as smp
from cutsell_worker import source_normalization_plan as snp


def _source_without_docstrings(path: str) -> str:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(getattr(node.body[0], "value", None), ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body[0] = ast.Expr(value=ast.Constant(value=""))
    return ast.unparse(tree)


def _run_git_diff(rel_path: str) -> str:
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", rel_path], capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _base_profile(**overrides) -> smp.SourceMediaProfile:
    fields = dict(
        path="unused", probe_status=smp.PROBE_STATUS_COMPLETE,
        container_name=smp.CONTAINER_MP4, raw_format_name="mov,mp4", duration_sec=5.0, file_size_bytes=1000,
        video_presence=smp.VIDEO_PRESENT, audio_presence=smp.AUDIO_PRESENT,
        video_stream_count=1, audio_stream_count=1,
        video_codec=smp.VIDEO_CODEC_H264, raw_video_codec="h264", video_profile=None,
        pixel_format="yuv420p", bit_depth=8,
        coded_width=1920, coded_height=1080, display_width=1920, display_height=1080,
        rotation_degrees=0, rotation_source=smp.ROTATION_SOURCE_NONE,
        avg_frame_rate=30.0, r_frame_rate=30.0, effective_fps=30.0, vfr_status=smp.VFR_STATUS_CFR,
        color_primaries="bt709", color_transfer="bt709", color_space="bt709", color_range="tv",
        hdr_status=smp.HDR_STATUS_SDR,
        audio_codec="aac", raw_audio_codec="aac", audio_sample_rate_hz=48000, audio_channels=2,
        audio_channel_layout="stereo",
        format_start_time=0.0, video_stream_start_time=0.0, audio_stream_start_time=0.0,
        video_time_base="1/30000", audio_time_base="1/48000",
    )
    fields.update(overrides)
    return smp.SourceMediaProfile(**fields)


def _decide(profile, **kwargs) -> sfp.SourceFormatPolicyDecision:
    return sfp.evaluate_source_format_policy(profile, **kwargs)


# ---------------------------------------------------------------------------
# Contract shape / canonical values (Stage 1-5, test contract items 1-14)
# ---------------------------------------------------------------------------

def test_one_contract_version_owner():
    assert snp.SOURCE_NORMALIZATION_CONTRACT_VERSION == 1
    assert snp.CANONICAL_SOURCE_MEDIA_CONTRACT_V1.contract_version == snp.SOURCE_NORMALIZATION_CONTRACT_VERSION


def test_canonical_container_mp4():
    assert snp.CANONICAL_SOURCE_MEDIA_CONTRACT_V1.container == snp.CONTRACT_CONTAINER_MP4 == "MP4"


def test_canonical_video_codec_h264():
    assert snp.CANONICAL_SOURCE_MEDIA_CONTRACT_V1.video_codec == "H264"


def test_canonical_pixel_format_yuv420p():
    assert snp.CANONICAL_SOURCE_MEDIA_CONTRACT_V1.pixel_format == "YUV420P"


def test_canonical_bit_depth_8():
    assert snp.CANONICAL_SOURCE_MEDIA_CONTRACT_V1.bit_depth == 8


def test_physical_orientation_target():
    assert snp.CANONICAL_SOURCE_MEDIA_CONTRACT_V1.orientation_mode == snp.CONTRACT_ORIENTATION_PHYSICAL_PIXELS_ORIENTED


def test_rotation_metadata_zero_or_absent():
    assert snp.CANONICAL_SOURCE_MEDIA_CONTRACT_V1.rotation_metadata_expected == "0_OR_ABSENT"


def test_cfr_preserved_policy_in_contract():
    assert "PRESERVE_CFR" in snp.CANONICAL_SOURCE_MEDIA_CONTRACT_V1.frame_rate_policy


def test_not_fixed_30fps_in_contract():
    assert "30" not in snp.CANONICAL_SOURCE_MEDIA_CONTRACT_V1.frame_rate_policy


def test_sdr_bt709_target():
    c = snp.CANONICAL_SOURCE_MEDIA_CONTRACT_V1
    assert c.hdr_target == "SDR"
    assert c.color_primaries == c.color_transfer == c.color_space == "BT709"


def test_limited_tv_range():
    assert snp.CANONICAL_SOURCE_MEDIA_CONTRACT_V1.color_range == "TV"


def test_audio_preserve_policy():
    assert snp.CANONICAL_SOURCE_MEDIA_CONTRACT_V1.audio_policy == snp.CONTRACT_AUDIO_POLICY_PRESERVE_SOURCE_NO_MANDATORY_NORMALIZATION


def test_missing_audio_valid_no_audio_action():
    profile = _base_profile(audio_presence=smp.AUDIO_MISSING, audio_stream_count=0, audio_codec=None,
                             raw_audio_codec=None, audio_sample_rate_hz=None, audio_channels=None,
                             audio_channel_layout=None, rotation_degrees=90,
                             rotation_source=smp.ROTATION_SOURCE_DISPLAY_MATRIX)
    decision = _decide(profile)
    result = snp.build_source_normalization_plan("s1", profile, decision)
    assert result.plan.audio_action == snp.ACTION_NO_ACTION


def test_plan_frozen():
    profile = _base_profile(rotation_degrees=90, rotation_source=smp.ROTATION_SOURCE_DISPLAY_MATRIX)
    decision = _decide(profile)
    result = snp.build_source_normalization_plan("s1", profile, decision)
    with pytest.raises(Exception):
        result.plan.rotation_action = snp.ACTION_NO_ACTION  # type: ignore[misc]


def test_plan_deterministic():
    profile = _base_profile(rotation_degrees=90, rotation_source=smp.ROTATION_SOURCE_DISPLAY_MATRIX)
    decision = _decide(profile)
    r1 = snp.build_source_normalization_plan("s1", profile, decision)
    r2 = snp.build_source_normalization_plan("s1", profile, decision)
    assert r1.plan.plan_identity == r2.plan.plan_identity


def test_filename_independent():
    """The builder never accepts or reads a filename at all -- confirmed
    by signature inspection (no 'filename'/'path' parameter)."""
    import inspect
    params = inspect.signature(snp.build_source_normalization_plan).parameters
    assert "filename" not in params
    assert "path" not in params


def test_path_independent():
    profile = _base_profile(rotation_degrees=90, rotation_source=smp.ROTATION_SOURCE_DISPLAY_MATRIX)
    decision = _decide(profile)
    r_a = snp.build_source_normalization_plan("opaque-id-1", profile, decision)
    r_b = snp.build_source_normalization_plan("opaque-id-1", profile, decision)
    assert r_a.plan.plan_identity == r_b.plan.plan_identity
    r_c = snp.build_source_normalization_plan("opaque-id-2", profile, decision)
    assert r_c.plan.plan_identity != r_a.plan.plan_identity


# ---------------------------------------------------------------------------
# ACCEPT / NORMALIZE_REQUIRED / REJECT / INSUFFICIENT_EVIDENCE behavior
# ---------------------------------------------------------------------------

def test_accept_maps_to_not_required():
    profile = _base_profile()
    decision = _decide(profile)
    assert decision.decision == sfp.DECISION_ACCEPT
    result = snp.build_source_normalization_plan("s1", profile, decision)
    assert result.outcome == snp.NORMALIZATION_NOT_REQUIRED
    assert result.plan is None


def test_normalize_required_produces_plan():
    profile = _base_profile(rotation_degrees=90, rotation_source=smp.ROTATION_SOURCE_DISPLAY_MATRIX)
    decision = _decide(profile)
    assert decision.decision == sfp.DECISION_NORMALIZE_REQUIRED
    result = snp.build_source_normalization_plan("s1", profile, decision)
    assert result.plan is not None


def test_reject_produces_no_plan():
    profile = _base_profile(video_presence=smp.VIDEO_MISSING)
    decision = _decide(profile)
    assert decision.decision == sfp.DECISION_REJECT
    result = snp.build_source_normalization_plan("s1", profile, decision)
    assert result.plan is None
    assert result.outcome == snp.NORMALIZATION_UNSUPPORTED


def test_insufficient_evidence_produces_no_unsafe_plan():
    profile = _base_profile(video_codec=None, raw_video_codec="mystery")
    decision = _decide(profile)
    assert decision.decision == sfp.DECISION_INSUFFICIENT_EVIDENCE
    result = snp.build_source_normalization_plan("s1", profile, decision)
    assert result.plan is None


# ---------------------------------------------------------------------------
# Per-property action mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("degrees,action", [(90, "ROTATE_90"), (180, "ROTATE_180"), (270, "ROTATE_270")])
def test_rotation_mapping(degrees, action):
    profile = _base_profile(rotation_degrees=degrees, rotation_source=smp.ROTATION_SOURCE_DISPLAY_MATRIX,
                             display_width=1080, display_height=1920)
    decision = _decide(profile)
    result = snp.build_source_normalization_plan("s1", profile, decision)
    assert result.plan.rotation_action == action
    assert result.plan.executability == snp.EXECUTABILITY_EXECUTABLE


def test_vfr_mapping_uses_effective_fps():
    profile = _base_profile(vfr_status=smp.VFR_STATUS_LIKELY_VFR, effective_fps=23.976)
    decision = _decide(profile)
    result = snp.build_source_normalization_plan("s1", profile, decision)
    assert result.plan.frame_rate_action == snp.ACTION_VFR_TO_CFR
    assert result.plan.executability == snp.EXECUTABILITY_EXECUTABLE


def test_vfr_without_effective_fps_is_invalid_state():
    profile = _base_profile(vfr_status=smp.VFR_STATUS_LIKELY_VFR, effective_fps=None)
    decision = _decide(profile)
    result = snp.build_source_normalization_plan("s1", profile, decision)
    assert result.plan.executability == snp.EXECUTABILITY_INVALID_SOURCE_STATE
    assert result.outcome == snp.NORMALIZATION_UNSUPPORTED


def test_pq_mapping():
    profile = _base_profile(hdr_status=smp.HDR_STATUS_HDR_PQ, color_transfer="smpte2084")
    decision = _decide(profile)
    result = snp.build_source_normalization_plan("s1", profile, decision, tonemap_available=True)
    assert result.plan.hdr_action == snp.ACTION_HDR_PQ_TO_SDR_BT709
    assert result.plan.executability == snp.EXECUTABILITY_EXECUTABLE


def test_hlg_mapping():
    profile = _base_profile(hdr_status=smp.HDR_STATUS_HDR_HLG, color_transfer="arib-std-b67")
    decision = _decide(profile)
    result = snp.build_source_normalization_plan("s1", profile, decision, tonemap_available=True)
    assert result.plan.hdr_action == snp.ACTION_HDR_HLG_TO_SDR_BT709


def test_pq_capability_unknown_blocked():
    profile = _base_profile(hdr_status=smp.HDR_STATUS_HDR_PQ, color_transfer="smpte2084")
    decision = _decide(profile)
    result = snp.build_source_normalization_plan("s1", profile, decision)
    assert result.plan.executability == snp.EXECUTABILITY_CAPABILITY_UNVERIFIED
    assert "tonemap_available" in result.blocking_capability_gaps


def test_dolby_vision_unsupported():
    profile = _base_profile(hdr_status=smp.HDR_STATUS_HDR_DOLBY_VISION)
    decision = _decide(profile)
    result = snp.build_source_normalization_plan("s1", profile, decision, tonemap_available=True)
    assert result.plan.executability == snp.EXECUTABILITY_UNSUPPORTED
    assert result.plan.hdr_action == snp.ACTION_NO_ACTION
    assert result.outcome == snp.NORMALIZATION_UNSUPPORTED


def test_hdr_other_unsupported():
    profile = _base_profile(hdr_status=smp.HDR_STATUS_HDR_OTHER)
    decision = _decide(profile)
    result = snp.build_source_normalization_plan("s1", profile, decision, tonemap_available=True)
    assert result.plan.executability == snp.EXECUTABILITY_UNSUPPORTED


def test_ten_bit_mapping():
    profile = _base_profile(bit_depth=10, pixel_format="yuv420p10le")
    decision = _decide(profile)
    result = snp.build_source_normalization_plan("s1", profile, decision)
    assert result.plan.bit_depth_action == snp.ACTION_TEN_BIT_TO_EIGHT_BIT


def test_yuv422_mapping():
    profile = _base_profile(pixel_format="yuv422p")
    decision = _decide(profile)
    result = snp.build_source_normalization_plan("s1", profile, decision)
    assert result.plan.pixel_format_action == snp.ACTION_PIXEL_FORMAT_TO_YUV420P


def test_yuv444_mapping():
    profile = _base_profile(pixel_format="yuv444p")
    decision = _decide(profile)
    result = snp.build_source_normalization_plan("s1", profile, decision)
    assert result.plan.pixel_format_action == snp.ACTION_PIXEL_FORMAT_TO_YUV420P


def test_hevc_capability_confirmed_mapping():
    # HEVC alone resolves to ACCEPT under D-272 (this module's own
    # documented gap); combine with rotation to reach NORMALIZE_REQUIRED
    # so the codec_action mapping itself can be proven.
    profile = _base_profile(video_codec=smp.VIDEO_CODEC_HEVC, raw_video_codec="hevc",
                             rotation_degrees=90, rotation_source=smp.ROTATION_SOURCE_DISPLAY_MATRIX,
                             display_width=1080, display_height=1920)
    cap = sfp.RuntimeCapabilityInput(hevc_decode_confirmed=True)
    decision = _decide(profile, runtime_capability=cap)
    assert decision.decision == sfp.DECISION_NORMALIZE_REQUIRED
    result = snp.build_source_normalization_plan("s1", profile, decision, runtime_capability=cap)
    assert result.plan.codec_action == snp.ACTION_HEVC_TO_H264
    assert result.plan.executability == snp.EXECUTABILITY_EXECUTABLE


def test_hevc_capability_unknown_blocked_via_policy():
    """A plain HEVC source with unconfirmed capability never even
    reaches NORMALIZE_REQUIRED -- D-272 itself returns INSUFFICIENT_
    EVIDENCE, so no plan is built at all (proves the boundary, not a
    plan-level capability gap)."""
    profile = _base_profile(video_codec=smp.VIDEO_CODEC_HEVC, raw_video_codec="hevc")
    decision = _decide(profile)
    assert decision.decision == sfp.DECISION_INSUFFICIENT_EVIDENCE
    result = snp.build_source_normalization_plan("s1", profile, decision)
    assert result.plan is None


def test_hevc_capability_unknown_blocked_within_plan():
    """When HEVC rides along with a genuine NORMALIZE_REQUIRED reason
    (e.g. rotation) but capability is unconfirmed, the codec_action is
    still planned (semantically required) but flagged capability-
    unverified at the plan level."""
    profile = _base_profile(video_codec=smp.VIDEO_CODEC_HEVC, raw_video_codec="hevc",
                             rotation_degrees=90, rotation_source=smp.ROTATION_SOURCE_DISPLAY_MATRIX,
                             display_width=1080, display_height=1920)
    cap = sfp.RuntimeCapabilityInput(hevc_decode_confirmed=True)
    decision = _decide(profile, runtime_capability=cap)
    # Build the PLAN without passing the capability confirmation through,
    # simulating a stale/missing capability signal at plan-build time.
    result = snp.build_source_normalization_plan("s1", profile, decision)
    assert result.plan.codec_action == snp.ACTION_HEVC_TO_H264
    assert result.plan.executability == snp.EXECUTABILITY_CAPABILITY_UNVERIFIED
    assert "hevc_decode_confirmed" in result.blocking_capability_gaps


def test_pure_clean_hevc_now_normalize_required_via_d272b():
    """D-272B reconciled the gap this test used to document: a
    capability-confirmed, otherwise-clean HEVC source now resolves to
    NORMALIZE_REQUIRED (via D-272's own REASON_HEVC_TO_H264_
    NORMALIZATION_REQUIRED) and produces an executable HEVC_TO_H264
    plan, closing the D-272/D-273/D-274A reconciliation gap."""
    profile = _base_profile(video_codec=smp.VIDEO_CODEC_HEVC, raw_video_codec="hevc")
    cap = sfp.RuntimeCapabilityInput(hevc_decode_confirmed=True)
    decision = _decide(profile, runtime_capability=cap)
    assert decision.decision == sfp.DECISION_NORMALIZE_REQUIRED
    assert sfp.REASON_HEVC_TO_H264_NORMALIZATION_REQUIRED in decision.normalization_reasons
    result = snp.build_source_normalization_plan("s1", profile, decision, runtime_capability=cap)
    assert result.plan is not None
    assert result.plan.codec_action == snp.ACTION_HEVC_TO_H264
    assert result.plan.executability == snp.EXECUTABILITY_EXECUTABLE
    assert result.outcome == snp.NORMALIZATION_PLANNED


def test_malformed_rotation_no_executable_plan():
    profile = _base_profile(rotation_degrees=None, rotation_source=smp.ROTATION_SOURCE_UNKNOWN)
    decision = _decide(profile)
    assert decision.decision == sfp.DECISION_NORMALIZE_REQUIRED
    result = snp.build_source_normalization_plan("s1", profile, decision)
    assert result.plan.executability == snp.EXECUTABILITY_INVALID_SOURCE_STATE


def test_timeline_action_triggers_on_nonzero_start():
    profile = _base_profile(rotation_degrees=90, rotation_source=smp.ROTATION_SOURCE_DISPLAY_MATRIX,
                             format_start_time=0.5)
    decision = _decide(profile)
    result = snp.build_source_normalization_plan("s1", profile, decision)
    assert result.plan.timeline_action == snp.ACTION_TIMELINE_TO_ZERO


def test_timeline_action_not_triggered_on_zero_start():
    profile = _base_profile(rotation_degrees=90, rotation_source=smp.ROTATION_SOURCE_DISPLAY_MATRIX)
    decision = _decide(profile)
    result = snp.build_source_normalization_plan("s1", profile, decision)
    assert result.plan.timeline_action == snp.ACTION_NO_ACTION


# ---------------------------------------------------------------------------
# Multi-action composition, canonical order, identity sensitivity
# ---------------------------------------------------------------------------

def test_multi_action_plan_supported():
    profile = _base_profile(
        rotation_degrees=90, rotation_source=smp.ROTATION_SOURCE_DISPLAY_MATRIX,
        display_width=1080, display_height=1920,
        vfr_status=smp.VFR_STATUS_LIKELY_VFR, effective_fps=24.0,
        hdr_status=smp.HDR_STATUS_HDR_PQ, color_transfer="smpte2084",
        bit_depth=10, pixel_format="yuv420p10le",
    )
    decision = _decide(profile)
    result = snp.build_source_normalization_plan("s1", profile, decision, tonemap_available=True)
    plan = result.plan
    assert plan.rotation_action == snp.ACTION_ROTATE_90
    assert plan.frame_rate_action == snp.ACTION_VFR_TO_CFR
    assert plan.hdr_action == snp.ACTION_HDR_PQ_TO_SDR_BT709
    assert plan.bit_depth_action == snp.ACTION_TEN_BIT_TO_EIGHT_BIT
    assert plan.executability == snp.EXECUTABILITY_EXECUTABLE


def test_stable_canonical_action_order():
    assert snp._CANONICAL_ACTION_FIELD_ORDER == (
        "timeline_action", "rotation_action", "codec_action", "hdr_action",
        "bit_depth_action", "pixel_format_action", "frame_rate_action",
        "container_action", "audio_action",
    )


def test_plan_identity_changes_when_action_changes():
    profile_a = _base_profile(rotation_degrees=90, rotation_source=smp.ROTATION_SOURCE_DISPLAY_MATRIX,
                               display_width=1080, display_height=1920)
    profile_b = _base_profile(rotation_degrees=180, rotation_source=smp.ROTATION_SOURCE_DISPLAY_MATRIX)
    decision_a = _decide(profile_a)
    decision_b = _decide(profile_b)
    r_a = snp.build_source_normalization_plan("s1", profile_a, decision_a)
    r_b = snp.build_source_normalization_plan("s1", profile_b, decision_b)
    assert r_a.plan.plan_identity != r_b.plan.plan_identity


# ---------------------------------------------------------------------------
# Normalized SHA / original preservation / re-probe / re-evaluation
# ---------------------------------------------------------------------------

def test_normalized_sha_field_distinct_type():
    ref = snp.NormalizedSourceReference(
        original_source_identity="orig-1", normalization_plan_identity="normplan_x",
    )
    assert ref.normalized_output_sha256 is None
    ref2 = replace(ref, normalized_output_sha256="a" * 64)
    assert ref2.normalized_output_sha256 != ref.original_source_identity


def test_original_source_preserved_no_overwrite_semantics():
    """SourceNormalizationPlan carries no field that could represent
    overwriting the original -- confirmed by field inspection."""
    profile = _base_profile(rotation_degrees=90, rotation_source=smp.ROTATION_SOURCE_DISPLAY_MATRIX)
    decision = _decide(profile)
    result = snp.build_source_normalization_plan("s1", profile, decision)
    field_names = set(result.plan.__dataclass_fields__.keys())
    for banned in ("overwrite", "delete_original", "in_place"):
        assert not any(banned in name for name in field_names)


def test_reprobe_required_via_verification_contract():
    accepted_profile = _base_profile()
    accepted_decision = _decide(accepted_profile)
    verification = snp.verify_normalized_source(accepted_profile, accepted_decision)
    assert verification.verified is True
    assert snp.verification_outcome(verification) == snp.NORMALIZATION_SUCCEEDED


def test_reevaluation_required_only_accept_verifies():
    still_blocked_profile = _base_profile(rotation_degrees=90, rotation_source=smp.ROTATION_SOURCE_DISPLAY_MATRIX)
    still_blocked_decision = _decide(still_blocked_profile)
    verification = snp.verify_normalized_source(still_blocked_profile, still_blocked_decision)
    assert verification.verified is False


def test_blocked_after_normalization_fails_verification():
    rejected_profile = _base_profile(video_presence=smp.VIDEO_MISSING)
    rejected_decision = _decide(rejected_profile)
    verification = snp.verify_normalized_source(rejected_profile, rejected_decision)
    assert verification.verified is False
    assert snp.verification_outcome(verification) == snp.NORMALIZATION_VERIFICATION_FAILED
    assert verification.errors


def test_one_pass_invariant():
    assert snp.is_normalization_attempt_allowed(0) is True
    assert snp.is_normalization_attempt_allowed(1) is False
    assert snp.is_normalization_attempt_allowed(2) is False


# ---------------------------------------------------------------------------
# No execution, no ffmpeg, no filtergraph, no transcode, no mutation
# ---------------------------------------------------------------------------

def test_no_execution_source_scan():
    source = _source_without_docstrings("cutsell_worker/source_normalization_plan.py")
    for banned in ("subprocess.run(", "subprocess.Popen(", "import subprocess", "ffmpeg", "-vf", '"-c:v"',
                   "scale=", "pad=", "hflip", "vflip", "zscale", "tonemap="):
        assert banned not in source


def test_no_filesystem_access_source_scan():
    source = _source_without_docstrings("cutsell_worker/source_normalization_plan.py")
    for banned in ("open(", "os.path", "Path(", "import os\n"):
        assert banned not in source


def test_no_media_mutation():
    """The plan builder never accepts a media file path/bytes at all --
    confirmed by signature inspection."""
    import inspect
    params = list(inspect.signature(snp.build_source_normalization_plan).parameters)
    assert params == ["source_identity", "profile", "policy_decision", "runtime_capability", "tonemap_available", "target_contract"]


def test_no_normalization_succeeded_fabricated_without_verification():
    """This gate's own module never constructs NORMALIZATION_SUCCEEDED
    except through the explicit verify_normalized_source contract."""
    source = _source_without_docstrings("cutsell_worker/source_normalization_plan.py")
    # NORMALIZATION_SUCCEEDED is only ever returned by verification_outcome,
    # never assigned as a literal outcome inside build_source_normalization_plan.
    build_fn_source = source[source.index("def build_source_normalization_plan"):source.index("def verify_normalized_source")]
    assert "NORMALIZATION_SUCCEEDED" not in build_fn_source


# ---------------------------------------------------------------------------
# Firewall -- unrelated authorities unchanged
# ---------------------------------------------------------------------------

_FIREWALL_FILES = [
    "cutsell_worker/render.py",
    "cutsell_worker/render_delivery.py",
    "cutsell_worker/render_plan.py",
    "cutsell_worker/media_probe.py",
    "cutsell_worker/source_media_profile.py",
    # source_format_policy.py deliberately removed from this list: D-272B
    # (a narrow, separately-authorized reconciliation gate) legitimately
    # extends it -- see test_cutsell_d272_source_format_policy.py's own
    # updated expectations for the current authoritative HEVC behavior.
    "cutsell_worker/post_render_media_qc.py",
    "cutsell_worker/visual_finishing_measurement.py",
    "cutsell_worker/audio_finishing_measurement.py",
    "cutsell_worker/boundary_engine_pass.py",
    "cutsell_worker/pacing_transition_decision.py",
    "cutsell_worker/post_render_watch_listen_qc.py",
    "cutsell_worker/live_render_qc.py",
    "cutsell_worker/finishing_contract.py",
    "cutsell_worker/export_job.py",
    "cutsell_worker/exports.py",
    "cutsell_worker/tenant_safe_delivery.py",
    "cutsell_worker/uploads.py",
    "cutsell_worker/worker_job.py",
    "cutsell_worker/flow_b.py",
    "gpu_execution_provider.py",
]


@pytest.mark.parametrize("rel_path", _FIREWALL_FILES)
def test_unrelated_authorities_unchanged(rel_path):
    if not Path(rel_path).exists():
        pytest.skip(f"{rel_path} not present in this checkout")
    assert _run_git_diff(rel_path) == "", f"D-274A must not touch {rel_path}"


def test_renderer_timeout_constant_unchanged():
    from cutsell_worker import render as render_module
    assert render_module.RENDER_FFMPEG_TIMEOUT_SEC == 1200.0


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

def test_no_secrets_in_source():
    source = Path("cutsell_worker/source_normalization_plan.py").read_text(encoding="utf-8")
    for banned in ("AKIA", "aws_secret", "BEGIN PRIVATE KEY", "api_key="):
        assert banned not in source


def test_compileall_clean():
    import sys
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", "cutsell_worker/source_normalization_plan.py"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
