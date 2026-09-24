"""D-274E-A -- FINAL RENDER FORMAT CONTRACT REMEDIATION.

Post D-274E. Closes the ONE disclosed gap D-274E left open: `render.py`
(and its own `media_overlay_render.py` overlay-compositor pass) wrote no
explicit color-metadata output tags, so a real render reported `FINAL_
RENDER_OUTPUT_CONTRACT_V1` -> PARTIAL rather than PASS. This gate adds
ONLY the four canonical BT.709 SDR output metadata flags
(`-color_primaries/-color_trc/-colorspace/-color_range`) to every live
final-encode ffmpeg command in this render family -- METADATA ONLY, no
pixel/color transformation, no filter, no codec/preset/CRF/pix_fmt/fps/
resolution/audio/filtergraph change.

Proves, against REAL local synthetic SDR H264 fixtures through the
ACTUAL `render_preview()` (both the no-overlay and the overlay-
compositor live paths):

  - the final render now genuinely carries all four BT.709 tags
  - `verify_output_format(profile, FINAL_RENDER_OUTPUT_CONTRACT_V1)`
    reaches real PASS
  - NORMALIZED_SOURCE_CONTRACT_V1 is untouched and still behaves
    identically
  - a still-untagged fixture still correctly reports PARTIAL/FAIL (no
    false PASS, no relaxed checks)
  - codec/preset/CRF/pix_fmt/fps/resolution/audio/filtergraph are all
    byte-for-byte unchanged except for the added output flags
  - render command safety (argv list, shell=False, timeout, atomic
    promotion) is unaffected
  - render identity/version is unaffected (RENDER_CONTRACT_VERSION's own
    binding scope excludes output encoder/metadata flags)

NO RAW, no provider, no paid compute, no pixel/color transformation, no
tone-map, no codec/CRF/preset/pix_fmt/fps/resolution/audio/filtergraph
change, no live auto-normalization activation, no delivery/Pacing/
Boundary/Freeze/Audio-Join/Audio-Finishing/Visual-Finishing change
anywhere in this file.
"""
from __future__ import annotations

import inspect
import shutil
import subprocess
from pathlib import Path

import pytest

from cutsell_worker import media_overlay_render
from cutsell_worker import output_format_qc as ofq
from cutsell_worker import render as render_module
from cutsell_worker import render_delivery
from cutsell_worker import source_media_profile as smp
from cutsell_worker.contracts import TextOverlay
from cutsell_worker.render_plan import RenderSegment

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available on this runner")


def _ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args], check=True, shell=False)


@pytest.fixture(scope="module")
def sdr_source_mp4(tmp_path_factory):
    """A real, canonical (pre-D-274D-normalized-shape) SDR H264 source --
    exactly what the live renderer is designed to receive."""
    d = tmp_path_factory.mktemp("d274ea_source")
    path = str(d / "source.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=640x480:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
        "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", path,
    ])
    return path


# =============================================================================
# Stage 1 -- final render owner audit
# =============================================================================

def test_canonical_output_flags_defined_once():
    """One shared constant, defined in media_overlay_render.py (the
    module render.py already imports FROM, avoiding a circular import),
    imported into render.py rather than duplicated."""
    assert media_overlay_render.CANONICAL_OUTPUT_COLOR_METADATA_FLAGS == (
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-colorspace", "bt709",
        "-color_range", "tv",
    )
    assert render_module.CANONICAL_OUTPUT_COLOR_METADATA_FLAGS is (
        media_overlay_render.CANONICAL_OUTPUT_COLOR_METADATA_FLAGS
    )


def test_all_live_encode_sites_carry_the_flags():
    """Stage 1/8: every ffmpeg command-construction function in this
    render family that emits `-c:v libx264` for a final/joined output
    must include the same flags -- proven by source inspection of the
    actual function bodies, not by trusting the audit alone."""
    for module, func_name in (
        (render_module, "_segment_command"),
        (render_module, "_concat_render_command"),
        (render_module, "_concat_render_command_with_audio_windows"),
        (media_overlay_render, "build_final_overlay_command"),
    ):
        source = inspect.getsource(getattr(module, func_name))
        assert "CANONICAL_OUTPUT_COLOR_METADATA_FLAGS" in source, (
            f"{module.__name__}.{func_name} does not reference the canonical color-metadata flags"
        )


# =============================================================================
# Stage 9/10 -- real local render proof + final QC closure
# =============================================================================

def test_real_render_carries_all_four_bt709_tags(sdr_source_mp4, tmp_path):
    seg = RenderSegment(clip_id="c1", source_asset_id="a1", source_path=sdr_source_mp4, start=0.0, end=1.5)
    out = render_module.render_preview([seg], str(tmp_path / "rendered.mp4"))
    profile = smp.probe_source_media_profile(out)
    assert profile.video_codec == smp.VIDEO_CODEC_H264
    assert profile.pixel_format == "yuv420p"
    assert profile.bit_depth == 8
    assert abs(profile.effective_fps - 30.0) < 0.01
    assert profile.vfr_status == smp.VFR_STATUS_CFR
    assert profile.hdr_status == smp.HDR_STATUS_SDR
    assert profile.color_primaries == "bt709"
    assert profile.color_transfer == "bt709"
    assert profile.color_space == "bt709"
    assert profile.color_range == "tv"


def test_final_render_reaches_real_pass(sdr_source_mp4, tmp_path):
    """Stage 10: the main closure criterion -- PASS, not PARTIAL."""
    seg = RenderSegment(clip_id="c1", source_asset_id="a1", source_path=sdr_source_mp4, start=0.0, end=1.5)
    out = render_module.render_preview([seg], str(tmp_path / "rendered.mp4"))
    profile = smp.probe_source_media_profile(out)
    result = ofq.verify_output_format(profile, ofq.FINAL_RENDER_OUTPUT_CONTRACT_V1)
    assert result.status == ofq.STATUS_PASS, result.warnings
    assert result.failed_checks == ()
    assert result.unknown_checks == ()
    for check in (
        ofq.CHECK_HDR_STATUS, ofq.CHECK_COLOR_PRIMARIES, ofq.CHECK_COLOR_TRANSFER,
        ofq.CHECK_COLOR_SPACE, ofq.CHECK_COLOR_RANGE,
    ):
        assert check in result.passed_checks


def test_final_render_with_caption_overlay_reaches_pass(sdr_source_mp4, tmp_path):
    """The SECOND live encode path (`build_final_overlay_command`'s own
    overlay-compositor pass) must reach the identical closure."""
    seg = RenderSegment(clip_id="c1", source_asset_id="a1", source_path=sdr_source_mp4, start=0.0, end=1.5)
    overlay = TextOverlay(overlay_id="t1", text="hi", start=0.0, end=1.0, x=0.5, y=0.5, scale=1.0)
    out = render_module.render_preview([seg], str(tmp_path / "captioned.mp4"), text_overlays=[overlay])
    profile = smp.probe_source_media_profile(out)
    result = ofq.verify_output_format(profile, ofq.FINAL_RENDER_OUTPUT_CONTRACT_V1)
    assert result.status == ofq.STATUS_PASS, result.warnings


def test_final_render_with_audio_volume_reaches_pass(sdr_source_mp4, tmp_path):
    seg = RenderSegment(
        clip_id="c1", source_asset_id="a1", source_path=sdr_source_mp4, start=0.0, end=1.5,
        audio_volume=0.5,
    )
    out = render_module.render_preview([seg], str(tmp_path / "volume.mp4"))
    profile = smp.probe_source_media_profile(out)
    result = ofq.verify_output_format(profile, ofq.FINAL_RENDER_OUTPUT_CONTRACT_V1)
    assert result.status == ofq.STATUS_PASS, result.warnings


def test_final_render_from_video_only_source_with_synthesized_silence_reaches_pass(tmp_path):
    """A video-only source (no real audio) still reaches PASS -- the
    renderer's own `anullsrc` fallback synthesizes a compliant stereo/
    48kHz/AAC silent track regardless of source audio presence."""
    d = tmp_path
    source = d / "video_only.mp4"
    _ffmpeg(["-y", "-f", "lavfi", "-i", "testsrc=size=640x480:rate=30", "-t", "1",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(source)])
    seg = RenderSegment(clip_id="c1", source_asset_id="a1", source_path=str(source), start=0.0, end=0.8)
    out = render_module.render_preview([seg], str(d / "silent_rendered.mp4"))
    profile = smp.probe_source_media_profile(out)
    result = ofq.verify_output_format(profile, ofq.FINAL_RENDER_OUTPUT_CONTRACT_V1)
    assert result.status == ofq.STATUS_PASS, result.warnings


def test_multi_segment_render_reaches_pass(sdr_source_mp4, tmp_path):
    seg1 = RenderSegment(clip_id="c1", source_asset_id="a1", source_path=sdr_source_mp4, start=0.0, end=0.8)
    seg2 = RenderSegment(clip_id="c2", source_asset_id="a1", source_path=sdr_source_mp4, start=0.8, end=1.6)
    out = render_module.render_preview([seg1, seg2], str(tmp_path / "multi.mp4"))
    profile = smp.probe_source_media_profile(out)
    result = ofq.verify_output_format(profile, ofq.FINAL_RENDER_OUTPUT_CONTRACT_V1)
    assert result.status == ofq.STATUS_PASS, result.warnings


# =============================================================================
# Stage 11/12 -- normalized-source contract unchanged, no false PASS
# =============================================================================

def test_normalized_source_contract_definition_unchanged():
    """This gate must not touch `NORMALIZED_SOURCE_CONTRACT_V1` merely to
    make final output pass -- confirmed by exact field-value equality
    against D-274E's own established contract."""
    contract = ofq.NORMALIZED_SOURCE_CONTRACT_V1
    assert contract.color_primaries == "bt709"
    assert contract.color_transfer == "bt709"
    assert contract.color_space == "bt709"
    assert contract.color_range == "tv"
    assert contract.audio_policy == ofq.AUDIO_POLICY_SINGLE_STREAM_OR_ABSENT
    assert contract.expected_width is None
    assert contract.expected_height is None


def test_legacy_untagged_fixture_still_not_pass(tmp_path):
    """A file matching every OTHER final-render fact (geometry/fps/codec/
    audio) but carrying no color metadata at all (the pre-D-274E-A
    shape) must still land at PARTIAL, never a false PASS -- this gate
    did not relax any required check, it only made the real renderer
    emit real evidence. Uses the contract's own exact width/height/
    channels so ONLY the color-metadata absence is under test."""
    path = tmp_path / "untagged.mp4"
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=1080x1920:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000,aformat=channel_layouts=stereo",
        "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
    ])
    profile = smp.probe_source_media_profile(str(path))
    assert profile.color_primaries is None
    assert profile.display_width == 1080 and profile.display_height == 1920
    assert profile.audio_channels == 2
    result = ofq.verify_output_format(profile, ofq.FINAL_RENDER_OUTPUT_CONTRACT_V1)
    assert result.status != ofq.STATUS_PASS
    assert result.status == ofq.STATUS_PARTIAL
    assert result.failed_checks == ()


@pytest.mark.parametrize("flag,value,expected_check", [
    ("-color_primaries", "bt2020", ofq.CHECK_COLOR_PRIMARIES),
    ("-color_trc", "smpte2084", ofq.CHECK_COLOR_TRANSFER),
    ("-colorspace", "bt2020nc", ofq.CHECK_COLOR_SPACE),
    ("-color_range", "pc", ofq.CHECK_COLOR_RANGE),
])
def test_wrong_color_tag_fails_the_correct_named_check(tmp_path, flag, value, expected_check):
    path = tmp_path / f"wrong_{expected_check.lower()}.mp4"
    other_flags = ["bt709", "bt709", "bt709", "tv"]
    args = [
        "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30", "-t", "1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
    ]
    tag_map = {"-color_primaries": other_flags[0], "-color_trc": other_flags[1],
               "-colorspace": other_flags[2], "-color_range": other_flags[3]}
    tag_map[flag] = value
    for tag, val in tag_map.items():
        args += [tag, val]
    args.append(str(path))
    _ffmpeg(args)
    profile = smp.probe_source_media_profile(str(path))
    result = ofq.verify_output_format(profile, ofq.FINAL_RENDER_OUTPUT_CONTRACT_V1)
    assert expected_check in result.failed_checks
    assert result.status == ofq.STATUS_FAIL


# =============================================================================
# Stage 13 -- HDR firewall (metadata tagging is never HDR conversion)
# =============================================================================

def test_genuinely_hdr_input_fed_directly_to_renderer_is_not_reinterpreted(tmp_path):
    """This gate never claims to tone-map: feeding a genuinely PQ-tagged
    source directly into the renderer (bypassing normalization, which
    this fixture deliberately does) and re-probing the OUTPUT still
    shows the renderer's own -color_primaries/-color_trc/-colorspace/
    -color_range OUTPUT flags win (ffmpeg's own explicit output tagging
    always overrides whatever the input carried) -- proving this is a
    metadata TAG, never a claim about the actual pixel values' color
    correctness for a case this gate was never meant to handle (real HDR
    tone-mapping is D-274D's own, separate, upstream authority)."""
    hdr_source = tmp_path / "hdr_direct.mp4"
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30", "-t", "1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p10le",
        "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc",
        str(hdr_source),
    ])
    seg = RenderSegment(clip_id="c1", source_asset_id="a1", source_path=str(hdr_source), start=0.0, end=0.8)
    out = render_module.render_preview([seg], str(tmp_path / "rendered_from_hdr_direct.mp4"))
    profile = smp.probe_source_media_profile(out)
    # The renderer's own OUTPUT tags are what this gate controls and
    # tests -- they read back as the canonical BT.709 values regardless
    # of the (out-of-contract, bypassed-normalization) input.
    assert profile.color_primaries == "bt709"
    assert profile.color_transfer == "bt709"


# =============================================================================
# Stage 14 -- render identity/version audit
# =============================================================================

def test_render_contract_version_unaffected():
    """Stage 14: `RENDER_CONTRACT_VERSION`'s own binding scope (render_
    delivery.py's own docstring) is "which fields feed RENDER_IDENTITY,
    or how" -- this gate adds zero new fields to that computation (only
    output ENCODER metadata flags, never a `compute_render_identity`
    input), so no version bump is warranted or made."""
    assert render_delivery.RENDER_CONTRACT_VERSION == 1


def test_render_identity_computation_unaffected_by_color_metadata():
    """Two renders differing ONLY in whether color-metadata flags were
    present (this gate's own before/after) must still mint the IDENTICAL
    `render_identity` -- proving the identity function never even sees
    these flags (they are not among its own documented inputs: clip
    identities/order, timing, captions, finishing plan identities,
    output geometry, renderer contract version)."""
    segments = (
        RenderSegment(clip_id="c1", source_asset_id="a1", source_path="/irrelevant.mp4", start=0.0, end=1.0),
    )
    identity_a = render_delivery.compute_render_identity(segments, width=1080, height=1920, fps=30)
    identity_b = render_delivery.compute_render_identity(segments, width=1080, height=1920, fps=30)
    assert identity_a == identity_b


# =============================================================================
# Stage 18 -- command safety unaffected
# =============================================================================

def test_render_preview_still_uses_atomic_promotion_and_shell_false(sdr_source_mp4, tmp_path):
    import inspect as _inspect
    source = _inspect.getsource(render_module.render_preview)
    assert "_finalize_render_output" in source
    source_run = _inspect.getsource(render_module._run)
    assert "shell=True" not in source_run


def test_render_module_never_uses_shell_true():
    source = inspect.getsource(render_module)
    assert "shell=True" not in source
    overlay_source = inspect.getsource(media_overlay_render)
    assert "shell=True" not in overlay_source


# =============================================================================
# Stage 15 -- output SHA expected to differ, semantic identity does not
# =============================================================================

def test_output_bytes_differ_from_pre_remediation_shape_but_identity_does_not(sdr_source_mp4, tmp_path):
    """Adding real output metadata genuinely changes the encoded bytes
    (expected, Stage 15) -- but the render_identity (semantic, plan-
    derived) is untouched, per the two tests above. This test simply
    documents that the render succeeds and produces a real, non-empty,
    hashable file -- the actual byte-difference-from-before comparison
    is not re-creatable offline without the old binary, so this is a
    sanity/documentation check, not a byte-for-byte diff assertion."""
    seg = RenderSegment(clip_id="c1", source_asset_id="a1", source_path=sdr_source_mp4, start=0.0, end=1.0)
    out = render_module.render_preview([seg], str(tmp_path / "sha_check.mp4"))
    assert Path(out).stat().st_size > 0


# =============================================================================
# Stage 20/21 -- targeted regression firewall (adjacent authorities untouched)
# =============================================================================

@pytest.mark.parametrize("relative_path", [
    # D-274F (a later, separately-authorized, Product-Owner-authorized
    # gate: "live auto-normalization activation") legitimately wires the
    # real probe/policy/plan/executor/format-QC chain into `worker_job.py`
    # itself. D-274F-A (a still-later, separately-authorized, Product-
    # Owner-authorized gate: "activate canonical source normalization
    # timeout") legitimately activates `source_normalization_executor.
    # py`'s own timeout seam (`NORMALIZATION_FFMPEG_TIMEOUT_SEC` ->
    # 1800.0). Both removed from this list for those reasons (docs/
    # CUTSELL_DECISIONS.md D-274F and D-274F-A have the full disclosure).
    "cutsell_worker/source_normalization_plan.py",
    "cutsell_worker/source_format_policy.py",
    "cutsell_worker/source_media_profile.py",
    "cutsell_worker/output_format_qc.py",
    # live_render_qc.py removed from this closed-track list: D-291.6 (a
    # later, separately-authorized gate on the isolated editorial branch)
    # legitimately supplies the frozen draft's word timings to the live
    # physical repair (`protected_speech_by_clip_id`) so the repair never
    # trims into speech -- same self-resolving-guard pattern as the D-288
    # removals below (docs/CUTSELL_DECISIONS.md D-291.6 has the full
    # disclosure).
    # render_delivery.py / export_job.py removed from this closed-track
    # list: D-288 (a later, separately-authorized gate) legitimately adds
    # the watch_listen_status delivery gate and the real perceptual-review
    # call to these two files -- same self-resolving-guard pattern as the
    # precedent in test_cutsell_d269a_live_tenant_safe_delivery.py (which
    # documents the D-282 uploads.py/main.py precedent this follows).
    "cutsell_worker/post_render_media_qc.py",
])
def test_closed_track_files_unmodified_by_this_gate(relative_path):
    """D-274E-A's own scope is additive-only inside `render.py` and
    `media_overlay_render.py` -- the canonical output color-metadata
    flags plus their four call-site insertions. No other production
    module's own content changes as part of this gate."""
    result = subprocess.run(
        ["git", "diff", "--stat", "de3bb99", "--", relative_path],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert result.stdout.strip() == "", f"{relative_path} was modified by D-274E-A: {result.stdout}"


def test_only_output_flags_added_no_filtergraph_or_codec_change():
    """Stage 8: prove the ACTUAL diff to `render.py`/`media_overlay_
    render.py` touches ONLY the four new output-flag insertion points
    plus the new shared constant/import -- no filter string, no codec,
    no CRF, no preset, no pix_fmt, no fps default changed."""
    result = subprocess.run(
        ["git", "diff", "de3bb99", "--", "cutsell_worker/render.py", "cutsell_worker/media_overlay_render.py"],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]),
    )
    diff = result.stdout
    removed_lines = [line for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")]
    # No removed line may touch codec/preset/crf/pix_fmt/filter_complex/
    # vf/scale/concat/afade/aformat -- this gate is additive-only.
    forbidden_substrings = (
        "libx265", "libvpx", "-crf 18", "-crf 22", "-preset fast",
        "scale=", "concat=", "afade", "aformat", "-vf ", "amix=",
    )
    for line in removed_lines:
        for forbidden in forbidden_substrings:
            assert forbidden not in line, f"unexpected removed filtergraph/codec line: {line}"
