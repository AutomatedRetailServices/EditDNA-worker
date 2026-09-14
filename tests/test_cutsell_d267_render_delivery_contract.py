"""D-267 -- RENDER IDENTITY + DELIVERY CONTRACT FOUNDATION.

Post D-266 (structured render failure observability + atomic promotion) /
D-266A (1200s ffmpeg execution timeout). Renderer execution safety is
CLOSED; this file proves the new render-identity/output-hash/delivery-
contract layer on top of it, offline, with real ffmpeg renders where a
render is actually needed and deterministic fixtures everywhere else.

No RAW, no provider, no S3/network call anywhere in this file.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from cutsell_worker import render, render_delivery as rd
from cutsell_worker.render_plan import RenderSegment

pytestmark_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available on this runner")


def _ffmpeg(args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args], check=True)


@pytest.fixture(scope="module")
def source_clip(tmp_path_factory):
    directory = tmp_path_factory.mktemp("d267_render")
    path = str(directory / "source.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000,volume=0.3,aformat=channel_layouts=stereo",
        "-t", "3", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-b:a", "96k", path,
    ])
    return path


def _segment(source_path: str, clip_id: str = "c1", start: float = 0.0, end: float = 1.5, **kwargs) -> RenderSegment:
    return RenderSegment(clip_id=clip_id, source_asset_id="s1", source_path=source_path, start=start, end=end, **kwargs)


def _source_without_docstrings(path: str) -> str:
    """D-262/D-263's own established false-positive-proofing technique --
    strip every module/function/class docstring before scanning for
    forbidden vocabulary, since this file's own scope-discipline prose
    legitimately names things it says it does NOT do."""
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


# =============================================================================
# Stage 1/2/10/12 -- render identity
# =============================================================================

def test_render_identity_deterministic_same_input():
    segs = (_segment("/a/x.mp4"), _segment("/a/y.mp4", "c2", 1.5, 3.0))
    a = rd.compute_render_identity(segs, width=1080, height=1920, fps=30)
    b = rd.compute_render_identity(segs, width=1080, height=1920, fps=30)
    assert a == b
    assert a.startswith("render_")


def test_render_identity_filename_path_independent():
    """Two segments with IDENTICAL semantic content but DIFFERENT source
    paths/filenames must mint the SAME render_identity (Stage 10)."""
    seg_a = _segment("/machine-one/tmp/x.mp4")
    seg_b = _segment("/completely/different/machine/y.mp4")
    identity_a = rd.compute_render_identity((seg_a,), width=1080, height=1920, fps=30)
    identity_b = rd.compute_render_identity((seg_b,), width=1080, height=1920, fps=30)
    assert identity_a == identity_b


def test_render_identity_source_asset_id_matters():
    seg_a = _segment("/a/x.mp4")
    seg_b = dataclasses.replace(seg_a, source_asset_id="different_source")
    assert rd.compute_render_identity((seg_a,), width=1080, height=1920, fps=30) != \
        rd.compute_render_identity((seg_b,), width=1080, height=1920, fps=30)


def test_render_identity_clip_order_matters():
    seg_a = _segment("/a/x.mp4", "c1", 0.0, 1.0)
    seg_b = _segment("/a/x.mp4", "c2", 1.0, 2.0)
    forward = rd.compute_render_identity((seg_a, seg_b), width=1080, height=1920, fps=30)
    backward = rd.compute_render_identity((seg_b, seg_a), width=1080, height=1920, fps=30)
    assert forward != backward


def test_render_identity_selection_matters():
    seg_a = _segment("/a/x.mp4", "c1", 0.0, 1.0)
    seg_b = _segment("/a/x.mp4", "c2", 1.0, 2.0)
    two_clips = rd.compute_render_identity((seg_a, seg_b), width=1080, height=1920, fps=30)
    one_clip = rd.compute_render_identity((seg_a,), width=1080, height=1920, fps=30)
    assert two_clips != one_clip


def test_render_identity_visual_plan_identity_matters():
    seg = _segment("/a/x.mp4")
    with_plan = rd.compute_render_identity((seg,), width=1080, height=1920, fps=30, visual_finishing_plan_identity="vfplan_abc")
    without_plan = rd.compute_render_identity((seg,), width=1080, height=1920, fps=30, visual_finishing_plan_identity=None)
    different_plan = rd.compute_render_identity((seg,), width=1080, height=1920, fps=30, visual_finishing_plan_identity="vfplan_xyz")
    assert with_plan != without_plan
    assert with_plan != different_plan


def test_render_identity_audio_plan_identity_matters():
    seg = _segment("/a/x.mp4")
    a = rd.compute_render_identity((seg,), width=1080, height=1920, fps=30, audio_finishing_plan_identity="afplan_1")
    b = rd.compute_render_identity((seg,), width=1080, height=1920, fps=30, audio_finishing_plan_identity="afplan_2")
    assert a != b


def test_render_identity_renderer_contract_version_matters():
    seg = _segment("/a/x.mp4")
    v1 = rd.compute_render_identity((seg,), width=1080, height=1920, fps=30, renderer_contract_version=1)
    v2 = rd.compute_render_identity((seg,), width=1080, height=1920, fps=30, renderer_contract_version=2)
    assert v1 != v2


def test_render_identity_visual_transform_spec_matters(source_clip):
    from cutsell_worker.visual_finishing_executor import VisualTransformSpec
    transform = VisualTransformSpec(
        action="PUNCH_IN", source_width=160, source_height=120, scale_factor=1.25,
        scaled_width=200, scaled_height=150, crop_x=20, crop_y=15, crop_width=160, crop_height=120,
    )
    seg_plain = _segment(source_clip)
    seg_transformed = _segment(source_clip, visual_transform=transform)
    plain = rd.compute_render_identity((seg_plain,), width=1080, height=1920, fps=30)
    transformed = rd.compute_render_identity((seg_transformed,), width=1080, height=1920, fps=30)
    assert plain != transformed


def test_render_identity_output_geometry_matters():
    seg = _segment("/a/x.mp4")
    a = rd.compute_render_identity((seg,), width=1080, height=1920, fps=30)
    b = rd.compute_render_identity((seg,), width=720, height=1280, fps=30)
    assert a != b


# =============================================================================
# Stage 3/13 -- output SHA-256 (real bytes)
# =============================================================================

@pytestmark_ffmpeg
def test_output_sha256_computed_from_real_final_file(source_clip, tmp_path):
    out = tmp_path / "out.mp4"
    render.render_preview((_segment(source_clip),), str(out))
    digest = rd.compute_output_sha256(str(out))
    expected = hashlib.sha256(out.read_bytes()).hexdigest()
    assert digest == expected
    assert len(digest) == 64


def test_output_sha256_deterministic_on_same_bytes(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"identical content" * 1000)
    a = rd.compute_output_sha256(str(path))
    b = rd.compute_output_sha256(str(path))
    assert a == b


def test_output_sha256_raises_on_missing_file(tmp_path):
    with pytest.raises(OSError):
        rd.compute_output_sha256(str(tmp_path / "does-not-exist.mp4"))


# =============================================================================
# Stage 8 -- technical QC integration (consumes, never re-derives)
# =============================================================================

def test_qc_status_from_deliverable_true():
    fake_result = SimpleNamespace(deliverable=True)
    assert rd.technical_qc_status_from_live_render_qc(fake_result) == rd.TECHNICAL_QC_STATUS_PASS


def test_qc_status_from_deliverable_false():
    fake_result = SimpleNamespace(deliverable=False)
    assert rd.technical_qc_status_from_live_render_qc(fake_result) == rd.TECHNICAL_QC_STATUS_FAIL


def test_qc_status_none_is_not_run():
    assert rd.technical_qc_status_from_live_render_qc(None) == rd.TECHNICAL_QC_STATUS_NOT_RUN


def test_qc_status_never_reads_status_string_directly():
    """Even a `.status` attribute that LOOKS like PASS must not matter --
    only `.deliverable` is ever read (D-036 item 7 authority boundary)."""
    fake_result = SimpleNamespace(deliverable=False, status="PASS")
    assert rd.technical_qc_status_from_live_render_qc(fake_result) == rd.TECHNICAL_QC_STATUS_FAIL


# =============================================================================
# Stage 4/5/6/7/9 -- delivery record construction and DELIVERY_READY gate
# =============================================================================

@pytestmark_ffmpeg
def test_full_success_local_only_reaches_ready_for_upload(source_clip, tmp_path):
    out = tmp_path / "out.mp4"
    render.render_preview((_segment(source_clip),), str(out))
    identity = rd.compute_render_identity((_segment(source_clip),), width=1080, height=1920, fps=30)
    record = rd.build_render_delivery_record(
        render_identity=identity,
        render_execution_status=rd.RENDER_EXECUTION_STATUS_SUCCEEDED,
        final_path=str(out),
        technical_qc_status=rd.TECHNICAL_QC_STATUS_PASS,
        require_upload=False,
    )
    assert record.delivery_status == rd.DELIVERY_STATUS_READY_FOR_UPLOAD
    assert record.ready_for_delivery is False  # Stage 7: local-only never fakes DELIVERY_READY
    assert record.output_sha256 is not None
    assert record.output_size_bytes and record.output_size_bytes > 0
    assert record.errors == ()


@pytestmark_ffmpeg
def test_full_success_with_upload_reaches_delivery_ready(source_clip, tmp_path):
    out = tmp_path / "out.mp4"
    render.render_preview((_segment(source_clip),), str(out))
    identity = rd.compute_render_identity((_segment(source_clip),), width=1080, height=1920, fps=30)
    record = rd.build_render_delivery_record(
        render_identity=identity,
        render_execution_status=rd.RENDER_EXECUTION_STATUS_SUCCEEDED,
        final_path=str(out),
        technical_qc_status=rd.TECHNICAL_QC_STATUS_PASS,
        require_upload=True,
        upload_status=rd.UPLOAD_STATUS_SUCCEEDED,
    )
    assert record.delivery_status == rd.DELIVERY_STATUS_DELIVERY_READY
    assert record.ready_for_delivery is True


def test_render_execution_failed_blocks_delivery():
    record = rd.build_render_delivery_record(
        render_identity="render_x", render_execution_status=rd.RENDER_EXECUTION_STATUS_FAILED,
        final_path=None, technical_qc_status=rd.TECHNICAL_QC_STATUS_NOT_RUN,
    )
    assert record.delivery_status == rd.DELIVERY_STATUS_RENDER_FAILED
    assert record.ready_for_delivery is False
    assert record.output_sha256 is None


def test_missing_output_blocks_delivery(tmp_path):
    record = rd.build_render_delivery_record(
        render_identity="render_x", render_execution_status=rd.RENDER_EXECUTION_STATUS_SUCCEEDED,
        final_path=str(tmp_path / "missing.mp4"), technical_qc_status=rd.TECHNICAL_QC_STATUS_PASS,
    )
    assert record.delivery_status == rd.DELIVERY_STATUS_RENDER_FAILED
    assert "final_output_missing" in record.errors


def test_empty_output_blocks_delivery(tmp_path):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    record = rd.build_render_delivery_record(
        render_identity="render_x", render_execution_status=rd.RENDER_EXECUTION_STATUS_SUCCEEDED,
        final_path=str(empty), technical_qc_status=rd.TECHNICAL_QC_STATUS_PASS,
    )
    assert record.delivery_status == rd.DELIVERY_STATUS_RENDER_FAILED
    assert "final_output_empty" in record.errors
    assert record.output_sha256 is None


def test_hash_failure_blocks_delivery(tmp_path, monkeypatch):
    real_file = tmp_path / "out.mp4"
    real_file.write_bytes(b"some bytes")

    def _raise(*_a, **_k):
        raise OSError("simulated disk read failure")

    monkeypatch.setattr(rd, "compute_output_sha256", _raise)
    record = rd.build_render_delivery_record(
        render_identity="render_x", render_execution_status=rd.RENDER_EXECUTION_STATUS_SUCCEEDED,
        final_path=str(real_file), technical_qc_status=rd.TECHNICAL_QC_STATUS_PASS,
    )
    assert record.delivery_status == rd.DELIVERY_STATUS_HASH_FAILED
    assert record.output_sha256 is None
    assert record.ready_for_delivery is False


def test_qc_fail_blocks_delivery(tmp_path):
    real_file = tmp_path / "out.mp4"
    real_file.write_bytes(b"some bytes")
    record = rd.build_render_delivery_record(
        render_identity="render_x", render_execution_status=rd.RENDER_EXECUTION_STATUS_SUCCEEDED,
        final_path=str(real_file), technical_qc_status=rd.TECHNICAL_QC_STATUS_FAIL,
    )
    assert record.delivery_status == rd.DELIVERY_STATUS_QC_FAILED
    assert record.ready_for_delivery is False
    # even though QC failed, the file WAS hashed -- hashing and QC are independent gates
    assert record.output_sha256 is not None


def test_qc_not_run_blocks_delivery(tmp_path):
    real_file = tmp_path / "out.mp4"
    real_file.write_bytes(b"some bytes")
    record = rd.build_render_delivery_record(
        render_identity="render_x", render_execution_status=rd.RENDER_EXECUTION_STATUS_SUCCEEDED,
        final_path=str(real_file), technical_qc_status=rd.TECHNICAL_QC_STATUS_NOT_RUN,
    )
    assert record.delivery_status == rd.DELIVERY_STATUS_QC_FAILED
    assert record.ready_for_delivery is False


def test_blocking_error_forces_delivery_blocked(tmp_path):
    real_file = tmp_path / "out.mp4"
    real_file.write_bytes(b"some bytes")
    record = rd.build_render_delivery_record(
        render_identity="render_x", render_execution_status=rd.RENDER_EXECUTION_STATUS_SUCCEEDED,
        final_path=str(real_file), technical_qc_status=rd.TECHNICAL_QC_STATUS_PASS,
        require_upload=True, upload_status=rd.UPLOAD_STATUS_SUCCEEDED,
        blocking_error="ownership_mismatch_detected",
    )
    assert record.delivery_status == rd.DELIVERY_STATUS_DELIVERY_BLOCKED
    assert record.ready_for_delivery is False
    assert "ownership_mismatch_detected" in record.errors


def test_unrecognized_render_execution_status_is_unknown():
    record = rd.build_render_delivery_record(
        render_identity="render_x", render_execution_status="SOMETHING_ELSE",
        final_path=None, technical_qc_status=rd.TECHNICAL_QC_STATUS_NOT_RUN,
    )
    assert record.delivery_status == rd.DELIVERY_STATUS_UNKNOWN


# =============================================================================
# Stage 5/15/17/18 -- upload state transitions
# =============================================================================

def _ready_for_upload_record(tmp_path) -> rd.RenderDeliveryRecord:
    real_file = tmp_path / "out.mp4"
    real_file.write_bytes(b"some bytes")
    return rd.build_render_delivery_record(
        render_identity="render_x", render_execution_status=rd.RENDER_EXECUTION_STATUS_SUCCEEDED,
        final_path=str(real_file), technical_qc_status=rd.TECHNICAL_QC_STATUS_PASS,
        require_upload=True, upload_status=rd.UPLOAD_STATUS_NOT_ATTEMPTED,
    )


def test_upload_in_progress_state(tmp_path):
    record = _ready_for_upload_record(tmp_path)
    assert record.delivery_status == rd.DELIVERY_STATUS_READY_FOR_UPLOAD
    in_progress = rd.with_upload_result(record, upload_status=rd.UPLOAD_STATUS_IN_PROGRESS)
    assert in_progress.delivery_status == rd.DELIVERY_STATUS_UPLOAD_IN_PROGRESS
    assert in_progress.ready_for_delivery is False


def test_upload_success_transitions_to_delivery_ready(tmp_path):
    record = _ready_for_upload_record(tmp_path)
    succeeded = rd.with_upload_result(
        record, upload_status=rd.UPLOAD_STATUS_SUCCEEDED,
        remote_reference="bucket/key/abc.mp4", remote_etag="\"deadbeef\"", remote_size_bytes=10,
    )
    assert succeeded.delivery_status == rd.DELIVERY_STATUS_DELIVERY_READY
    assert succeeded.ready_for_delivery is True
    assert succeeded.remote_reference == "bucket/key/abc.mp4"


def test_upload_failure_never_yields_delivery_ready(tmp_path):
    record = _ready_for_upload_record(tmp_path)
    failed = rd.with_upload_result(record, upload_status=rd.UPLOAD_STATUS_FAILED)
    assert failed.delivery_status == rd.DELIVERY_STATUS_UPLOAD_FAILED
    assert failed.ready_for_delivery is False
    assert "upload_failed" in failed.errors


def test_retry_after_upload_failure_can_still_succeed(tmp_path):
    record = _ready_for_upload_record(tmp_path)
    failed = rd.with_upload_result(record, upload_status=rd.UPLOAD_STATUS_FAILED)
    retried = rd.with_upload_result(failed, upload_status=rd.UPLOAD_STATUS_SUCCEEDED)
    assert retried.delivery_status == rd.DELIVERY_STATUS_DELIVERY_READY


def test_with_upload_result_ineligible_state_returns_unchanged_copy(tmp_path):
    real_file = tmp_path / "out.mp4"
    real_file.write_bytes(b"x")
    render_failed = rd.build_render_delivery_record(
        render_identity="render_x", render_execution_status=rd.RENDER_EXECUTION_STATUS_FAILED,
        final_path=None, technical_qc_status=rd.TECHNICAL_QC_STATUS_NOT_RUN,
    )
    result = rd.with_upload_result(render_failed, upload_status=rd.UPLOAD_STATUS_SUCCEEDED)
    assert result.delivery_status == rd.DELIVERY_STATUS_RENDER_FAILED  # never upgraded
    assert result is not render_failed  # still a new object (Stage 14)


def test_with_upload_result_never_mutates_original_record(tmp_path):
    record = _ready_for_upload_record(tmp_path)
    original_status = record.delivery_status
    rd.with_upload_result(record, upload_status=rd.UPLOAD_STATUS_SUCCEEDED)
    assert record.delivery_status == original_status  # original untouched


def test_unrecognized_upload_status_is_unknown(tmp_path):
    record = _ready_for_upload_record(tmp_path)
    result = rd.with_upload_result(record, upload_status="WEIRD_STATUS")
    assert result.delivery_status == rd.DELIVERY_STATUS_UNKNOWN


# =============================================================================
# Stage 14 -- immutability
# =============================================================================

def test_render_delivery_record_is_frozen():
    record = rd.build_render_delivery_record(
        render_identity="render_x", render_execution_status=rd.RENDER_EXECUTION_STATUS_FAILED,
        final_path=None, technical_qc_status=rd.TECHNICAL_QC_STATUS_NOT_RUN,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.delivery_status = rd.DELIVERY_STATUS_DELIVERY_READY  # type: ignore[misc]


# =============================================================================
# Stage 16/19 -- S3 ETag semantics
# =============================================================================

@pytest.mark.parametrize("etag", ["\"9bb58f26192e4ba00f01e2e7b136bbd8\"", "\"abc-3\"", None, ""])
def test_etag_never_treated_as_sha256(etag):
    assert rd.is_etag_valid_sha256_proxy(etag) is False


# =============================================================================
# Stage 21 -- ownership linkage (foundation only, no auth invented)
# =============================================================================

def test_ownership_fields_carried_when_supplied(tmp_path):
    real_file = tmp_path / "out.mp4"
    real_file.write_bytes(b"x")
    record = rd.build_render_delivery_record(
        render_identity="render_x", render_execution_status=rd.RENDER_EXECUTION_STATUS_SUCCEEDED,
        final_path=str(real_file), technical_qc_status=rd.TECHNICAL_QC_STATUS_PASS,
        project_id="proj_1", job_id="job_42",
    )
    assert record.project_id == "proj_1"
    assert record.job_id == "job_42"


def test_two_records_with_distinct_ownership_do_not_collide(tmp_path):
    real_file = tmp_path / "out.mp4"
    real_file.write_bytes(b"x")
    record_a = rd.build_render_delivery_record(
        render_identity="render_same", render_execution_status=rd.RENDER_EXECUTION_STATUS_SUCCEEDED,
        final_path=str(real_file), technical_qc_status=rd.TECHNICAL_QC_STATUS_PASS, project_id="tenant_a",
    )
    record_b = rd.build_render_delivery_record(
        render_identity="render_same", render_execution_status=rd.RENDER_EXECUTION_STATUS_SUCCEEDED,
        final_path=str(real_file), technical_qc_status=rd.TECHNICAL_QC_STATUS_PASS, project_id="tenant_b",
    )
    assert record_a.project_id != record_b.project_id
    assert record_a is not record_b


# =============================================================================
# Stage 20 -- observability diagnostics
# =============================================================================

def test_diagnostics_payload_shape(tmp_path):
    real_file = tmp_path / "out.mp4"
    real_file.write_bytes(b"x")
    record = rd.build_render_delivery_record(
        render_identity="render_x", render_execution_status=rd.RENDER_EXECUTION_STATUS_SUCCEEDED,
        final_path=str(real_file), technical_qc_status=rd.TECHNICAL_QC_STATUS_PASS,
    )
    diagnostics = rd.render_delivery_diagnostics(record)
    for key in (
        "render_identity", "render_status", "qc_status", "hash_status", "upload_status",
        "delivery_status", "output_size_bytes", "output_sha256", "errors", "warnings", "created_at",
    ):
        assert key in diagnostics
    assert diagnostics["hash_status"] == "OK"


def test_diagnostics_hash_status_failed(tmp_path, monkeypatch):
    real_file = tmp_path / "out.mp4"
    real_file.write_bytes(b"x")
    monkeypatch.setattr(rd, "compute_output_sha256", lambda *_a, **_k: (_ for _ in ()).throw(OSError("boom")))
    record = rd.build_render_delivery_record(
        render_identity="render_x", render_execution_status=rd.RENDER_EXECUTION_STATUS_SUCCEEDED,
        final_path=str(real_file), technical_qc_status=rd.TECHNICAL_QC_STATUS_PASS,
    )
    diagnostics = rd.render_delivery_diagnostics(record)
    assert diagnostics["hash_status"] == "FAILED"


# =============================================================================
# Stage 22/23 -- path safety, no secrets, no shell
# =============================================================================

@pytestmark_ffmpeg
@pytest.mark.parametrize("name", ["out with spaces.mp4", "sortie-éè.mp4", "it's-a-take.mp4"])
def test_unicode_and_special_char_paths_hash_correctly(source_clip, tmp_path, name):
    out = tmp_path / name
    render.render_preview((_segment(source_clip),), str(out))
    digest = rd.compute_output_sha256(str(out))
    assert digest == hashlib.sha256(out.read_bytes()).hexdigest()


def test_no_shell_true_in_render_delivery_module():
    source = Path("cutsell_worker/render_delivery.py").read_text(encoding="utf-8")
    assert "shell=True" not in source


def test_no_network_or_credential_construction():
    source = _source_without_docstrings("cutsell_worker/render_delivery.py")
    for needle in ("boto3", "s3_client", "AWS_SECRET", "Authorization", "requests.", "urllib"):
        assert needle not in source


def test_no_actual_upload_call_in_module():
    source = _source_without_docstrings("cutsell_worker/render_delivery.py")
    for needle in ("multipart_uploads", "upload_part", "put_object"):
        assert needle not in source


# =============================================================================
# Stage 23 items 21-33 -- unrelated-authority firewalls
# =============================================================================

def _run_git_diff(rel_path: str) -> str:
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", rel_path], capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


@pytest.mark.parametrize("rel_path", [
    "cutsell_worker/render.py",
    "cutsell_worker/render_plan.py",
    "cutsell_worker/audio_finishing_executor.py",
    "cutsell_worker/audio_finishing_composition.py",
    "cutsell_worker/visual_finishing_executor.py",
    "cutsell_worker/visual_finishing_composition.py",
    "cutsell_worker/boundary_engine_pass.py",
    "cutsell_worker/pacing_transition_decision.py",
    "cutsell_worker/post_render_media_qc.py",
    "cutsell_worker/post_render_watch_listen_qc.py",
    "cutsell_worker/live_render_qc.py",
    "cutsell_worker/media_probe.py",
    "cutsell_worker/finishing_contract.py",
    "cutsell_worker/multipart_uploads.py",
])
def test_unrelated_authorities_unchanged(rel_path):
    assert _run_git_diff(rel_path) == "", f"D-267 must not touch {rel_path}"


def test_render_timeout_still_1200():
    assert render.RENDER_FFMPEG_TIMEOUT_SEC == 1200.0


def test_codec_filtergraph_still_unchanged():
    source = Path("cutsell_worker/render.py").read_text(encoding="utf-8")
    assert '"libx264"' in source
    assert '"-crf", "20"' in source
    assert "RENDER_FPS_DEFAULT = 30" in source
