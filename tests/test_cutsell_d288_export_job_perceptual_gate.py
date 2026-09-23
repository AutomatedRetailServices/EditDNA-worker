"""D-288 -- the real production export path now runs perceptual review.

Audit finding this closes (`docs/CUTSELL_DECISIONS.md` D-288): direct
reading of `cutsell_worker/export_job.py` (the real mobile-app export job)
found it NEVER called `perceptual_watch_listen.review_rendered_candidate`
(or anything in `universal_clean_cut_validation.py`) at all -- only the
Video00 RAW *validation harness* ran the perceptual reviewer. A technical
post-render QC PASS alone was sufficient for `run_export_job` to mark a
project "finished", upload the file, and hand back a real `download_url`
to the mobile client, with zero perceptual gate in between.

This file proves the fix end-to-end through `export_job._tenant_safe_
deliver`/`run_export_job`, monkeypatching `export_job.perceptual_review_
for_rendered_candidate` (the same call `_tenant_safe_deliver` now makes)
to control the perceptual verdict deterministically, without needing a
real ffmpeg render or real audio/video decoding. No Video00 fact/id
anywhere below.
"""
from __future__ import annotations

import functools
from types import SimpleNamespace

import pytest

from cutsell_worker import export_job
from cutsell_worker import exports
from cutsell_worker import render_delivery as rd
from cutsell_worker import tenant_safe_delivery as tsd
from cutsell_worker.perceptual_watch_listen import (
    WATCH_LISTEN_BLOCKED,
    WATCH_LISTEN_HUMAN_REVIEW_REQUIRED,
    WATCH_LISTEN_SYSTEM_PASS,
)
from cutsell_worker.render_plan import RenderSegment


class FakeS3Client:
    """Minimal stand-in for `boto3.client("s3", ...)` -- same shape as this
    repo's own D-269A fixture (`test_cutsell_d269a_live_tenant_safe_
    delivery.py`), reused here rather than reinvented."""

    def __init__(self):
        self.objects: dict[str, dict] = {}

    def upload_file(self, filename, bucket, key, **_kwargs):
        with open(filename, "rb") as handle:
            data = handle.read()
        self.objects[key] = {"body": data, "size": len(data)}

    def head_object(self, Bucket, Key):  # noqa: N803 -- matches boto3's own signature
        obj = self.objects.get(Key)
        if obj is None:
            from botocore.exceptions import ClientError
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        return {"ContentLength": obj["size"], "Metadata": {}}

    def generate_presigned_url(self, *_args, **_kwargs):
        return "https://x.invalid/presigned"


@pytest.fixture
def fake_s3(monkeypatch):
    client = FakeS3Client()
    monkeypatch.setattr(exports, "load_runtime_config", lambda: SimpleNamespace(s3_bucket="test-bucket", aws_region="us-east-1"))
    return client


@pytest.fixture
def wire_real_store_export(monkeypatch, fake_s3):
    monkeypatch.setattr(export_job, "store_export", functools.partial(exports.store_export, client=fake_s3))
    return fake_s3


def _plan():
    return (RenderSegment(clip_id="clip-1", source_asset_id="src-1", source_path="/tmp/x.mp4", start=1.0, end=2.0, caption_text="hello"),)


def _rendered_file(tmp_path, content=b"rendered-mp4-bytes") -> str:
    path = tmp_path / "out.mp4"
    path.write_bytes(content)
    return str(path)


def _fake_draft():
    return SimpleNamespace(selected=(), diagnostics={})


def _fake_qc_result():
    return SimpleNamespace(status="PASS", attempts=(), output_path=None)


def _stub_review(monkeypatch, watch_listen_status: str | None):
    def _fake(output_path, draft, local_paths, qc_result):
        if watch_listen_status is None:
            return None
        return {"watch_listen_status": watch_listen_status, "status": "FAIL" if watch_listen_status == WATCH_LISTEN_BLOCKED else "UNCERTAIN"}
    monkeypatch.setattr(export_job, "perceptual_review_for_rendered_candidate", _fake)


# =============================================================================
# Positive: a real draft/qc_result is passed through -> perceptual review runs
# =============================================================================

def test_system_pass_reaches_delivery_ready(tmp_path, wire_real_store_export, monkeypatch):
    _stub_review(monkeypatch, WATCH_LISTEN_SYSTEM_PASS)
    output = _rendered_file(tmp_path)
    result = export_job._tenant_safe_deliver(
        output_path=output, plan=_plan(), draft=_fake_draft(), local_paths={}, qc_result=_fake_qc_result(),
        project_id="proj-1", user_id="user-1", job_id="job-1",
    )
    assert result["delivery_status"] == rd.DELIVERY_STATUS_DELIVERY_READY
    assert result["watch_listen_status"] == WATCH_LISTEN_SYSTEM_PASS
    assert result["download_url"] is not None


# =============================================================================
# Negative (required): BLOCKED never delivers
# =============================================================================

def test_watch_listen_blocked_never_delivers(tmp_path, wire_real_store_export, monkeypatch):
    _stub_review(monkeypatch, WATCH_LISTEN_BLOCKED)
    output = _rendered_file(tmp_path)
    with pytest.raises(export_job.TenantSafeDeliveryBlocked) as exc_info:
        export_job._tenant_safe_deliver(
            output_path=output, plan=_plan(), draft=_fake_draft(), local_paths={}, qc_result=_fake_qc_result(),
            project_id="proj-1", user_id="user-1", job_id="job-1",
        )
    assert exc_info.value.record.delivery_status == rd.DELIVERY_STATUS_DELIVERY_BLOCKED
    assert exc_info.value.perceptual_review["watch_listen_status"] == WATCH_LISTEN_BLOCKED
    # The upload never happened -- BLOCKED is caught BEFORE store_export runs.
    assert wire_real_store_export.objects == {}


def test_watch_listen_human_review_required_never_auto_delivers(tmp_path, wire_real_store_export, monkeypatch):
    _stub_review(monkeypatch, WATCH_LISTEN_HUMAN_REVIEW_REQUIRED)
    output = _rendered_file(tmp_path)
    with pytest.raises(export_job.TenantSafeDeliveryBlocked) as exc_info:
        export_job._tenant_safe_deliver(
            output_path=output, plan=_plan(), draft=_fake_draft(), local_paths={}, qc_result=_fake_qc_result(),
            project_id="proj-1", user_id="user-1", job_id="job-1",
        )
    assert exc_info.value.record.delivery_status == rd.DELIVERY_STATUS_WATCH_LISTEN_PENDING
    assert wire_real_store_export.objects == {}


def test_a_perceptual_review_exception_fails_closed_never_silently_delivers(tmp_path, wire_real_store_export, monkeypatch):
    """`review_rendered_candidate` itself never raises (its own contract),
    but if this call site's own wiring ever did, it must fail CLOSED --
    never silently proceed to delivery as though nothing happened."""
    def _raise(*_args, **_kwargs):
        raise RuntimeError("simulated decode failure")
    monkeypatch.setattr(export_job, "perceptual_review_for_rendered_candidate", _raise)
    output = _rendered_file(tmp_path)
    with pytest.raises(RuntimeError, match="simulated decode failure"):
        export_job._tenant_safe_deliver(
            output_path=output, plan=_plan(), draft=_fake_draft(), local_paths={}, qc_result=_fake_qc_result(),
            project_id="proj-1", user_id="user-1", job_id="job-1",
        )
    assert wire_real_store_export.objects == {}


# =============================================================================
# Negative (required): technical QC PASS is never a substitute for Watch+Listen
# =============================================================================

def test_technical_qc_pass_alone_no_longer_sufficient_for_delivery(tmp_path, wire_real_store_export, monkeypatch):
    """The exact real production gap this gate closes: before D-288, a
    technical-QC-PASSed render (`technical_qc_status=PASS`, proven by this
    call reaching `_tenant_safe_deliver` at all -- `run_export_job` only
    calls it after `qc_result.status == "PASS"`) was sufficient for
    DELIVERY_READY on its own. It no longer is, once a real perceptual
    verdict is supplied and that verdict is not SYSTEM_PASS/HUMAN_APPROVED."""
    _stub_review(monkeypatch, WATCH_LISTEN_HUMAN_REVIEW_REQUIRED)
    output = _rendered_file(tmp_path)
    with pytest.raises(export_job.TenantSafeDeliveryBlocked):
        export_job._tenant_safe_deliver(
            output_path=output, plan=_plan(), draft=_fake_draft(), local_paths={}, qc_result=_fake_qc_result(),
            project_id="proj-1", user_id="user-1", job_id="job-1",
        )


# =============================================================================
# Backward compatibility: a caller that never supplies a draft is unaffected
# (this module's own D-269A remote/ownership/upload test suite -- a
# DIFFERENT concern from perceptual review)
# =============================================================================

def test_omitting_draft_leaves_the_perceptual_gate_unapplied(tmp_path, wire_real_store_export):
    output = _rendered_file(tmp_path)
    result = export_job._tenant_safe_deliver(
        output_path=output, plan=_plan(), project_id="proj-1", user_id="user-1", job_id="job-1",
    )
    assert result["delivery_status"] == rd.DELIVERY_STATUS_DELIVERY_READY
    assert result["watch_listen_status"] is None
