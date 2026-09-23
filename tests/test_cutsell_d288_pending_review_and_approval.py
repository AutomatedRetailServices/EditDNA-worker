"""D-288 (findings 1, 2, 6) -- persistent pending review + the real,
authenticated human-approval flow, proven end to end through
`export_job.run_export_job` / `_tenant_safe_deliver` / `resume_delivery_
after_approval`.

Covers the exact behaviors the correction gate requires:
- a HUMAN_REVIEW_REQUIRED render survives `TemporaryDirectory` cleanup
  (persisted before the job's own `with` block exits) and is recoverable
  after the fact -- never terminates as a failed render;
- a valid, exact-artifact-bound approval lets delivery continue;
- a stale approval (wrong render_identity, wrong output_sha256, wrong
  plan_id/plan_version, or bound to a DIFFERENT job's pending record)
  never does;
- an empty requesting/approver identity is rejected;
- BLOCKED is never approvable, even by an otherwise-valid approval;
- `run_export_job` itself reports `state="pending_review"`, never
  `"failed"`, for a HUMAN_REVIEW_REQUIRED candidate.

No Video00 fact/id anywhere below.
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from cutsell_worker import export_job
from cutsell_worker import exports
from cutsell_worker import pending_watch_listen_review as pwl
from cutsell_worker import render_delivery as rd
from cutsell_worker import tenant_safe_delivery as tsd
from cutsell_worker.perceptual_watch_listen import (
    WATCH_LISTEN_BLOCKED,
    WATCH_LISTEN_HUMAN_APPROVED,
    WATCH_LISTEN_HUMAN_REVIEW_REQUIRED,
    WATCH_LISTEN_SYSTEM_PASS,
)
from cutsell_worker.render_plan import RenderSegment


# =============================================================================
# Shared fakes (same shape as this repo's own D-269A/D-288 fixtures)
# =============================================================================

class FakeS3Client:
    def __init__(self):
        self.objects: dict[str, dict] = {}

    def upload_file(self, filename, bucket, key, **_kwargs):
        with open(filename, "rb") as handle:
            data = handle.read()
        self.objects[key] = {"body": data, "size": len(data)}

    def download_file(self, bucket, key, destination):
        obj = self.objects[key]
        Path(destination).write_bytes(obj["body"])

    def head_object(self, Bucket, Key):  # noqa: N803
        obj = self.objects.get(Key)
        if obj is None:
            from botocore.exceptions import ClientError
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        return {"ContentLength": obj["size"], "Metadata": {}}

    def generate_presigned_url(self, *_args, **_kwargs):
        return "https://x.invalid/presigned"


class FakeRedis:
    def __init__(self):
        self.data: dict[str, str] = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, **_kwargs):
        self.data[key] = value
        return True


@pytest.fixture
def fake_s3(monkeypatch):
    client = FakeS3Client()
    monkeypatch.setattr(exports, "load_runtime_config", lambda: SimpleNamespace(s3_bucket="test-bucket", aws_region="us-east-1"))
    return client


@pytest.fixture
def wire_real_store_export(monkeypatch, fake_s3):
    monkeypatch.setattr(export_job, "store_export", functools.partial(exports.store_export, client=fake_s3))
    return fake_s3


@pytest.fixture
def fake_redis():
    return FakeRedis()


def _plan():
    return (RenderSegment(clip_id="clip-1", source_asset_id="src-1", source_path="/tmp/x.mp4", start=1.0, end=2.0, caption_text="hello"),)


def _rendered_file(tmp_path, content=b"rendered-mp4-bytes") -> str:
    path = tmp_path / "out.mp4"
    path.write_bytes(content)
    return str(path)


def _fake_draft():
    return SimpleNamespace(selected=(), diagnostics={})


def _fake_qc_result(plan_id="plan_test", plan_version=1):
    return SimpleNamespace(status="PASS", attempts=(), output_path=None, plan_id=plan_id, plan_version=plan_version)


def _stub_review(monkeypatch, watch_listen_status: str):
    def _fake(output_path, draft, local_paths, qc_result):
        return {"watch_listen_status": watch_listen_status, "status": "FAIL" if watch_listen_status == WATCH_LISTEN_BLOCKED else "UNCERTAIN"}
    monkeypatch.setattr(export_job, "perceptual_review_for_rendered_candidate", _fake)


# =============================================================================
# Finding 1: the render survives TemporaryDirectory cleanup, recoverable
# =============================================================================

def test_pending_review_file_is_available_after_temporarydirectory_closes(tmp_path, wire_real_store_export, fake_redis, monkeypatch):
    _stub_review(monkeypatch, WATCH_LISTEN_HUMAN_REVIEW_REQUIRED)
    with tempdir_that_actually_closes(tmp_path) as directory:
        output = _rendered_file(Path(directory))
        with pytest.raises(export_job.PendingHumanWatchListenReview) as exc_info:
            export_job._tenant_safe_deliver(
                output_path=output, plan=_plan(), draft=_fake_draft(), local_paths={}, qc_result=_fake_qc_result(),
                project_id="proj-1", user_id="user-1", job_id="job-1",
                pending_review_redis_client=fake_redis,
            )
        record_id = exc_info.value.record.record_id
    # The directory is now gone (closed by the `with` block above) -- the
    # local file no longer exists. The RECORD, and the object it points
    # at in (fake) S3, must still be there.
    assert not Path(output).exists()
    loaded = pwl.load_pending_review(user_id="user-1", project_id="proj-1", job_id="job-1", client=fake_redis)
    assert loaded is not None
    assert loaded.record_id == record_id
    assert any(loaded.pending_s3_uri.endswith(key.split("/")[-1]) for key in wire_real_store_export.objects)


import contextlib
import tempfile as _tempfile


@contextlib.contextmanager
def tempdir_that_actually_closes(tmp_path):
    directory = _tempfile.mkdtemp(dir=str(tmp_path))
    try:
        yield directory
    finally:
        import shutil
        shutil.rmtree(directory, ignore_errors=True)


def test_run_export_job_reports_pending_review_state_never_failed(tmp_path, wire_real_store_export, fake_redis, monkeypatch):
    fake_job = SimpleNamespace(id="job-1", started_at=1_700_000_000.0, meta={}, save_meta=lambda: None)
    rq_module = ModuleType("rq")
    rq_module.get_current_job = lambda: fake_job
    monkeypatch.setitem(sys.modules, "rq", rq_module)

    monkeypatch.setattr(export_job, "validate_product_source_uri", lambda uri, **kwargs: ("bucket", "key"))
    monkeypatch.setattr(export_job, "download_source", lambda uri, destination: Path(destination).write_bytes(b"source") or destination)
    fake_plan = (RenderSegment(clip_id="clip-1", source_asset_id="src-1", source_path="/tmp/x.mp4", start=1.0, end=2.0),)
    monkeypatch.setattr(export_job, "build_render_plan", lambda draft, local_paths: fake_plan)

    def fake_render_with_qc(draft, plan, output, *, text_overlays=(), media_overlays=(), **kwargs):
        Path(output).write_bytes(b"mp4")
        return SimpleNamespace(status="PASS", output_path=output, plan_id="plan_test", plan_version=1, semantic_hash="hash_test", attempts=(), deliverable=True)
    monkeypatch.setattr(export_job, "render_with_post_render_qc", fake_render_with_qc)
    _stub_review(monkeypatch, WATCH_LISTEN_HUMAN_REVIEW_REQUIRED)

    project_states = []
    def _tracking_safe_update_project(**kwargs):
        project_states.append(kwargs.get("state"))
        return {"state": kwargs.get("state")}
    monkeypatch.setattr(export_job, "safe_update_project", _tracking_safe_update_project)
    monkeypatch.setattr(export_job, "_safe_notify", lambda **kwargs: {"status": "queued", "notification_id": "n1"})

    # Route the pending-review Redis client through the same fake -- the
    # real call site takes it via `_tenant_safe_deliver`'s own kwarg
    # (default `None` -> real Redis), so we monkeypatch `pending_watch_
    # listen_review._redis_client` directly (the module-level default-
    # client resolver) rather than threading a new parameter through
    # `run_export_job`'s own public payload shape.
    monkeypatch.setattr(pwl, "_redis_client", lambda client=None: fake_redis if client is None else client)

    draft_payload = {
        "schema_version": "cutsell.v1",
        "project_id": "project-1",
        "strategy": "mixed",
        "selected": [{
            "clip_id": "clip-1", "source_asset_id": "src-1", "source_order": 0,
            "start": 1.0, "end": 2.0, "text": "hello", "caption_text": "hello",
            "semantic_role": "OTHER", "selected": True,
        }],
        "alternates": [], "discarded": [], "diagnostics": {}, "text_overlays": [],
    }

    result = export_job.run_export_job({
        "project_id": "project-1",
        "user_id": "user-1",
        "draft": draft_payload,
        "sources": [{"source_asset_id": "src-1", "original_name": "one.mov", "uri": "s3://bucket/cutsell/uploads/u/p/one.mov"}],
    })

    assert result["state"] == "pending_review"
    assert result["watch_listen_status"] == WATCH_LISTEN_HUMAN_REVIEW_REQUIRED
    assert "pending_review_record_id" in result
    assert "failed" not in project_states
    assert "pending_review" in project_states


# =============================================================================
# Finding 2: the real approval flow -- auth, exact-artifact binding, and
# BLOCKED-never-approvable, all unit-tested directly against
# `pending_watch_listen_review.apply_human_approval`.
# =============================================================================

def _persist_record(fake_redis, fake_s3, *, watch_listen_status=WATCH_LISTEN_HUMAN_REVIEW_REQUIRED,
                     render_identity="render_aaaa", output_sha256="sha-aaaa",
                     plan_id="plan_1", plan_version=1, user_id="user-1", project_id="proj-1", job_id="job-1",
                     local_path=None):
    return pwl.persist_pending_review(
        local_path=local_path,
        user_id=user_id, project_id=project_id, job_id=job_id,
        render_identity=render_identity, output_sha256=output_sha256,
        plan_id=plan_id, plan_version=plan_version,
        watch_listen_status=watch_listen_status, perceptual_review={"status": "UNCERTAIN"},
        client=fake_redis, s3_client=fake_s3,
    )


def test_valid_exact_artifact_approval_promotes_to_human_approved(tmp_path, fake_s3, fake_redis):
    local_path = _rendered_file(tmp_path)
    record = _persist_record(fake_redis, fake_s3, local_path=local_path)
    updated = pwl.apply_human_approval(
        user_id="user-1", project_id="proj-1", job_id="job-1",
        requesting=tsd.DeliveryOwnershipScope(user_id="user-1", project_id="proj-1", job_id="job-1"),
        approved=True, approver="qa@example.com",
        expected_render_identity=record.render_identity, expected_output_sha256=record.output_sha256,
        expected_plan_id=record.plan_id, expected_plan_version=record.plan_version,
        client=fake_redis,
    )
    assert updated.watch_listen_status == WATCH_LISTEN_HUMAN_APPROVED
    assert updated.approval_status == "APPROVED"
    assert updated.approver == "qa@example.com"
    reloaded = pwl.load_pending_review(user_id="user-1", project_id="proj-1", job_id="job-1", client=fake_redis)
    assert reloaded.watch_listen_status == WATCH_LISTEN_HUMAN_APPROVED


def test_stale_approval_wrong_render_identity_is_rejected(tmp_path, fake_s3, fake_redis):
    local_path = _rendered_file(tmp_path)
    record = _persist_record(fake_redis, fake_s3, local_path=local_path)
    with pytest.raises(pwl.PendingReviewError, match="approval_bound_to_different_artifact_or_plan"):
        pwl.apply_human_approval(
            user_id="user-1", project_id="proj-1", job_id="job-1",
            requesting=tsd.DeliveryOwnershipScope(user_id="user-1", project_id="proj-1", job_id="job-1"),
            approved=True, approver="qa@example.com",
            expected_render_identity="render_DIFFERENT", expected_output_sha256=record.output_sha256,
            expected_plan_id=record.plan_id, expected_plan_version=record.plan_version,
            client=fake_redis,
        )
    reloaded = pwl.load_pending_review(user_id="user-1", project_id="proj-1", job_id="job-1", client=fake_redis)
    assert reloaded.watch_listen_status == WATCH_LISTEN_HUMAN_REVIEW_REQUIRED  # unchanged


def test_stale_approval_wrong_output_sha256_is_rejected(tmp_path, fake_s3, fake_redis):
    local_path = _rendered_file(tmp_path)
    record = _persist_record(fake_redis, fake_s3, local_path=local_path)
    with pytest.raises(pwl.PendingReviewError, match="approval_bound_to_different_artifact_or_plan"):
        pwl.apply_human_approval(
            user_id="user-1", project_id="proj-1", job_id="job-1",
            requesting=tsd.DeliveryOwnershipScope(user_id="user-1", project_id="proj-1", job_id="job-1"),
            approved=True, approver="qa@example.com",
            expected_render_identity=record.render_identity, expected_output_sha256="sha-DIFFERENT",
            expected_plan_id=record.plan_id, expected_plan_version=record.plan_version,
            client=fake_redis,
        )


def test_stale_approval_wrong_plan_version_is_rejected(tmp_path, fake_s3, fake_redis):
    """A re-render of the same plan_id at a NEWER plan_version (e.g. a
    repaired v2) must not be approvable by an approval that names the
    OLD version -- the "plan final ejecutado" binding, not just the
    render/hash."""
    local_path = _rendered_file(tmp_path)
    record = _persist_record(fake_redis, fake_s3, local_path=local_path, plan_version=1)
    with pytest.raises(pwl.PendingReviewError, match="approval_bound_to_different_artifact_or_plan"):
        pwl.apply_human_approval(
            user_id="user-1", project_id="proj-1", job_id="job-1",
            requesting=tsd.DeliveryOwnershipScope(user_id="user-1", project_id="proj-1", job_id="job-1"),
            approved=True, approver="qa@example.com",
            expected_render_identity=record.render_identity, expected_output_sha256=record.output_sha256,
            expected_plan_id=record.plan_id, expected_plan_version=2,
            client=fake_redis,
        )


def test_approving_a_different_jobs_pending_record_is_rejected_no_such_record(tmp_path, fake_s3, fake_redis):
    """Approving against a job_id that has no pending record at all --
    never fabricates a record to approve."""
    with pytest.raises(pwl.PendingReviewError, match="no_pending_review_found"):
        pwl.apply_human_approval(
            user_id="user-1", project_id="proj-1", job_id="job-does-not-exist",
            requesting=tsd.DeliveryOwnershipScope(user_id="user-1", project_id="proj-1", job_id="job-does-not-exist"),
            approved=True, approver="qa@example.com",
            expected_render_identity="render_x", expected_output_sha256="sha-x",
            expected_plan_id="plan_x", expected_plan_version=1,
            client=fake_redis,
        )


def test_empty_approver_identity_is_rejected(tmp_path, fake_s3, fake_redis):
    local_path = _rendered_file(tmp_path)
    record = _persist_record(fake_redis, fake_s3, local_path=local_path)
    with pytest.raises(pwl.PendingReviewError, match="approval_requires_non_empty_approver_identity"):
        pwl.apply_human_approval(
            user_id="user-1", project_id="proj-1", job_id="job-1",
            requesting=tsd.DeliveryOwnershipScope(user_id="user-1", project_id="proj-1", job_id="job-1"),
            approved=True, approver="   ",
            expected_render_identity=record.render_identity, expected_output_sha256=record.output_sha256,
            expected_plan_id=record.plan_id, expected_plan_version=record.plan_version,
            client=fake_redis,
        )


def test_empty_requesting_identity_is_rejected():
    """`DeliveryOwnershipScope` itself rejects empty identities at
    construction -- reused, not reinvented, as the auth primitive here."""
    with pytest.raises(ValueError):
        tsd.DeliveryOwnershipScope(user_id="", project_id="proj-1", job_id="job-1")


def test_mismatched_requesting_principal_is_denied_not_silently_approved(tmp_path, fake_s3, fake_redis):
    local_path = _rendered_file(tmp_path)
    record = _persist_record(fake_redis, fake_s3, local_path=local_path)
    with pytest.raises(PermissionError):
        pwl.apply_human_approval(
            user_id="user-1", project_id="proj-1", job_id="job-1",
            requesting=tsd.DeliveryOwnershipScope(user_id="attacker", project_id="proj-1", job_id="job-1"),
            approved=True, approver="qa@example.com",
            expected_render_identity=record.render_identity, expected_output_sha256=record.output_sha256,
            expected_plan_id=record.plan_id, expected_plan_version=record.plan_version,
            client=fake_redis,
        )
    reloaded = pwl.load_pending_review(user_id="user-1", project_id="proj-1", job_id="job-1", client=fake_redis)
    assert reloaded.watch_listen_status == WATCH_LISTEN_HUMAN_REVIEW_REQUIRED  # unchanged


def test_blocked_is_never_approvable_even_with_a_valid_exact_artifact_approval(tmp_path, fake_s3, fake_redis):
    local_path = _rendered_file(tmp_path)
    record = _persist_record(fake_redis, fake_s3, local_path=local_path, watch_listen_status=WATCH_LISTEN_BLOCKED)
    with pytest.raises(pwl.PendingReviewError, match="blocked_never_approvable"):
        pwl.apply_human_approval(
            user_id="user-1", project_id="proj-1", job_id="job-1",
            requesting=tsd.DeliveryOwnershipScope(user_id="user-1", project_id="proj-1", job_id="job-1"),
            approved=True, approver="qa@example.com",
            expected_render_identity=record.render_identity, expected_output_sha256=record.output_sha256,
            expected_plan_id=record.plan_id, expected_plan_version=record.plan_version,
            client=fake_redis,
        )
    reloaded = pwl.load_pending_review(user_id="user-1", project_id="proj-1", job_id="job-1", client=fake_redis)
    assert reloaded.watch_listen_status == WATCH_LISTEN_BLOCKED  # unchanged, never promoted


def test_rejection_records_a_real_rejected_state_not_a_silent_noop(tmp_path, fake_s3, fake_redis):
    local_path = _rendered_file(tmp_path)
    record = _persist_record(fake_redis, fake_s3, local_path=local_path)
    updated = pwl.apply_human_approval(
        user_id="user-1", project_id="proj-1", job_id="job-1",
        requesting=tsd.DeliveryOwnershipScope(user_id="user-1", project_id="proj-1", job_id="job-1"),
        approved=False, approver="qa@example.com",
        expected_render_identity=record.render_identity, expected_output_sha256=record.output_sha256,
        expected_plan_id=record.plan_id, expected_plan_version=record.plan_version,
        client=fake_redis,
    )
    assert updated.approval_status == "REJECTED"
    assert updated.watch_listen_status == WATCH_LISTEN_HUMAN_REVIEW_REQUIRED  # never silently promoted


# =============================================================================
# Finding 2 ("reanudación de entrega"): resume_delivery_after_approval
# =============================================================================

def test_resume_delivery_after_approval_completes_real_tenant_safe_delivery(tmp_path, wire_real_store_export, fake_redis):
    local_path = _rendered_file(tmp_path, content=b"approved-bytes")
    output_sha256 = rd.compute_output_sha256(local_path)
    record = pwl.persist_pending_review(
        local_path=local_path, user_id="user-1", project_id="proj-1", job_id="job-1",
        render_identity="render_" + "b" * 24, output_sha256=output_sha256,
        plan_id="plan_1", plan_version=1, watch_listen_status=WATCH_LISTEN_HUMAN_REVIEW_REQUIRED,
        perceptual_review={}, client=fake_redis, store_export_fn=export_job.store_export,
    )
    pwl.apply_human_approval(
        user_id="user-1", project_id="proj-1", job_id="job-1",
        requesting=tsd.DeliveryOwnershipScope(user_id="user-1", project_id="proj-1", job_id="job-1"),
        approved=True, approver="qa@example.com",
        expected_render_identity=record.render_identity, expected_output_sha256=record.output_sha256,
        expected_plan_id=record.plan_id, expected_plan_version=record.plan_version,
        client=fake_redis,
    )

    result = export_job.resume_delivery_after_approval(
        user_id="user-1", project_id="proj-1", job_id="job-1", client=fake_redis, s3_client=wire_real_store_export,
    )
    assert result["delivery_status"] == rd.DELIVERY_STATUS_DELIVERY_READY
    assert result["watch_listen_status"] == WATCH_LISTEN_HUMAN_APPROVED
    assert result["download_url"] is not None
    assert any(k.startswith("cutsell/exports/") for k in wire_real_store_export.objects)

    reloaded = pwl.load_pending_review(user_id="user-1", project_id="proj-1", job_id="job-1", client=fake_redis)
    assert reloaded.resumed_delivery_at is not None


def test_resume_delivery_refuses_an_unapproved_pending_review(tmp_path, fake_s3, fake_redis):
    local_path = _rendered_file(tmp_path)
    _persist_record(fake_redis, fake_s3, local_path=local_path)  # never approved
    with pytest.raises(pwl.PendingReviewError, match="pending_review_not_approved"):
        export_job.resume_delivery_after_approval(
            user_id="user-1", project_id="proj-1", job_id="job-1", client=fake_redis, s3_client=fake_s3,
        )
