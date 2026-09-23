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


def test_full_walkthrough_export_to_finished_via_authenticated_approval(tmp_path, wire_real_store_export, fake_redis, monkeypatch):
    """D-288.2 blocker 3's own required test, end to end: exportación ->
    archivo pendiente conservado tras cleanup -> acceso autenticado ->
    aprobación -> reanudación -> versión registrada y proyecto terminado."""
    fake_job = SimpleNamespace(id="job-walk", started_at=1_700_000_000.0, meta={}, save_meta=lambda: None)
    rq_module = ModuleType("rq")
    rq_module.get_current_job = lambda: fake_job
    monkeypatch.setitem(sys.modules, "rq", rq_module)

    monkeypatch.setattr(export_job, "validate_product_source_uri", lambda uri, **kwargs: ("bucket", "key"))

    def fake_download(uri, destination):
        Path(destination).write_bytes(b"source-bytes")
        return destination
    monkeypatch.setattr(export_job, "download_source", fake_download)
    fake_plan = (RenderSegment(clip_id="clip-1", source_asset_id="src-1", source_path="/tmp/x.mp4", start=1.0, end=2.0),)
    monkeypatch.setattr(export_job, "build_render_plan", lambda draft, local_paths: fake_plan)

    rendered_bytes = b"walkthrough-rendered-mp4-bytes"

    def fake_render_with_qc(draft, plan, output, *, text_overlays=(), media_overlays=(), **kwargs):
        Path(output).write_bytes(rendered_bytes)
        return SimpleNamespace(status="PASS", output_path=output, plan_id="plan_walk", plan_version=1, semantic_hash="hash_walk", attempts=(), deliverable=True)
    monkeypatch.setattr(export_job, "render_with_post_render_qc", fake_render_with_qc)
    _stub_review(monkeypatch, WATCH_LISTEN_HUMAN_REVIEW_REQUIRED)
    monkeypatch.setattr(export_job, "safe_update_project", lambda **kwargs: {"state": kwargs.get("state")})
    monkeypatch.setattr(export_job, "_safe_notify", lambda **kwargs: {"status": "queued", "notification_id": "n1"})
    monkeypatch.setattr(pwl, "_redis_client", lambda client=None: fake_redis if client is None else client)

    draft_payload = {
        "schema_version": "cutsell.v1", "project_id": "project-walk", "strategy": "mixed",
        "selected": [{
            "clip_id": "clip-1", "source_asset_id": "src-1", "source_order": 0,
            "start": 1.0, "end": 2.0, "text": "hello", "caption_text": "hello",
            "semantic_role": "OTHER", "selected": True,
        }],
        "alternates": [], "discarded": [], "diagnostics": {}, "text_overlays": [],
    }

    # 1. Export -- lands as a pending review (HUMAN_REVIEW_REQUIRED).
    exported = export_job.run_export_job({
        "project_id": "project-walk", "user_id": "user-walk", "draft": draft_payload,
        "sources": [{"source_asset_id": "src-1", "original_name": "one.mov", "uri": "s3://bucket/cutsell/uploads/u/p/one.mov"}],
    })
    assert exported["state"] == "pending_review"

    # 2. The file survives the job's own TemporaryDirectory cleanup --
    # already closed by the time run_export_job returned.
    render_identity = exported["pending_review_render_identity"]
    output_sha256 = exported["pending_review_output_sha256"]
    plan_id = exported["pending_review_plan_id"]
    plan_version = exported["pending_review_plan_version"]

    # 3. Acceso autenticado -- the real owner can query it privately.
    owner = tsd.DeliveryOwnershipScope(user_id="user-walk", project_id="project-walk", job_id="job-walk")
    queried = pwl.get_pending_review_for_authenticated_caller(
        user_id="user-walk", project_id="project-walk", job_id="job-walk", requesting=owner, client=fake_redis,
    )
    assert queried is not None
    assert queried.render_identity == render_identity

    # Identidad ausente/ajena: rejected both ways.
    with pytest.raises(pwl.PendingReviewError, match="requesting_identity_required_no_none_bypass"):
        pwl.get_pending_review_for_authenticated_caller(
            user_id="user-walk", project_id="project-walk", job_id="job-walk", requesting=None, client=fake_redis,
        )
    stranger = tsd.DeliveryOwnershipScope(user_id="someone-else", project_id="project-walk", job_id="job-walk")
    with pytest.raises(PermissionError):
        pwl.get_pending_review_for_authenticated_caller(
            user_id="user-walk", project_id="project-walk", job_id="job-walk", requesting=stranger, client=fake_redis,
        )

    # A first REJECT, to prove rejection doesn't wreck the flow (and, per
    # blocker 1, never leaves a resumable HUMAN_APPROVED state behind).
    pwl.apply_human_approval(
        user_id="user-walk", project_id="project-walk", job_id="job-walk", requesting=owner,
        approved=False, approver="user-walk",
        expected_render_identity=render_identity, expected_output_sha256=output_sha256,
        expected_plan_id=plan_id, expected_plan_version=plan_version, client=fake_redis,
    )
    with pytest.raises(pwl.PendingReviewError, match="pending_review_not_approved"):
        export_job.resume_delivery_after_approval(
            user_id="user-walk", project_id="project-walk", job_id="job-walk", requesting=owner,
            client=fake_redis, s3_client=wire_real_store_export,
        )

    # 4. Aprobación -- the real decision.
    approved = pwl.apply_human_approval(
        user_id="user-walk", project_id="project-walk", job_id="job-walk", requesting=owner,
        approved=True, approver="user-walk",
        expected_render_identity=render_identity, expected_output_sha256=output_sha256,
        expected_plan_id=plan_id, expected_plan_version=plan_version, client=fake_redis,
    )
    assert approved.watch_listen_status == WATCH_LISTEN_HUMAN_APPROVED

    version_calls = []
    monkeypatch.setattr(export_job, "add_render_version", lambda **kwargs: version_calls.append(kwargs) or {"render_version_id": "rv_walk", "created_at": "t", "size_bytes": kwargs.get("size_bytes")})
    project_states = []
    monkeypatch.setattr(export_job, "safe_update_project", lambda **kwargs: project_states.append(kwargs.get("state")) or {"state": kwargs.get("state")})
    notifications = []
    monkeypatch.setattr(export_job, "_safe_notify", lambda **kwargs: notifications.append(kwargs) or {"status": "queued", "notification_id": "n1"})

    # 5. Reanudación -- delivery actually completes.
    resumed = export_job.resume_delivery_after_approval(
        user_id="user-walk", project_id="project-walk", job_id="job-walk", requesting=owner,
        client=fake_redis, s3_client=wire_real_store_export,
    )

    # 6. Versión registrada y proyecto terminado.
    assert resumed["state"] == "finished"
    assert resumed["delivery_status"] == rd.DELIVERY_STATUS_DELIVERY_READY
    assert len(version_calls) == 1
    assert project_states == ["finished"]
    assert len(notifications) == 1 and notifications[0]["kind"] == "render_finished"
    assert any(k.startswith("cutsell/exports/") for k in wire_real_store_export.objects)


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

    _requesting = tsd.DeliveryOwnershipScope(user_id="user-1", project_id="proj-1", job_id="job-1")
    result = export_job.resume_delivery_after_approval(
        user_id="user-1", project_id="proj-1", job_id="job-1", requesting=_requesting,
        client=fake_redis, s3_client=wire_real_store_export,
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
    _requesting = tsd.DeliveryOwnershipScope(user_id="user-1", project_id="proj-1", job_id="job-1")
    with pytest.raises(pwl.PendingReviewError, match="pending_review_not_approved"):
        export_job.resume_delivery_after_approval(
            user_id="user-1", project_id="proj-1", job_id="job-1", requesting=_requesting,
            client=fake_redis, s3_client=fake_s3,
        )


# =============================================================================
# D-288.2 blocker 1: the revocation bug -- approve -> reject -> resume MUST
# NOT deliver anything.
# =============================================================================

def test_approve_then_reject_then_resume_delivers_nothing(tmp_path, wire_real_store_export, fake_redis):
    """The EXACT required test: aprobar -> rechazar -> reanudar NO entrega
    ningún archivo. Before the fix, `apply_human_approval(approved=False)`
    left `watch_listen_status` stuck at HUMAN_APPROVED (only `approval_
    status` moved to REJECTED), and `resume_delivery_after_approval` only
    checked that one field -- so a REVOKED approval still resumed
    delivery. Both are fixed: rejection reverts `watch_listen_status` to
    the record's own immutable `automated_watch_listen_status`, and
    resume now requires BOTH fields to agree."""
    local_path = _rendered_file(tmp_path, content=b"revocation-test-bytes")
    output_sha256 = rd.compute_output_sha256(local_path)
    record = pwl.persist_pending_review(
        local_path=local_path, user_id="user-1", project_id="proj-1", job_id="job-1",
        render_identity="render_" + "c" * 24, output_sha256=output_sha256,
        plan_id="plan_1", plan_version=1, watch_listen_status=WATCH_LISTEN_HUMAN_REVIEW_REQUIRED,
        perceptual_review={}, client=fake_redis, store_export_fn=export_job.store_export,
    )
    requesting = tsd.DeliveryOwnershipScope(user_id="user-1", project_id="proj-1", job_id="job-1")

    # 1. Approve.
    approved_record = pwl.apply_human_approval(
        user_id="user-1", project_id="proj-1", job_id="job-1", requesting=requesting,
        approved=True, approver="user-1",
        expected_render_identity=record.render_identity, expected_output_sha256=record.output_sha256,
        expected_plan_id=record.plan_id, expected_plan_version=record.plan_version,
        client=fake_redis,
    )
    assert approved_record.watch_listen_status == WATCH_LISTEN_HUMAN_APPROVED

    # 2. Reject (revoke) the SAME record.
    rejected_record = pwl.apply_human_approval(
        user_id="user-1", project_id="proj-1", job_id="job-1", requesting=requesting,
        approved=False, approver="user-1",
        expected_render_identity=record.render_identity, expected_output_sha256=record.output_sha256,
        expected_plan_id=record.plan_id, expected_plan_version=record.plan_version,
        client=fake_redis,
    )
    assert rejected_record.watch_listen_status == WATCH_LISTEN_HUMAN_REVIEW_REQUIRED  # reverted, never stuck at APPROVED
    assert rejected_record.approval_status == "REJECTED"

    reloaded = pwl.load_pending_review(user_id="user-1", project_id="proj-1", job_id="job-1", client=fake_redis)
    assert reloaded.watch_listen_status != WATCH_LISTEN_HUMAN_APPROVED

    # 3. Resume MUST refuse -- no file delivered.
    with pytest.raises(pwl.PendingReviewError, match="pending_review_not_approved"):
        export_job.resume_delivery_after_approval(
            user_id="user-1", project_id="proj-1", job_id="job-1", requesting=requesting,
            client=fake_redis, s3_client=wire_real_store_export,
        )
    assert not any(k.startswith("cutsell/exports/") for k in wire_real_store_export.objects)


def test_reapproving_after_a_revocation_works_and_then_resume_succeeds(tmp_path, wire_real_store_export, fake_redis):
    """A revoked (REJECTED) record can still be legitimately re-approved
    later -- rejection is not a permanent block (that is what BLOCKED
    alone means); a genuine change-of-mind must be possible."""
    local_path = _rendered_file(tmp_path, content=b"reapproval-bytes")
    output_sha256 = rd.compute_output_sha256(local_path)
    record = pwl.persist_pending_review(
        local_path=local_path, user_id="user-1", project_id="proj-1", job_id="job-1",
        render_identity="render_" + "d" * 24, output_sha256=output_sha256,
        plan_id="plan_1", plan_version=1, watch_listen_status=WATCH_LISTEN_HUMAN_REVIEW_REQUIRED,
        perceptual_review={}, client=fake_redis, store_export_fn=export_job.store_export,
    )
    requesting = tsd.DeliveryOwnershipScope(user_id="user-1", project_id="proj-1", job_id="job-1")
    approval_kwargs = dict(
        user_id="user-1", project_id="proj-1", job_id="job-1", requesting=requesting,
        expected_render_identity=record.render_identity, expected_output_sha256=record.output_sha256,
        expected_plan_id=record.plan_id, expected_plan_version=record.plan_version, client=fake_redis,
    )
    pwl.apply_human_approval(approved=True, approver="user-1", **approval_kwargs)
    pwl.apply_human_approval(approved=False, approver="user-1", **approval_kwargs)
    reapproved = pwl.apply_human_approval(approved=True, approver="user-1", **approval_kwargs)
    assert reapproved.watch_listen_status == WATCH_LISTEN_HUMAN_APPROVED

    result = export_job.resume_delivery_after_approval(
        user_id="user-1", project_id="proj-1", job_id="job-1", requesting=requesting,
        client=fake_redis, s3_client=wire_real_store_export,
    )
    assert result["delivery_status"] == rd.DELIVERY_STATUS_DELIVERY_READY


# =============================================================================
# D-288.2 blocker 2: real authenticated entry -- requesting=None rejected
# outright, never derived from unauthenticated IDs.
# =============================================================================

def test_apply_human_approval_rejects_requesting_none(tmp_path, fake_s3, fake_redis):
    local_path = _rendered_file(tmp_path)
    record = _persist_record(fake_redis, fake_s3, local_path=local_path)
    with pytest.raises(pwl.PendingReviewError, match="requesting_identity_required_no_none_bypass"):
        pwl.apply_human_approval(
            user_id="user-1", project_id="proj-1", job_id="job-1", requesting=None,
            approved=True, approver="user-1",
            expected_render_identity=record.render_identity, expected_output_sha256=record.output_sha256,
            expected_plan_id=record.plan_id, expected_plan_version=record.plan_version,
            client=fake_redis,
        )


def test_resume_delivery_rejects_requesting_none(tmp_path, fake_redis):
    with pytest.raises(pwl.PendingReviewError, match="requesting_identity_required_no_none_bypass"):
        export_job.resume_delivery_after_approval(
            user_id="user-1", project_id="proj-1", job_id="job-1", requesting=None, client=fake_redis,
        )


def test_get_pending_review_rejects_requesting_none(tmp_path, fake_s3, fake_redis):
    local_path = _rendered_file(tmp_path)
    _persist_record(fake_redis, fake_s3, local_path=local_path)
    with pytest.raises(pwl.PendingReviewError, match="requesting_identity_required_no_none_bypass"):
        pwl.get_pending_review_for_authenticated_caller(
            user_id="user-1", project_id="proj-1", job_id="job-1", requesting=None, client=fake_redis,
        )


def test_get_pending_review_for_authenticated_caller_succeeds_for_the_real_owner(tmp_path, fake_s3, fake_redis):
    local_path = _rendered_file(tmp_path)
    record = _persist_record(fake_redis, fake_s3, local_path=local_path)
    requesting = tsd.DeliveryOwnershipScope(user_id="user-1", project_id="proj-1", job_id="job-1")
    loaded = pwl.get_pending_review_for_authenticated_caller(
        user_id="user-1", project_id="proj-1", job_id="job-1", requesting=requesting, client=fake_redis,
    )
    assert loaded.record_id == record.record_id


def test_get_pending_review_denies_a_mismatched_authenticated_caller(tmp_path, fake_s3, fake_redis):
    local_path = _rendered_file(tmp_path)
    _persist_record(fake_redis, fake_s3, local_path=local_path)
    attacker = tsd.DeliveryOwnershipScope(user_id="attacker", project_id="proj-1", job_id="job-1")
    with pytest.raises(PermissionError):
        pwl.get_pending_review_for_authenticated_caller(
            user_id="user-1", project_id="proj-1", job_id="job-1", requesting=attacker, client=fake_redis,
        )


# =============================================================================
# D-288.2 blocker 3: full finalization reuse + idempotency
# =============================================================================

def test_resumed_delivery_registers_render_version_and_finishes_project(tmp_path, wire_real_store_export, fake_redis, monkeypatch):
    local_path = _rendered_file(tmp_path, content=b"finalization-bytes")
    output_sha256 = rd.compute_output_sha256(local_path)
    record = pwl.persist_pending_review(
        local_path=local_path, user_id="user-1", project_id="proj-1", job_id="job-1",
        render_identity="render_" + "e" * 24, output_sha256=output_sha256,
        plan_id="plan_1", plan_version=1, watch_listen_status=WATCH_LISTEN_HUMAN_REVIEW_REQUIRED,
        perceptual_review={}, client=fake_redis, store_export_fn=export_job.store_export,
        selected_count=3, text_overlay_count=1, media_overlay_count=0,
    )
    requesting = tsd.DeliveryOwnershipScope(user_id="user-1", project_id="proj-1", job_id="job-1")
    pwl.apply_human_approval(
        user_id="user-1", project_id="proj-1", job_id="job-1", requesting=requesting,
        approved=True, approver="user-1",
        expected_render_identity=record.render_identity, expected_output_sha256=record.output_sha256,
        expected_plan_id=record.plan_id, expected_plan_version=record.plan_version, client=fake_redis,
    )

    version_calls = []
    real_add_render_version = export_job.add_render_version
    def _tracking_add_render_version(**kwargs):
        version_calls.append(kwargs)
        return {"render_version_id": "rv_test", "created_at": "2026-01-01T00:00:00Z", "size_bytes": kwargs.get("size_bytes")}
    monkeypatch.setattr(export_job, "add_render_version", _tracking_add_render_version)

    project_states = []
    monkeypatch.setattr(export_job, "safe_update_project", lambda **kwargs: project_states.append(kwargs.get("state")) or {"state": kwargs.get("state")})

    notifications = []
    monkeypatch.setattr(export_job, "_safe_notify", lambda **kwargs: notifications.append(kwargs) or {"status": "queued", "notification_id": "n1"})

    result = export_job.resume_delivery_after_approval(
        user_id="user-1", project_id="proj-1", job_id="job-1", requesting=requesting,
        client=fake_redis, s3_client=wire_real_store_export,
    )

    assert result["state"] == "finished"
    assert result["render_version_status"] == "saved"
    assert len(version_calls) == 1
    assert version_calls[0]["selected_count"] == 3
    assert version_calls[0]["text_overlay_count"] == 1
    assert project_states == ["finished"]
    assert len(notifications) == 1
    assert notifications[0]["kind"] == "render_finished"


def test_resume_is_idempotent_no_duplicate_version_or_notification(tmp_path, wire_real_store_export, fake_redis, monkeypatch):
    local_path = _rendered_file(tmp_path, content=b"idempotency-bytes")
    output_sha256 = rd.compute_output_sha256(local_path)
    record = pwl.persist_pending_review(
        local_path=local_path, user_id="user-1", project_id="proj-1", job_id="job-1",
        render_identity="render_" + "f" * 24, output_sha256=output_sha256,
        plan_id="plan_1", plan_version=1, watch_listen_status=WATCH_LISTEN_HUMAN_REVIEW_REQUIRED,
        perceptual_review={}, client=fake_redis, store_export_fn=export_job.store_export,
    )
    requesting = tsd.DeliveryOwnershipScope(user_id="user-1", project_id="proj-1", job_id="job-1")
    pwl.apply_human_approval(
        user_id="user-1", project_id="proj-1", job_id="job-1", requesting=requesting,
        approved=True, approver="user-1",
        expected_render_identity=record.render_identity, expected_output_sha256=record.output_sha256,
        expected_plan_id=record.plan_id, expected_plan_version=record.plan_version, client=fake_redis,
    )

    version_calls = []
    monkeypatch.setattr(export_job, "add_render_version", lambda **kwargs: version_calls.append(kwargs) or {"render_version_id": "rv_test", "created_at": "t", "size_bytes": 1})
    notifications = []
    monkeypatch.setattr(export_job, "_safe_notify", lambda **kwargs: notifications.append(kwargs) or {"status": "queued", "notification_id": "n1"})

    first = export_job.resume_delivery_after_approval(
        user_id="user-1", project_id="proj-1", job_id="job-1", requesting=requesting,
        client=fake_redis, s3_client=wire_real_store_export,
    )
    second = export_job.resume_delivery_after_approval(
        user_id="user-1", project_id="proj-1", job_id="job-1", requesting=requesting,
        client=fake_redis, s3_client=wire_real_store_export,
    )
    assert first == second
    assert len(version_calls) == 1  # never called twice
    assert len(notifications) == 1  # never called twice


def test_resume_refuses_a_modified_pending_artifact(tmp_path, wire_real_store_export, fake_redis):
    """"Archivo modificado": if the bytes at the private pending location
    ever differ from what was approved (tamper, corruption, or a bug
    elsewhere), resume must refuse rather than deliver something a human
    never actually reviewed."""
    local_path = _rendered_file(tmp_path, content=b"original-bytes")
    output_sha256 = rd.compute_output_sha256(local_path)
    record = pwl.persist_pending_review(
        local_path=local_path, user_id="user-1", project_id="proj-1", job_id="job-1",
        render_identity="render_" + "1" * 24, output_sha256=output_sha256,
        plan_id="plan_1", plan_version=1, watch_listen_status=WATCH_LISTEN_HUMAN_REVIEW_REQUIRED,
        perceptual_review={}, client=fake_redis, store_export_fn=export_job.store_export,
    )
    requesting = tsd.DeliveryOwnershipScope(user_id="user-1", project_id="proj-1", job_id="job-1")
    pwl.apply_human_approval(
        user_id="user-1", project_id="proj-1", job_id="job-1", requesting=requesting,
        approved=True, approver="user-1",
        expected_render_identity=record.render_identity, expected_output_sha256=record.output_sha256,
        expected_plan_id=record.plan_id, expected_plan_version=record.plan_version, client=fake_redis,
    )

    # Tamper with the stored object AFTER approval.
    key = record.pending_s3_uri.split("test-bucket/", 1)[1]
    wire_real_store_export.objects[key]["body"] = b"tampered-bytes"
    wire_real_store_export.objects[key]["size"] = len(b"tampered-bytes")

    with pytest.raises(pwl.PendingReviewError, match="resumed_artifact_hash_mismatch"):
        export_job.resume_delivery_after_approval(
            user_id="user-1", project_id="proj-1", job_id="job-1", requesting=requesting,
            client=fake_redis, s3_client=wire_real_store_export,
        )
    assert not any(k.startswith("cutsell/exports/") for k in wire_real_store_export.objects)
