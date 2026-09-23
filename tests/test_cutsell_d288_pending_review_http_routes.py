"""D-288.2 (finding 2, blocker 2) -- the REAL, authenticated HTTP handlers
for pending-review query/approval/resumption.

Proves, through real HTTP requests (`TestClient` + the real `AuthScope
Middleware`, never a bare function call): no authenticated session -> 401
on all three endpoints; a mismatched authenticated user -> 403/404, never
another user's record; the approver identity is ALWAYS the authenticated
caller (the request body has no field that could supply one); a full
approve -> resume walkthrough succeeds over real HTTP. Same `_secure_app`
pattern this repo's own `test_cutsell_clean_worker_auth.py` already
establishes (`CUTSELL_AUTH_REQUIRED=1` + a faked `resolve_session`) rather
than a new one. No Video00 fact/id anywhere below.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import cutsell_app.auth_middleware as middleware
from cutsell_app.auth_middleware import AuthScopeMiddleware
from cutsell_app.pending_review_routes import router as pending_review_router
from cutsell_worker import export_job
from cutsell_worker import pending_watch_listen_review as pwl
from cutsell_worker import render_delivery as rd
from cutsell_worker import tenant_safe_delivery as tsd
from cutsell_worker.perceptual_watch_listen import WATCH_LISTEN_HUMAN_REVIEW_REQUIRED
from tests.fake_atomic_redis import FakeAtomicRedis


class FakeS3Client:
    def __init__(self):
        self.objects: dict[str, dict] = {}

    def upload_file(self, filename, bucket, key, **_kwargs):
        with open(filename, "rb") as handle:
            data = handle.read()
        self.objects[key] = {"body": data, "size": len(data)}

    def download_file(self, bucket, key, destination):
        Path(destination).write_bytes(self.objects[key]["body"])

    def head_object(self, Bucket, Key):  # noqa: N803
        obj = self.objects.get(Key)
        if obj is None:
            from botocore.exceptions import ClientError
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        return {"ContentLength": obj["size"], "Metadata": {}}

    def generate_presigned_url(self, *_args, **_kwargs):
        return "https://x.invalid/presigned"


FakeRedis = FakeAtomicRedis  # D-288.4: shared CAS + append-if-absent emulation


@pytest.fixture
def fake_s3():
    return FakeS3Client()


@pytest.fixture
def fake_redis():
    return FakeRedis()


@pytest.fixture
def client(monkeypatch, fake_redis, fake_s3):
    """The real `pending_review_routes.router` behind the real
    `AuthScopeMiddleware`, with `CUTSELL_AUTH_REQUIRED=1` (forcing real
    enforcement regardless of this test session's own global disable
    flag) and a faked `resolve_session` standing in for a real bearer-
    token verification service -- same pattern as `test_cutsell_clean_
    worker_auth.py`'s own `_secure_app`."""
    monkeypatch.setenv("CUTSELL_AUTH_REQUIRED", "1")
    monkeypatch.setattr(
        middleware, "resolve_session",
        lambda token: {"user_id": "user-http"} if token == "good-token" else (_ for _ in ()).throw(PermissionError("invalid token")),
    )
    monkeypatch.setattr(middleware, "fetch_job_snapshot", lambda job_id, **kwargs: object())
    monkeypatch.setattr(pwl, "_redis_client", lambda c=None: fake_redis if c is None else c)
    monkeypatch.setattr(export_job, "store_export", lambda *a, **k: _real_store_export(fake_s3, *a, **k))
    # `resume_delivery_after_approval`'s own real-boto3 default (no
    # `s3_client` kwarg exposed at the route layer, matching production --
    # a route handler never takes a test-only fake as a parameter) is
    # redirected here so the route-level walkthrough test can exercise it
    # for real without a genuine AWS credential/network call.
    import boto3 as _boto3
    monkeypatch.setattr(_boto3, "client", lambda service, **kwargs: fake_s3 if service == "s3" else None)

    app = FastAPI()
    app.add_middleware(AuthScopeMiddleware)
    app.include_router(pending_review_router)
    return TestClient(app)


def _real_store_export(fake_s3, path, *, project_id, user_id, object_key=None, object_metadata=None, **_kwargs):
    from cutsell_worker.exports import store_export
    return store_export(path, project_id=project_id, user_id=user_id, object_key=object_key, object_metadata=object_metadata, client=fake_s3)


@pytest.fixture(autouse=True)
def _wire_s3_config(monkeypatch, fake_s3):
    from cutsell_worker import exports
    monkeypatch.setattr(exports, "load_runtime_config", lambda: SimpleNamespace(s3_bucket="test-bucket", aws_region="us-east-1"))


def _persist(fake_redis, fake_s3, tmp_path, *, user_id="user-http", project_id="proj-http", job_id="job-http", content=b"http-route-bytes"):
    path = tmp_path / "rendered.mp4"
    path.write_bytes(content)
    output_sha256 = rd.compute_output_sha256(str(path))
    return pwl.persist_pending_review(
        local_path=str(path), user_id=user_id, project_id=project_id, job_id=job_id,
        render_identity="render_" + "a" * 24, output_sha256=output_sha256,
        plan_id="plan_http", plan_version=1, watch_listen_status=WATCH_LISTEN_HUMAN_REVIEW_REQUIRED,
        perceptual_review={"status": "UNCERTAIN"}, client=fake_redis,
        store_export_fn=lambda *a, **k: _real_store_export(fake_s3, *a, **k),
    )


def _decision_body(record, *, approved: bool):
    return {
        "approved": approved,
        "expected_render_identity": record.render_identity,
        "expected_output_sha256": record.output_sha256,
        "expected_plan_id": record.plan_id,
        "expected_plan_version": record.plan_version,
    }


# =============================================================================
# No authenticated session -> 401 on every endpoint
# =============================================================================

def test_get_pending_review_requires_auth(client):
    resp = client.get("/v1/projects/proj-http/jobs/job-http/pending-review")
    assert resp.status_code == 401


def test_submit_decision_requires_auth(client):
    resp = client.post(
        "/v1/projects/proj-http/jobs/job-http/pending-review/decision",
        json={"approved": True, "expected_render_identity": "x", "expected_output_sha256": "y", "expected_plan_id": "z", "expected_plan_version": 1},
    )
    assert resp.status_code == 401


def test_resume_delivery_requires_auth(client):
    resp = client.post("/v1/projects/proj-http/jobs/job-http/pending-review/resume-delivery")
    assert resp.status_code == 401


# =============================================================================
# A mismatched authenticated user never sees or acts on another user's record
# =============================================================================

def test_get_pending_review_denies_a_different_authenticated_user(client, fake_redis, fake_s3, tmp_path, monkeypatch):
    _persist(fake_redis, fake_s3, tmp_path, user_id="the-real-owner")
    monkeypatch.setattr(middleware, "resolve_session", lambda token: {"user_id": "someone-else"} if token == "good-token" else (_ for _ in ()).throw(PermissionError("bad")))
    resp = client.get(
        "/v1/projects/proj-http/jobs/job-http/pending-review",
        headers={"Authorization": "Bearer good-token"},
    )
    # The lookup key is scoped by the AUTHENTICATED user_id ("someone-
    # else"), which has no such record -- 404, never another user's data.
    assert resp.status_code == 404


# =============================================================================
# The authenticated owner CAN query, decide, and resume -- and the approver
# is always the authenticated identity, never body-controllable
# =============================================================================

def test_full_http_walkthrough_query_approve_resume(client, fake_redis, fake_s3, tmp_path):
    record = _persist(fake_redis, fake_s3, tmp_path)
    headers = {"Authorization": "Bearer good-token"}

    got = client.get("/v1/projects/proj-http/jobs/job-http/pending-review", headers=headers)
    assert got.status_code == 200
    assert got.json()["render_identity"] == record.render_identity

    decided = client.post(
        "/v1/projects/proj-http/jobs/job-http/pending-review/decision",
        json=_decision_body(record, approved=True), headers=headers,
    )
    assert decided.status_code == 200
    body = decided.json()
    assert body["watch_listen_status"] == "HUMAN_APPROVED"
    # The approver is the AUTHENTICATED user -- never accepted from the
    # request body (the schema has no such field at all).
    assert body["approver"] == "user-http"

    resumed = client.post("/v1/projects/proj-http/jobs/job-http/pending-review/resume-delivery", headers=headers)
    assert resumed.status_code == 200
    assert resumed.json()["delivery_status"] == rd.DELIVERY_STATUS_DELIVERY_READY


def test_approver_field_in_request_body_is_ignored_not_a_real_field(client, fake_redis, fake_s3, tmp_path):
    """Even if a client tries to smuggle an `approver` field into the
    decision body, the Pydantic request model has no such field -- it is
    silently dropped by FastAPI's own schema validation, and the recorded
    approver is still the real authenticated identity."""
    record = _persist(fake_redis, fake_s3, tmp_path)
    headers = {"Authorization": "Bearer good-token"}
    body = _decision_body(record, approved=True)
    body["approver"] = "attacker@example.com"

    resp = client.post(
        "/v1/projects/proj-http/jobs/job-http/pending-review/decision",
        json=body, headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["approver"] == "user-http"  # never "attacker@example.com"


def test_user_id_smuggled_into_the_body_is_rejected_by_the_auth_layer_itself(client, fake_redis, fake_s3, tmp_path):
    """A `user_id` field in the body is a DIFFERENT (and even earlier)
    layer of the same protection: `AuthScopeMiddleware` itself cross-
    checks any body `user_id` against the authenticated session and
    rejects a mismatch BEFORE this route (or any route) ever runs --
    proof "nunca de IDs... como sustituto" holds at more than one layer."""
    record = _persist(fake_redis, fake_s3, tmp_path)
    headers = {"Authorization": "Bearer good-token"}
    body = _decision_body(record, approved=True)
    body["user_id"] = "attacker"

    resp = client.post(
        "/v1/projects/proj-http/jobs/job-http/pending-review/decision",
        json=body, headers=headers,
    )
    assert resp.status_code == 403


def test_reject_then_resume_over_http_delivers_nothing(client, fake_redis, fake_s3, tmp_path):
    record = _persist(fake_redis, fake_s3, tmp_path)
    headers = {"Authorization": "Bearer good-token"}
    client.post("/v1/projects/proj-http/jobs/job-http/pending-review/decision", json=_decision_body(record, approved=True), headers=headers)
    client.post("/v1/projects/proj-http/jobs/job-http/pending-review/decision", json=_decision_body(record, approved=False), headers=headers)

    resp = client.post("/v1/projects/proj-http/jobs/job-http/pending-review/resume-delivery", headers=headers)
    assert resp.status_code == 422
    assert not any(k.startswith("cutsell/exports/") for k in fake_s3.objects)


def test_stale_decision_body_is_rejected_over_http(client, fake_redis, fake_s3, tmp_path):
    _persist(fake_redis, fake_s3, tmp_path)
    headers = {"Authorization": "Bearer good-token"}
    stale_body = {
        "approved": True, "expected_render_identity": "render_DIFFERENT",
        "expected_output_sha256": "wrong-sha", "expected_plan_id": "wrong-plan", "expected_plan_version": 99,
    }
    resp = client.post("/v1/projects/proj-http/jobs/job-http/pending-review/decision", json=stale_body, headers=headers)
    assert resp.status_code == 422


def test_resume_without_a_prior_approval_over_http_is_rejected(client, fake_redis, fake_s3, tmp_path):
    _persist(fake_redis, fake_s3, tmp_path)
    headers = {"Authorization": "Bearer good-token"}
    resp = client.post("/v1/projects/proj-http/jobs/job-http/pending-review/resume-delivery", headers=headers)
    assert resp.status_code == 422


# =============================================================================
# D-288.3 (blocker 4): the media-preview endpoint, over real HTTP.
# =============================================================================

def test_media_preview_requires_auth(client):
    resp = client.get(
        "/v1/projects/proj-http/jobs/job-http/pending-review/media",
        params={"expected_render_identity": "x", "expected_output_sha256": "y"},
    )
    assert resp.status_code == 401


def test_owner_can_fetch_a_media_preview_url_over_http(client, fake_redis, fake_s3, tmp_path):
    record = _persist(fake_redis, fake_s3, tmp_path)
    headers = {"Authorization": "Bearer good-token"}
    resp = client.get(
        "/v1/projects/proj-http/jobs/job-http/pending-review/media",
        params={"expected_render_identity": record.render_identity, "expected_output_sha256": record.output_sha256},
        headers=headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["preview_url"]
    assert body["render_identity"] == record.render_identity
    assert "delivery_status" not in body


def test_a_different_authenticated_user_cannot_fetch_the_media_preview(client, fake_redis, fake_s3, tmp_path, monkeypatch):
    record = _persist(fake_redis, fake_s3, tmp_path, user_id="the-real-owner")
    monkeypatch.setattr(middleware, "resolve_session", lambda token: {"user_id": "someone-else"} if token == "good-token" else (_ for _ in ()).throw(PermissionError("bad")))
    headers = {"Authorization": "Bearer good-token"}
    resp = client.get(
        "/v1/projects/proj-http/jobs/job-http/pending-review/media",
        params={"expected_render_identity": record.render_identity, "expected_output_sha256": record.output_sha256},
        headers=headers,
    )
    # The lookup key is scoped by the AUTHENTICATED user_id -- no such
    # record for "someone-else", so this surfaces as the module's own
    # "no pending review found" (422), never another user's preview URL.
    assert resp.status_code == 422


def test_media_preview_bound_to_exact_artifact_refuses_a_stale_reference_over_http(client, fake_redis, fake_s3, tmp_path):
    _persist(fake_redis, fake_s3, tmp_path)
    headers = {"Authorization": "Bearer good-token"}
    resp = client.get(
        "/v1/projects/proj-http/jobs/job-http/pending-review/media",
        params={"expected_render_identity": "render_STALE", "expected_output_sha256": "sha-STALE"},
        headers=headers,
    )
    assert resp.status_code == 422
