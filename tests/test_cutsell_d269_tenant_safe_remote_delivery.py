"""D-269 -- TENANT-SAFE REMOTE DELIVERY FOUNDATION.

Post D-268 (export/storage delivery integration audit, Verdict A: local
delivery-record foundation functional, tenant/remote gaps identified: P0
auth-fails-open, P1 D-267-unwired, P1 no-remote-verification, P1 IDOR
invisible at handler seam, P1 stale-latest-job-pointer race, P2 raw-URI/
no-binding). This file proves, entirely offline:

- `cutsell_app.auth_middleware`'s new fail-closed `_auth_required()` default
  and its one explicit, bounded local-dev disable flag;
- `cutsell_app.main`'s handler-level defense-in-depth ownership check on
  `GET /v1/jobs/{job_id}` and `POST /v1/jobs/{job_id}/cancel`;
- `cutsell_worker.tenant_safe_delivery`'s full ownership/render-identity/
  remote-object binding, remote verification, presign authorization, and
  stale-job-pointer guard;
- `cutsell_worker.project_store.update_project`'s new (backward-compatible,
  currently-inert-for-every-existing-caller) stale-job guard wiring.

No real S3 mutation, no network call, no provider, no RAW anywhere in this
file -- `RemoteObjectMetadataFixture` stands in for a real `head_object`
response throughout.
"""
from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cutsell_worker import render_delivery as rd
from cutsell_worker import tenant_safe_delivery as tsd


class _FakePipeline:
    """D-269: the same minimal Redis-pipeline test double already
    established in `tests/test_cutsell_clean_worker_projects.py` -- no new
    dependency (`fakeredis` is not installed in this environment)."""

    def __init__(self, redis):
        self.redis = redis
        self.ops = []

    def set(self, key, value):
        self.ops.append(("set", key, value))
        return self

    def zadd(self, key, mapping):
        self.ops.append(("zadd", key, dict(mapping)))
        return self

    def execute(self):
        for op in self.ops:
            if op[0] == "set":
                self.redis.set(op[1], op[2])
            else:
                self.redis.zadd(op[1], op[2])
        self.ops = []
        return [True]


class _FakeRedis:
    def __init__(self):
        self.data = {}
        self.zsets = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value
        return True

    def zadd(self, key, mapping):
        bucket = self.zsets.setdefault(key, {})
        bucket.update(mapping)
        return len(mapping)

    def zrevrange(self, key, start, end):
        ordered = sorted(self.zsets.get(key, {}).items(), key=lambda item: item[1], reverse=True)
        stop = None if end < 0 else end + 1
        return [item[0].encode() for item in ordered[start:stop]]

    def pipeline(self):
        return _FakePipeline(self)


def _source_without_docstrings(path: str) -> str:
    """D-262/D-263/D-266/D-267's own established false-positive-proofing
    technique -- strip every module/function/class docstring before
    scanning for forbidden vocabulary, since this codebase's own
    scope-discipline prose legitimately names things it says it does NOT
    do."""
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


def _scope(user_id="user-1", project_id="proj-1", job_id="job-1") -> tsd.DeliveryOwnershipScope:
    return tsd.DeliveryOwnershipScope(user_id=user_id, project_id=project_id, job_id=job_id)


def _ready_delivery(tmp_path, render_identity="render_" + "a" * 24) -> rd.RenderDeliveryRecord:
    out = tmp_path / "out.mp4"
    if not out.exists():
        out.write_bytes(b"fake-mp4-bytes")
    return rd.build_render_delivery_record(
        render_identity=render_identity,
        render_execution_status=rd.RENDER_EXECUTION_STATUS_SUCCEEDED,
        final_path=str(out),
        technical_qc_status=rd.TECHNICAL_QC_STATUS_PASS,
        require_upload=True,
        upload_status=rd.UPLOAD_STATUS_SUCCEEDED,
    )


def _remote_for(ownership, delivery, *, bucket="cutsell-exports", key=None, upload_status=rd.UPLOAD_STATUS_SUCCEEDED):
    key = key or tsd.build_tenant_safe_export_key(ownership=ownership, render_identity=delivery.render_identity)
    return tsd.RemoteDeliveryObject(
        ownership=ownership, render_identity=delivery.render_identity,
        output_sha256=delivery.output_sha256, output_size_bytes=delivery.output_size_bytes,
        bucket=bucket, object_key=key, upload_status=upload_status,
    )


def _matching_remote_metadata(remote: tsd.RemoteDeliveryObject) -> tsd.RemoteObjectMetadataFixture:
    return tsd.RemoteObjectMetadataFixture(
        exists=True, key=remote.object_key, size_bytes=remote.output_size_bytes,
        metadata_render_identity=remote.render_identity, metadata_sha256=remote.output_sha256,
    )


# =============================================================================
# Stage 2 -- DeliveryOwnershipScope
# =============================================================================

def test_ownership_scope_requires_all_three_fields_nonempty():
    with pytest.raises(ValueError):
        tsd.DeliveryOwnershipScope(user_id="", project_id="p", job_id="j")
    with pytest.raises(ValueError):
        tsd.DeliveryOwnershipScope(user_id="u", project_id="", job_id="j")
    with pytest.raises(ValueError):
        tsd.DeliveryOwnershipScope(user_id="u", project_id="p", job_id="")


def test_ownership_scope_is_frozen_and_comparable_by_value():
    a = _scope()
    b = _scope()
    assert a == b
    assert a is not b
    with pytest.raises(Exception):
        a.user_id = "someone-else"  # type: ignore[misc]


def test_ownership_scope_has_no_tenant_or_org_field():
    """Stage 2: no invented tenant_id/organization_id -- this product has
    no tenant/org model today (D-268's own finding)."""
    fields = {f for f in _scope().__dataclass_fields__}
    assert "tenant_id" not in fields
    assert "organization_id" not in fields


# =============================================================================
# Stage 4/5/6/7/18/23 -- deterministic, tenant-safe export key
# =============================================================================

def test_export_key_deterministic_for_same_scope_and_identity():
    ownership = _scope()
    identity = "render_" + "b" * 24
    key_a = tsd.build_tenant_safe_export_key(ownership=ownership, render_identity=identity)
    key_b = tsd.build_tenant_safe_export_key(ownership=ownership, render_identity=identity)
    assert key_a == key_b


def test_export_key_isolates_different_users():
    identity = "render_" + "b" * 24
    key_user1 = tsd.build_tenant_safe_export_key(ownership=_scope(user_id="user-1"), render_identity=identity)
    key_user2 = tsd.build_tenant_safe_export_key(ownership=_scope(user_id="user-2"), render_identity=identity)
    assert key_user1 != key_user2


def test_export_key_isolates_different_projects_same_user():
    identity = "render_" + "b" * 24
    key_p1 = tsd.build_tenant_safe_export_key(ownership=_scope(project_id="proj-1"), render_identity=identity)
    key_p2 = tsd.build_tenant_safe_export_key(ownership=_scope(project_id="proj-2"), render_identity=identity)
    assert key_p1 != key_p2


def test_export_key_isolates_different_jobs_same_user_and_project():
    identity = "render_" + "b" * 24
    key_j1 = tsd.build_tenant_safe_export_key(ownership=_scope(job_id="job-1"), render_identity=identity)
    key_j2 = tsd.build_tenant_safe_export_key(ownership=_scope(job_id="job-2"), render_identity=identity)
    assert key_j1 != key_j2


def test_export_key_two_users_same_project_id_text_still_isolated():
    """Same literal project_id string across two different users must not
    collide -- the key incorporates the hashed USER id too."""
    identity = "render_" + "b" * 24
    key_a = tsd.build_tenant_safe_export_key(ownership=_scope(user_id="user-a", project_id="shared-name"), render_identity=identity)
    key_b = tsd.build_tenant_safe_export_key(ownership=_scope(user_id="user-b", project_id="shared-name"), render_identity=identity)
    assert key_a != key_b


def test_export_key_rejects_non_render_identity_value():
    with pytest.raises(ValueError):
        tsd.build_tenant_safe_export_key(ownership=_scope(), render_identity="not-a-render-identity")


def test_export_key_never_embeds_raw_malicious_filename():
    """Stage 4/18: a raw user-controlled filename (path traversal, control
    chars, etc.) is never a key input at all -- the key is built purely
    from server-side identity + render_identity, so this is a structural
    guarantee, not a sanitizer."""
    ownership = _scope()
    identity = "render_" + "c" * 24
    key = tsd.build_tenant_safe_export_key(ownership=ownership, render_identity=identity)
    for hostile in ("../../etc/passwd", "..", "\\", "\x00", "; rm -rf /"):
        assert hostile not in key


def test_export_key_unicode_user_and_project_values_hash_safely():
    ownership = tsd.DeliveryOwnershipScope(user_id="ユーザー-1", project_id="проект-1", job_id="job-1")
    identity = "render_" + "d" * 24
    key = tsd.build_tenant_safe_export_key(ownership=ownership, render_identity=identity)
    # every segment must be pure lowercase hex/underscore/render_ prefix
    for segment in key.removeprefix(tsd.DEFAULT_TENANT_SAFE_EXPORT_PREFIX).removesuffix(".mp4").split("/"):
        assert tsd._SAFE_KEY_SEGMENT_RE.match(segment) or segment.startswith("render_")


def test_export_key_rejects_unsafe_prefix():
    with pytest.raises(ValueError):
        tsd.build_tenant_safe_export_key(ownership=_scope(), render_identity="render_" + "e" * 24, prefix="../escape/")


def test_export_key_ends_with_mp4():
    key = tsd.build_tenant_safe_export_key(ownership=_scope(), render_identity="render_" + "f" * 24)
    assert key.endswith(".mp4")


def test_export_key_duplicate_delivery_idempotent():
    """Stage 23: re-running the SAME job against the SAME render plan
    produces the SAME key -- duplicate delivery overwrites, never
    accumulates."""
    ownership = _scope()
    identity = "render_" + "g" * 24
    key_first = tsd.build_tenant_safe_export_key(ownership=ownership, render_identity=identity)
    key_second = tsd.build_tenant_safe_export_key(ownership=ownership, render_identity=identity)
    assert key_first == key_second


# =============================================================================
# Stage 9/16/19 -- remote verification (no live S3 request)
# =============================================================================

def test_remote_verification_object_missing():
    ownership = _scope()
    delivery = _remote_expected(ownership)
    result = tsd.verify_remote_delivery(delivery, tsd.RemoteObjectMetadataFixture(exists=False))
    assert result.verification_status == tsd.VERIFICATION_STATUS_FAIL
    assert "remote_object_missing" in result.errors


def _remote_expected(ownership) -> tsd.RemoteDeliveryObject:
    identity = "render_" + "h" * 24
    key = tsd.build_tenant_safe_export_key(ownership=ownership, render_identity=identity)
    return tsd.RemoteDeliveryObject(
        ownership=ownership, render_identity=identity, output_sha256="a" * 64, output_size_bytes=1000,
        bucket="cutsell-exports", object_key=key,
    )


def test_remote_verification_key_mismatch():
    ownership = _scope()
    expected = _remote_expected(ownership)
    remote = tsd.RemoteObjectMetadataFixture(exists=True, key="wrong/key.mp4", size_bytes=1000)
    result = tsd.verify_remote_delivery(expected, remote)
    assert result.verification_status == tsd.VERIFICATION_STATUS_FAIL
    assert "remote_key_mismatch" in result.errors


def test_remote_verification_size_mismatch():
    ownership = _scope()
    expected = _remote_expected(ownership)
    remote = tsd.RemoteObjectMetadataFixture(exists=True, key=expected.object_key, size_bytes=999)
    result = tsd.verify_remote_delivery(expected, remote)
    assert result.verification_status == tsd.VERIFICATION_STATUS_FAIL
    assert "remote_size_mismatch" in result.errors


def test_remote_verification_render_identity_mismatch():
    ownership = _scope()
    expected = _remote_expected(ownership)
    remote = tsd.RemoteObjectMetadataFixture(
        exists=True, key=expected.object_key, size_bytes=1000, metadata_render_identity="render_" + "z" * 24,
    )
    result = tsd.verify_remote_delivery(expected, remote)
    assert result.verification_status == tsd.VERIFICATION_STATUS_FAIL
    assert "remote_render_identity_mismatch" in result.errors


def test_remote_verification_hash_unknown_when_absent():
    ownership = _scope()
    expected = _remote_expected(ownership)
    remote = tsd.RemoteObjectMetadataFixture(exists=True, key=expected.object_key, size_bytes=1000)
    result = tsd.verify_remote_delivery(expected, remote)
    assert result.remote_hash_status == tsd.REMOTE_HASH_STATUS_NOT_AVAILABLE
    # absence of hash metadata is NOT itself a failure (Stage 8: never fabricate a match, but also never punish absence)
    assert result.verification_status == tsd.VERIFICATION_STATUS_PASS


def test_remote_verification_hash_mismatch_fails():
    ownership = _scope()
    expected = _remote_expected(ownership)
    remote = tsd.RemoteObjectMetadataFixture(
        exists=True, key=expected.object_key, size_bytes=1000, metadata_sha256="b" * 64,
    )
    result = tsd.verify_remote_delivery(expected, remote)
    assert result.remote_hash_status == tsd.REMOTE_HASH_STATUS_MISMATCHED
    assert result.verification_status == tsd.VERIFICATION_STATUS_FAIL


def test_remote_verification_hash_matched_passes():
    ownership = _scope()
    expected = _remote_expected(ownership)
    remote = tsd.RemoteObjectMetadataFixture(
        exists=True, key=expected.object_key, size_bytes=1000, metadata_sha256=expected.output_sha256,
    )
    result = tsd.verify_remote_delivery(expected, remote)
    assert result.remote_hash_status == tsd.REMOTE_HASH_STATUS_MATCHED
    assert result.verification_status == tsd.VERIFICATION_STATUS_PASS


@pytest.mark.parametrize("etag", ["\"9bb58f26192e4ba00f01e2e7b136bbd8\"", "\"abc-3\"", None, ""])
def test_etag_never_treated_as_sha256(etag):
    assert tsd.is_etag_valid_sha256_proxy(etag) is False


# =============================================================================
# Stage 10/12 -- evaluate_tenant_safe_delivery: the fail-closed bridge
# =============================================================================

def test_tenant_bridge_valid_full_chain_reaches_delivery_ready(tmp_path):
    ownership = _scope()
    delivery = _ready_delivery(tmp_path)
    remote = _remote_for(ownership, delivery)
    verification = tsd.verify_remote_delivery(
        tsd.RemoteDeliveryObject(
            ownership=ownership, render_identity=remote.render_identity, output_sha256=remote.output_sha256,
            output_size_bytes=remote.output_size_bytes, bucket=remote.bucket, object_key=remote.object_key,
        ),
        _matching_remote_metadata(remote),
    )
    record = tsd.evaluate_tenant_safe_delivery(
        ownership=ownership, expected_render_identity=delivery.render_identity, delivery=delivery,
        remote=remote, remote_verification=verification, require_remote=True,
    )
    assert record.delivery_status == rd.DELIVERY_STATUS_DELIVERY_READY
    assert record.ready_for_delivery is True


def test_tenant_bridge_render_identity_mismatch_blocks(tmp_path):
    ownership = _scope()
    delivery = _ready_delivery(tmp_path)
    record = tsd.evaluate_tenant_safe_delivery(
        ownership=ownership, expected_render_identity="render_" + "z" * 24, delivery=delivery,
    )
    assert record.delivery_status == tsd.TENANT_DELIVERY_STATUS_WRONG_OBJECT
    assert record.ready_for_delivery is False


def test_tenant_bridge_wrong_project_blocks(tmp_path):
    ownership = _scope(project_id="proj-1")
    other_ownership = _scope(project_id="proj-2")
    delivery = _ready_delivery(tmp_path)
    remote = _remote_for(other_ownership, delivery)
    record = tsd.evaluate_tenant_safe_delivery(
        ownership=ownership, expected_render_identity=delivery.render_identity, delivery=delivery,
        remote=remote, require_remote=True,
    )
    assert record.delivery_status == tsd.TENANT_DELIVERY_STATUS_OWNERSHIP_MISMATCH
    assert record.ready_for_delivery is False


def test_tenant_bridge_wrong_job_blocks(tmp_path):
    ownership = _scope(job_id="job-1")
    other_ownership = _scope(job_id="job-2")
    delivery = _ready_delivery(tmp_path)
    remote = _remote_for(other_ownership, delivery)
    record = tsd.evaluate_tenant_safe_delivery(
        ownership=ownership, expected_render_identity=delivery.render_identity, delivery=delivery,
        remote=remote, require_remote=True,
    )
    assert record.delivery_status == tsd.TENANT_DELIVERY_STATUS_OWNERSHIP_MISMATCH
    assert record.ready_for_delivery is False


def test_tenant_bridge_wrong_sha_on_remote_fixture_blocks(tmp_path):
    ownership = _scope()
    delivery = _ready_delivery(tmp_path)
    remote = _remote_for(ownership, delivery)
    bad_metadata = tsd.RemoteObjectMetadataFixture(
        exists=True, key=remote.object_key, size_bytes=remote.output_size_bytes,
        metadata_render_identity=remote.render_identity, metadata_sha256="f" * 64,
    )
    verification = tsd.verify_remote_delivery(
        tsd.RemoteDeliveryObject(
            ownership=ownership, render_identity=remote.render_identity, output_sha256=remote.output_sha256,
            output_size_bytes=remote.output_size_bytes, bucket=remote.bucket, object_key=remote.object_key,
        ),
        bad_metadata,
    )
    record = tsd.evaluate_tenant_safe_delivery(
        ownership=ownership, expected_render_identity=delivery.render_identity, delivery=delivery,
        remote=remote, remote_verification=verification, require_remote=True,
    )
    assert record.delivery_status == tsd.TENANT_DELIVERY_STATUS_REMOTE_VERIFY_FAILED
    assert record.ready_for_delivery is False


def test_tenant_bridge_wrong_remote_size_blocks(tmp_path):
    ownership = _scope()
    delivery = _ready_delivery(tmp_path)
    remote = _remote_for(ownership, delivery)
    bad_metadata = tsd.RemoteObjectMetadataFixture(
        exists=True, key=remote.object_key, size_bytes=remote.output_size_bytes + 500,
    )
    verification = tsd.verify_remote_delivery(
        tsd.RemoteDeliveryObject(
            ownership=ownership, render_identity=remote.render_identity, output_sha256=remote.output_sha256,
            output_size_bytes=remote.output_size_bytes, bucket=remote.bucket, object_key=remote.object_key,
        ),
        bad_metadata,
    )
    record = tsd.evaluate_tenant_safe_delivery(
        ownership=ownership, expected_render_identity=delivery.render_identity, delivery=delivery,
        remote=remote, remote_verification=verification, require_remote=True,
    )
    assert record.delivery_status == tsd.TENANT_DELIVERY_STATUS_REMOTE_VERIFY_FAILED
    assert record.ready_for_delivery is False


def test_tenant_bridge_missing_remote_object_blocks_when_required(tmp_path):
    ownership = _scope()
    delivery = _ready_delivery(tmp_path)
    record = tsd.evaluate_tenant_safe_delivery(
        ownership=ownership, expected_render_identity=delivery.render_identity, delivery=delivery,
        remote=None, require_remote=True,
    )
    assert record.delivery_status == tsd.TENANT_DELIVERY_STATUS_REMOTE_VERIFY_FAILED
    assert record.ready_for_delivery is False


def test_tenant_bridge_local_only_stops_at_ready_for_upload(tmp_path):
    """require_remote=False (default) mirrors D-267's own local-only
    contract -- never fabricates DELIVERY_READY without a remote object."""
    ownership = _scope()
    delivery = _ready_delivery(tmp_path)
    record = tsd.evaluate_tenant_safe_delivery(
        ownership=ownership, expected_render_identity=delivery.render_identity, delivery=delivery,
    )
    assert record.delivery_status == rd.DELIVERY_STATUS_READY_FOR_UPLOAD
    assert record.ready_for_delivery is False


def test_tenant_bridge_upstream_qc_fail_passes_through_unchanged(tmp_path):
    """D-267's own QC/hash/render blocker is NEVER re-derived or
    overridden -- it passes straight through."""
    out = tmp_path / "qc_fail.mp4"
    out.write_bytes(b"bytes")
    delivery = rd.build_render_delivery_record(
        render_identity="render_" + "q" * 24, render_execution_status=rd.RENDER_EXECUTION_STATUS_SUCCEEDED,
        final_path=str(out), technical_qc_status=rd.TECHNICAL_QC_STATUS_FAIL,
    )
    ownership = _scope()
    record = tsd.evaluate_tenant_safe_delivery(
        ownership=ownership, expected_render_identity=delivery.render_identity, delivery=delivery,
    )
    assert record.delivery_status == rd.DELIVERY_STATUS_QC_FAILED
    assert record.ready_for_delivery is False


def test_tenant_bridge_upstream_upload_fail_passes_through(tmp_path):
    out = tmp_path / "upload_fail.mp4"
    out.write_bytes(b"bytes")
    delivery = rd.build_render_delivery_record(
        render_identity="render_" + "r" * 24, render_execution_status=rd.RENDER_EXECUTION_STATUS_SUCCEEDED,
        final_path=str(out), technical_qc_status=rd.TECHNICAL_QC_STATUS_PASS,
        require_upload=True, upload_status=rd.UPLOAD_STATUS_FAILED,
    )
    ownership = _scope()
    record = tsd.evaluate_tenant_safe_delivery(
        ownership=ownership, expected_render_identity=delivery.render_identity, delivery=delivery,
    )
    assert record.delivery_status == rd.DELIVERY_STATUS_UPLOAD_FAILED
    assert record.ready_for_delivery is False


def test_tenant_bridge_same_user_different_jobs_isolated(tmp_path):
    delivery_a = _ready_delivery(tmp_path, render_identity="render_" + "1" * 24)
    delivery_b = _ready_delivery(tmp_path, render_identity="render_" + "2" * 24)
    ownership_a = _scope(job_id="job-a")
    ownership_b = _scope(job_id="job-b")
    key_a = tsd.build_tenant_safe_export_key(ownership=ownership_a, render_identity=delivery_a.render_identity)
    key_b = tsd.build_tenant_safe_export_key(ownership=ownership_b, render_identity=delivery_b.render_identity)
    assert key_a != key_b


def test_tenant_record_is_frozen(tmp_path):
    ownership = _scope()
    delivery = _ready_delivery(tmp_path)
    record = tsd.evaluate_tenant_safe_delivery(
        ownership=ownership, expected_render_identity=delivery.render_identity, delivery=delivery,
    )
    with pytest.raises(Exception):
        record.delivery_status = rd.DELIVERY_STATUS_DELIVERY_READY  # type: ignore[misc]


def test_tenant_record_no_secret_fields():
    record_fields = set(tsd.TenantSafeDeliveryRecord.__dataclass_fields__)
    remote_fields = set(tsd.RemoteDeliveryObject.__dataclass_fields__)
    for forbidden in ("credentials", "access_key", "secret_key", "signed_url", "presigned_url", "aws_secret"):
        assert forbidden not in record_fields
        assert forbidden not in remote_fields


# =============================================================================
# Stage 13/14/17 -- ownership authorization + defense-in-depth
# =============================================================================

def test_authorize_delivery_access_exact_match_true():
    scope = _scope()
    assert tsd.authorize_delivery_access(requesting=scope, record_ownership=scope) is True


def test_authorize_delivery_access_no_id_alone_grants_access():
    """Stage 17: matching just the user_id (with different project/job)
    must NOT authorize access -- the full scope must match."""
    scope = _scope()
    same_user_different_job = _scope(job_id="different-job")
    assert tsd.authorize_delivery_access(requesting=same_user_different_job, record_ownership=scope) is False


def test_assert_delivery_access_raises_on_mismatch():
    scope = _scope()
    other = _scope(user_id="user-2")
    with pytest.raises(PermissionError):
        tsd.assert_delivery_access(requesting=other, record_ownership=scope)


def test_assert_delivery_access_none_requesting_skips_check():
    """Mirrors jobs.py's own `_assert_job_owner(user_id=None)` precedent --
    only skips when there is genuinely no identity to compare, never when
    one exists and disagrees."""
    scope = _scope()
    tsd.assert_delivery_access(requesting=None, record_ownership=scope)  # must not raise


def test_assert_delivery_access_matching_scope_does_not_raise():
    scope = _scope()
    tsd.assert_delivery_access(requesting=scope, record_ownership=scope)  # must not raise


# =============================================================================
# Stage 13/26/27 -- presign authorization (pure decision, no real network)
# =============================================================================

def test_presign_authorized_when_owner_and_delivery_ready(tmp_path):
    ownership = _scope()
    delivery = _ready_delivery(tmp_path)
    remote = _remote_for(ownership, delivery)
    verification = tsd.verify_remote_delivery(
        tsd.RemoteDeliveryObject(
            ownership=ownership, render_identity=remote.render_identity, output_sha256=remote.output_sha256,
            output_size_bytes=remote.output_size_bytes, bucket=remote.bucket, object_key=remote.object_key,
        ),
        _matching_remote_metadata(remote),
    )
    record = tsd.evaluate_tenant_safe_delivery(
        ownership=ownership, expected_render_identity=delivery.render_identity, delivery=delivery,
        remote=remote, remote_verification=verification, require_remote=True,
    )
    ok, reason = tsd.authorize_presign_issuance(requesting=ownership, record=record)
    assert ok is True
    assert reason is None


def test_presign_denied_for_wrong_user(tmp_path):
    ownership = _scope()
    other = _scope(user_id="attacker")
    delivery = _ready_delivery(tmp_path)
    remote = _remote_for(ownership, delivery)
    verification = tsd.verify_remote_delivery(
        tsd.RemoteDeliveryObject(
            ownership=ownership, render_identity=remote.render_identity, output_sha256=remote.output_sha256,
            output_size_bytes=remote.output_size_bytes, bucket=remote.bucket, object_key=remote.object_key,
        ),
        _matching_remote_metadata(remote),
    )
    record = tsd.evaluate_tenant_safe_delivery(
        ownership=ownership, expected_render_identity=delivery.render_identity, delivery=delivery,
        remote=remote, remote_verification=verification, require_remote=True,
    )
    ok, reason = tsd.authorize_presign_issuance(requesting=other, record=record)
    assert ok is False
    assert reason == "ownership_mismatch"


def test_presign_denied_when_delivery_not_ready(tmp_path):
    ownership = _scope()
    delivery = _ready_delivery(tmp_path)
    record = tsd.evaluate_tenant_safe_delivery(
        ownership=ownership, expected_render_identity=delivery.render_identity, delivery=delivery,
    )
    ok, reason = tsd.authorize_presign_issuance(requesting=ownership, record=record)
    assert ok is False
    assert reason is not None and reason.startswith("delivery_not_ready:")


def test_presign_authorization_performs_no_real_network():
    source = _source_without_docstrings("cutsell_worker/tenant_safe_delivery.py")
    for needle in ("boto3", "requests.", "urllib", "generate_presigned_url", "http://", "https://"):
        assert needle not in source


# =============================================================================
# Stage 21/22 -- stale-job pointer protection
# =============================================================================

def test_stale_older_job_cannot_overwrite_newer_current():
    assert tsd.is_job_still_current(
        current_latest_job_id="job-new", current_latest_job_started_at=200.0,
        candidate_job_id="job-old", candidate_job_started_at=100.0,
    ) is False


def test_newer_job_can_become_current():
    assert tsd.is_job_still_current(
        current_latest_job_id="job-old", current_latest_job_started_at=100.0,
        candidate_job_id="job-new", candidate_job_started_at=200.0,
    ) is True


def test_same_job_always_current_idempotent():
    assert tsd.is_job_still_current(
        current_latest_job_id="job-1", current_latest_job_started_at=200.0,
        candidate_job_id="job-1", candidate_job_started_at=50.0,
    ) is True


def test_no_ordering_evidence_allows_through():
    assert tsd.is_job_still_current(
        current_latest_job_id=None, current_latest_job_started_at=None,
        candidate_job_id="job-1", candidate_job_started_at=None,
    ) is True


def test_missing_candidate_timestamp_allows_through():
    assert tsd.is_job_still_current(
        current_latest_job_id="job-old", current_latest_job_started_at=100.0,
        candidate_job_id="job-new", candidate_job_started_at=None,
    ) is True


@pytest.fixture
def fake_redis_client():
    return _FakeRedis()


def test_project_store_stale_job_guard_skips_older_job(fake_redis_client):
    from cutsell_worker import project_store as ps
    project = ps.create_project(user_id="user-1", title="T", client=fake_redis_client)
    project_id = project["project_id"]
    ps.update_project(
        user_id="user-1", project_id=project_id, latest_job_id="job-new",
        latest_job_started_at=200.0, state="rendering", client=fake_redis_client,
    )
    stale = ps.update_project(
        user_id="user-1", project_id=project_id, latest_job_id="job-old",
        latest_job_started_at=100.0, state="failed", client=fake_redis_client,
    )
    assert stale["latest_job_id"] == "job-new"
    assert stale["state"] == "rendering"


def test_project_store_stale_job_guard_accepts_newer_job(fake_redis_client):
    from cutsell_worker import project_store as ps
    project = ps.create_project(user_id="user-1", title="T", client=fake_redis_client)
    project_id = project["project_id"]
    ps.update_project(
        user_id="user-1", project_id=project_id, latest_job_id="job-old",
        latest_job_started_at=100.0, state="rendering", client=fake_redis_client,
    )
    fresh = ps.update_project(
        user_id="user-1", project_id=project_id, latest_job_id="job-new",
        latest_job_started_at=200.0, state="completed", client=fake_redis_client,
    )
    assert fresh["latest_job_id"] == "job-new"
    assert fresh["state"] == "completed"


def test_project_store_every_existing_caller_omitting_timestamp_is_unconditional_overwrite(fake_redis_client):
    """Backward compatibility: a caller that never passes
    `latest_job_started_at` (i.e. every existing call site) always
    overwrites, exactly like before this gate."""
    from cutsell_worker import project_store as ps
    project = ps.create_project(user_id="user-1", title="T", client=fake_redis_client)
    project_id = project["project_id"]
    ps.update_project(user_id="user-1", project_id=project_id, latest_job_id="job-1", state="rendering", client=fake_redis_client)
    result = ps.update_project(user_id="user-1", project_id=project_id, latest_job_id="job-2", state="completed", client=fake_redis_client)
    assert result["latest_job_id"] == "job-2"
    assert result["state"] == "completed"


# =============================================================================
# Stage 20/30 -- observability diagnostics (no secrets, no media contents)
# =============================================================================

def test_diagnostics_payload_shape(tmp_path):
    ownership = _scope()
    delivery = _ready_delivery(tmp_path)
    record = tsd.evaluate_tenant_safe_delivery(
        ownership=ownership, expected_render_identity=delivery.render_identity, delivery=delivery,
    )
    diag = tsd.tenant_safe_delivery_diagnostics(record)
    for key in (
        "tenant_delivery_status", "ready_for_delivery", "user_id", "project_id", "job_id",
        "expected_render_identity", "remote_object_key", "remote_verification_status",
        "remote_hash_status", "tenant_errors",
    ):
        assert key in diag


def test_diagnostics_never_carry_credentials(tmp_path):
    ownership = _scope()
    delivery = _ready_delivery(tmp_path)
    remote = _remote_for(ownership, delivery)
    record = tsd.evaluate_tenant_safe_delivery(
        ownership=ownership, expected_render_identity=delivery.render_identity, delivery=delivery, remote=remote,
    )
    diag = tsd.tenant_safe_delivery_diagnostics(record)
    serialized = str(diag).lower()
    for forbidden in ("secret", "access_key", "presigned", "authorization"):
        assert forbidden not in serialized


# =============================================================================
# Stage 1/32 -- secure auth default + explicit local bypass
# =============================================================================

def test_auth_required_by_default(monkeypatch):
    from cutsell_app import auth_middleware as am
    monkeypatch.delenv("CUTSELL_AUTH_REQUIRED", raising=False)
    monkeypatch.delenv(am._LOCAL_DEV_AUTH_DISABLE_ENV, raising=False)
    assert am._auth_required() is True


def test_auth_disabled_only_via_explicit_local_dev_flag(monkeypatch):
    from cutsell_app import auth_middleware as am
    monkeypatch.delenv("CUTSELL_AUTH_REQUIRED", raising=False)
    monkeypatch.setenv(am._LOCAL_DEV_AUTH_DISABLE_ENV, "1")
    assert am._auth_required() is False


def test_auth_required_env_always_wins_over_local_dev_disable(monkeypatch):
    from cutsell_app import auth_middleware as am
    monkeypatch.setenv("CUTSELL_AUTH_REQUIRED", "1")
    monkeypatch.setenv(am._LOCAL_DEV_AUTH_DISABLE_ENV, "1")
    assert am._auth_required() is True


def test_mistyped_auth_required_env_does_not_disable_security(monkeypatch):
    """D-268's own P0 finding: a missing/mistyped CUTSELL_AUTH_REQUIRED
    must never silently disable enforcement."""
    from cutsell_app import auth_middleware as am
    monkeypatch.setenv("CUTSELL_AUTH_REQUIRED", "TRUE-ish-typo")
    monkeypatch.delenv(am._LOCAL_DEV_AUTH_DISABLE_ENV, raising=False)
    assert am._auth_required() is True


# =============================================================================
# Stage 14 -- handler-level defense-in-depth (IDOR never invisible)
# =============================================================================

def test_get_job_handler_passes_auth_user_id_through(monkeypatch):
    import cutsell_app.main as api
    from cutsell_worker.jobs import JobSnapshot
    captured = {}

    def fake_fetch(job_id, **kwargs):
        captured.update(kwargs)
        return JobSnapshot(job_id, "analyzing", progress=10)

    monkeypatch.setattr(api, "fetch_job_snapshot", fake_fetch)
    response = TestClient(api.app).get("/v1/jobs/job-1")
    assert response.status_code == 200
    assert "user_id" in captured


def test_get_job_handler_403_on_permission_error(monkeypatch):
    import cutsell_app.main as api

    def deny(_job_id, **_kwargs):
        raise PermissionError("not yours")

    monkeypatch.setattr(api, "fetch_job_snapshot", deny)
    response = TestClient(api.app).get("/v1/jobs/job-1")
    assert response.status_code == 403


def test_cancel_job_handler_403_on_permission_error(monkeypatch):
    import cutsell_app.main as api

    def deny(_job_id, **_kwargs):
        raise PermissionError("not yours")

    monkeypatch.setattr(api, "cancel_job", deny)
    response = TestClient(api.app).post("/v1/jobs/job-1/cancel")
    assert response.status_code == 403


def test_cancel_job_handler_passes_auth_user_id_through(monkeypatch):
    import cutsell_app.main as api
    from cutsell_worker.jobs import JobSnapshot
    captured = {}

    def fake_cancel(job_id, **kwargs):
        captured.update(kwargs)
        return JobSnapshot(job_id, "canceled")

    monkeypatch.setattr(api, "cancel_job", fake_cancel)
    response = TestClient(api.app).post("/v1/jobs/job-1/cancel")
    assert response.status_code == 200
    assert "user_id" in captured


# =============================================================================
# Stage 22/23/29 -- no secrets, no shell, no network anywhere in this module
# =============================================================================

def test_no_shell_true_in_tenant_safe_delivery_module():
    source = Path("cutsell_worker/tenant_safe_delivery.py").read_text(encoding="utf-8")
    assert "shell=True" not in source


def test_no_network_or_credential_construction_in_tenant_safe_delivery():
    source = _source_without_docstrings("cutsell_worker/tenant_safe_delivery.py")
    for needle in ("boto3", "AWS_SECRET", "Authorization", "requests.", "urllib.request"):
        assert needle not in source


def test_no_actual_upload_or_presign_network_call():
    source = _source_without_docstrings("cutsell_worker/tenant_safe_delivery.py")
    for needle in ("put_object", "generate_presigned_url", "upload_part"):
        assert needle not in source


# =============================================================================
# Stage 23 -- unrelated-authority firewalls (Pacing/Boundary/Freeze/Audio/
# Visual Finishing/QC/renderer/codec/filtergraph/provider all unchanged)
# =============================================================================

@pytest.mark.parametrize("rel_path", [
    "cutsell_worker/render.py",
    "cutsell_worker/render_delivery.py",
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
    "cutsell_worker/gpu_execution_provider.py",
    "cutsell_worker/jobs.py",
    "cutsell_worker/exports.py",
    "cutsell_worker/uploads.py",
])
def test_unrelated_authorities_unchanged(rel_path):
    assert _run_git_diff(rel_path) == "", f"D-269 must not touch {rel_path}"


def test_render_timeout_still_1200():
    from cutsell_worker import render
    assert render.RENDER_FFMPEG_TIMEOUT_SEC == 1200.0


def test_codec_filtergraph_still_unchanged():
    source = Path("cutsell_worker/render.py").read_text(encoding="utf-8")
    assert '"libx264"' in source
    assert '"-crf", "20"' in source
    assert "RENDER_FPS_DEFAULT = 30" in source


def test_no_provider_no_raw_no_paid_compute_in_module():
    source = _source_without_docstrings("cutsell_worker/tenant_safe_delivery.py")
    for needle in ("runpod", "RunPod", "modal.", "GPUExecutionProvider"):
        assert needle not in source
