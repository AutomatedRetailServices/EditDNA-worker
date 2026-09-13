"""D-269A -- LIVE TENANT-SAFE DELIVERY ACTIVATION.

Post D-269 (offline tenant-safe delivery foundation: ownership scope,
deterministic key builder, remote verification contract, presign
authorization, stale-job guard -- all proven in isolation, NOT wired into
the real export path). This file proves the LIVE wiring in
`cutsell_worker/export_job.py`'s `_tenant_safe_deliver` and
`cutsell_worker/exports.py`'s extended `store_export`: render_identity
bound from the actual render plan, the D-269 tenant-safe key replacing
the legacy uuid4 key, a real post-upload `head_object` call consumed
through `verify_remote_delivery`, and `DELIVERY_READY` reached only when
every invariant holds -- against a FAKE S3 client throughout. No real
S3 write, no network, no provider, no RAW anywhere in this file.
"""
from __future__ import annotations

import ast
import functools
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from cutsell_worker import export_job
from cutsell_worker import exports
from cutsell_worker import render_delivery as rd
from cutsell_worker import tenant_safe_delivery as tsd
from cutsell_worker.render_plan import RenderSegment


def _source_without_docstrings(path: str) -> str:
    """D-262/D-263/D-266/D-267/D-269's own established false-positive-
    proofing technique."""
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


# =============================================================================
# Fake S3 client -- Stage 27: no real network, no AWS, ever
# =============================================================================

class FakeS3Client:
    """Stands in for `boto3.client("s3", ...)`. `upload_file` really writes
    to an in-memory object table so `head_object` can honestly report on
    what was actually uploaded; every override below lets a test simulate
    a specific, real-world remote-evidence disagreement (Stage 15/17/18)
    without ever touching a network."""

    def __init__(self):
        self.objects: dict[str, dict] = {}
        self.uploads: list[tuple] = []
        self.presign_calls: list[tuple] = []
        self.upload_should_fail = False
        self.head_should_report_missing = False
        self.head_size_override: int | None = None
        self.head_metadata_override: dict | None = None

    def upload_file(self, source, bucket, key, ExtraArgs=None):
        self.uploads.append((source, bucket, key, ExtraArgs))
        if self.upload_should_fail:
            raise RuntimeError("simulated_upload_failure")
        size = Path(source).stat().st_size
        metadata = dict((ExtraArgs or {}).get("Metadata") or {})
        self.objects[key] = {"size": size, "metadata": metadata}

    def head_object(self, Bucket, Key):
        if self.head_should_report_missing or Key not in self.objects:
            raise RuntimeError("simulated_404_not_found")
        stored = self.objects[Key]
        return {
            "ContentLength": self.head_size_override if self.head_size_override is not None else stored["size"],
            "Metadata": self.head_metadata_override if self.head_metadata_override is not None else stored["metadata"],
        }

    def generate_presigned_url(self, operation, Params, ExpiresIn):
        self.presign_calls.append((operation, Params, ExpiresIn))
        return f"https://download.invalid/{Params['Key']}"


@pytest.fixture
def fake_s3(monkeypatch):
    client = FakeS3Client()
    monkeypatch.setattr(exports, "load_runtime_config", lambda: SimpleNamespace(s3_bucket="test-bucket", aws_region="us-east-1"))
    return client


@pytest.fixture
def wire_real_store_export(monkeypatch, fake_s3):
    """D-269A Stage 5/27: `export_job._tenant_safe_deliver` calls the
    module-level `store_export` name -- this binds it to the REAL
    `exports.store_export` (Stage 5: no parallel uploader) with the fake
    S3 client pre-bound, so the whole chain (tenant-safe key -> real
    upload_file/head_object/generate_presigned_url code -> remote
    verification) runs for real except for the actual network call."""
    monkeypatch.setattr(export_job, "store_export", functools.partial(exports.store_export, client=fake_s3))
    return fake_s3


def _plan(clip_id="clip-1", source_asset_id="src-1", text="hello"):
    return (RenderSegment(clip_id=clip_id, source_asset_id=source_asset_id, source_path="/tmp/x.mp4", start=1.0, end=2.0, caption_text=text),)


def _rendered_file(tmp_path, name="out.mp4", content=b"rendered-mp4-bytes") -> str:
    path = tmp_path / name
    path.write_bytes(content)
    return str(path)


# =============================================================================
# Stage 1-8 -- the live deliver seam, full happy path
# =============================================================================

def test_live_deliver_full_valid_flow_reaches_delivery_ready(tmp_path, wire_real_store_export):
    output = _rendered_file(tmp_path)
    result = export_job._tenant_safe_deliver(
        output_path=output, plan=_plan(), project_id="proj-1", user_id="user-1", job_id="job-1",
    )
    assert result["delivery_status"] == rd.DELIVERY_STATUS_DELIVERY_READY
    assert result["render_identity"].startswith("render_")
    assert result["output_sha256"] is not None
    assert result["export_uri"].startswith("s3://test-bucket/")
    assert result["remote_reference"] == result["export_uri"]
    assert result["download_url"] is not None
    assert "presign_denied_reason" not in result


def test_live_deliver_binds_job_id_into_tenant_safe_key(tmp_path, wire_real_store_export):
    output = _rendered_file(tmp_path)
    result_a = export_job._tenant_safe_deliver(
        output_path=output, plan=_plan(), project_id="proj-1", user_id="user-1", job_id="job-a",
    )
    output_b = _rendered_file(tmp_path, name="out2.mp4")
    result_b = export_job._tenant_safe_deliver(
        output_path=output_b, plan=_plan(), project_id="proj-1", user_id="user-1", job_id="job-b",
    )
    assert result_a["export_uri"] != result_b["export_uri"]


def test_live_deliver_uses_tenant_safe_key_not_legacy_uuid_scheme(tmp_path, wire_real_store_export):
    output = _rendered_file(tmp_path)
    result = export_job._tenant_safe_deliver(
        output_path=output, plan=_plan(), project_id="proj-1", user_id="user-1", job_id="job-1",
    )
    key = result["export_uri"].split("test-bucket/", 1)[1]
    ownership = tsd.DeliveryOwnershipScope(user_id="user-1", project_id="proj-1", job_id="job-1")
    expected_key = tsd.build_tenant_safe_export_key(ownership=ownership, render_identity=result["render_identity"])
    assert key == expected_key
    # never the legacy scheme's own prefix shape (scope_hash/scope_hash/uuid4.mp4 with a bare uuid,
    # not a render_-prefixed segment)
    assert "render_" in key


def test_live_deliver_binds_render_identity_from_plan_not_filename(tmp_path, wire_real_store_export):
    plan_a = _plan(text="hello")
    plan_b = _plan(text="different caption")
    output_a = _rendered_file(tmp_path, name="same-name.mp4")
    result_a = export_job._tenant_safe_deliver(
        output_path=output_a, plan=plan_a, project_id="proj-1", user_id="user-1", job_id="job-1",
    )
    output_b = _rendered_file(tmp_path, name="same-name-2.mp4")
    result_b = export_job._tenant_safe_deliver(
        output_path=output_b, plan=plan_b, project_id="proj-1", user_id="user-1", job_id="job-2",
    )
    assert result_a["render_identity"] != result_b["render_identity"]


def test_live_deliver_binds_output_sha256_from_real_bytes(tmp_path, wire_real_store_export):
    import hashlib
    output = _rendered_file(tmp_path, content=b"specific-bytes-for-hash-check")
    result = export_job._tenant_safe_deliver(
        output_path=output, plan=_plan(), project_id="proj-1", user_id="user-1", job_id="job-1",
    )
    assert result["output_sha256"] == hashlib.sha256(b"specific-bytes-for-hash-check").hexdigest()


# =============================================================================
# Stage 6/7 -- upload success alone is not enough; remote verification required
# =============================================================================

def test_upload_success_alone_not_sufficient_missing_remote_object(tmp_path, monkeypatch, fake_s3):
    """A `store_export` stand-in that reports upload success but NO real
    remote evidence (Stage 6/7: never mark ready merely because upload
    returned success)."""
    output = _rendered_file(tmp_path)

    def fake_store_export(path, **kwargs):
        return {
            "export_uri": "s3://test-bucket/some/key.mp4", "download_url": "https://x.invalid/k.mp4",
            "expires_in": 3600, "size_bytes": Path(path).stat().st_size,
            "bucket": "test-bucket", "object_key": kwargs.get("object_key"),
            "remote_head": {"exists": False},
        }
    monkeypatch.setattr(export_job, "store_export", fake_store_export)
    with pytest.raises(export_job.TenantSafeDeliveryBlocked) as exc_info:
        export_job._tenant_safe_deliver(
            output_path=output, plan=_plan(), project_id="proj-1", user_id="user-1", job_id="job-1",
        )
    assert exc_info.value.record.delivery_status == tsd.TENANT_DELIVERY_STATUS_REMOTE_VERIFY_FAILED


def test_missing_remote_object_via_real_head_object_404(tmp_path, wire_real_store_export):
    output = _rendered_file(tmp_path)
    wire_real_store_export.head_should_report_missing = True
    with pytest.raises(export_job.TenantSafeDeliveryBlocked) as exc_info:
        export_job._tenant_safe_deliver(
            output_path=output, plan=_plan(), project_id="proj-1", user_id="user-1", job_id="job-1",
        )
    assert exc_info.value.record.delivery_status == tsd.TENANT_DELIVERY_STATUS_REMOTE_VERIFY_FAILED
    assert exc_info.value.record.ready_for_delivery is False


def test_remote_size_mismatch_blocks_delivery(tmp_path, wire_real_store_export):
    output = _rendered_file(tmp_path)
    wire_real_store_export.head_size_override = 999999
    with pytest.raises(export_job.TenantSafeDeliveryBlocked) as exc_info:
        export_job._tenant_safe_deliver(
            output_path=output, plan=_plan(), project_id="proj-1", user_id="user-1", job_id="job-1",
        )
    assert exc_info.value.record.delivery_status == tsd.TENANT_DELIVERY_STATUS_REMOTE_VERIFY_FAILED


def test_remote_render_identity_mismatch_blocks_delivery(tmp_path, wire_real_store_export):
    output = _rendered_file(tmp_path)
    wire_real_store_export.head_metadata_override = {"render_identity": "render_" + "z" * 24, "sha256": "irrelevant"}
    with pytest.raises(export_job.TenantSafeDeliveryBlocked) as exc_info:
        export_job._tenant_safe_deliver(
            output_path=output, plan=_plan(), project_id="proj-1", user_id="user-1", job_id="job-1",
        )
    assert exc_info.value.record.delivery_status == tsd.TENANT_DELIVERY_STATUS_REMOTE_VERIFY_FAILED


def test_remote_hash_mismatch_blocks_delivery(tmp_path, wire_real_store_export):
    output = _rendered_file(tmp_path)
    result_probe = None
    # First render for real to learn the real render_identity, then force a wrong sha256 in metadata.
    wire_real_store_export.head_metadata_override = None
    plan = _plan()
    ownership = tsd.DeliveryOwnershipScope(user_id="user-1", project_id="proj-1", job_id="job-1")
    render_identity = rd.compute_render_identity(tuple(plan), width=1080, height=1920, fps=30)
    wire_real_store_export.head_metadata_override = {"render_identity": render_identity, "sha256": "f" * 64}
    with pytest.raises(export_job.TenantSafeDeliveryBlocked) as exc_info:
        export_job._tenant_safe_deliver(
            output_path=output, plan=plan, project_id="proj-1", user_id="user-1", job_id="job-1",
        )
    assert exc_info.value.record.delivery_status == tsd.TENANT_DELIVERY_STATUS_REMOTE_VERIFY_FAILED


def test_remote_hash_unavailable_is_honest_not_fabricated_pass(tmp_path, wire_real_store_export):
    """No metadata at all (an object predating this feature) -- absence of
    remote hash evidence is honestly UNVERIFIED/NOT_AVAILABLE, and does
    NOT itself block delivery (Stage 18: "if SHA unavailable: preserve
    exact REMOTE_HASH_UNVERIFIED semantics")."""
    output = _rendered_file(tmp_path)
    plan = _plan()
    render_identity = rd.compute_render_identity(tuple(plan), width=1080, height=1920, fps=30)
    # metadata carries render_identity (so that check passes) but no sha256 key at all
    wire_real_store_export.head_metadata_override = {"render_identity": render_identity}
    result = export_job._tenant_safe_deliver(
        output_path=output, plan=plan, project_id="proj-1", user_id="user-1", job_id="job-1",
    )
    assert result["delivery_status"] == rd.DELIVERY_STATUS_DELIVERY_READY


def test_upload_failure_blocks_delivery_no_ready_state(tmp_path, wire_real_store_export):
    output = _rendered_file(tmp_path)
    wire_real_store_export.upload_should_fail = True
    with pytest.raises(RuntimeError, match="simulated_upload_failure"):
        export_job._tenant_safe_deliver(
            output_path=output, plan=_plan(), project_id="proj-1", user_id="user-1", job_id="job-1",
        )


def test_qc_failure_never_reaches_tenant_safe_delivery(monkeypatch, tmp_path):
    """QC failure happens BEFORE `_tenant_safe_deliver` is ever called
    (existing D-030 wiring, unchanged) -- proven here at the
    `run_export_job` level with a real `FakeJob` context."""
    from types import ModuleType
    import sys

    fake_job = SimpleNamespace(id="job-1", started_at=1_700_000_000.0, meta={}, save_meta=lambda: None)
    rq_module = ModuleType("rq")
    rq_module.get_current_job = lambda: fake_job
    monkeypatch.setitem(sys.modules, "rq", rq_module)
    monkeypatch.setattr(export_job, "validate_product_source_uri", lambda uri, **kwargs: ("bucket", "key"))
    monkeypatch.setattr(export_job, "download_source", lambda uri, destination: Path(destination).write_bytes(b"x") or destination)
    monkeypatch.setattr(export_job, "build_render_plan", lambda draft, local_paths: _plan())

    called_tenant_safe = []
    monkeypatch.setattr(export_job, "_tenant_safe_deliver", lambda **kwargs: called_tenant_safe.append(1))

    from cutsell_worker.live_render_qc import PostRenderQCFailure

    def fake_render_with_qc(draft, plan, output, **kwargs):
        Path(output).write_bytes(b"x")
        result = SimpleNamespace(status="NEEDS_HUMAN_REVIEW", output_path=None, plan_id="p", plan_version=1, semantic_hash="h", attempts=())
        raise PostRenderQCFailure(result)
    monkeypatch.setattr(export_job, "render_with_post_render_qc", fake_render_with_qc)

    draft_dict = {
        "schema_version": "cutsell.v1", "project_id": "project-1", "strategy": "mixed",
        "selected": [{"clip_id": "c1", "source_asset_id": "src-1", "source_order": 0, "start": 1.0, "end": 2.0,
                      "text": "hi", "caption_text": "hi", "semantic_role": "OTHER", "selected": True}],
        "alternates": [], "discarded": [], "diagnostics": {},
    }
    with pytest.raises(PostRenderQCFailure):
        export_job.run_export_job({
            "project_id": "project-1", "user_id": "user-1", "draft": draft_dict,
            "sources": [{"source_asset_id": "src-1", "original_name": "one.mov", "uri": "s3://bucket/one.mov"}],
        })
    assert called_tenant_safe == []


# =============================================================================
# Stage 11/13 -- ownership mismatch / presign authorization
# =============================================================================

def test_wrong_requesting_principal_denied_presign(tmp_path, wire_real_store_export):
    output = _rendered_file(tmp_path)
    wrong_scope = tsd.DeliveryOwnershipScope(user_id="attacker", project_id="proj-1", job_id="job-1")
    result = export_job._tenant_safe_deliver(
        output_path=output, plan=_plan(), project_id="proj-1", user_id="user-1", job_id="job-1",
        requesting=wrong_scope,
    )
    assert result["delivery_status"] == rd.DELIVERY_STATUS_DELIVERY_READY  # delivery itself still succeeded
    assert result["download_url"] is None  # but no presign published to the wrong principal
    assert result["presign_denied_reason"] == "ownership_mismatch"


def test_matching_requesting_principal_gets_presign(tmp_path, wire_real_store_export):
    output = _rendered_file(tmp_path)
    own_scope = tsd.DeliveryOwnershipScope(user_id="user-1", project_id="proj-1", job_id="job-1")
    result = export_job._tenant_safe_deliver(
        output_path=output, plan=_plan(), project_id="proj-1", user_id="user-1", job_id="job-1",
        requesting=own_scope,
    )
    assert result["download_url"] is not None


def test_real_export_flow_default_requesting_is_own_ownership(tmp_path, wire_real_store_export):
    """The real RQ export flow never diverges: default `requesting` is the
    export's own ownership scope, so it always receives its own presign."""
    output = _rendered_file(tmp_path)
    result = export_job._tenant_safe_deliver(
        output_path=output, plan=_plan(), project_id="proj-1", user_id="user-1", job_id="job-1",
    )
    assert result["download_url"] is not None


# =============================================================================
# Stage 14 -- duplicate export determinism
# =============================================================================

def test_duplicate_equivalent_export_deterministic_key(tmp_path, wire_real_store_export):
    output = _rendered_file(tmp_path, content=b"identical-bytes")
    plan = _plan()
    result_a = export_job._tenant_safe_deliver(
        output_path=output, plan=plan, project_id="proj-1", user_id="user-1", job_id="job-1",
    )
    output2 = _rendered_file(tmp_path, name="retry.mp4", content=b"identical-bytes")
    result_b = export_job._tenant_safe_deliver(
        output_path=output2, plan=plan, project_id="proj-1", user_id="user-1", job_id="job-1",
    )
    assert result_a["export_uri"] == result_b["export_uri"]
    assert result_a["render_identity"] == result_b["render_identity"]


# =============================================================================
# Stage 22/23 -- two-job / two-user isolation via the live path
# =============================================================================

def test_same_project_two_jobs_distinct_delivery(tmp_path, wire_real_store_export):
    plan = _plan()
    output_a = _rendered_file(tmp_path, name="a.mp4")
    result_a = export_job._tenant_safe_deliver(
        output_path=output_a, plan=plan, project_id="proj-1", user_id="user-1", job_id="job-a",
    )
    output_b = _rendered_file(tmp_path, name="b.mp4")
    result_b = export_job._tenant_safe_deliver(
        output_path=output_b, plan=plan, project_id="proj-1", user_id="user-1", job_id="job-b",
    )
    assert result_a["export_uri"] != result_b["export_uri"]


def test_two_users_same_textual_project_and_job_ids_isolated(tmp_path, wire_real_store_export):
    plan = _plan()
    output_a = _rendered_file(tmp_path, name="ua.mp4")
    result_a = export_job._tenant_safe_deliver(
        output_path=output_a, plan=plan, project_id="shared-project", user_id="user-a", job_id="shared-job",
    )
    output_b = _rendered_file(tmp_path, name="ub.mp4")
    result_b = export_job._tenant_safe_deliver(
        output_path=output_b, plan=plan, project_id="shared-project", user_id="user-b", job_id="shared-job",
    )
    assert result_a["export_uri"] != result_b["export_uri"]


# =============================================================================
# Stage 9/10/21 -- stale-job guard wired into project_store via job_started_at
# =============================================================================

def test_job_started_epoch_reads_real_started_at_datetime():
    from datetime import datetime, timezone
    job = SimpleNamespace(started_at=datetime(2024, 1, 1, tzinfo=timezone.utc))
    epoch = export_job._job_started_epoch(job)
    assert epoch == datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()


def test_job_started_epoch_accepts_plain_float():
    job = SimpleNamespace(started_at=1700000000.0)
    assert export_job._job_started_epoch(job) == 1700000000.0


def test_job_started_epoch_none_when_absent():
    job = SimpleNamespace()
    assert export_job._job_started_epoch(job) is None


class _FakePipeline:
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


def test_stale_job_pointer_race_via_safe_update_project():
    """Stage 21: job_old starts first, job_new starts later, job_new
    completes first, job_old completes later -- job_new must remain
    current."""
    from cutsell_worker.project_store import create_project, update_project

    redis = _FakeRedis()
    project = create_project(user_id="user-1", title="T", client=redis)
    project_id = project["project_id"]

    # job_new (started later) completes FIRST
    update_project(
        user_id="user-1", project_id=project_id, latest_job_id="job_new",
        latest_job_started_at=200.0, state="finished", client=redis,
    )
    # job_old (started earlier) completes LATE
    result = update_project(
        user_id="user-1", project_id=project_id, latest_job_id="job_old",
        latest_job_started_at=100.0, state="finished", client=redis,
    )
    assert result["latest_job_id"] == "job_new"
    assert result["state"] == "finished"  # job_old's own state write was skipped


# =============================================================================
# Stage 26 -- immutability
# =============================================================================

def test_tenant_safe_delivery_blocked_carries_immutable_record(tmp_path, monkeypatch, fake_s3):
    output = _rendered_file(tmp_path)

    def fake_store_export(path, **kwargs):
        return {
            "export_uri": "s3://test-bucket/x.mp4", "download_url": "https://x.invalid",
            "expires_in": 3600, "size_bytes": Path(path).stat().st_size,
            "bucket": "test-bucket", "object_key": kwargs.get("object_key"),
            "remote_head": {"exists": False},
        }
    monkeypatch.setattr(export_job, "store_export", fake_store_export)
    with pytest.raises(export_job.TenantSafeDeliveryBlocked) as exc_info:
        export_job._tenant_safe_deliver(
            output_path=output, plan=_plan(), project_id="proj-1", user_id="user-1", job_id="job-1",
        )
    with pytest.raises(Exception):
        exc_info.value.record.delivery_status = "SOMETHING_ELSE"  # type: ignore[misc]


# =============================================================================
# Stage 25 -- backward compatibility (export_uri retained)
# =============================================================================

def test_export_uri_still_present_alongside_remote_reference(tmp_path, wire_real_store_export):
    output = _rendered_file(tmp_path)
    result = export_job._tenant_safe_deliver(
        output_path=output, plan=_plan(), project_id="proj-1", user_id="user-1", job_id="job-1",
    )
    assert result["export_uri"] == result["remote_reference"]
    assert result["export_uri"].startswith("s3://")


# =============================================================================
# Stage 28 -- security diagnostics
# =============================================================================

def test_diagnostics_available_for_blocked_delivery(tmp_path, wire_real_store_export):
    wire_real_store_export.head_should_report_missing = True
    output = _rendered_file(tmp_path)
    with pytest.raises(export_job.TenantSafeDeliveryBlocked) as exc_info:
        export_job._tenant_safe_deliver(
            output_path=output, plan=_plan(), project_id="proj-1", user_id="user-1", job_id="job-1",
        )
    diag = tsd.tenant_safe_delivery_diagnostics(exc_info.value.record)
    assert diag["tenant_delivery_status"] == tsd.TENANT_DELIVERY_STATUS_REMOTE_VERIFY_FAILED
    assert diag["ready_for_delivery"] is False
    serialized = str(diag).lower()
    for forbidden in ("secret", "access_key", "authorization"):
        assert forbidden not in serialized


# =============================================================================
# Stage 27/29 -- no real network anywhere in the modified modules
# =============================================================================

def test_no_network_or_credential_construction_in_export_job():
    source = _source_without_docstrings("cutsell_worker/export_job.py")
    for needle in ("requests.", "urllib.request", "AWS_SECRET", "Authorization:"):
        assert needle not in source


def test_no_network_or_credential_construction_in_exports():
    source = _source_without_docstrings("cutsell_worker/exports.py")
    for needle in ("requests.", "urllib.request", "AWS_SECRET", "Authorization:"):
        assert needle not in source


def test_no_shell_true_in_modified_modules():
    for path in ("cutsell_worker/export_job.py", "cutsell_worker/exports.py"):
        assert "shell=True" not in Path(path).read_text(encoding="utf-8")


def test_no_provider_no_raw_reference_in_modified_modules():
    for path in ("cutsell_worker/export_job.py", "cutsell_worker/exports.py"):
        source = _source_without_docstrings(path)
        for needle in ("runpod", "RunPod", "modal.", "GPUExecutionProvider"):
            assert needle not in source


# =============================================================================
# Stage 30 -- regression firewall (Pacing/Boundary/Freeze/Audio/Visual
# Finishing/QC authority/renderer/codec/filtergraph/provider unchanged)
# =============================================================================

@pytest.mark.parametrize("rel_path", [
    "cutsell_worker/render.py",
    "cutsell_worker/render_delivery.py",
    "cutsell_worker/render_plan.py",
    "cutsell_worker/tenant_safe_delivery.py",
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
    "cutsell_worker/uploads.py",
    "cutsell_worker/project_store.py",
    "cutsell_app/auth_middleware.py",
    "cutsell_app/main.py",
])
def test_unrelated_authorities_unchanged(rel_path):
    assert _run_git_diff(rel_path) == "", f"D-269A must not touch {rel_path}"


def test_render_timeout_still_1200():
    from cutsell_worker import render
    assert render.RENDER_FFMPEG_TIMEOUT_SEC == 1200.0


def test_codec_filtergraph_still_unchanged():
    source = Path("cutsell_worker/render.py").read_text(encoding="utf-8")
    assert '"libx264"' in source
    assert '"-crf", "20"' in source
    assert "RENDER_FPS_DEFAULT = 30" in source


def test_render_versions_export_uri_contract_unchanged():
    """`add_render_version` still requires an `s3://`-prefixed
    `export_uri` -- the tenant-safe key changes WHAT the key looks like,
    never the URI scheme/contract this module already validates."""
    from cutsell_worker.render_versions import add_render_version
    with pytest.raises(ValueError):
        add_render_version(
            user_id="user-1", project_id="proj-1", export_uri="not-s3://bad",
            size_bytes=10, selected_count=1,
        )
