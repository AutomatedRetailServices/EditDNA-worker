"""D-288.4 -- the races the D-288.3 review named, reproduced FIRST as
failing tests against the D-288.3 head (`89796e9f`), then closed.

Each test forces its race deterministically -- a hook after the last read
or inside a write, or two real OS threads held at a barrier -- never by
timing. Test names carry the review's own numbering:

1. approve -> deliver -> reject -> resume must REFUSE the new request
   while keeping the delivered history on the record;
2. a revocation landing after resume's LAST read but before its upload
   must stop publication (the old post-upload CAS was too late);
3. a revocation landing INSIDE the upload write must be refused (the
   atomic publish claim already happened) and delivery must complete;
4/5. two concurrent `add_render_version` / `publish_notification` calls
   for the same render must return ONE stored id;
6. a retried finalize must not duplicate the render version in the
   project's own history;
7. a pending-review link issued for render A must still show A's bytes
   after render B (same job, same render identity, different bytes) is
   generated;
8. (D-288.4.1) a keyless notification write must never overwrite a
   concurrent atomic insert -- every notification is appended
   atomically, keyed by its own unique `notification_id`.

No Video00 fact/id anywhere below.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from cutsell_worker import export_job
from cutsell_worker import exports
from cutsell_worker import notifications as notif
from cutsell_worker import pending_watch_listen_review as pwl
from cutsell_worker import project_store
from cutsell_worker import render_delivery as rd
from cutsell_worker import render_versions as rv
from cutsell_worker import tenant_safe_delivery as tsd
from cutsell_worker.perceptual_watch_listen import WATCH_LISTEN_HUMAN_APPROVED, WATCH_LISTEN_HUMAN_REVIEW_REQUIRED
from tests.fake_atomic_redis import FakeAtomicRedis


class FakeS3Client:
    def __init__(self):
        self.objects: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.download_hook = None
        self.upload_hook = None

    def upload_file(self, filename, bucket, key, **_kwargs):
        if self.upload_hook is not None:
            self.upload_hook(key)
        with open(filename, "rb") as handle:
            data = handle.read()
        with self._lock:
            self.objects[key] = {"body": data, "size": len(data)}

    def download_file(self, bucket, key, destination):
        if self.download_hook is not None:
            self.download_hook(key)
        with self._lock:
            obj = self.objects[key]
        Path(destination).write_bytes(obj["body"])

    def head_object(self, Bucket, Key):  # noqa: N803
        obj = self.objects.get(Key)
        if obj is None:
            from botocore.exceptions import ClientError
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        return {"ContentLength": obj["size"], "Metadata": {}}

    def generate_presigned_url(self, operation, Params, ExpiresIn):  # noqa: N803
        # The key is embedded so a test can see WHICH object a link points at.
        return f"https://x.invalid/{Params['Key']}"


@pytest.fixture
def fake_s3(monkeypatch):
    client = FakeS3Client()
    monkeypatch.setattr(exports, "load_runtime_config", lambda: SimpleNamespace(s3_bucket="test-bucket", aws_region="us-east-1"))
    monkeypatch.setattr(export_job, "store_export", lambda path, **kwargs: exports.store_export(path, client=client, **kwargs))
    return client


@pytest.fixture
def fake_redis(monkeypatch):
    redis = FakeAtomicRedis()
    monkeypatch.setattr(export_job, "add_render_version", lambda **kwargs: rv.add_render_version(client=redis, **kwargs))
    monkeypatch.setattr(export_job, "publish_notification", lambda **kwargs: notif.publish_notification(client=redis, **kwargs))
    monkeypatch.setattr(project_store, "_redis_client", lambda client=None: redis if client is None else client)
    return redis


USER, PROJECT, JOB = "user-r", "proj-r", "job-r"
OWNER = tsd.DeliveryOwnershipScope(user_id=USER, project_id=PROJECT, job_id=JOB)


def _seed_project(fake_redis, *, latest_job_id=None, latest_job_started_at=None):
    record = {
        "schema_version": "cutsell.project.v1", "project_id": PROJECT, "user_id": USER,
        "title": "Untitled Cut", "state": "created", "created_at": "t", "updated_at": "t",
        "sources": [], "latest_job_id": latest_job_id, "render_versions": [],
    }
    if latest_job_started_at is not None:
        record["latest_job_started_at"] = latest_job_started_at
    fake_redis.set(project_store.project_key(user_id=USER, project_id=PROJECT), json.dumps(record))


def _persist(fake_redis, tmp_path, *, content=b"reviewed-bytes", render_identity="render_" + "a" * 24, name="out.mp4", job_started_at=1.0):
    path = tmp_path / name
    path.write_bytes(content)
    return pwl.persist_pending_review(
        local_path=str(path), user_id=USER, project_id=PROJECT, job_id=JOB,
        render_identity=render_identity, output_sha256=rd.compute_output_sha256(str(path)),
        plan_id="plan_1", plan_version=1, watch_listen_status=WATCH_LISTEN_HUMAN_REVIEW_REQUIRED,
        perceptual_review={}, client=fake_redis, store_export_fn=export_job.store_export,
        job_started_at=job_started_at,
    )


def _decide(fake_redis, record, *, approved: bool):
    return pwl.apply_human_approval(
        user_id=USER, project_id=PROJECT, job_id=JOB, requesting=OWNER, approved=approved, approver=USER,
        expected_render_identity=record.render_identity, expected_output_sha256=record.output_sha256,
        expected_plan_id=record.plan_id, expected_plan_version=record.plan_version, client=fake_redis,
    )


def _resume(fake_redis, fake_s3):
    return export_job.resume_delivery_after_approval(
        user_id=USER, project_id=PROJECT, job_id=JOB, requesting=OWNER, client=fake_redis, s3_client=fake_s3,
    )


def _tenant_safe_keys(fake_s3):
    return [k for k in fake_s3.objects if k.startswith("cutsell/exports/")]


# 1 ---------------------------------------------------------------------------

def test_repro_1_approve_deliver_reject_then_resume_is_refused_history_preserved(tmp_path, fake_s3, fake_redis):
    _seed_project(fake_redis)
    record = _persist(fake_redis, tmp_path)
    _decide(fake_redis, record, approved=True)
    delivered = _resume(fake_redis, fake_s3)
    assert delivered["state"] == "finished"

    rejected = _decide(fake_redis, record, approved=False)
    assert rejected.approval_status == pwl.APPROVAL_STATUS_REJECTED
    assert rejected.resumed_delivery_result == delivered  # history preserved

    with pytest.raises(pwl.PendingReviewError, match="pending_review_not_approved"):
        _resume(fake_redis, fake_s3)

    reloaded = pwl.load_pending_review(user_id=USER, project_id=PROJECT, job_id=JOB, client=fake_redis)
    assert reloaded.resumed_delivery_result == delivered  # still there
    assert reloaded.approval_status == pwl.APPROVAL_STATUS_REJECTED
    assert len(rv.list_render_versions(user_id=USER, project_id=PROJECT, client=fake_redis)) == 1


# 2 ---------------------------------------------------------------------------

def test_repro_2_revocation_after_the_last_read_before_upload_never_publishes(tmp_path, fake_s3, fake_redis, monkeypatch):
    _seed_project(fake_redis)
    record = _persist(fake_redis, tmp_path)
    _decide(fake_redis, record, approved=True)

    real_load = pwl.load_pending_review
    state = {"loads_in_resume": 0, "armed": True}

    def _load_then_revoke(**kwargs):
        loaded = real_load(**kwargs)
        if state["armed"]:
            state["loads_in_resume"] += 1
            # The 2nd load inside resume is its LAST read before the
            # publish step (the 1st is the top-of-function snapshot).
            # Revoke right AFTER that read returns, so resume holds a
            # snapshot that says APPROVED while the store says REJECTED.
            if state["loads_in_resume"] == 2:
                state["armed"] = False
                _decide(fake_redis, record, approved=False)
        return loaded
    monkeypatch.setattr(pwl, "load_pending_review", _load_then_revoke)

    with pytest.raises(pwl.PendingReviewError):
        _resume(fake_redis, fake_s3)

    assert _tenant_safe_keys(fake_s3) == []  # never published
    assert rv.list_render_versions(user_id=USER, project_id=PROJECT, client=fake_redis) == []
    reloaded = real_load(user_id=USER, project_id=PROJECT, job_id=JOB, client=fake_redis)
    assert reloaded.approval_status == pwl.APPROVAL_STATUS_REJECTED  # the revocation was never clobbered
    assert reloaded.resumed_delivery_result is None


# 3 ---------------------------------------------------------------------------

def test_repro_3_revocation_inside_the_upload_write_is_refused_and_delivery_completes(tmp_path, fake_s3, fake_redis):
    _seed_project(fake_redis)
    record = _persist(fake_redis, tmp_path)
    _decide(fake_redis, record, approved=True)

    outcome = {}

    def _revoke_inside_upload(key):
        # Only the tenant-safe write -- the pending-review upload already
        # happened at persist time, before this hook was installed.
        if key.startswith("cutsell/exports/"):
            try:
                _decide(fake_redis, record, approved=False)
                outcome["revocation"] = "accepted"
            except pwl.PendingReviewConflict as exc:
                outcome["revocation"] = f"refused:{exc}"
    fake_s3.upload_hook = _revoke_inside_upload

    result = _resume(fake_redis, fake_s3)
    assert result["state"] == "finished"
    assert outcome["revocation"].startswith("refused:")  # publish claim already held

    reloaded = pwl.load_pending_review(user_id=USER, project_id=PROJECT, job_id=JOB, client=fake_redis)
    assert reloaded.approval_status == pwl.APPROVAL_STATUS_APPROVED
    assert reloaded.watch_listen_status == WATCH_LISTEN_HUMAN_APPROVED
    assert reloaded.resumed_delivery_result == result
    assert len(_tenant_safe_keys(fake_s3)) == 1


# 4 / 5 ------------------------------------------------------------------------

def _run_two(fn):
    results, errors = {}, {}

    def _call(name):
        try:
            results[name] = fn()
        except Exception as exc:  # captured for assertion, never swallowed
            errors[name] = exc
    threads = [threading.Thread(target=_call, args=(n,)) for n in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not errors, errors
    return results


def _barrier_on(fake_redis, key_prefix):
    """Hold BOTH threads at the WRITE that follows their own read -- the
    D-288.3 read-modify-write then lets each append its own id. The
    D-288.4 atomic path never performs a Python-side `set` for this key,
    so the hook simply never fires there."""
    barrier = threading.Barrier(2, timeout=2)

    def _hook(key):
        if key.startswith(key_prefix):
            try:
                barrier.wait(timeout=2)
            except threading.BrokenBarrierError:
                pass
    fake_redis.set_hook = _hook


def test_repro_4_two_concurrent_add_render_version_calls_return_one_stored_id(fake_redis):
    _barrier_on(fake_redis, "cutsell:v1:renders:")
    export_uri = "s3://test-bucket/cutsell/exports/u/p/j/render_x.mp4"
    results = _run_two(lambda: rv.add_render_version(
        user_id=USER, project_id=PROJECT, export_uri=export_uri, size_bytes=3, selected_count=1, client=fake_redis,
    ))
    history = rv.list_render_versions(user_id=USER, project_id=PROJECT, client=fake_redis)
    assert len(history) == 1
    assert results["a"]["render_version_id"] == results["b"]["render_version_id"] == history[0]["render_version_id"]


def test_repro_5_two_concurrent_publish_notification_calls_return_one_stored_id(fake_redis):
    _barrier_on(fake_redis, "cutsell:v1:notifications:")
    results = _run_two(lambda: notif.publish_notification(
        user_id=USER, project_id=PROJECT, kind="render_finished", payload={}, idempotency_key="render_x", client=fake_redis,
    ))
    stored = [n for n in notif.list_notifications(user_id=USER, client=fake_redis) if n["kind"] == "render_finished"]
    assert len(stored) == 1
    assert results["a"]["notification_id"] == results["b"]["notification_id"] == stored[0]["notification_id"]


# 6 ---------------------------------------------------------------------------

def test_repro_6_retried_finalize_never_duplicates_the_version_in_the_project_history(fake_redis):
    _seed_project(fake_redis)
    delivery = {
        "delivery_status": rd.DELIVERY_STATUS_DELIVERY_READY, "watch_listen_status": WATCH_LISTEN_HUMAN_APPROVED,
        "render_identity": "render_" + "b" * 24, "output_sha256": "s", "remote_reference": "r", "expires_in": 60,
        "size_bytes": 3, "export_uri": "s3://test-bucket/cutsell/exports/u/p/j/render_b.mp4",
    }
    kwargs = dict(user_id=USER, project_id=PROJECT, job_id=JOB, job_started_at=1.0, delivery=delivery,
                  selected_count=1, text_overlay_count=0, media_overlay_count=0)
    first = export_job._finalize_successful_delivery(**kwargs)
    second = export_job._finalize_successful_delivery(**kwargs)  # the retry
    assert first["render_version_id"] == second["render_version_id"]

    project = project_store.get_project(user_id=USER, project_id=PROJECT, client=fake_redis)
    assert len(project["render_versions"]) == 1
    assert project["render_versions"][0]["render_version_id"] == first["render_version_id"]
    assert len(rv.list_render_versions(user_id=USER, project_id=PROJECT, client=fake_redis)) == 1


# 7 ---------------------------------------------------------------------------

def test_repro_7_a_pending_link_issued_for_a_still_shows_a_after_b_is_generated(tmp_path, fake_s3, fake_redis):
    a = _persist(fake_redis, tmp_path, content=b"render-A-bytes", name="a.mp4")
    link_a = pwl.get_pending_review_media_access(
        user_id=USER, project_id=PROJECT, job_id=JOB, requesting=OWNER,
        expected_render_identity=a.render_identity, expected_output_sha256=a.output_sha256,
        client=fake_redis, s3_client=fake_s3,
    )
    key_a = link_a["preview_url"].split("https://x.invalid/", 1)[1]
    assert fake_s3.objects[key_a]["body"] == b"render-A-bytes"

    # D-267: two encodes of the identical plan can hash differently --
    # SAME render identity, DIFFERENT bytes, same job slot.
    b = _persist(fake_redis, tmp_path, content=b"render-B-bytes", name="b.mp4", render_identity=a.render_identity)
    assert b.output_sha256 != a.output_sha256

    assert fake_s3.objects[key_a]["body"] == b"render-A-bytes"  # the link for A still shows A
    assert b.pending_s3_uri != a.pending_s3_uri
    assert a.output_sha256 in a.pending_s3_uri and b.output_sha256 in b.pending_s3_uri


# 8 ---------------------------------------------------------------------------

def test_repro_8_a_keyless_notification_write_never_overwrites_a_concurrent_atomic_insert(fake_redis):
    """`draft_ready` (no idempotency_key) reads the list -> `render_
    finished` inserts itself atomically -> `draft_ready` writes. Both
    must remain stored. The D-288.4 keyless path was still a Python-side
    GET -> insert -> SET, so its stale SET erased the concurrent insert;
    now every notification is appended atomically (unique `notification_
    id` as the match field), so there is no Python-side write left for
    the hook to interleave with at all."""
    key = notif.notification_key(USER)
    seen = {"python_side_sets": 0}

    def _insert_render_finished_between_read_and_write(written_key):
        if written_key == key:
            seen["python_side_sets"] += 1
            notif.publish_notification(
                user_id=USER, project_id=PROJECT, kind="render_finished", payload={},
                idempotency_key="render_x", client=fake_redis,
            )
    fake_redis.set_hook = _insert_render_finished_between_read_and_write

    notif.publish_notification(user_id=USER, project_id=PROJECT, kind="draft_ready", payload={}, client=fake_redis)
    if seen["python_side_sets"] == 0:
        # Fixed path: nothing to interleave with -- insert it afterwards
        # so the "both remain stored" assertion is exercised either way.
        notif.publish_notification(
            user_id=USER, project_id=PROJECT, kind="render_finished", payload={},
            idempotency_key="render_x", client=fake_redis,
        )

    kinds = sorted(n["kind"] for n in notif.list_notifications(user_id=USER, client=fake_redis))
    assert kinds == ["draft_ready", "render_finished"]
    assert seen["python_side_sets"] == 0  # the keyless write is atomic too, never a read-modify-write
