"""22 Oct: server storage kept to the minimum -- 3 days after the last use (report only), restore at the same key."""
import datetime as dt
import json
from types import SimpleNamespace

import pytest

from cutsell_worker import storage_lifecycle as sl
from cutsell_worker.uploads import scoped_upload_prefix, upload_prefix


class FakeRedis:
    def __init__(self):
        self.z = {}

    def zadd(self, key, mapping):
        self.z.update(mapping)

    def zscore(self, key, member):
        return self.z.get(member)


class FakeS3:
    def __init__(self, objects):
        self.objects = objects          # key -> (size, last_modified)
        self.deleted = []

    def get_paginator(self, name):
        objs = self.objects

        class P:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": k, "Size": v[0], "LastModified": v[1]} for k, v in objs.items() if k.startswith(Prefix)]}
        return P()

    def delete_object(self, Bucket, Key):
        self.deleted.append(Key)


def _ts(days_ago, now):
    return dt.datetime.fromtimestamp(now - days_ago * 86400, tz=dt.timezone.utc)


def test_only_files_unused_for_3_days_are_reported_and_opening_a_cut_counts_as_use(monkeypatch):
    now = 1_800_000_000.0
    p = upload_prefix()
    s3 = FakeS3({p + "a/old.mp4": (100, _ts(40, now)), p + "a/opened.mp4": (200, _ts(40, now)),
                 p + "a/new.mp4": (300, _ts(1, now)), "engine-ab-tests/qa.mp4": (999, _ts(400, now))})
    r = FakeRedis()
    sl.touch([f"s3://b/{p}a/opened.mp4"], r, now=now - 2 * 86400)   # opened 2 days ago: the count started again
    out = sl.candidates(s3, "b", r, now=now)
    assert out["keys"] == [p + "a/old.mp4"] and out["old_bytes"] == 100
    assert out["bucket_files"] == 3                      # QA / benchmark material is never even listed


def test_nothing_is_deleted_without_the_switch_and_the_confirmation(monkeypatch):
    p = upload_prefix()
    s3 = FakeS3({})
    assert sl.remove(s3, "b", [p + "x.mp4"], confirm=True) == 0
    monkeypatch.setenv("CUTSELL_STORAGE_LIFECYCLE_DELETE", "1")
    assert sl.remove(s3, "b", [p + "x.mp4"], confirm=False) == 0
    assert sl.remove(s3, "b", [p + "x.mp4", "engine-ab-tests/qa.mp4"], confirm=True) == 1
    assert s3.deleted == [p + "x.mp4"]                   # never outside the app's uploads


def test_restore_goes_to_the_same_key_and_only_inside_her_project():
    key = scoped_upload_prefix(user_id="u1", project_id="p1") + "abc-video.mp4"
    assert sl.restore_target(uri=f"s3://b/{key}", user_id="u1", project_id="p1", bucket="b") == key
    with pytest.raises(PermissionError):
        sl.restore_target(uri=f"s3://b/{key}", user_id="u2", project_id="p1", bucket="b")
    with pytest.raises(ValueError):
        sl.restore_target(uri=f"s3://other/{key}", user_id="u1", project_id="p1", bucket="b")


def test_restore_upload_refuses_a_copy_that_is_still_there(monkeypatch):
    import cutsell_worker.multipart_uploads as mp
    key = scoped_upload_prefix(user_id="u1", project_id="p1") + "abc-video.mp4"
    monkeypatch.setattr(mp, "prepare_upload_target", lambda **k: {"bucket": "b", "region": "auto", "object_key": "new",
                                                                   "source_uri": "s3://b/new", "content_type": "video/mp4",
                                                                   "size_bytes": k["size_bytes"]})

    class S3:
        def __init__(self, exists): self.exists = exists; self.created = []
        def head_object(self, Bucket, Key):
            if not self.exists:
                raise type("E", (Exception,), {})("nf") if False else _nf()
        def create_multipart_upload(self, Bucket, Key, ContentType):
            self.created.append(Key); return {"UploadId": "up1"}

    def _nf():
        e = Exception("not found"); e.response = {"Error": {"Code": "404"}}; return e

    class R:
        def __init__(self): self.d = {}
        def set(self, k, v, ex=None): self.d[k] = v

    with pytest.raises(FileExistsError):
        mp.start_multipart_upload(project_id="p1", user_id="u1", original_name="v.mp4", content_type="video/mp4",
                                  size_bytes=10, s3=S3(True), redis_client=R(), restore_uri=f"s3://b/{key}")
    s3 = S3(False)
    out = mp.start_multipart_upload(project_id="p1", user_id="u1", original_name="v.mp4", content_type="video/mp4",
                                    size_bytes=10, s3=s3, redis_client=R(), restore_uri=f"s3://b/{key}")
    assert s3.created == [key] and out["source_uri"] == f"s3://b/{key}" and out["object_key"] == key


def test_opening_a_cut_marks_its_files_used_and_says_if_a_copy_was_removed(monkeypatch):
    from fastapi.testclient import TestClient
    import cutsell_app.main as api
    snap = {"project_id": "p1", "user_id": "u1", "revision": 1, "draft": {},
            "sources": [{"source_asset_id": "s1", "uri": "s3://b/k1.mp4", "metadata": {"audio_uri": "s3://b/k1.m4a"}},
                        {"source_asset_id": "s2", "uri": "s3://b/k2.mp4"}]}
    monkeypatch.setattr(api, "get_draft_snapshot", lambda **k: json.loads(json.dumps(snap)))
    monkeypatch.setattr(api, "_hydrate_snapshot_assets", lambda s: s)
    monkeypatch.setattr(api, "_source_available", lambda uri: uri != "s3://b/k2.mp4")
    touched = []
    monkeypatch.setattr(sl, "touch", lambda uris, *a, **k: touched.extend(uris))
    out = TestClient(api.app).get("/v1/projects/p1/draft", params={"user_id": "u1"}).json()
    assert [s["available"] for s in out["sources"]] == [True, False]
    assert touched == ["s3://b/k1.mp4", "s3://b/k1.m4a", "s3://b/k2.mp4"]


def test_daily_cleanup_looks_only_at_the_iphone_video_copies_after_24_hours(monkeypatch):
    """PO 7 Nov 2026: 24 h after the last use, only the copies the iPhone can upload again; report only at first."""
    import fakeredis
    now = 1_800_000_000.0
    p = upload_prefix()
    u = "0" * 16 + "/" + "1" * 16 + "/"
    copy_old = p + u + "a" * 32 + "-IMG_1.MOV"
    copy_used = p + u + "b" * 32 + "-IMG_2.mov"
    copy_new = p + u + "c" * 32 + "-IMG_3.mp4"
    keep = [p + u + "d" * 32 + "-voice.m4a",                  # voice file
            p + u + "clean/v2/source-abc.m4a",                 # clean audio
            p + u + "timeline/asset1.mov",                     # editor media
            p + u + "mix-001-A-B.mp4",                         # a Mix video (no original in Photos)
            p + u + "mix/clean-x.mov"]
    objects = {copy_old: (500, _ts(2, now)), copy_used: (400, _ts(5, now)), copy_new: (300, _ts(0.5, now))}
    objects.update({k: (999, _ts(30, now)) for k in keep})
    s3 = FakeS3(objects)
    r = fakeredis.FakeRedis()
    monkeypatch.delenv("CUTSELL_STORAGE_KEEP_HOURS", raising=False)
    monkeypatch.delenv("CUTSELL_STORAGE_LIFECYCLE_DELETE", raising=False)
    sl.touch([f"s3://b/{copy_used}"], r, now=now - 3600)        # used an hour ago
    report = sl.daily(s3, "b", r, now=now)
    assert report["mode"] == "report_only" and report["removed"] == 0 and s3.deleted == []
    assert report["copies"] == 3 and report["unused"] == 1 and report["unused_bytes"] == 500
    assert json.loads(r.get(sl.REPORT_KEY))["unused"] == 1
    monkeypatch.setenv("CUTSELL_STORAGE_LIFECYCLE_DELETE", "1")
    assert sl.daily(s3, "b", r, now=now)["removed"] == 1 and s3.deleted == [copy_old]


def test_any_server_use_of_a_file_counts_as_use(monkeypatch, tmp_path):
    from cutsell_worker import storage
    seen = []
    monkeypatch.setattr(sl, "touch", lambda uris, client=None, now=None: seen.extend(uris))
    monkeypatch.setattr("cutsell_worker.storage_routing.download", lambda b, k, d, client=None: d)
    storage.download_source("s3://b/cutsell/uploads/x/y/z.mov", str(tmp_path / "z.mov"))
    assert seen == ["s3://b/cutsell/uploads/x/y/z.mov"]
