"""Account deletion after the Cloudflare R2 move (Product Owner, 15 Oct 2026): the user's files
from before the move are deleted from the old Amazon bucket too, ONLY that user's prefixes, and
ONLY on account deletion. Deleting a single cut never touches the old bucket."""
from types import SimpleNamespace

import cutsell_worker.account_lifecycle as lifecycle
from cutsell_worker.account_lifecycle import _project_prefixes


class FakeS3:
    def __init__(self, keys):
        self.keys = list(keys)
        self.deleted = []

    def list_objects_v2(self, **kwargs):
        return {"Contents": [{"Key": k} for k in self.keys if k.startswith(kwargs["Prefix"])],
                "IsTruncated": False}

    def delete_objects(self, **kwargs):
        for item in kwargs["Delete"]["Objects"]:
            self.deleted.append(item["Key"])
            self.keys.remove(item["Key"])
        return {}


class FakePipe:
    def __getattr__(self, name):
        return lambda *a, **k: None


class FakeRedis:
    def pipeline(self):
        return FakePipe()

    def scan_iter(self, match):
        return iter(())


def _setup(monkeypatch, fallback=True):
    monkeypatch.setattr(lifecycle, "load_runtime_config",
                        lambda: SimpleNamespace(s3_bucket="cutsell-videos", aws_region="auto",
                                                redis_url="redis://x", database_url=None))
    monkeypatch.setattr(lifecycle, "get_project", lambda **k: {"project_id": k["project_id"]})
    monkeypatch.setattr(lifecycle, "list_projects", lambda **k: [{"project_id": "prj_a"}])
    monkeypatch.setattr(lifecycle, "revoke_all_sessions", lambda **k: {"revoked": 1})
    monkeypatch.setattr(lifecycle, "delete_apple_binding", lambda **k: False)
    monkeypatch.setattr(lifecycle, "delete_project_timeline_assets", lambda **k: {"timeline_assets_deleted": 0})
    if fallback:
        monkeypatch.setenv("FALLBACK_S3_BUCKET", "old-amazon-bucket")
        monkeypatch.setenv("FALLBACK_AWS_ACCESS_KEY_ID", "id")
        monkeypatch.setenv("FALLBACK_AWS_SECRET_ACCESS_KEY", "secret")
    else:
        for name in ("FALLBACK_S3_BUCKET", "FALLBACK_AWS_ACCESS_KEY_ID", "FALLBACK_AWS_SECRET_ACCESS_KEY"):
            monkeypatch.delenv(name, raising=False)


def _keys():
    mine = [p + "f.mp4" for p in _project_prefixes(user_id="usr_me", project_id="prj_a")]
    other = [p + "f.mp4" for p in _project_prefixes(user_id="usr_other", project_id="prj_b")]
    return mine, other


def test_account_deletion_deletes_only_this_users_files_in_the_old_bucket(monkeypatch):
    _setup(monkeypatch)
    mine, other = _keys()
    current, old = FakeS3(mine), FakeS3(mine + other + ["assets/sfx/v1/good.m4a"])
    result = lifecycle.delete_account_data(user_id="usr_me", redis_client=FakeRedis(), s3_client=current,
                                           old_storage_client=old)
    assert sorted(old.deleted) == sorted(mine)
    assert sorted(old.keys) == sorted(other + ["assets/sfx/v1/good.m4a"])
    assert result["old_storage_objects_deleted"] == len(mine)
    assert result["s3_objects_deleted"] == len(mine)


def test_deleting_one_cut_never_touches_the_old_bucket(monkeypatch):
    _setup(monkeypatch)
    mine, _ = _keys()
    old = FakeS3(mine)
    lifecycle.delete_project_data(user_id="usr_me", project_id="prj_a", redis_client=FakeRedis(),
                                  s3_client=FakeS3(mine), old_storage_client=old)
    assert old.deleted == []


def test_no_old_bucket_configured_means_no_old_bucket_calls(monkeypatch):
    _setup(monkeypatch, fallback=False)
    mine, _ = _keys()
    old = FakeS3(mine)
    result = lifecycle.delete_account_data(user_id="usr_me", redis_client=FakeRedis(), s3_client=FakeS3(mine),
                                           old_storage_client=old)
    assert old.deleted == [] and result["old_storage_objects_deleted"] == 0
