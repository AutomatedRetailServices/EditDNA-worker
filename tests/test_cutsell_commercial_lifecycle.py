import os

from cutsell_worker.account_lifecycle import _project_prefixes, _delete_s3_prefix
from cutsell_worker.usage_limits import release_processing_slot, reserve_processing_slot


class FakeConcurrencyRedis:
    def __init__(self):
        self.values = {}

    def eval(self, script, numkeys, key, *args):
        if "INCR" in script:
            limit = int(args[0])
            current = int(self.values.get(key, 0))
            if current >= limit:
                return [0, current]
            current += 1
            self.values[key] = current
            return [1, current]
        current = int(self.values.get(key, 0))
        if current <= 1:
            self.values.pop(key, None)
            return 0
        current -= 1
        self.values[key] = current
        return current


class FakeS3:
    def __init__(self, keys):
        self.keys = list(keys)
        self.deleted = []

    def list_objects_v2(self, **kwargs):
        prefix = kwargs["Prefix"]
        matches = [{"Key": key} for key in self.keys if key.startswith(prefix)]
        return {"Contents": matches, "IsTruncated": False}

    def delete_objects(self, **kwargs):
        for item in kwargs["Delete"]["Objects"]:
            key = item["Key"]
            self.deleted.append(key)
            if key in self.keys:
                self.keys.remove(key)
        return {}


def test_concurrency_slots_enforce_limit_and_release(monkeypatch):
    monkeypatch.setenv("CUTSELL_MAX_CONCURRENT_JOBS_PER_USER", "2")
    client = FakeConcurrencyRedis()
    first = reserve_processing_slot(user_id="usr_test", client=client)
    second = reserve_processing_slot(user_id="usr_test", client=client)
    third = reserve_processing_slot(user_id="usr_test", client=client)
    assert first["allowed"] is True and first["active"] == 1
    assert second["allowed"] is True and second["active"] == 2
    assert third["allowed"] is False and third["status"] == "concurrency_limit"
    assert release_processing_slot(user_id="usr_test", client=client)["active"] == 1
    assert reserve_processing_slot(user_id="usr_test", client=client)["allowed"] is True


def test_project_deletion_prefixes_cover_all_project_media(monkeypatch):
    monkeypatch.setenv("CUTSELL_UPLOAD_PREFIX", "cutsell/uploads/")
    prefixes = _project_prefixes(user_id="usr_1", project_id="prj_1")
    assert len(prefixes) == 4
    assert prefixes[0].startswith("cutsell/uploads/")
    assert prefixes[1].startswith("cutsell/overlay-assets/")
    assert prefixes[2].startswith("cutsell/timeline-assets/")
    assert prefixes[3].startswith("cutsell/exports/")
    assert len(set(prefixes)) == 4


def test_s3_prefix_delete_never_touches_other_project_objects():
    target = "cutsell/uploads/u1/p1/"
    client = FakeS3([
        target + "a.mp4",
        target + "b.mp4",
        "cutsell/uploads/u1/p2/keep.mp4",
    ])
    count = _delete_s3_prefix(client, bucket="bucket", prefix=target)
    assert count == 2
    assert set(client.deleted) == {target + "a.mp4", target + "b.mp4"}
    assert client.keys == ["cutsell/uploads/u1/p2/keep.mp4"]
