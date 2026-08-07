import json

from fastapi.testclient import TestClient

import cutsell_app.project_routes as routes
from cutsell_worker.project_store import create_project, get_project, list_projects, update_project


class FakePipeline:
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


class FakeRedis:
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
        return FakePipeline(self)


def test_project_library_is_scoped_by_user_and_keeps_render_history():
    redis = FakeRedis()
    first = create_project(user_id="u1", title="Lotus Wheel", client=redis)
    second = create_project(user_id="u1", title="Costco Card", client=redis)
    other = create_project(user_id="u2", title="Private", client=redis)

    listed = list_projects(user_id="u1", client=redis)
    assert {item["project_id"] for item in listed} == {first["project_id"], second["project_id"]}
    assert other["project_id"] not in {item["project_id"] for item in listed}

    updated = update_project(
        user_id="u1",
        project_id=first["project_id"],
        state="finished",
        latest_job_id="job-1",
        sources=[{"source_asset_id": "src-1"}],
        render_version={"render_id": "r1", "export_uri": "s3://bucket/final.mp4"},
        client=redis,
    )
    assert updated["state"] == "finished"
    assert updated["latest_job_id"] == "job-1"
    assert updated["render_versions"][0]["render_id"] == "r1"

    try:
        get_project(user_id="u2", project_id=first["project_id"], client=redis)
    except KeyError:
        pass
    else:
        raise AssertionError("cross-user project access must fail")


def test_project_routes_create_list_get_and_rename(monkeypatch):
    records = {}
    def fake_create(**kwargs):
        record = {"project_id": "prj_1", "user_id": kwargs["user_id"], "title": kwargs.get("title") or "Untitled Cut", "state": "created"}
        records["prj_1"] = record
        return dict(record)
    monkeypatch.setattr(routes, "create_project", fake_create)
    monkeypatch.setattr(routes, "list_projects", lambda **kwargs: [dict(records["prj_1"])])
    monkeypatch.setattr(routes, "get_project", lambda **kwargs: dict(records[kwargs["project_id"]]))
    def fake_update(**kwargs):
        records[kwargs["project_id"]]["title"] = kwargs["title"]
        return dict(records[kwargs["project_id"]])
    monkeypatch.setattr(routes, "update_project", fake_update)

    client = TestClient(routes.router)
    # APIRouter alone is not an ASGI app; mount it in a minimal FastAPI instance.
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)

    created = client.post("/v1/projects", json={"user_id": "u1", "title": "My Cut"})
    assert created.status_code == 200 and created.json()["project_id"] == "prj_1"
    listed = client.get("/v1/projects?user_id=u1")
    assert listed.status_code == 200 and len(listed.json()["projects"]) == 1
    fetched = client.get("/v1/projects/prj_1?user_id=u1")
    assert fetched.status_code == 200 and fetched.json()["title"] == "My Cut"
    renamed = client.patch("/v1/projects/prj_1", json={"user_id": "u1", "title": "Renamed"})
    assert renamed.status_code == 200 and renamed.json()["title"] == "Renamed"
